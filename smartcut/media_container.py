from dataclasses import dataclass, field
from fractions import Fraction
from itertools import chain
import av
import av.stream
import av.video
import numpy as np

def ts_to_time(ts):
    return Fraction(round(ts*1000), 1000)

@dataclass
class AudioTrack():
    av_stream: av.stream.Stream
    audio_load_stream: av.stream.Stream
    path: str
    index: int

    index_in_source: int = 0

    packets: list[av.Packet] = field(default_factory = lambda: [])
    frame_times: np.array = field(default_factory = lambda: [])
    pts_to_samples: dict = field(default_factory = lambda: {})

    controls: object = None
    error_msg: str = None
    audio_16k: np.array = None

    duration: Fraction = None
    shift: float = 0.0
    max_tree: np.array = None

    level_ignoring_mute: float = None
    muted: bool = None

    def selected_for_transcript(self):
        return self.controls is None or self.controls.transcript_button.isChecked()

class MediaContainer:
    av_containers: list[av.container.Container]
    video_stream: av.video.stream.VideoStream | None
    path: str

    eof_time: Fraction

    video_stream: av.stream.Stream | None

    video_frame_times: np.ndarray
    video_keyframe_indices: list[int]
    gop_start_times: np.ndarray # Smallest pts in a GOP

    audio_tracks: list[AudioTrack]

    chat_url: str | None
    chat_history: np.ndarray | None
    chat_cumsum: np.ndarray | None
    chat_visualize: bool

    def __init__(self, path) -> None:
        self.path = path

        frame_pts = []
        self.video_keyframe_indices = []

        est_eof_time = 0
        av_container = av.open(path, 'r', metadata_errors='ignore')
        audio_loading_container = av.open(path, 'r', metadata_errors='ignore')
        self.av_containers = [av_container, audio_loading_container]

        self.chat_url = None
        self.chat_history = None
        self.chat_visualize = True

        if len(av_container.streams.video) == 0:
            self.video_stream = None
            streams = av_container.streams.audio
        else:
            self.video_stream = av_container.streams.video[0]
            self.video_stream.thread_type = "FRAME"
            streams = [self.video_stream] + list(av_container.streams.audio)

        self.audio_tracks = []
        stream_index_to_audio_track = {}
        for i, (a_s, loading_s) in enumerate(zip(av_container.streams.audio, audio_loading_container.streams.audio)):
            a_s.thread_type = "FRAME"
            loading_s.thread_type = "FRAME"
            track = AudioTrack(a_s, loading_s, path, i, i)
            self.audio_tracks.append(track)
            stream_index_to_audio_track[a_s.index] = track

        video_keyframe_indices = []

        for packet in av_container.demux(streams):
            if packet.pts is None:
                continue
            est_eof_time = max(est_eof_time, (packet.pts + packet.duration) * packet.time_base)
            if packet.stream.type == 'video':
                if packet.is_keyframe:
                    video_keyframe_indices.append(len(frame_pts))

                frame_pts.append(packet.pts)
            else:
                track = stream_index_to_audio_track[packet.stream_index]
                track.last_packet = packet

                # NOTE: storing the audio packets like this keeps the whole compressed audio loaded in RAM
                track.packets.append(packet)
                track.frame_times.append(packet.pts)

        self.eof_time = est_eof_time

        if self.video_stream is not None:
            self.video_frame_times = np.sort(np.array(frame_pts)) * self.video_stream.time_base

            self.gop_start_times = self.video_frame_times[video_keyframe_indices]

        for t in self.audio_tracks:
            frame_times = np.array(t.frame_times)
            t.frame_times = frame_times * t.av_stream.time_base
            # last_packet = t.packets[-1]
            last_packet = t.last_packet
            t.duration = (last_packet.pts + last_packet.duration) * last_packet.time_base

    def close(self):
        for c in self.av_containers:
            c.close()

    def get_video_frame_from_time(self, t):
        idx = np.searchsorted(self.video_frame_times, t)
        if idx == len(self.video_frame_times):
            return self.eof_time, idx
        elif idx == 0:
            return self.video_frame_times[0], 0
        # Otherwise, find the closest of the two possible candidates: arr[idx-1] and arr[idx]
        else:
            prev_val = self.video_frame_times[idx - 1]
            next_val = self.video_frame_times[idx]
            if t - prev_val <= next_val - t:
                return prev_val, idx - 1
            else:
                return next_val, idx

    def get_video_frame_from_ts(self, ts):
        return self.get_video_frame_from_time(ts_to_time(ts))

    def add_audio_file(self, path):
        av_container = av.open(path, 'r', metadata_errors='ignore')
        self.av_containers.append(av_container)
        audio_load_container = av.open(path, 'r', metadata_errors='ignore')
        self.av_containers.append(audio_load_container)
        idx = 0
        stream = av_container.streams.audio[idx]
        stream.thread_type = "FRAME"
        audio_load_stream = audio_load_container.streams.audio[idx]
        audio_load_stream.thread_type = "FRAME"
        track = AudioTrack(stream, audio_load_stream, path, len(self.audio_tracks), 0)
        self.audio_tracks.append(track)

        est_eof_time = 0
        for packet in av_container.demux(stream):
            if packet.pts is None:
                continue
            est_eof_time = max(est_eof_time, (packet.pts + packet.duration) * packet.time_base)
            track.packets.append(packet)
            track.frame_times.append(packet.pts)

        if self.video_stream is None:
            self.eof_time = max(self.eof_time, est_eof_time)

        track.frame_times = np.array(track.frame_times)
        track.frame_times = track.frame_times * stream.time_base
        last_packet = track.packets[-1]
        track.duration = (last_packet.pts + last_packet.duration) * last_packet.time_base
        return track

class AudioReader:
    def __init__(self, track: AudioTrack, use_loading_stream: bool = False):
        self.track = track
        if use_loading_stream:
            self.stream = track.audio_load_stream
        else:
            self.stream = track.av_stream

        self.rate = self.stream.rate
        self.codec = self.stream.codec_context

        self.cache_time = -1
        self.packet_i = 0
        self.resampler = None

    def read(self, start, end) -> np.ndarray:
        dur = end - start
        buffer = np.zeros((round(self.stream.rate * dur), self.stream.channels), np.float32)
        start_in_samples = round(start * self.stream.rate)
        end_in_samples = start_in_samples + buffer.shape[0]

        # Decode 1 sec extra.
        # NOTE: TODO: This could be lower, but does it matter?
        start = np.searchsorted(self.track.frame_times, start - 1)

        self.codec.flush_buffers()

        first = True
        sample_pos = 0
        time_pos = -100
        for p in chain(self.track.packets[start:], [None]):
            # print(p)
            for f in self.codec.decode(p):
                # print(f)

                # NOTE: Why is time_base so *#!*
                # time_base = f.time_base if f.time_base is not None else self.stream.time_base
                f_start = f.pts * self.stream.time_base
                f_end = f_start + Fraction(f.samples, f.sample_rate)
                # print(start, f_start, f_end, time_base)
                # exit()

                if first:
                    if f.pts in self.track.pts_to_samples:
                        sample_pos = self.track.pts_to_samples[f.pts]
                    else:
                        sample_pos = round(f_start * f.sample_rate)
                    first = False
                elif abs(time_pos - f_start) > 0.02:
                    print('Skipping a gap in audio pts', time_pos, f_start)
                    if f.pts in self.track.pts_to_samples:
                        sample_pos = self.track.pts_to_samples[f.pts]
                    else:
                        sample_pos = round(f_start * f.sample_rate)

                self.track.pts_to_samples[f.pts] = sample_pos
                if sample_pos >= end_in_samples:
                    break

                if sample_pos + f.samples > start_in_samples:
                    if f.format.name != 'fltp':
                        if self.resampler is None:
                            self.resampler = av.AudioResampler('fltp', f.layout, f.rate)
                        frames = self.resampler.resample(f)
                        # frames.extend(self.resampler.resample(None))
                        data = [rsf.to_ndarray() for rsf in frames]
                        decoded = np.concatenate(data, axis=-1)
                    else:
                        decoded = f.to_ndarray()
                    decoded = decoded.T
                    if sample_pos < start_in_samples:
                        decoded = decoded[start_in_samples - sample_pos:]
                        sample_pos = start_in_samples

                    l = min(end_in_samples - sample_pos, decoded.shape[0])
                    buffer_pos = sample_pos - start_in_samples
                    buffer[buffer_pos:buffer_pos+l] = decoded[:l]
                    sample_pos += decoded.shape[0]
                else:
                    sample_pos += f.samples
                time_pos = f_end
            else: # Break from the inner loop
                continue
            break

        return buffer.T

def layout_from_channels(channels):
    if channels == 1:
        return 'mono'
    if channels == 2:
        return 'stereo'
    if channels == 6:
        return '5.1'
    raise AssertionError("Invalid audio track layout. Some audio formats are not supported.")

def channels_from_layout(layout):
    match layout:
        case 'mono':
            return 1
        case 'stereo':
            return 2
        case '5.1':
            return 6
        case _:
            raise ValueError

def upmix(audio: np.array):
    if audio.shape[0] == 1:
        return np.repeat(audio, 2, axis=0)
    elif audio.shape[0] == 2:
        return np.concatenate([audio, np.zeros((4, audio.shape[0]))])
    raise ValueError

def downmix(audio: np.array):
    if audio.shape[0] == 2:
        return np.mean(audio, axis=0, keepdims=True)
    elif audio.shape[0] == 6:
        # 5.1 audio
        stereo = np.zeros((2, audio.shape[1]), dtype=np.float32)
        # Channels order: FL, FR, C, LFE, SL, SR
        # LFE is removed. Alternatively it could be mixed in at 0.1

        surround_channel_weight = 0.25
        center_weight = 0.5

        stereo[0] = audio[0] + audio[2] * center_weight + audio[4] * surround_channel_weight
        stereo[1] = audio[1] + audio[2] * center_weight + audio[5] * surround_channel_weight
        return stereo
    raise ValueError

def channel_conversion(audio: np.array, layout):
    c = audio.shape[0]
    target = channels_from_layout(layout)
    if c == target:
        return audio

    while c < target:
        audio = upmix(audio)
        c = audio.shape[0]
    while c > target:
        audio = downmix(audio)
        c = audio.shape[0]

    return audio
