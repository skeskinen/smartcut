from enum import Enum
from fractions import Fraction

from math import e
import os
from typing import List
from dataclasses import dataclass, field
import av
import av.bitstream
import av.container
import numpy as np

from smartcut.media_container import MediaContainer

from smartcut.misc_data import AudioExportInfo, AudioExportSettings, CutSegment, MixInfo, VideoTransform, VideoViewTransform, WatermarkView

try:
    from smc.audio_handling import MixAudioCutter, RecodeTrackAudioCutter
except ImportError:
    pass

class CancelObject:
    cancelled: bool = False

def is_annexb(packet):
        data = bytes(packet)
        return data[:3] == b'\0\0\x01' or data[:4] == b'\0\0\0\x01'

def copy_packet(p: av.packet.Packet) -> av.packet.Packet:
    # return p
    packet = av.packet.Packet(p)
    packet.pts = p.pts
    packet.dts = p.dts
    packet.duration = p.duration
    # packet.pos = p.pos
    packet.time_base = p.time_base
    packet.stream = p.stream
    packet.is_keyframe = p.is_keyframe
    # packet.is_discard = p.is_discard

    return packet

def make_cut_segments(media_container: MediaContainer,
        positive_segments: List[tuple[Fraction, Fraction]],
        keyframe_mode: bool = False
        ) -> List[CutSegment]:
    cut_segments = []
    if media_container.video_stream is None:
        for p in positive_segments:
            s = p[0]
            while s + 5 < p[1]:
                cut_segments.append(CutSegment(False, s, s + 4))
                s += 4
            cut_segments.append(CutSegment(False, s, p[1]))
        return cut_segments

    source_cutpoints = list(media_container.gop_start_times) + [media_container.eof_time]
    p = 0
    for (i, o) in zip(source_cutpoints[:-1], source_cutpoints[1:]):
        while p < len(positive_segments) and positive_segments[p][1] <= i:
            p += 1

        # Three cases: no overlap, complete overlap, and partial overlap
        if p == len(positive_segments) or o <= positive_segments[p][0]:
            pass
        elif keyframe_mode or (i >= positive_segments[p][0] and o <= positive_segments[p][1]):
            cut_segments.append(CutSegment(False, i, o))
        else:
            if i > positive_segments[p][0]:
                cut_segments.append(CutSegment(True, i, positive_segments[p][1]))
                p += 1
            while p < len(positive_segments) and positive_segments[p][1] < o:
                cut_segments.append(CutSegment(True, positive_segments[p][0], positive_segments[p][1]))
                p += 1
            if p < len(positive_segments) and positive_segments[p][0] < o:
                cut_segments.append(CutSegment(True, positive_segments[p][0], o))

    return cut_segments

class PassthruAudioCutter:
    def __init__(self, media_container: MediaContainer, output_av_container: av.container.Container,
                track_index: int, export_settings: AudioExportSettings):
        self.track = media_container.audio_tracks[track_index]
        self.out_stream = output_av_container.add_stream(template=self.track.av_stream)
        self.out_stream.metadata.update(self.track.av_stream.metadata)
        self.segment_start_in_output = 0
        self.prev_dts = -100_000
        self.prev_pts = -100_000

    def segment(self, cut_segment: CutSegment) -> list[av.Packet]:
        if cut_segment.start_time <= 0:
            start = 0
        else:
            start = np.searchsorted(self.track.frame_times, cut_segment.start_time)
        end = np.searchsorted(self.track.frame_times, cut_segment.end_time)
        in_packets = self.track.packets[start : end]

        segment_start_pts = int(cut_segment.start_time / self.track.av_stream.time_base)

        packets = []
        for p in in_packets:
            packet = copy_packet(p)
            # packet = p
            packet.stream = self.out_stream
            packet.pts = int(packet.pts - segment_start_pts + self.segment_start_in_output)
            packet.dts = int(packet.dts - segment_start_pts + self.segment_start_in_output)
            if packet.pts <= self.prev_pts:
                print("Correcting for too low pts in audio passthru")
                packet.pts = self.prev_pts + 1
            if packet.dts <= self.prev_dts:
                print("Correcting for too low dts in audio passthru")
                packet.dts = self.prev_dts + 1
            self.prev_pts = packet.pts
            self.prev_dts = packet.dts
            packets.append(packet)

        segment_duration = cut_segment.end_time - cut_segment.start_time
        # NOTE: Packet timestamps are still in input time_base
        self.segment_start_in_output += segment_duration / self.track.av_stream.time_base
        return packets

    def finish(self):
        return []

class VideoExportMode(Enum):
    SMARTCUT = 1
    KEYFRAMES = 2
    RECODE = 3

class VideoExportQuality(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    INDISTINGUISHABLE = 4
    NEAR_LOSSLESS = 5
    LOSSLESS = 6

@dataclass
class VideoSettings:
    mode: VideoExportMode
    quality: VideoExportQuality
    transform: VideoTransform
    codec_override: str = 'copy'

class VideoCutter:
    def __init__(self, media_container, output_av_container, video_settings: VideoSettings, log_level):
        self.media_container = media_container
        self.log_level = log_level
        self.encoder_inited = False
        self.video_settings = video_settings
        self.transform_graph = None

        self.enc_codec = None

        self.in_stream = media_container.video_stream
        self.input_av_container: av.container.Container = self.in_stream.container

        if video_settings.mode == VideoExportMode.RECODE and video_settings.codec_override != 'copy':
            self.out_stream = output_av_container.add_stream(video_settings.codec_override, rate=self.in_stream.guessed_rate)
            self.out_stream.width = self.in_stream.width
            self.out_stream.height = self.in_stream.height
            self.codec_name = video_settings.codec_override

            self.init_encoder()
            self.enc_codec = self.out_stream.codec_context
            self.enc_codec.options.update(self.encoding_options)
            self.enc_codec.thread_type = "FRAME"
        else:
            self.out_stream = output_av_container.add_stream(template=self.in_stream)
            self.codec_name = self.in_stream.codec_context.name

            self.remux_bitstream_filter = av.bitstream.BitStreamFilterContext('null', self.in_stream, self.out_stream)
            if self.in_stream.codec_context.name == 'h264':
                if not is_annexb(self.in_stream.codec_context.extradata):
                    self.remux_bitstream_filter = av.bitstream.BitStreamFilterContext('h264_mp4toannexb', self.in_stream, self.out_stream)
            elif self.in_stream.codec_context.name == 'hevc':
                if not is_annexb(self.in_stream.codec_context.extradata):
                    self.remux_bitstream_filter = av.bitstream.BitStreamFilterContext('hevc_mp4toannexb', self.in_stream, self.out_stream)

        if video_settings.transform is not None:
            res = video_settings.transform.resolution
            self.out_stream.width = res[0]
            self.out_stream.height = res[1]

        self.last_dts = -100_000_000

        self.segment_start_in_output = 0

    def init_encoder(self):
        self.encoder_inited = True
        # v_codec = self.in_stream.codec_context
        profile = self.out_stream.codec_context.profile

        if 'av1' in self.codec_name:
            self.codec_name = 'av1'
            profile = None
        if 'vp9' == self.codec_name:
            if profile is not None:
                profile = profile[-1:]
                if int(profile) > 1:
                    raise ValueError("VP9 Profile 2 and Profile 3 are not supported by the encoder. Please select cutting on keyframes mode.")
        elif profile is not None:
            if 'Baseline' in profile:
                profile = 'baseline'
            elif 'High 4:4:4' in profile:
                profile = 'high444'
            elif 'Rext' in profile: # This is some sort of h265 extension. This might be the source of some issues I've had?
                profile = None
            else:
                profile = profile.lower().replace(':', '').replace(' ', '')

        # NOTE: crf 11 is a really high quality. I think this is ok because we typically only encode short segments.
        # I didn't see a nice way to get the original quality settings from the input video stream.
        # NOTE: 11 is too much. Try 16 for now

        crf_value = 23
        match self.video_settings.quality:
            case VideoExportQuality.LOW:
                crf_value = 23
            case VideoExportQuality.NORMAL:
                crf_value = 18
            case VideoExportQuality.HIGH:
                crf_value = 14
            case VideoExportQuality.INDISTINGUISHABLE:
                crf_value = 8
            case VideoExportQuality.NEAR_LOSSLESS:
                crf_value = 3

        if self.codec_name in ['hevc', 'av1', 'vp9']:
            crf_value += 4
        if self.video_settings.quality == VideoExportQuality.LOSSLESS:
            crf_value = 0

        self.encoding_options = {'crf': str(crf_value)}
        if self.codec_name == 'vp9' and self.video_settings.quality == VideoExportQuality.LOSSLESS:
            self.encoding_options['lossless'] = '1'
        # encoding_options = {}
        if profile is not None:
            self.encoding_options['profile'] = profile

        if self.codec_name == 'hevc':
            # Get the encoder settings from input stream extradata.
            # In theory this should not work. The stuff in extradata is technically just comments set by the encoder.
            # Another issue is that the extradata format is going to be different depending on the encoder.
            # So this will likely only work if the input stream is encoded with x265 ¯\_(ツ)_/¯
            # However, this does make the testcases from fails -> passes.
            # And I've tested that it works on some real videos as well.
            # Maybe there is some option that I'm not setting correctly and there is a better way to get the correct value?

            extradata = self.in_stream.codec_context.extradata
            x265_params = []
            try:
                options_str = str(extradata.split(b'options: ')[1][:-1], 'ascii')
                x265_params = options_str.split(' ')
                for i, o in enumerate(x265_params):
                    if ':' in o:
                        x265_params[i] = o.replace(':', ',')
                    if not '=' in o:
                        x265_params[i] = o + '=1'
            except:
                pass

            # Repeat headers. This should be the same as `global_headers = False`,
            # but for some reason setting this explicitly is necessary with x265.
            x265_params.append('repeat-headers=1')

            if self.log_level is not None:
                x265_params.append(f'log_level={self.log_level}')

            if self.video_settings.quality == VideoExportQuality.LOSSLESS:
                x265_params.append('lossless=1')

            self.encoding_options['x265-params'] = ':'.join(x265_params)

        if self.video_settings.transform is not None:
            transform = self.video_settings.transform
            views = transform.views
            n = len(views)
            n_transform = len([v for v in views if isinstance(v, VideoViewTransform)])

            res_w = transform.resolution[0]
            res_h = transform.resolution[1]

            graph = self.transform_graph = av.filter.Graph()
            src_buf = graph.add_buffer(template=self.in_stream)
            split = graph.add("split", f'{n_transform+1}')
            src_buf.link_to(split)

            bg_crop = graph.add("crop", "ih*9/16:ih")
            base = bg_scale = graph.add("scale", f"{res_w}:{res_h}")

            split.link_to(bg_crop, n_transform)
            bg_crop.link_to(bg_scale)

            for i, view in enumerate(views):
                if isinstance(view, VideoViewTransform):
                    crop_w = f'in_w*{view.input_w}'
                    crop_h = f'in_h*{view.input_h}'
                    crop_x = f'in_w*{view.input_x}'
                    crop_y = f'in_h*{view.input_y}'

                    view_crop = graph.add("crop", f"{crop_w}:{crop_h}:{crop_x}:{crop_y}")
                    view_scale = graph.add("scale", f"{res_w}:-1")
                    view_overlay = graph.add("overlay", f"0:{int(res_h * view.output_y)}")

                    split.link_to(view_crop, i)
                    view_crop.link_to(view_scale)
                    base.link_to(view_overlay)
                    view_scale.link_to(view_overlay, input_idx=1)

                    base = view_overlay
                elif isinstance(view, WatermarkView):
                    path = view.path
                    movie = graph.add("movie", f"filename='{path}'")
                    watermark_overlay = graph.add("overlay", f"{int(res_w * view.output_x)}:{int(res_h * view.output_y)}")
                    base.link_to(watermark_overlay)
                    movie.link_to(watermark_overlay, input_idx=1)

                    base = watermark_overlay

            sink = graph.add("buffersink")
            base.link_to(sink)
            graph.configure()

    def segment(self, cut_segment: CutSegment) -> list[av.Packet]:
        self.out_time_base = self.out_stream.time_base

        if cut_segment.require_recode:
            packets = self.recode_segment(cut_segment)
        else:
            packets = self.flush_encoder()
            packets.extend(self.remux_segment(cut_segment))
        segment_duration = cut_segment.end_time - cut_segment.start_time
        self.segment_start_in_output += segment_duration / self.out_stream.time_base

        for packet in packets:
            packet.stream = self.out_stream
            if packet.dts is not None:
                if packet.dts <= self.last_dts:
                    packet.dts = self.last_dts + 1
                self.last_dts = packet.dts
            else:
                packet.dts = self.last_dts + 1
                self.last_dts = packet.dts
        return packets

    def finish(self):
        packets = self.flush_encoder()
        for packet in packets:
            packet.stream = self.out_stream
            if packet.dts is not None:
                if packet.dts <= self.last_dts:
                    packet.dts = self.last_dts + 1
                self.last_dts = packet.dts
            else:
                packet.dts = self.last_dts + 1
                self.last_dts = packet.dts

        return packets

    def recode_segment(self, s: CutSegment) -> list[av.Packet]:
        if not self.encoder_inited:
            self.init_encoder()
        result_packets = []
        segment_start_pts = int(s.start_time / self.in_stream.time_base)
        segment_end_pts = int(s.end_time / self.in_stream.time_base)

        # Fallback to always reset the encoder
        # result_packets = self.flush_encoder()
        # self.enc_codec = None

        if self.enc_codec is None:
            muxing_codec = self.out_stream.codec_context
            enc_codec = av.CodecContext.create(self.codec_name, 'w')
            enc_codec.rate = muxing_codec.rate
            enc_codec.options.update(self.encoding_options)

            enc_codec.width = muxing_codec.width
            enc_codec.height = muxing_codec.height
            enc_codec.pix_fmt = muxing_codec.pix_fmt
            enc_codec.time_base = self.out_stream.time_base
            enc_codec.flags = muxing_codec.flags
            enc_codec.global_header = False # Force writing of headers to the output stream
            if muxing_codec.bit_rate is not None:
                enc_codec.bit_rate = muxing_codec.bit_rate
            if muxing_codec.bit_rate_tolerance is not None:
                enc_codec.bit_rate_tolerance = muxing_codec.bit_rate_tolerance
            enc_codec.codec_tag = muxing_codec.codec_tag
            enc_codec.thread_type = "FRAME"
            self.enc_codec = enc_codec

        self.input_av_container.seek(segment_start_pts, stream=self.in_stream)

        for frame in self.input_av_container.decode(self.in_stream):
            if frame.pts < segment_start_pts:
                continue
            if frame.pts >= segment_end_pts:
                break

            if self.transform_graph is not None:
                self.transform_graph.vpush(frame)
                frame = self.transform_graph.vpull()

            frame.pts -= segment_start_pts

            frame.pts = frame.pts * self.in_stream.time_base / self.out_time_base
            frame.time_base = self.out_time_base
            frame.pts += self.segment_start_in_output

            frame.pict_type = av.video.frame.PictureType.NONE
            result_packets.extend(self.enc_codec.encode(frame))

        return result_packets

    def flush_encoder(self):
        if self.enc_codec is None:
            return []

        r = self.enc_codec.encode()
        self.enc_codec = None
        return r

    def remux_segment(self, s: CutSegment) -> list[av.Packet]:
        result_packets = []
        segment_start_pts = int(s.start_time / self.in_stream.time_base)
        segment_end_pts = int(s.end_time / self.in_stream.time_base)

        # print(f"seeking to {segment_start_pts}")
        self.input_av_container.seek(segment_start_pts, stream=self.in_stream)

        for packet in self.input_av_container.demux(self.in_stream):
            # packet = copy_packet(p)
            # print("in packet:", packet)

            if packet.pts is None or packet.pts >= segment_end_pts:
                break

            if packet.pts < segment_start_pts:
                print("Skipping video packets. Seeking to wrong spot?")
                continue

            packet.pts -= segment_start_pts
            packet.pts = packet.pts * self.in_stream.time_base / self.out_time_base
            packet.pts += self.segment_start_in_output
            if packet.dts is not None:
                packet.dts -= segment_start_pts
                packet.dts = packet.dts * self.in_stream.time_base / self.out_time_base
                packet.dts += self.segment_start_in_output

            result_packets.extend(self.remux_bitstream_filter.filter(packet))

        result_packets.extend(self.remux_bitstream_filter.filter(None))

        self.remux_bitstream_filter.flush()
        return result_packets

def smart_cut(media_container: MediaContainer, positive_segments: List[tuple[Fraction, Fraction]],
              out_path: str, audio_export_info: AudioExportInfo = None, log_level = None, progress = None,
              video_settings=None, segment_mode=False, cancel_object: CancelObject | None = None):
    if video_settings is None:
        video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.NORMAL, None)

    cut_segments = make_cut_segments(media_container, positive_segments, video_settings.mode == VideoExportMode.KEYFRAMES)

    if video_settings.mode == VideoExportMode.RECODE:
        for c in cut_segments:
            c.require_recode = True

    if segment_mode:
        output_files = []
        padding = len(str(len(positive_segments)))
        for i, s in enumerate(positive_segments):
            segment_index = str(i + 1).zfill(padding)  # Zero-pad the segment index
            if "#" in out_path:
                pound_index = out_path.rfind("#")
                output_file = out_path[:pound_index] + segment_index + out_path[pound_index + 1:]
            else:
                # Insert the segment index right before the last '.'
                dot_index = out_path.rfind(".")
                if dot_index != -1:
                    output_file = out_path[:dot_index] + segment_index + out_path[dot_index:]
                else:
                    output_file = f"{out_path}{segment_index}"

            output_files.append((output_file, s))

    else:
        output_files = [(out_path, positive_segments[-1])]
    previously_done_segments = 0
    for output_path_segment in output_files:
        if cancel_object is not None and cancel_object.cancelled:
            break
        with av.open(output_path_segment[0], 'w') as output_av_container:

            include_video = True
            if output_av_container.format.name in ['ogg', 'mp3', 'm4a', 'ipod', 'flac', 'wav']: #ipod is the real name for m4a, I guess
                include_video = False
            generators = []
            if media_container.video_stream is not None and include_video:
                generators.append(VideoCutter(media_container, output_av_container, video_settings, log_level))

            if audio_export_info is not None:
                if audio_export_info.mix_export_settings is not None:
                    generators.append(MixAudioCutter(media_container, output_av_container,
                                                    audio_export_info.mix_info, audio_export_info.mix_export_settings))
                for track_i, track_export_settings in enumerate(audio_export_info.output_tracks):
                    if track_export_settings is not None:
                        if track_export_settings.codec == 'passthru':
                            generators.append(PassthruAudioCutter(media_container, output_av_container, track_i, track_export_settings))
                        else:
                            generators.append(RecodeTrackAudioCutter(media_container, output_av_container, track_i, track_export_settings))

            output_av_container.start_encoding()
            if progress is not None:
                progress.emit(len(cut_segments))
            for s in cut_segments[previously_done_segments:]:
                if cancel_object is not None and cancel_object.cancelled:
                    break
                if s.start_time >= output_path_segment[1][1]: # Go to the next output file
                    break

                if progress is not None:
                    progress.emit(previously_done_segments)
                previously_done_segments += 1
                assert s.start_time < s.end_time
                for g in generators:
                    for packet in g.segment(s):
                        # if isinstance(g, VideoCutter):
                        # print(packet)
                        output_av_container.mux(packet)
            for g in generators:
                for packet in g.finish():
                    # if isinstance(g, VideoCutter):
                    # print("finish packet: ", packet)
                    output_av_container.mux(packet)
            if progress is not None:
                progress.emit(previously_done_segments)

        if cancel_object is not None and cancel_object.cancelled:
            last_file_path = output_path_segment[0]

            if os.path.exists(last_file_path):
                os.remove(last_file_path)
