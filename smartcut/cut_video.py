from enum import Enum
from fractions import Fraction
import heapq

from math import e
import os
from typing import List
from dataclasses import dataclass, field
import av
import av.bitstream
import av.container
import numpy as np

from smartcut.media_container import MediaContainer
from smartcut.nal_tools import convert_hevc_cra_to_bla
from smartcut.media_utils import VideoExportMode, VideoExportQuality, get_crf_for_quality

from smartcut.misc_data import AudioExportInfo, AudioExportSettings, CutSegment, MixInfo

try:
    from smc.audio_handling import MixAudioCutter, RecodeTrackAudioCutter
except ImportError:
    pass

class CancelObject:
    cancelled: bool = False

@dataclass
class FrameHeapItem:
    """Wrapper for frames in the heap, sorted by PTS"""
    pts: int
    frame: 'av.VideoFrame'

    def __lt__(self, other):
        # Handle None PTS values by treating them as -1 (earliest)
        self_pts = self.pts if self.pts is not None else -1
        other_pts = other.pts if other.pts is not None else -1
        return self_pts < other_pts

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

    source_cutpoints = media_container.gop_start_times_pts_s + [media_container.eof_time]
    p = 0
    for gop_idx, (i, o, i_dts, o_dts) in enumerate(zip(source_cutpoints[:-1], source_cutpoints[1:], media_container.gop_start_times_dts, media_container.gop_end_times_dts)):
        while p < len(positive_segments) and positive_segments[p][1] <= i:
            p += 1

        # Three cases: no overlap, complete overlap, and partial overlap
        if p == len(positive_segments) or o <= positive_segments[p][0]:
            pass
        elif keyframe_mode or (i >= positive_segments[p][0] and o <= positive_segments[p][1]):
            cut_segments.append(CutSegment(False, i, o, i_dts, o_dts, gop_idx))
        else:
            if i > positive_segments[p][0]:
                cut_segments.append(CutSegment(True, i, positive_segments[p][1], i_dts, o_dts, gop_idx))
                p += 1
            while p < len(positive_segments) and positive_segments[p][1] < o:
                cut_segments.append(CutSegment(True, positive_segments[p][0], positive_segments[p][1], i_dts, o_dts, gop_idx))
                p += 1
            if p < len(positive_segments) and positive_segments[p][0] < o:
                cut_segments.append(CutSegment(True, positive_segments[p][0], o, i_dts, o_dts, gop_idx))

    return cut_segments

def map_codec_name_for_pyav15(codec_name: str) -> str:
    """Map decoder names to encoder names for PyAV 15.0"""
    codec_mapping = {
        'libdav1d': 'libaom-av1',  # AV1 decoder to encoder
    }
    return codec_mapping.get(codec_name, codec_name)

class PassthruAudioCutter:
    def __init__(self, media_container: MediaContainer, output_av_container: av.container.Container,
                track_index: int, export_settings: AudioExportSettings):
        self.track = media_container.audio_tracks[track_index]

        self.out_stream = output_av_container.add_stream_from_template(self.track.av_stream, options={'x265-params': 'log_level=error'})

        self.out_stream.metadata.update(self.track.av_stream.metadata)
        self.out_stream.disposition = self.track.av_stream.disposition.value
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

        in_tb = self.track.av_stream.time_base
        packets = []
        for p in in_packets:
            packet = copy_packet(p)
            # packet = p
            packet.stream = self.out_stream
            packet.pts = int(packet.pts + (self.segment_start_in_output - cut_segment.start_time) / in_tb)
            packet.dts = int(packet.dts + (self.segment_start_in_output - cut_segment.start_time) / in_tb)
            if packet.pts <= self.prev_pts:
                print("Correcting for too low pts in audio passthru")
                packet.pts = self.prev_pts + 1
            if packet.dts <= self.prev_dts:
                print("Correcting for too low dts in audio passthru")
                packet.dts = self.prev_dts + 1
            self.prev_pts = packet.pts
            self.prev_dts = packet.dts
            packets.append(packet)

        self.segment_start_in_output += cut_segment.end_time - cut_segment.start_time
        return packets

    def finish(self):
        return []

class SubtitleCutter:
    def __init__(self, media_container: MediaContainer, output_av_container: av.container.Container, subtitle_track_index: int):
        self.track_i = subtitle_track_index
        self.packets = media_container.subtitle_tracks[subtitle_track_index]

        self.in_stream = media_container.av_containers[0].streams.subtitles[subtitle_track_index]
        self.out_stream = output_av_container.add_stream_from_template(self.in_stream)
        self.out_stream.metadata.update(self.in_stream.metadata)
        self.out_stream.disposition = self.in_stream.disposition.value
        self.segment_start_in_output = 0
        self.prev_pts = -100_000

        self.current_packet_i = 0

    def segment(self, cut_segment: CutSegment) -> list[av.Packet]:
        segment_start_pts = int(cut_segment.start_time / self.in_stream.time_base)
        segment_end_pts = int(cut_segment.end_time / self.in_stream.time_base)

        out_packets = []

        # TODO: This is the simplest implementation of subtitle cutting. Investigate more complex logic.
        # We include subtitles for the whole original time if the subtitle start time is included in the output
        # Good: simple, Bad: 1) if start is cut it's not shown at all 2) we can show a subtitle for too long if there is cut after it's shown
        while self.current_packet_i < len(self.packets):
            p = self.packets[self.current_packet_i]
            if p.pts < segment_start_pts:
                self.current_packet_i += 1
            elif p.pts >= segment_start_pts and p.pts < segment_end_pts:
                out_packets.append(p)
                self.current_packet_i += 1
            else:
                break

        for packet in out_packets:
            packet.stream = self.out_stream
            packet.pts = int(packet.pts - segment_start_pts + self.segment_start_in_output / self.in_stream.time_base)

            if packet.pts < self.prev_pts:
                print("Correcting for too low pts in subtitle passthru. This should not happen.")
                packet.pts = self.prev_pts + 1
            packet.dts = packet.pts
            self.prev_pts = packet.pts
            self.prev_dts = packet.dts

        self.segment_start_in_output += cut_segment.end_time - cut_segment.start_time
        return out_packets

    def finish(self):
        return []



@dataclass
class VideoSettings:
    mode: VideoExportMode
    quality: VideoExportQuality
    codec_override: str = 'copy'

class VideoCutter:
    def __init__(self, media_container, output_av_container, video_settings: VideoSettings, log_level):
        self.media_container = media_container
        self.log_level = log_level
        self.encoder_inited = False
        self.video_settings = video_settings

        self.enc_codec = None

        self.in_stream = media_container.video_stream
        # Open another container because seeking to beginning of the file is unreliable...
        self.input_av_container: av.container.Container = av.open(media_container.path, 'r', metadata_errors='ignore')

        self.demux_iter = self.input_av_container.demux(self.in_stream)
        self.demux_saved_packet = None

        # Frame buffering for fetch_frame (using heap for efficient PTS ordering)
        self.frame_buffer = []
        self.frame_buffer_gop_dts = -1
        self.decoder = self.in_stream.codec_context

        if video_settings.mode == VideoExportMode.RECODE and video_settings.codec_override != 'copy':
            self.out_stream = output_av_container.add_stream(video_settings.codec_override, rate=self.in_stream.guessed_rate, options={'x265-params': 'log_level=error'})
            self.out_stream.width = self.in_stream.width
            self.out_stream.height = self.in_stream.height
            if self.in_stream.sample_aspect_ratio is not None:
                self.out_stream.sample_aspect_ratio = self.in_stream.sample_aspect_ratio
            self.out_stream.metadata.update(self.in_stream.metadata)
            self.out_stream.disposition = self.in_stream.disposition.value
            self.codec_name = video_settings.codec_override

            self.init_encoder()
            self.enc_codec = self.out_stream.codec_context
            self.enc_codec.options.update(self.encoding_options)
            self.enc_codec.thread_type = "FRAME"
            self.enc_last_pts = -1
        else:
            # Map codec name for decoder to encoder compatibility (AV1 case)
            original_codec_name = self.in_stream.codec_context.name
            mapped_codec_name = map_codec_name_for_pyav15(original_codec_name)

            if mapped_codec_name != original_codec_name:
                # Need to create stream with mapped codec name, not template
                self.out_stream = output_av_container.add_stream(mapped_codec_name, rate=self.in_stream.guessed_rate)
                self.out_stream.width = self.in_stream.width
                self.out_stream.height = self.in_stream.height
                if self.in_stream.sample_aspect_ratio is not None:
                    self.out_stream.sample_aspect_ratio = self.in_stream.sample_aspect_ratio
                self.out_stream.metadata.update(self.in_stream.metadata)
                self.out_stream.disposition = self.in_stream.disposition.value
                self.codec_name = mapped_codec_name
            else:
                # Use template if no mapping needed
                self.out_stream = output_av_container.add_stream_from_template(self.in_stream, options={'x265-params': 'log_level=error'})
                self.out_stream.metadata.update(self.in_stream.metadata)
                self.out_stream.disposition = self.in_stream.disposition.value
                self.codec_name = original_codec_name


            # if self.codec_name == 'mpeg2video':
                # self.out_stream.average_rate = self.in_stream.average_rate
                # self.out_stream.base_rate = self.in_stream.base_rate

            self.remux_bitstream_filter = av.bitstream.BitStreamFilterContext('null', self.in_stream, self.out_stream)
            if self.in_stream.codec_context.name == 'h264':
                if not is_annexb(self.in_stream.codec_context.extradata):
                    self.remux_bitstream_filter = av.bitstream.BitStreamFilterContext('h264_mp4toannexb', self.in_stream, self.out_stream)
            elif self.in_stream.codec_context.name == 'hevc':
                if not is_annexb(self.in_stream.codec_context.extradata):
                    self.remux_bitstream_filter = av.bitstream.BitStreamFilterContext('hevc_mp4toannexb', self.in_stream, self.out_stream)


        self._normalize_output_codec_tag(output_av_container)

        self.last_dts = -100_000_000

        self.segment_start_in_output = 0

        # Track stream continuity for CRA to BLA conversion
        self.last_remuxed_segment_gop_index = None
        self.is_first_remuxed_segment = True

    def _normalize_output_codec_tag(self, output_av_container: av.container.Container) -> None:
        """Ensure codec tags are compatible with MP4/MOV style containers."""

        codec_name = self.in_stream.codec_context.name
        container_name = output_av_container.format.name.lower() if output_av_container.format.name else ''

        if not any(name in container_name for name in ('mp4', 'mov', 'matroska', 'webm')):
            return

        if codec_name == 'h264':
            if self._is_mpegts_h264_tag(self.out_stream.codec_context.codec_tag):
                self.out_stream.codec_context.codec_tag = 'avc1'
        elif codec_name in ('hevc', 'h265'):
            if self._is_mpegts_hevc_tag(self.out_stream.codec_context.codec_tag):
                self.out_stream.codec_context.codec_tag = 'hvc1'

    @staticmethod
    def _is_mpegts_h264_tag(codec_tag) -> bool:
        if isinstance(codec_tag, int):
            return codec_tag == 27
        if isinstance(codec_tag, bytes):
            return codec_tag == b'\x1b\x00\x00\x00'
        if isinstance(codec_tag, str):
            return codec_tag == '\x1b\x00\x00\x00'
        return False

    @staticmethod
    def _is_mpegts_hevc_tag(codec_tag) -> bool:
        if isinstance(codec_tag, int):
            return codec_tag == 36
        if isinstance(codec_tag, bytes):
            return codec_tag in (b'HEVC', b'\x24\x00\x00\x00')
        if isinstance(codec_tag, str):
            return codec_tag in ('HEVC', '\x24\x00\x00\x00')
        return False

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
            elif 'Simple' in profile: # mpeg4 didn't like my profile strings
                profile = None
            else:
                profile = profile.lower().replace(':', '').replace(' ', '')

        # Get CRF value for quality setting
        crf_value = get_crf_for_quality(self.video_settings.quality)

        # Adjust CRF for newer codecs that are more efficient
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


    def segment(self, cut_segment: CutSegment) -> list[av.Packet]:
        self.out_time_base = self.out_stream.time_base

        if cut_segment.require_recode:
            packets = self.recode_segment(cut_segment)
        else:
            packets = self.flush_encoder()
            packets.extend(self.remux_segment(cut_segment))
            # Update tracking variables for CRA to BLA conversion
            self.last_remuxed_segment_gop_index = cut_segment.gop_index
            self.is_first_remuxed_segment = False

        self.segment_start_in_output += cut_segment.end_time - cut_segment.start_time

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

        self.input_av_container.close()

        return packets

    def recode_segment(self, s: CutSegment) -> list[av.Packet]:
        if not self.encoder_inited:
            self.init_encoder()
        result_packets = []

        if self.enc_codec is None:
            muxing_codec = self.out_stream.codec_context
            enc_codec = av.CodecContext.create(self.codec_name, 'w')

            if muxing_codec.rate is not None:
                enc_codec.rate = muxing_codec.rate
            enc_codec.options.update(self.encoding_options)

            enc_codec.width = muxing_codec.width
            enc_codec.height = muxing_codec.height
            enc_codec.pix_fmt = muxing_codec.pix_fmt

            if muxing_codec.sample_aspect_ratio is not None:
                enc_codec.sample_aspect_ratio = muxing_codec.sample_aspect_ratio
            if self.codec_name == 'mpeg2video':
                enc_codec.time_base = Fraction(1, muxing_codec.rate)
            else:
                enc_codec.time_base = self.out_stream.time_base
            enc_codec.flags = muxing_codec.flags
            if muxing_codec.bit_rate is not None:
                enc_codec.bit_rate = muxing_codec.bit_rate
            if muxing_codec.bit_rate_tolerance is not None:
                enc_codec.bit_rate_tolerance = muxing_codec.bit_rate_tolerance
            enc_codec.codec_tag = muxing_codec.codec_tag
            enc_codec.thread_type = "FRAME"
            self.enc_last_pts = -1
            self.enc_codec = enc_codec

        for frame in self.fetch_frame(s.gop_start_dts, s.gop_end_dts, s.end_time):
            in_tb = frame.time_base if frame.time_base is not None else self.in_stream.time_base
            if frame.pts * in_tb < s.start_time:
                continue
            if frame.pts * in_tb >= s.end_time:
                break

            out_tb = self.out_time_base if self.codec_name != 'mpeg2video' else self.enc_codec.time_base

            frame.pts -= int(s.start_time / in_tb)

            frame.pts = int(frame.pts * in_tb / out_tb)
            frame.time_base = out_tb
            frame.pts += int(self.segment_start_in_output / out_tb)

            if frame.pts <= self.enc_last_pts:
                frame.pts = self.enc_last_pts + 1
            self.enc_last_pts = frame.pts

            frame.pict_type = av.video.frame.PictureType.NONE
            result_packets.extend(self.enc_codec.encode(frame))

        if self.codec_name == 'mpeg2video':
            for p in result_packets:
                p.pts = p.pts * p.time_base / self.out_time_base
                p.dts = p.dts * p.time_base / self.out_time_base
                p.time_base = self.out_time_base
        return result_packets

    def remux_segment(self, s: CutSegment) -> list[av.Packet]:
        result_packets = []
        segment_start_pts = int(s.start_time / self.in_stream.time_base)

        # Check if we need CRA to BLA conversion for this segment
        should_convert_cra = self._should_convert_cra_for_segment(s)
        first_packet = True

        for packet in self.fetch_packet(s.gop_start_dts, s.gop_end_dts):

            # Apply CRA to BLA conversion only to the first packet if needed
            if first_packet and should_convert_cra:
                converted_data = convert_hevc_cra_to_bla(bytes(packet))
                if converted_data != bytes(packet):
                    # Create new packet with converted data using same pattern as copy_packet
                    new_packet = av.Packet(converted_data)
                    new_packet.pts = packet.pts
                    new_packet.dts = packet.dts
                    new_packet.duration = packet.duration
                    new_packet.time_base = packet.time_base
                    new_packet.stream = packet.stream
                    new_packet.is_keyframe = packet.is_keyframe
                    packet = new_packet

            # Mark that we've processed the first packet
            if first_packet:
                first_packet = False

            # Apply timing adjustments
            packet.pts -= segment_start_pts
            packet.pts = int(packet.pts * self.in_stream.time_base / self.out_time_base)
            packet.pts += int(self.segment_start_in_output / self.out_time_base)
            if packet.dts is not None:
                packet.dts -= segment_start_pts
                packet.dts = int(packet.dts * self.in_stream.time_base / self.out_time_base)
                packet.dts += int(self.segment_start_in_output / self.out_time_base)

            result_packets.extend(self.remux_bitstream_filter.filter(packet))

        result_packets.extend(self.remux_bitstream_filter.filter(None))

        self.remux_bitstream_filter.flush()
        return result_packets

    def _should_convert_cra_for_segment(self, s: CutSegment) -> bool:
        """
        Check if this segment needs CRA to BLA conversion.
        This is needed when:
        1. The stream is HEVC
        2. The GOP starts with a CRA frame (NAL type 21)
        3. The stream was cut before this point. Meaning, we had discontinuity in the remux sequence right before this point.
          Examples:
            beginning was skipped: -k 8,12. Current segment is the first segment without recoding (e.g. starting at 8.5 with a little bit of recoding before this).
            discontinuity in the stream: -c 20,30. Current segment is the first segment without recoding after the cut that happened at 30.
        """
        # Check 1: Must be HEVC
        if self.in_stream.codec_context.name != 'hevc':
            return False

        # Check 2: GOP must start with CRA (NAL type 21)
        if s.gop_index >= 0 and s.gop_index < len(self.media_container.gop_start_nal_types):
            nal_type = self.media_container.gop_start_nal_types[s.gop_index]
            if nal_type != 21:  # Not a CRA frame
                return False
        else:
            return False  # Invalid GOP index

        # Check 3: Check for discontinuity (missing stream content)
        # Case 1: First segment doesn't start at beginning (content was skipped)
        if self.is_first_remuxed_segment and s.gop_index >0:
            return True

        # Case 2: Gap between current segment and previous segment (content was cut)
        if self.last_remuxed_segment_gop_index is not None:
            if s.gop_index > self.last_remuxed_segment_gop_index + 1:
                return True

        return False

    def _apply_cra_to_bla_conversion(self, packets: list[av.Packet]) -> list[av.Packet]:
        """
        Apply CRA to BLA conversion to the packet list.
        This modifies the packet data to convert CRA frames to BLA frames.
        """

        converted_packets = []
        for packet in packets:
            if packet.stream.type == 'video':
                # Convert the packet data
                converted_data = convert_hevc_cra_to_bla(bytes(packet))
                if converted_data != bytes(packet):
                    # Create a new packet with converted data
                    new_packet = av.Packet(converted_data)
                    # Copy all attributes from original packet
                    new_packet.stream = packet.stream
                    new_packet.pts = packet.pts
                    new_packet.dts = packet.dts
                    new_packet.time_base = packet.time_base
                    converted_packets.append(new_packet)
                else:
                    converted_packets.append(packet)
            else:
                converted_packets.append(packet)

        return converted_packets

    def flush_encoder(self):
        if self.enc_codec is None:
            return []

        result_packets = self.enc_codec.encode()

        if self.codec_name == 'mpeg2video':
            for p in result_packets:
                p.pts = p.pts * p.time_base / self.out_time_base
                p.dts = p.dts * p.time_base / self.out_time_base
                p.time_base = self.out_time_base

        self.enc_codec = None
        return result_packets

    def fetch_packet(self, target_dts, end_dts):
        tb = self.in_stream.time_base

        # First, check if we have a saved packet from previous call
        if self.demux_saved_packet is not None:
            saved_dts = self.demux_saved_packet.dts if self.demux_saved_packet.dts is not None else -100_000_000
            if saved_dts >= target_dts:
                if saved_dts <= end_dts:
                    packet = self.demux_saved_packet
                    self.demux_saved_packet = None
                    yield packet
                else:
                    # Saved packet is beyond our end range, don't yield it
                    return
            else:
                # Saved packet is before our target, clear it
                self.demux_saved_packet = None

        for packet in self.demux_iter:
            in_dts = packet.dts if packet.dts is not None else -100_000_000

            # Skip packets before target_dts
            if packet.pts is None or in_dts < target_dts:
                diff = (target_dts - in_dts) * tb
                if in_dts > 0 and diff > 120:
                    t = int(target_dts - 30 / tb)
                    # print(f"Seeking to skip a gap: {float(t * tb)}")
                    self.input_av_container.seek(t, stream = self.in_stream)
                    # Clear saved packet after seek since iterator position changed
                    self.demux_saved_packet = None
                continue

            # Check if packet exceeds end_dts
            if in_dts > end_dts:
                # Save this packet for next call and stop iteration
                self.demux_saved_packet = packet
                return

            # Packet is in our target range, yield it
            yield packet

    def fetch_frame(self, gop_start_dts, gop_end_dts, end_time):
        # Initialize or reset for new GOP
        if self.frame_buffer_gop_dts != gop_start_dts:
            self.frame_buffer = []
            self.frame_buffer_gop_dts = gop_start_dts
            self.decoder.flush_buffers()

        # Get time_base for PTS calculations
        time_base = self.in_stream.time_base

        # Process packets and yield frames when safe
        current_dts = gop_start_dts

        for packet in self.fetch_packet(gop_start_dts, gop_end_dts):
            current_dts = packet.dts if packet.dts is not None else current_dts

            # Decode packet and add frames to buffer
            for frame in self.decoder.decode(packet):
                heap_item = FrameHeapItem(frame.pts, frame)
                heapq.heappush(self.frame_buffer, heap_item)

            # Release frames that are safe (buffer_lowest_pts <= current_dts)
            while len(self.frame_buffer) > 1:  # Keep at least one frame for ordering
                lowest_heap_item = self.frame_buffer[0]  # Peek at heap minimum
                frame = lowest_heap_item.frame
                frame_pts = lowest_heap_item.pts if lowest_heap_item.pts is not None else -1
                frame_time_base = frame.time_base if frame.time_base is not None else time_base

                # Only process frames that are safe to release (frame_pts <= current_dts)
                if frame_pts <= current_dts:
                    if frame_pts * frame_time_base < end_time:
                        heapq.heappop(self.frame_buffer)  # Remove from heap
                        yield frame
                    else:
                        # Safe frame is beyond end_time - we're done since all frames from now would be beyond end time
                        return
                else:
                    break

        # Final flush of the decoder
        try:
            for frame in self.decoder.decode(None):
                heap_item = FrameHeapItem(frame.pts, frame)
                heapq.heappush(self.frame_buffer, heap_item)
        except:
            pass

        # Yield remaining frames within time range
        while self.frame_buffer:
            # Peek at the next frame without popping it
            next_frame = self.frame_buffer[0]
            frame = next_frame.frame
            frame_time_base = frame.time_base if frame.time_base is not None else time_base

            if (next_frame.pts is not None and
                next_frame.pts * frame_time_base < end_time):
                # Frame is within time range, pop and yield it
                heapq.heappop(self.frame_buffer)
                yield frame
            else:
                # Frame is outside time range, stop processing (leave it in buffer)
                break

def smart_cut(media_container: MediaContainer, positive_segments: List[tuple[Fraction, Fraction]],
              out_path: str, audio_export_info: AudioExportInfo = None, log_level = None, progress = None,
              video_settings=None, segment_mode=False, cancel_object: CancelObject | None = None):
    if video_settings is None:
        video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.NORMAL)

    adjusted_segment_times = []
    for (s, e) in positive_segments:

        if media_container.video_stream is not None and s == 0:
            s = -1_000_000
        else:
            s = s + media_container.start_time
        adjusted_segment_times.append((s, e + media_container.start_time))

    cut_segments = make_cut_segments(media_container, adjusted_segment_times, video_settings.mode == VideoExportMode.KEYFRAMES)

    if video_settings.mode == VideoExportMode.RECODE:
        for c in cut_segments:
            c.require_recode = True

    if segment_mode:
        output_files = []
        padding = len(str(len(adjusted_segment_times)))
        for i, s in enumerate(adjusted_segment_times):
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
        output_files = [(out_path, adjusted_segment_times[-1])]
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

            for sub_track_i in range(len(media_container.subtitle_tracks)):
                generators.append(SubtitleCutter(media_container, output_av_container, sub_track_i))

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
                            # if packet.dts > 468468:
                        # print(float(packet.dts * output_av_container.streams.video[0].time_base))
                        if packet.dts < -900_000:
                            packet.dts = None

                        # Fix AVI format PTS/DTS ordering requirement
                        # AVI format requires PTS >= DTS (decode timestamp cannot be after presentation)
                        if (out_path.lower().endswith('.avi') and packet.dts is not None and
                            packet.pts is not None and packet.pts < packet.dts):
                            # Always adjust PTS up to match DTS to preserve decode order
                            packet.pts = packet.dts

                        # print(packet)
                        output_av_container.mux(packet)
            for g in generators:
                for packet in g.finish():
                    # if isinstance(g, VideoCutter):
                    # print("finish packet: ", packet)

                    # Fix AVI format PTS/DTS ordering requirement
                    # AVI format requires PTS >= DTS (decode timestamp cannot be after presentation)
                    if (out_path.lower().endswith('.avi') and packet.dts is not None and
                        packet.pts is not None and packet.pts < packet.dts):
                        # Always adjust PTS up to match DTS to preserve decode order
                        packet.pts = packet.dts

                    output_av_container.mux(packet)
            if progress is not None:
                progress.emit(previously_done_segments)

        if cancel_object is not None and cancel_object.cancelled:
            last_file_path = output_path_segment[0]

            if os.path.exists(last_file_path):
                os.remove(last_file_path)
