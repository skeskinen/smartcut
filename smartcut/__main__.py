import argparse
import av
import numpy as np
from fractions import Fraction
from smartcut.media_container import MediaContainer
from smartcut.cut_video import smart_cut, VideoSettings, VideoExportMode, VideoExportQuality, AudioExportSettings, AudioExportInfo
from tqdm import tqdm

def parse_time_segments(time_str):
    times = list(map(Fraction, time_str.split(',')))
    if len(times) % 2 != 0:
        raise ValueError("You must provide an even number of time points for segments.")
    return list(zip(times[::2], times[1::2]))

class Progress:
    def __init__(self):
        self.first_call = True
        self.tqdm = None

    def emit(self, value):
        if self.first_call:
            self.first_call = False
            self.tqdm = tqdm(total=value)
            return
        self.tqdm.update(1)

def main():
    parser = argparse.ArgumentParser(description="SmartCut CLI tool for video editing")
    parser.add_argument('input', type=str, help="Input media file path")
    parser.add_argument('output', type=str, help="Output media file path")
    parser.add_argument('--keep', type=str, help="Comma-separated list of start,end times to keep in seconds")
    parser.add_argument('--cut', type=str, help="Comma-separated list of start,end times to cut in seconds")
    parser.add_argument('--log-level', type=str, default='warning', help="Log level (default: warning)")
    parser.add_argument('--version', action='version', version='Smartcut 1.0.0')

    args = parser.parse_args()

    if args.keep and args.cut or not (args.keep or args.cut):
        raise ValueError("You must specify either --keep or --cut, not both.")

    source = MediaContainer(args.input)

    if args.keep:
        segments = parse_time_segments(args.keep)
    elif args.cut:
        cut_segments = parse_time_segments(args.cut)
        segments = [(Fraction(0), source.duration())]
        for c_start, c_end in cut_segments:
            last_segment = segments.pop()
            if c_start > last_segment[0]:
                segments.append((last_segment[0], c_start))
            if c_end < last_segment[1]:
                segments.append((c_end, last_segment[1]))
    else:
        raise ValueError("You must specify either --keep or --cut.")

    # Default audio settings: no mix, include all tracks with lossless passthru
    audio_settings = [AudioExportSettings(codec='passthru')] * len(source.audio_tracks)
    export_info = AudioExportInfo(output_tracks=audio_settings)

    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.NORMAL, None)

    progress = Progress()

    if args.log_level == 'warning':
        av.logging.set_level(av.logging.WARNING)
    if args.log_level == 'error':
        av.logging.set_level(av.logging.ERROR)
    if args.log_level == 'fatal':
        av.logging.set_level(av.logging.FATAL)

    exception_value = smart_cut(source, segments, args.output,
                                audio_export_info=export_info,
                                video_settings=video_settings,
                                progress=progress, log_level=args.log_level)

    progress.tqdm.close()

    if exception_value is not None:
        raise exception_value

    print(f"Smart cut completed successfully. Output saved to {args.output}")

if __name__ == '__main__':
    main()
