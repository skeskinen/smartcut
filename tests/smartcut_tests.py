from fractions import Fraction
import os
from time import time
import traceback
import av.logging
import ffmpeg
import platform

import numpy as np
import av
import scipy.signal
import soundfile as sf
import av.datasets as av_datasets
import scipy

from smartcut.misc_data import MixInfo, VideoTransform, VideoViewTransform
from smartcut.media_container import MediaContainer, AudioTrack, AudioReader
from smartcut.cut_video import AudioExportInfo, AudioExportSettings, VideoExportMode, VideoExportQuality, VideoSettings, make_cut_segments, smart_cut

np.random.seed(12345)

# Set the log level to silence the None dts warnings. I believe those can be ignored since
# we do set dts, except when it's not set in the source in which case it's not clear what
# value dts should take. It would be nice to occasionally check that there aren't more warnings.
av.logging.set_level(av.logging.ERROR)

data_dir = 'test_data'

os.chdir(os.path.dirname(__file__))
os.makedirs(data_dir, exist_ok=True)
os.chdir(data_dir)

short_h264_path = 'short_h264.mkv'
short_h265_path = 'short_h265.mkv'

def color_at_time(ts):
    c = np.empty((3,))
    c[0] = 0.5 + 0.5 * np.sin(2 * np.pi * (0 / 3 + ts / 10.))
    c[1] = 0.5 + 0.5 * np.sin(2 * np.pi * (1 / 3 + ts / 10.))
    c[2] = 0.5 + 0.5 * np.sin(2 * np.pi * (2 / 3 + ts / 10.))

    c = np.round(255 * c).astype(np.uint8)
    c = np.clip(c, 0, 255)

    return c

def create_test_video(path, target_duration, codec, pixel_format, fps, resolution, x265_options=[], profile=None):
    if os.path.exists(path):
        return
    total_frames = target_duration * fps

    container = av.open(path, mode="w")

    x265_options.append('log_level=warning')
    options = {'x265-params': ':'.join(x265_options)}
    if profile is not None:
        options['profile'] = profile
    stream = container.add_stream(codec, rate=fps, options=options)
    stream.width = resolution[0]
    stream.height = resolution[1]
    stream.pix_fmt = pixel_format

    for frame_i in range(total_frames):

        img = np.empty((stream.width, stream.height, 3), dtype=np.uint8)
        c = color_at_time(frame_i / fps)
        img[:, :] = c

        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()

def av_write_ogg(path, wave, sample_rate):
    with av.open(path, 'w') as out:
        s = out.add_stream('libvorbis', sample_rate, layout='mono')
        wave = wave.astype(np.float32)
        wave = np.expand_dims(wave, 0)
        frame = av.AudioFrame.from_ndarray(wave, format='fltp', layout='mono')
        frame.sample_rate = sample_rate
        frame.pts = 0
        packets = []
        packets.extend(s.encode(frame))
        packets.extend(s.encode(None))
        for p in packets:
            out.mux(p)

def generate_sine_wave(duration, path, frequency=440, sample_rate=44100):
    if os.path.exists(path):
        return

    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    wave = np.sin(2 * np.pi * frequency * t)

    if platform.system() == 'Windows' and os.path.splitext(path)[-1] == '.ogg':
        av_write_ogg(path, wave, sample_rate)
    else:
        sf.write(path, wave, sample_rate)

def generate_double_sine_wave(duration, path, frequency_0=440, frequency_1=440, sample_rate=44100):
    if os.path.exists(path):
        return
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    wave_0 = np.sin(2 * np.pi * frequency_0 * t)
    wave_1 = np.sin(2 * np.pi * frequency_1 * t)
    wave = 0.5 * (wave_0 + wave_1)

    if platform.system() == 'Windows' and os.path.splitext(path)[-1] == '.ogg':
        av_write_ogg(path, wave, sample_rate)
    else:
        sf.write(path, wave, sample_rate)

def assert_silence(track: AudioTrack):
    reader = AudioReader(track)

    dur = track.duration
    y_orig = reader.read(0., dur)

    sum = np.sum(np.abs(y_orig))
    assert sum < 0.3, f"Content should be silence, but wave sum is {sum}"

def compare_tracks(track_orig: AudioTrack, track_modified: AudioTrack, rms_threshold=0.07):
    orig = AudioReader(track_orig)
    modified = AudioReader(track_modified)

    # strict mode:
    # rms_threshold=0.05

    # assert orig.sr == modified.sr, "Sample rate changed"

    dur = track_orig.duration

    y_orig = orig.read(0., dur)
    y_modified = modified.read(0., dur)

    corr_ref = scipy.signal.correlate(y_orig, y_orig)
    corr_inp = scipy.signal.correlate(y_orig, y_modified)

    corr_ref = np.mean(np.abs(corr_ref))
    corr_inp = np.mean(np.abs(corr_inp))

    corr_diff = (corr_ref - corr_inp) / corr_ref
    assert corr_diff < 0.1, f"Audio contents have changed: {corr_diff} (correlation difference)"

    # TODO: come up with better audio similarity metric?

    diff = (y_orig - y_modified) ** 2
    rms = np.sqrt(np.mean(diff))

    assert rms < rms_threshold, f"Audio contents have changed: {rms} (rms error)"

def check_videos_equal(source_container: MediaContainer, result_container: MediaContainer):
    assert source_container.video_stream.width == result_container.video_stream.width
    assert source_container.video_stream.height == result_container.video_stream.height
    assert len(result_container.video_frame_times) == len(source_container.video_frame_times), f'Mismatch frame count. Exp: {len(source_container.video_frame_times)}, got: {len(result_container.video_frame_times)}'
    r = result_container.video_frame_times
    s = source_container.video_frame_times

    diff = np.abs(r - s)
    diff_i = np.argmax(diff)
    diff_amount = diff[diff_i]
    # NOTE: It would be nice to get a tighter bound on the frame timings.
    # But the difficulty is that we can't control the output stream time_base.
    # We have to just accept the value that av/ffmpeg gives us. So sometimes the
    # input and output timebases are not multiples of each other.
    diff_tolerance = Fraction(3, 1000)
    assert diff_amount <= diff_tolerance, f'Mismatch of {diff_amount} in frame timings, at frame {diff_i}.'

    with av.open(source_container.path, mode='r') as source_av:
        with av.open(result_container.path, mode='r') as result_av:
            for frame_i, (source_frame, result_frame) in enumerate(zip(source_av.decode(video=0), result_av.decode(video=0))):
                source_numpy = source_frame.to_ndarray(format='rgb24')
                result_numpy = result_frame.to_ndarray(format='rgb24')
                assert source_numpy.shape == result_numpy.shape, f'Video resolution or channel count changed. Exp: {source_numpy.shape}, got: {result_numpy.shape}'
                for y in [0, source_numpy.shape[0] // 2, source_numpy.shape[0] - 1]:
                    for x in [0, source_numpy.shape[1] // 2, source_numpy.shape[1] - 1]:
                        source_color = source_numpy[y, x]
                        result_color = result_numpy[y, x]
                        diff = np.abs(source_color.astype(np.int16) - result_color)
                        max_diff = np.max(diff)
                        assert max_diff <= 20, f'Large color deviation at frame {frame_i}. Exp: {source_color}, got: {result_color}'

def test_cut_on_keyframes(input_path, output_path):
    source = MediaContainer(input_path)
    cutpoints = list(source.gop_start_times) + [source.eof_time]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    cut_segments = make_cut_segments(source, segments)
    for c in cut_segments:
        assert not c.require_recode, "Cutting on a keyframe should not require recoding"

    smart_cut(source, segments, output_path)

    result_container = MediaContainer(output_path)
    check_videos_equal(source, result_container)

def test_smart_cut(input_path, output_path, n_cuts, audio_export_info = None, video_settings = None):
    source = MediaContainer(input_path)
    cutpoints = source.video_frame_times
    cutpoints = [0] + list(np.sort(np.random.choice(cutpoints, n_cuts, replace=False))) + [source.eof_time]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    smart_cut(source, segments, output_path,
        audio_export_info=audio_export_info, video_settings=video_settings, log_level='warning')

    result_container = MediaContainer(output_path)
    check_videos_equal(source, result_container)


def test_h264_cut_on_keyframes():
    create_test_video(short_h264_path, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_h264_cut_on_keyframes.__name__ + '.mkv'
    test_cut_on_keyframes(short_h264_path, output_path)

def test_h264_smart_cut():
    create_test_video(short_h264_path, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_h264_smart_cut.__name__ + '.mkv'
    for c in [1, 2, 3, 10, 30, 100]:
        test_smart_cut(short_h264_path, output_path, c)

def test_h264_multiple_cuts():
    create_test_video(short_h264_path, 30, 'h264', 'yuv420p', 30, (32, 18))
    source = MediaContainer(short_h264_path)

    output_path = test_h264_smart_cut.__name__ + '.mkv'
    for c in [1, 2, 3, 10, 30, 100]:
        cutpoints = source.video_frame_times
        cutpoints = [0] + list(np.sort(np.random.choice(cutpoints, c, replace=False))) + [source.eof_time]

        segments = list(zip(cutpoints[:-1], cutpoints[1:]))

        smart_cut(source, segments, output_path, log_level='warning')
        result_container = MediaContainer(output_path)
        check_videos_equal(source, result_container)

def test_h265_cut_on_keyframes():
    create_test_video(short_h265_path, 30, 'hevc', 'yuv422p10le', 60, (256, 144))
    output_path = test_h265_cut_on_keyframes.__name__ + '.mkv'
    test_cut_on_keyframes(short_h265_path, output_path)

def test_h265_smart_cut():
    create_test_video(short_h265_path, 30, 'hevc', 'yuv422p10le', 60, (256, 144))
    output_path = test_h265_smart_cut.__name__ + '.mkv'
    for c in [1, 2]:
        test_smart_cut(short_h265_path, output_path, c)

def test_h265_smart_cut_large():
    input_file = 'h265_large.mkv'
    create_test_video(input_file, 17, 'hevc', 'yuv422p10le', 25, (1280, 720))
    output_path = test_h265_smart_cut_large.__name__ + '.mkv'
    for c in [1, 2]:
        test_smart_cut(input_file, output_path, c)

def test_h264_24_fps_long():
    filename = 'long_h264.mkv'
    # 15 mins
    create_test_video(filename, 60 * 15, 'h264', 'yuv420p', 24, (32, 18))
    output_path = test_h264_24_fps_long.__name__ + '.mkv'
    test_smart_cut(filename, output_path, n_cuts=3)

def test_h264_1080p():
    filename = '1080p_h264.mkv'
    create_test_video(filename, 15, 'h264', 'yuv420p', 30, (1920, 1080))
    output_path = test_h264_1080p.__name__ + '.mkv'
    test_smart_cut(filename, output_path, n_cuts=3)

def test_h264_profile_baseline():
    filename = 'h264_baseline.mkv'
    create_test_video(filename, 15, 'h264', 'yuv420p', 30, (32, 18), profile='baseline')
    output_path = test_h264_profile_baseline.__name__ + '.mkv'
    test_smart_cut(filename, output_path, n_cuts=3)

def test_h264_profile_main():
    filename = 'h264_main.mkv'
    create_test_video(filename, 15, 'h264', 'yuv420p', 30, (32, 18), profile='main')
    output_path = test_h264_profile_main.__name__ + '.mkv'
    test_smart_cut(filename, output_path, n_cuts=3)

def test_h264_profile_high():
    filename = 'h264_high.mkv'
    create_test_video(filename, 15, 'h264', 'yuv420p', 30, (32, 18), profile='high')
    output_path = test_h264_profile_high.__name__ + '.mkv'
    test_smart_cut(filename, output_path, n_cuts=3)

def test_h264_profile_high10():
    filename = 'h264_high10.mkv'
    create_test_video(filename, 15, 'h264', 'yuv420p10le', 30, (32, 18), profile='high10')
    output_path = test_h264_profile_high10.__name__ + '.mkv'
    test_smart_cut(filename, output_path, n_cuts=3)

def test_h264_profile_high422():
    filename = 'h264_high422.mkv'
    create_test_video(filename, 15, 'h264', 'yuv422p', 30, (32, 18), profile='high422')
    output_path = test_h264_profile_high422.__name__ + '.mkv'
    test_smart_cut(filename, output_path, n_cuts=3)

def test_h264_profile_high444():
    filename = 'h264_high444.mkv'
    create_test_video(filename, 15, 'h264', 'yuv444p', 30, (32, 18), profile='high444')
    output_path = test_h264_profile_high444.__name__ + '.mkv'
    test_smart_cut(filename, output_path, n_cuts=3)

def test_mp4_cut_on_keyframe():
    filename = 'basic.mp4'
    create_test_video(filename, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_mp4_cut_on_keyframe.__name__ + '.mp4'
    test_cut_on_keyframes(filename, output_path)

def test_mp4_smart_cut():
    filename = 'basic.mp4'
    create_test_video(filename, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_mp4_smart_cut.__name__ + '.mp4'
    for c in [1, 2, 3, 10]:
        test_smart_cut(filename, output_path, c)

def test_mp4_to_mkv_smart_cut():
    filename = 'basic.mp4'
    create_test_video(filename, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_mp4_to_mkv_smart_cut.__name__ + '.mkv'
    for c in [1, 2, 3, 10]:
        test_smart_cut(filename, output_path, c)

def test_mkv_to_mp4_smart_cut():
    create_test_video(short_h264_path, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_mkv_to_mp4_smart_cut.__name__ + '.mp4'
    for c in [1, 2, 3, 10]:
        test_smart_cut(short_h264_path, output_path, c)

def test_mp4_h265_smart_cut():
    filename = 'h265.mp4'
    create_test_video(filename, 30, 'hevc', 'yuv420p', 30, (256, 144))
    output_path = test_mp4_h265_smart_cut.__name__ + '.mp4'
    for c in [1, 2, 3, 10]:
        test_smart_cut(filename, output_path, c)

def test_vertical_transform():
    input_path = 'vertical_in.mkv'
    file_duration = 5
    n_cuts = 5
    create_test_video(input_path, file_duration, 'h264', 'yuv420p', 30, (1920, 1080))
    reference_path = 'vertical_ref.mkv'
    create_test_video(reference_path, file_duration, 'h264', 'yuv420p', 30, (1080, 1920))
    reference_container = MediaContainer(reference_path)

    output_path = test_vertical_transform.__name__ + '.mkv'
    source_container = MediaContainer(input_path)

    cutpoints = source_container.video_frame_times
    cutpoints = [0] + list(np.sort(np.random.choice(cutpoints, n_cuts, replace=False))) + [source_container.eof_time]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    video_transform = VideoTransform((1080, 1920), views=[])
    video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.NORMAL, video_transform)
    smart_cut(source_container, segments, output_path, video_settings=video_settings)

    output_container = MediaContainer(output_path)
    check_videos_equal(reference_container, output_container)

    video_transform = VideoTransform((1080, 1920), views=[VideoViewTransform('test', 0.5, 0.5, 0.5, 0.5, 0.5, True)])
    video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.NORMAL, video_transform)
    smart_cut(source_container, segments, output_path, video_settings=video_settings)

    output_container = MediaContainer(output_path)
    check_videos_equal(reference_container, output_container)

    # test hevc override

    video_transform = VideoTransform((1080, 1920), views=[VideoViewTransform('test', 0.5, 0.5, 0.5, 0.5, 0.5, True)])
    video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.NORMAL, video_transform, codec_override='hevc')
    smart_cut(source_container, segments, output_path, video_settings=video_settings, log_level='warning')

    output_container = MediaContainer(output_path)
    assert output_container.video_stream.codec_context.name == 'hevc', f'codec should be hevc, found {output_container.video_stream.codec_context.name}'
    check_videos_equal(reference_container, output_container)

def test_video_recode_codec_override():
    input_path = 'video_settings_in.mkv'
    file_duration = 10
    n_cuts = 5
    create_test_video(input_path, file_duration, 'h264', 'yuv420p', 30, (854, 480))

    source_container = MediaContainer(input_path)

    cutpoints = source_container.video_frame_times
    cutpoints = [0] + list(np.sort(np.random.choice(cutpoints, n_cuts, replace=False))) + [source_container.eof_time]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    output_path_a = test_video_recode_codec_override.__name__ + 'a.mkv'
    output_path_b = test_video_recode_codec_override.__name__ + 'b.mkv'

    video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.NORMAL, None, codec_override='hevc')
    smart_cut(source_container, segments, output_path_a, video_settings=video_settings, log_level='warning')

    output_container = MediaContainer(output_path_a)
    assert output_container.video_stream.codec_context.name == 'hevc', f'codec should be hevc, found {output_container.video_stream.codec_context.name}'
    check_videos_equal(source_container, output_container)

    video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.HIGH, None, codec_override='hevc')
    smart_cut(source_container, segments, output_path_b, video_settings=video_settings, log_level='warning')

    output_container = MediaContainer(output_path_b)
    assert output_container.video_stream.codec_context.name == 'hevc', f'codec should be hevc, found {output_container.video_stream.codec_context.name}'
    check_videos_equal(source_container, output_container)

    assert os.path.getsize(output_path_b) > os.path.getsize(output_path_a)

    video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.NORMAL, None, codec_override='vp9')
    smart_cut(source_container, segments, output_path_a, video_settings=video_settings, log_level='warning')

    output_container = MediaContainer(output_path_a)
    assert output_container.video_stream.codec_context.name == 'vp9', f'codec should be vp9, found {output_container.video_stream.codec_context.name}'
    check_videos_equal(source_container, output_container)

    video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.HIGH, None, codec_override='vp9')
    smart_cut(source_container, segments, output_path_b, video_settings=video_settings, log_level='warning')

    output_container = MediaContainer(output_path_b)
    assert output_container.video_stream.codec_context.name == 'vp9', f'codec should be vp9, found {output_container.video_stream.codec_context.name}'
    check_videos_equal(source_container, output_container)

    assert os.path.getsize(output_path_b) > os.path.getsize(output_path_a)

    # These tests are very slow because the encoders are slow
    # video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.NORMAL, None, codec_override='av1')
    # smart_cut(source_container, segments, output_path_a, video_settings=video_settings, log_level='warning')

    # output_container = MediaContainer(output_path_a)
    # assert output_container.video_stream.codec_context.name == 'libdav1d', f'codec should be av1, found {output_container.video_stream.codec_context.name}'
    # check_videos_equal(source_container, output_container)

    # video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.HIGH, None, codec_override='vp9')
    # smart_cut(source_container, segments, output_path_b, video_settings=video_settings, log_level='warning')

    # output_container = MediaContainer(output_path_b)
    # assert output_container.video_stream.codec_context.name == 'libdav1d', f'codec should be av1 found {output_container.video_stream.codec_context.name}'
    # check_videos_equal(source_container, output_container)

    # assert os.path.getsize(output_path_b) > os.path.getsize(output_path_a)



def test_vorbis_passthru():
    filename = 'basic.ogg'
    freq = 440

    file_duration = 30
    generate_sine_wave(file_duration, filename, frequency=freq)
    output_path = test_vorbis_passthru.__name__ + '.ogg'

    n_cuts = 10
    source_container = MediaContainer(filename)
    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [file_duration]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    settings = AudioExportSettings(codec='passthru')
    export_info = AudioExportInfo(output_tracks=[settings])

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)
    output_container = MediaContainer(output_path)

    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0])

    # partial file i.e. suffix
    cutpoints = [15, file_duration]
    segments = list(zip(cutpoints[:-1], cutpoints[1:]))
    smart_cut(source_container, segments, output_path, audio_export_info=export_info)
    suffix_container = MediaContainer(output_path)
    assert suffix_container.eof_time > 14.9 and suffix_container.eof_time < 15.1
     # The cut point is not on packet boundary so the audio stream doesn't start at 0
    assert suffix_container.audio_tracks[0].packets[0].pts < 1000

def test_vorbis_track_cut():
    filename = 'basic.ogg'
    freq = 440

    file_duration = 30
    generate_sine_wave(file_duration, filename, frequency=freq)
    output_path = test_vorbis_track_cut.__name__ + '.ogg'

    n_cuts = 10
    source_container = MediaContainer(filename)
    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [source_container.eof_time]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    settings = AudioExportSettings(codec='libvorbis', channels = 'mono', bitrate=64000, sample_rate=44100)
    export_info = AudioExportInfo(output_tracks=[settings])

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    output_container = MediaContainer(output_path)
    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0])

    # partial file i.e. suffix
    cutpoints = [15, file_duration]
    segments = list(zip(cutpoints[:-1], cutpoints[1:]))
    smart_cut(source_container, segments, output_path, audio_export_info=export_info)
    suffix_container = MediaContainer(output_path)
    assert suffix_container.eof_time > 14.9 and suffix_container.eof_time < 15.1
     # The cut point is not on packet boundary so the audio stream doesn't start at 0
    assert suffix_container.audio_tracks[0].packets[0].pts < 1000

def test_mp3_track_cut():
    filename = 'basic.mp3'
    freq = 440

    file_duration = 30
    generate_sine_wave(file_duration, filename, frequency=freq)
    output_path = test_mp3_track_cut.__name__ + '.mp3'

    n_cuts = 10
    source_container = MediaContainer(filename)
    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [source_container.eof_time]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    settings = AudioExportSettings(codec='mp3', channels = 'mono', bitrate=128000, sample_rate=44100)
    export_info = AudioExportInfo(output_tracks=[settings])

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    output_container = MediaContainer(output_path)
    # NOTE: mp3 output has a timing issue at the beginning that I can't be arsed to fix.
    # Namely, the mp3 encoder adds some silence to the beginning (encoder delay).
    # Therefore we loosen the rms threshold so the mp3 test pass
    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0], rms_threshold=0.15)

    # partial file i.e. suffix
    cutpoints = [15, file_duration]
    segments = list(zip(cutpoints[:-1], cutpoints[1:]))
    smart_cut(source_container, segments, output_path, audio_export_info=export_info)
    suffix_container = MediaContainer(output_path)
    assert suffix_container.eof_time > 14.9 and suffix_container.eof_time < 15.1
     # The cut point is not on packet boundary so the audio stream doesn't start at 0
    assert suffix_container.audio_tracks[0].packets[0].pts < 1000

def test_mp3_passthru():
    filename = 'basic.mp3'
    freq = 440

    file_duration = 30
    generate_sine_wave(file_duration, filename, frequency=freq)
    output_path = test_mp3_passthru.__name__ + '.mp3'

    n_cuts = 10
    source_container = MediaContainer(filename)
    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [file_duration]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    settings = AudioExportSettings(codec='passthru')
    export_info = AudioExportInfo(output_tracks=[settings])

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)
    output_container = MediaContainer(output_path)

    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0])

    # partial file i.e. suffix
    cutpoints = [15, file_duration]
    segments = list(zip(cutpoints[:-1], cutpoints[1:]))
    smart_cut(source_container, segments, output_path, audio_export_info=export_info)
    suffix_container = MediaContainer(output_path)
    assert suffix_container.eof_time > 14.9 and suffix_container.eof_time < 15.1
     # The cut point is not on packet boundary so the audio stream doesn't start at 0
    assert suffix_container.audio_tracks[0].packets[0].pts < 1000


def test_vorbis_encode_mix():
    filename = 'basic.ogg'
    freq = 440

    file_duration = 30
    generate_sine_wave(file_duration, filename, frequency=freq)
    output_path = test_vorbis_encode_mix.__name__ + '.ogg'

    n_cuts = 10
    source_container = MediaContainer(filename)
    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [source_container.eof_time]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    mix = MixInfo([1.])
    settings = AudioExportSettings(codec='libvorbis', channels = 'mono', bitrate=64000, sample_rate=44100)
    export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    output_container = MediaContainer(output_path)
    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0])

def test_flac_conversions():
    filename = 'basic.ogg'
    freq = 440

    file_duration = 30
    generate_sine_wave(file_duration, filename, frequency=freq)
    output_path = test_flac_conversions.__name__ + '.flac'

    n_cuts = 10
    source_container = MediaContainer(filename)
    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [source_container.eof_time]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    mix = MixInfo([1.])
    settings = AudioExportSettings(codec='flac', channels = 'mono', sample_rate=44_100, bitrate=64_000)
    export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    output_container = MediaContainer(output_path)
    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0])

    ogg_output_path = test_flac_conversions.__name__ + '.ogg'

    mix = MixInfo([1.])
    settings = AudioExportSettings(codec='libvorbis', channels = 'mono', sample_rate=44_100, bitrate=64_000)
    export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

    smart_cut(output_container, segments, ogg_output_path, audio_export_info=export_info)

    vorbis_output_container = MediaContainer(ogg_output_path)
    compare_tracks(source_container.audio_tracks[0], vorbis_output_container.audio_tracks[0], rms_threshold=0.075)

def test_wav_conversions():
    filename = 'basic.ogg'
    freq = 440

    file_duration = 30
    generate_sine_wave(file_duration, filename, frequency=freq)
    output_path = test_wav_conversions.__name__ + '.wav'

    n_cuts = 10
    source_container = MediaContainer(filename)
    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [source_container.eof_time]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    mix = MixInfo([1.])
    settings = AudioExportSettings(codec='pcm_f32le', channels = 'mono', sample_rate=44_100, bitrate=64_000)
    export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    output_container = MediaContainer(output_path)
    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0], rms_threshold=0.075)

    settings = AudioExportSettings(codec='pcm_s16le', channels = 'mono', sample_rate=44_100, bitrate=64_000)
    export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    output_container = MediaContainer(output_path)
    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0], rms_threshold=0.075)

    # convert back to vorbis from s16
    ogg_output_path = test_wav_conversions.__name__ + '.ogg'

    settings = AudioExportSettings(codec='libvorbis', channels = 'mono', sample_rate=44_100, bitrate=64_000)
    export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

    smart_cut(output_container, segments, ogg_output_path, audio_export_info=export_info)

    ogg_output_container = MediaContainer(ogg_output_path)
    # TODO: flaky rms test
    compare_tracks(source_container.audio_tracks[0], ogg_output_container.audio_tracks[0], rms_threshold=0.15)


def make_video_and_audio_mkv(path, file_duration):

    audio_file_440 = 'tmp_440.ogg'
    # audio_file_440 = 'tmp_440.aac'
    generate_sine_wave(file_duration, audio_file_440, frequency=440)

    audio_file_630 = 'tmp_630.ogg'
    # audio_file_630 = 'tmp_630.aac'
    generate_sine_wave(file_duration, audio_file_630, frequency=630)

    tmp_video = 'tmp_video.mkv'
    create_test_video(tmp_video, file_duration, 'h264', 'yuv420p', 30, (32, 18))

    (
        ffmpeg
        .input(tmp_video)
        .output(ffmpeg.input(audio_file_440), ffmpeg.input(audio_file_630),
                path, vcodec='copy', acodec='aac', audio_bitrate=92_000, y=None)
        .run(quiet=True)
    )

def test_mkv_with_video_and_audio_passthru():
    file_duration = 30

    final_input = 'video_and_two_audio.mkv'
    make_video_and_audio_mkv(final_input, file_duration)

    output_path = test_mkv_with_video_and_audio_passthru.__name__ + '.mkv'

    source_container = MediaContainer(final_input)

    passthru_settings = AudioExportSettings(codec='passthru')
    export_info = AudioExportInfo(output_tracks=[passthru_settings, passthru_settings])
    test_smart_cut(final_input, output_path, n_cuts=5, audio_export_info=export_info)
    result_container = MediaContainer(output_path)

    assert len(result_container.audio_tracks) == 2
    compare_tracks(source_container.audio_tracks[0], result_container.audio_tracks[0])
    compare_tracks(source_container.audio_tracks[1], result_container.audio_tracks[1])

    export_info = AudioExportInfo(output_tracks=[None, passthru_settings])
    test_smart_cut(final_input, output_path, n_cuts=5, audio_export_info=export_info)
    result_container = MediaContainer(output_path)
    assert len(result_container.audio_tracks) == 1
    compare_tracks(source_container.audio_tracks[1], result_container.audio_tracks[0])

    export_info = AudioExportInfo(output_tracks=[passthru_settings, None])
    test_smart_cut(final_input, output_path, n_cuts=5, audio_export_info=export_info)
    result_container = MediaContainer(output_path)
    assert len(result_container.audio_tracks) == 1
    compare_tracks(source_container.audio_tracks[0], result_container.audio_tracks[0])

def test_mkv_with_video_and_audio_mix():
    file_duration = 30

    final_input = 'video_and_two_audio.mkv'
    make_video_and_audio_mkv(final_input, file_duration)

    output_path = test_mkv_with_video_and_audio_mix.__name__ + '.mkv'

    source_container = MediaContainer(final_input)

    mix = MixInfo([1., 0.])
    mix_export_settings = AudioExportSettings(codec='aac', channels='mono', bitrate=92_000, sample_rate=44_100)
    export_info = AudioExportInfo(mix, mix_export_settings)
    test_smart_cut(final_input, output_path, n_cuts=5, audio_export_info=export_info)

    result_container = MediaContainer(output_path)

    assert len(result_container.audio_tracks) == 1
    compare_tracks(source_container.audio_tracks[0], result_container.audio_tracks[0], rms_threshold=0.5)

    mix = MixInfo([0., 1.])
    export_info = AudioExportInfo(mix, mix_export_settings)
    test_smart_cut(final_input, output_path, n_cuts=5, audio_export_info=export_info)

    result_container = MediaContainer(output_path)
    assert len(result_container.audio_tracks) == 1
    compare_tracks(source_container.audio_tracks[1], result_container.audio_tracks[0], rms_threshold=0.6)

    mix = MixInfo([0.5, 0.5])
    export_info = AudioExportInfo(mix, mix_export_settings)
    test_smart_cut(final_input, output_path, n_cuts=5, audio_export_info=export_info)

    result_container = MediaContainer(output_path)
    assert len(result_container.audio_tracks) == 1

    reference_mix_path = 'reference_mix.ogg'
    generate_double_sine_wave(file_duration, reference_mix_path, 440, 630)
    reference_container = MediaContainer(reference_mix_path)
    compare_tracks(reference_container.audio_tracks[0], result_container.audio_tracks[0], rms_threshold=0.7)

def test_mix_with_rate_conversion():
    in_1 = '48k.ogg'
    freq_1 = 440
    in_2 = '26k.ogg'
    freq_2 = 600

    out_sr = 44_100

    file_duration = 30
    generate_sine_wave(file_duration, in_1, frequency=freq_1, sample_rate=48_000)
    generate_sine_wave(file_duration, in_2, frequency=freq_2, sample_rate=26_000)

    output_path = test_mix_with_rate_conversion.__name__ + '.ogg'

    n_cuts = 10
    source_container = MediaContainer(in_1)
    source_container.add_audio_file(in_2)

    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [source_container.eof_time]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    mix = MixInfo([0.5, 0.5])
    settings = AudioExportSettings(codec='libvorbis', channels = 'mono', bitrate=64000, sample_rate=out_sr)
    export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    reference_mix_path = test_mix_with_rate_conversion.__name__ + '_reference_mix.ogg'
    generate_double_sine_wave(file_duration, reference_mix_path, freq_1, freq_2, sample_rate=out_sr)
    ref_container = MediaContainer(reference_mix_path)

    output_container = MediaContainer(output_path)
    compare_tracks(ref_container.audio_tracks[0], output_container.audio_tracks[0], rms_threshold=0.12)

def test_denoiser():
    out_sr = 48_000
    test_sample_rates = [8_000, 16_000, 24_000, 36_000, 44_100, 48_000]
    if platform.system() == 'Windows':
        test_sample_rates = [36_000, 44_100, 48_000]
    for sr in test_sample_rates:
        in_file = f'denoiser_in_{sr}.ogg'
        file_duration = 3
        generate_sine_wave(file_duration, in_file, frequency=440, sample_rate=sr)

        output_path = test_denoiser.__name__ + f'_{sr}.ogg'

        n_cuts = 3
        source_container = MediaContainer(in_file)

        cutpoints = np.arange(file_duration*1000)[1:-1]
        cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [source_container.eof_time]

        segments = list(zip(cutpoints[:-1], cutpoints[1:]))

        mix = MixInfo([1.])
        settings = AudioExportSettings(codec='libvorbis', channels = 'mono', bitrate=64000, sample_rate=out_sr, denoise=1)
        export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

        smart_cut(source_container, segments, output_path, audio_export_info=export_info)

        output_container = MediaContainer(output_path)
        assert len(output_container.audio_tracks) == 1

    file_duration = 10

    audio_file_440 = 'denoise_in_440.ogg'
    generate_sine_wave(file_duration, audio_file_440, frequency=440, sample_rate=48_000)
    audio_file_630 = 'denoise_in_630.ogg'
    generate_sine_wave(file_duration, audio_file_630, frequency=630, sample_rate=48_000)

    output_path = test_denoiser.__name__ + '.ogg'

    source_container = MediaContainer(audio_file_440)
    source_container.add_audio_file(audio_file_630)

    mix = MixInfo([1., 1.])

    segments = [(0, source_container.eof_time)]

    # output denoise
    mix_export_settings = AudioExportSettings(codec='libvorbis', channels='mono', bitrate=92_000, sample_rate=48_000, denoise=2)
    export_info = AudioExportInfo(mix, mix_export_settings)
    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    result_container = MediaContainer(output_path)
    assert_silence(result_container.audio_tracks[0])

    # input 0 denoise
    mix_export_settings = AudioExportSettings(codec='libvorbis', channels='mono', bitrate=92_000, sample_rate=48_000, denoise=0)
    export_info = AudioExportInfo(mix, mix_export_settings)
    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    result_container = MediaContainer(output_path)
    compare_tracks(source_container.audio_tracks[1], result_container.audio_tracks[0])

    # input 1 denoise
    mix_export_settings = AudioExportSettings(codec='libvorbis', channels='mono', bitrate=92_000, sample_rate=48_000, denoise=1)
    export_info = AudioExportInfo(mix, mix_export_settings)
    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    result_container = MediaContainer(output_path)
    compare_tracks(source_container.audio_tracks[0], result_container.audio_tracks[0])


def test_vp9_smart_cut():
    filename = 'vp9.mkv'

    create_test_video(filename, 30, 'vp9', 'yuv420p', 30, (256, 144))
    output_path = test_vp9_smart_cut.__name__ + '.mkv'
    for c in [2, 6]:
        test_smart_cut(filename, output_path, n_cuts=c)

def test_vp9_profile_1():
    filename = 'vp9_p1_422.mkv'

    create_test_video(filename, 30, 'vp9', 'yuv422p', 30, (256, 144), profile='1')
    output_path = test_vp9_profile_1.__name__ + '.mkv'
    for c in [2, 6]:
        test_smart_cut(filename, output_path, n_cuts=c)

def test_av1_smart_cut():
    filename = 'av1.mkv'

    create_test_video(filename, 30, 'av1', 'yuv420p', 30, (256, 144))
    output_path = test_av1_smart_cut.__name__ + '.mkv'
    for c in [1, 2]:
        test_smart_cut(filename, output_path, n_cuts=c)

def test_night_sky():
    os.environ["PYAV_TESTDATA_DIR"] = 'pyav_datasets'
    filename = av_datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4")
    output_path = test_night_sky.__name__ + '.mp4'
    for c in [1, 2, 3]:
        test_smart_cut(filename, output_path, n_cuts=c)

def test_night_sky_to_mkv():
    os.environ["PYAV_TESTDATA_DIR"] = 'pyav_datasets'
    filename = av_datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4")
    output_path = test_night_sky_to_mkv.__name__ + '.mkv'
    for c in [1, 2, 3]:
        test_smart_cut(filename, output_path, n_cuts=c)

def test_sunset():
    os.environ["PYAV_TESTDATA_DIR"] = 'pyav_datasets'
    filename = av_datasets.curated("pexels/time-lapse-video-of-sunset-by-the-sea-854400.mp4")
    output_path = test_sunset.__name__ + '.mp4'
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH, None)
    for c in [1, 2, 3]:
        test_smart_cut(filename, output_path, n_cuts=c, video_settings=video_settings)

# This tests cutting of interlaced video and it fails. I don't see a way to make it work.
# Therefore interlacing is unsupported, probably forever. Leaving the test here for future reference.
def test_fate_interlaced_crop():
    os.environ["PYAV_TESTDATA_DIR"] = 'pyav_datasets'
    filename = av_datasets.fate("h264/interlaced_crop.mp4")
    output_path = test_fate_interlaced_crop.__name__ + '.mp4'
    for c in [1, 2, 3]:
        test_smart_cut(filename, output_path, n_cuts=c)

def test_broken_ref_vid():
    # os.environ["PYAV_TESTDATA_DIR"] = 'pyav_datasets'
    filename = '../ref_videos/remove_mistakes_short.mkv'
    output_path = test_broken_ref_vid.__name__ + '.mkv'
    for c in [1, 2, 3]:
        test_smart_cut(filename, output_path, n_cuts=c)


def test_manual():
    source_container = MediaContainer('../../out.flac')

    output_path = test_manual.__name__ + '.ogg'
    n_cuts = 3
    cutpoints = np.arange(source_container.eof_time*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [source_container.eof_time]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    mix = MixInfo([1.])
    settings = AudioExportSettings(codec='libopus', channels = 'mono', bitrate=64000, sample_rate=48000)
    export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    output_container = MediaContainer(output_path)
    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0])


def run_tests():
    """
    Runs all the tests and prints their status.
    """
    tests = [
        test_h264_cut_on_keyframes,
        test_h264_smart_cut,
        test_h264_24_fps_long,
        test_h264_1080p,
        test_h264_multiple_cuts,

        test_h264_profile_baseline,
        test_h264_profile_main,
        test_h264_profile_high,
        test_h264_profile_high10,
        test_h264_profile_high422,
        test_h264_profile_high444,

        test_mp4_cut_on_keyframe,
        test_mp4_smart_cut,
        test_mp4_to_mkv_smart_cut,
        test_mkv_to_mp4_smart_cut,

        test_vp9_smart_cut,
        test_av1_smart_cut,

        test_vp9_profile_1,

        test_night_sky,
        test_night_sky_to_mkv,
        test_sunset,

        test_h265_cut_on_keyframes,
        test_h265_smart_cut,
        test_h265_smart_cut_large,
        test_mp4_h265_smart_cut,

        test_vertical_transform,
        test_video_recode_codec_override,

        test_vorbis_passthru,

        test_mkv_with_video_and_audio_passthru,

        test_mp3_passthru,


        # test_broken_ref_vid,

        # test_manual,
    ]

    smc_tests = [
        test_vorbis_encode_mix,

        test_flac_conversions,
        test_wav_conversions,

        test_mkv_with_video_and_audio_mix,

        test_mix_with_rate_conversion,

        test_denoiser,

        test_vorbis_track_cut,
        test_mp3_track_cut,
    ]

    try:
        # Audio mixing, etc, is ommited from the CLI version, because Librosa and some other libs add a lot of bloat to the binary
        from smc.audio_handling import MixAudioCutter, RecodeTrackAudioCutter
        tests = tests + smc_tests
        print("Including smc tests")

    except ImportError:
        print("Skipping smc tests")
        pass

    # tests = [test_broken_ref_vid]
    # tests = smc_tests
    # tests = [test_denoiser]

    perf_timer = time()

    for test in tests:
        test_name = test.__name__
        try:
            test()
            print(f"{test_name}: PASS")
        except Exception as e:
            print(f"{test_name}: FAIL:")
            traceback.print_exc()
    print(f'Tests ran in {(time() - perf_timer):0.1f}s')

if __name__ == "__main__":
    run_tests()