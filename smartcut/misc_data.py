from dataclasses import dataclass, field

@dataclass
class MixInfo():
    track_levels: list[float]

@dataclass
class VideoViewTransform:
    name: str
    input_w: float
    input_h: float
    input_x: float
    input_y: float
    output_y: float
    enabled: bool

@dataclass
class WatermarkView:
    name: str
    path: str
    output_y: float
    output_x: float
    enabled: bool

@dataclass
class VideoTransform():
    resolution: tuple[int, int]
    views: list[VideoViewTransform | WatermarkView]

@dataclass
class AudioExportSettings:
    codec: str
    channels: str | None = None
    bitrate: int | None = None
    sample_rate: int | None = None
    denoise: int = -1

@dataclass
class AudioExportInfo:
    mix_info: MixInfo | None = None
    mix_export_settings: AudioExportSettings | None = None
    output_tracks: list[AudioExportSettings | None] = field(default_factory = lambda: [])

@dataclass
class CutSegment:
    require_recode: bool
    start_time: int
    end_time: int
