from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any

from utils.config_base import ConfigBase
from utils.frame_reader import FrameReader


@dataclass
class TimingConfig(ConfigBase):
    experiment_config: ExperimentConfig = field(repr=False)

    px_per_mm: int = field(init=False)
    mm_per_px: float = field(init=False)

    frames_per_sec: int = field(init=False)
    ms_per_frame: float = field(init=False)

    imaging_time_ms: float
    imaging_frame_num: int = field(init=False)

    pred_time_ms: float
    pred_frame_num: int = field(init=False)

    moving_time_ms: float
    moving_frame_num: int = field(init=False)

    camera_size_mm: tuple[float, float]
    camera_size_px: tuple[int, int] = field(init=False)

    micro_size_mm: tuple[float, float]
    micro_size_px: tuple[int, int] = field(init=False)

    def __post_init__(self):

        self.frames_per_sec = self.experiment_config.frames_per_sec
        self.ms_per_frame = self.experiment_config.ms_per_frame

        self.imaging_frame_num = math.ceil(self.imaging_time_ms / self.ms_per_frame)
        self.pred_frame_num = math.ceil(self.pred_time_ms / self.ms_per_frame)
        self.moving_frame_num = math.ceil(self.moving_time_ms / self.ms_per_frame)

        self.mm_per_px = self.experiment_config.mm_per_px
        self.px_per_mm = self.experiment_config.px_per_mm

        self.camera_size_px = (
            round(self.px_per_mm * self.camera_size_mm[0]),
            round(self.px_per_mm * self.camera_size_mm[1]),
        )

        self.micro_size_px = (
            round(self.px_per_mm * self.micro_size_mm[0]),
            round(self.px_per_mm * self.micro_size_mm[1]),
        )

        del self.experiment_config  # experiment_config was temporaty, only for the constructor

    @property
    def cycle_length(self) -> int:
        return self.imaging_frame_num + self.moving_frame_num

    @property
    def cycle_time_ms(self) -> float:
        return self.cycle_length * self.ms_per_frame


@dataclass
class ExperimentConfig(ConfigBase):
    name: str
    num_frames: int
    frames_per_sec: float
    orig_resolution: tuple[int, int]
    px_per_mm: float
    init_position: tuple[int, int]
    comments: str = ""
    mm_per_px: float = field(init=False)
    ms_per_frame: float = field(init=False)

    def __post_init__(self):
        self.ms_per_frame = 1000 / self.frames_per_sec
        self.mm_per_px = 1 / self.px_per_mm

    @classmethod
    def from_frame_reader(
        cls, reader: FrameReader, name: str, frames_per_sec: int, px_per_mm: float, init_position: tuple[int, int]
    ) -> ExperimentConfig:
        return ExperimentConfig(
            name=name,
            num_frames=len(reader),
            frames_per_sec=frames_per_sec,
            orig_resolution=reader.frame_size,
            px_per_mm=px_per_mm,
            init_position=init_position,
        )
