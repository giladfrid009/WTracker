from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any
from ultralytics import YOLO

from utils.config_base import ConfigBase
from utils.path_utils import join_paths, create_parent_directory
from frame_reader import FrameReader


@dataclass
class TimingConfig(ConfigBase):
    frames_per_sec: int
    ms_per_frame: float = field(init=False)

    imaging_time_ms: float
    imaging_frame_num: int = field(init=False)

    pred_time_ms: float
    pred_frame_num: int = field(init=False)

    moving_time_ms: float
    moving_frame_num: int = field(init=False)

    px_per_mm: int
    mm_per_px: float = field(init=False)

    camera_size_mm: tuple[float, float]
    camera_size_px: tuple[int, int] = field(init=False)

    micro_size_mm: tuple[float, float]
    micro_size_px: tuple[int, int] = field(init=False)

    def __post_init__(self):
        self.ms_per_frame = 1000 / self.frames_per_sec
        self.imaging_frame_num = math.ceil(self.imaging_time_ms / self.ms_per_frame)
        self.pred_frame_num = math.ceil(self.pred_time_ms / self.ms_per_frame)
        self.moving_frame_num = math.ceil(self.moving_time_ms / self.ms_per_frame)
        self.mm_per_px = 1 / self.px_per_mm

        self.camera_size_px = (
            round(self.px_per_mm * self.camera_size_mm[0]),
            round(self.px_per_mm * self.camera_size_mm[1]),
        )

        self.micro_size_px = (
            round(self.px_per_mm * self.micro_size_mm[0]),
            round(self.px_per_mm * self.micro_size_mm[1]),
        )

    @property
    def cycle_length(self) -> int:
        return self.imaging_frame_num + self.moving_frame_num

    @property
    def cycle_time_ms(self) -> float:
        return self.cycle_length * self.ms_per_frame


@dataclass
class LogConfig(ConfigBase):
    root_folder: str
    save_mic_view: bool = False
    save_cam_view: bool = False
    save_err_view: bool = True
    mic_folder_name: str = "micro"
    cam_folder_name: str = "camera"
    err_folder_name: str = "errors"
    bbox_file_name: str = "bboxes.csv"
    mic_file_name: str = "mic_{:09d}.png"
    cam_file_name: str = "cam_{:09d}.png"

    mic_file_path: str = field(init=False)
    cam_file_path: str = field(init=False)
    err_file_path: str = field(init=False)
    bbox_file_path: str = field(init=False)

    def __post_init__(self):
        self.mic_file_path = join_paths(self.root_folder, self.mic_folder_name, self.mic_file_name)
        self.cam_file_path = join_paths(self.root_folder, self.cam_folder_name, self.cam_file_name)
        self.err_file_path = join_paths(self.root_folder, self.err_folder_name, self.cam_file_name)
        self.bbox_file_path = join_paths(self.root_folder, self.bbox_file_name)

    def create_dirs(self):
        create_parent_directory(self.mic_file_path)
        create_parent_directory(self.cam_file_path)
        create_parent_directory(self.err_file_path)
        create_parent_directory(self.bbox_file_path)


@dataclass
class YoloConfig(ConfigBase):
    model_path: str
    device: str = "cpu"
    task: str = "detect"
    verbose: bool = False
    pred_kwargs: dict = field(
        default_factory=lambda: {
            "imgsz": 384,
            "conf": 0.1,
        }
    )

    model: YOLO = field(default=None, init=False, repr=False)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["model"]  # we dont want to serialize the model
        return state

    def load_model(self) -> YOLO:
        if self.model is None:
            self.model = YOLO(self.model_path, task=self.task, verbose=self.verbose)
        return self.model


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
