from __future__ import annotations
from collections.abc import Iterator
import numpy as np
import PIL.Image
from dataclasses import dataclass, field
from utils.config_base import ConfigBase

from frame_reader import FrameReader
from utils.path_utils import join_paths


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
    ) -> ExperimentDataset:
        return ExperimentConfig(
            name=name,
            num_frames=len(reader),
            frames_per_sec=frames_per_sec,
            orig_resolution=reader.frame_shape[:2],
            px_per_mm=px_per_mm,
            init_position=init_position,
        )


@dataclass(frozen=False)
class ImageMeta(ConfigBase):
    path: str
    shape: tuple[int, int, int]  # [w, h, c]
    pixel_size: float
    frame_number: int
    bbox: tuple[int, int, int, int]  # [x, y, w, h]

    @staticmethod
    def from_file(full_path: str, pixel_size: float, frame_number: int, bbox: tuple[int, int, int, int]) -> ImageMeta:
        with PIL.Image.open(full_path) as image:
            w, h, c = *image.size, len(image.getbands())
            return ImageMeta(full_path, (w, h, c), pixel_size, frame_number, bbox)


@dataclass
class ExperimentDataset:
    metadata: ExperimentConfig
    image_list: list[ImageMeta]
    is_sorted: bool = False

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx: int) -> ImageMeta:
        if not self.is_sorted:
            self.sort_items()
        return self.image_list[idx]

    def __iter__(self) -> Iterator[ImageMeta]:
        return self.image_list.__iter__()

    def add_sample(self, image: ImageMeta):
        self.is_sorted = False
        self.image_list.append(image)

    def remove_sample(self, idx: int):
        self.image_list.pop(idx)

    def extend(self, other: ExperimentDataset):
        self.image_list.extend(other.image_list)
        self.is_sorted = False

    def sort_items(self):
        self.image_list = sorted(self.image_list, key=lambda meta: meta.frame_number)
        self.is_sorted = True

    @staticmethod
    def create_from_frame_reader(
        reader: FrameReader,
        meta: ExperimentConfig,
        bbox_list: list[tuple[int, int, int, int]] = None,
    ):
        if bbox_list is None:
            bbox = (0, 0, reader.frame_shape[1], reader.frame_shape[0])

        image_list = []
        for i, image_name in enumerate(reader.files):
            full_path = join_paths(reader.root_folder, image_name)
            bbox = bbox if bbox_list is None else bbox_list[i]
            image_meta = ImageMeta.from_file(full_path, meta.px_per_mm, i, bbox)
            image_list.append(image_meta)

        return ExperimentDataset(meta, image_list, is_sorted=True)
