from __future__ import annotations
from collections.abc import Iterator
import numpy as np
import PIL.Image
from dataclasses import dataclass

from data.frame_reader import FrameReader
from data.file_utils import join_paths


# class for Raw data
@dataclass(frozen=True)
class ExperimentMeta:
    fps: float
    orig_resolution: tuple[int, int]
    pixel_size: float
    background: np.ndarray
    comments: str = ""


@dataclass(frozen=True)
class RawImageMeta:
    path: str
    shape: tuple[int, int, int]  # [w, h, c]
    pixel_size: float
    frame_number: int
    bbox: tuple[int, int, int, int]  # [x, y, w, h]

    @staticmethod
    def from_file(
        full_path: str, pixel_size: float, frame_number: int, bbox: tuple[int, int, int, int]
    ) -> RawImageMeta:
        with PIL.Image.open(full_path) as image:
            w, h, c = *image.size, len(image.getbands())
            return RawImageMeta(full_path, (w, h, c), pixel_size, frame_number, bbox)


@dataclass
class ExperimentDataset:
    metadata: ExperimentMeta
    image_list: list[RawImageMeta]
    is_sorted: bool = False

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx: int) -> RawImageMeta:
        if not self.is_sorted:
            self.sort_items()
        return self.image_list[idx]

    def __iter__(self) -> Iterator[RawImageMeta]:
        return self.image_list.__iter__()

    def add_sample(self, image: RawImageMeta):
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
    def from_frame_reader(
        reader: FrameReader,
        meta: ExperimentMeta,
        bbox_list: list[tuple[int, int, int, int]] = None,
    ):
        if bbox_list is None:
            bbox = (0, 0, reader.frame_shape[1], reader.frame_shape[0])

        image_list = []
        for i, image_name in enumerate(reader.files):
            full_path = join_paths(reader.root_folder, image_name)
            bbox = bbox if bbox_list is None else bbox_list[i]
            image_meta = RawImageMeta.from_file(full_path, meta.pixel_size, i, bbox)
            image_list.append(image_meta)

        return ExperimentDataset(meta, image_list, is_sorted=True)
