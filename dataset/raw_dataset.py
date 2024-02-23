
from __future__ import annotations
from collections.abc import Iterator
import numpy as np
import PIL.Image
from dataclasses import dataclass, field
from data.frame_reader import FrameReader
from dataset.bbox_utils import BoxFormat
from pathlib import Path

# class for Raw data
@dataclass(frozen=True)
class ExperimentMeta:
    fps: float
    orig_resolution: tuple[int, int]
    pixel_size: float
    background:np.ndarray
    comments: str = ''


# Generic class for images
@dataclass(frozen=True)
class RawImageMeta:
    path: str
    shape: tuple[int, int, int]  # [w, h, c]
    pixel_size: float
    frame_number: int
    bbox: tuple[int, int, int, int] # XYWH 
    
    @staticmethod
    def from_file(full_path: str, pixel_size: float, frame_number:int, bbox:tuple[int, int, int, int]) -> RawImageMeta:
        with PIL.Image.open(full_path) as image:
            w, h, c = *image.size, len(image.getbands())
            return RawImageMeta(full_path, (w, h, c), pixel_size, frame_number, bbox)


# Dataset of raw images
@dataclass
class ExperimentDataset:
    experiment_meta:ExperimentMeta
    image_list: list[RawImageMeta]
    _sorted:bool = field(default=False, init=False)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx: int) -> RawImageMeta:
        if not self._sorted:
            self.sort_items()
        return self.image_list[idx]

    def __iter__(self) -> Iterator[RawImageMeta]:
        return self.image_list.__iter__()

    def add_sample(self, image: RawImageMeta):
        self._sorted = False
        self.image_list.append(image)

    def remove_sample(self, idx: int):
        self.image_list.pop(idx)

    def extend(self, other: ExperimentDataset):
        self.image_list.extend(other.image_list)
        self._sorted = False
    
    def sort_items(self):
        self.image_list = sorted(self.image_list, key=lambda meta: meta.frame_number)
        self._sorted = True

    @staticmethod
    def from_frame_reader(reader:FrameReader, meta:ExperimentMeta, bbox_list:list[tuple[int, int, int, int]]=None):
        experiment = ExperimentDataset(meta, [])

        image_name_list = reader.files
        root = reader.root_folder
        h, w = reader.frame_shape
        for i, image_name in enumerate(image_name_list):
            full_path = image_name
            bbox = (0, 0, h, w) if bbox_list is None else bbox_list[i]
            image_meta = RawImageMeta.from_file(full_path, meta.pixel_size, i, bbox)
            experiment.add_sample(image_meta)
























