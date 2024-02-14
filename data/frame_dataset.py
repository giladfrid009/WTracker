from __future__ import annotations
from collections.abc import Iterator
import numpy as np
import PIL.Image
from dataclasses import dataclass
from data.bbox_utils import BoxFormat


@dataclass(frozen=True)
class ExperimentMeta:
    fps: float
    orig_resolution: tuple[int, int]
    pixel_size: float
    comments: str = ""


@dataclass(frozen=True)
class FrameMeta:
    path: str
    size: tuple[int, int]
    pixel_size: float

    @staticmethod
    def from_file(full_path: str, pixel_size: float) -> FrameMeta:
        with PIL.Image.open(full_path) as image:
            return FrameMeta(full_path, image.size, pixel_size)


@dataclass(frozen=True)
class Sample:
    metadata: FrameMeta
    bboxes: np.ndarray = None
    bbox_format: BoxFormat = None
    keypoints: np.ndarray = None

    # bboxes shape: [sample_number, 4]
    # keypoints shape: [sample_number, num_keypoints, 2]

    def __post_init__(self):
        have_bbox = self.bboxes is not None
        have_keypoints = self.keypoints is not None

        if have_bbox:
            if self.bboxes.ndim == 1:
                self.bboxes = self.bboxes[np.newaxis, :]
            
            self.bboxes.astype(int, copy=False)

            assert self.bboxes.shape[-1] == 4
            assert self.bboxes.ndim == 2
            assert self.bbox_format is not None

        if have_keypoints:
            if self.keypoints.ndim == 1:
                self.keypoints = self.keypoints[np.newaxis, np.newaxis, :]
            if self.keypoints.ndim == 2:
                self.keypoints = self.keypoints[np.newaxis, :]

            self.keypoints.astype(int, copy=False)

            assert self.keypoints.dtype
            assert self.keypoints.shape[-1] == 2
            assert self.keypoints.ndim == 3


@dataclass(frozen=True)
class FrameDataset:
    experiment: ExperimentMeta
    sample_list: list[Sample]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> Sample:
        return self.sample_list[idx]

    def __iter__(self) -> Iterator[Sample]:
        return self.sample_list.__iter__()

    def add_sample(self, sample: Sample):
        if sample.metadata.pixel_size != self.experiment.pixel_size:
            raise ValueError("sample has non-matching pixel size")

        self.sample_list.append(sample)

    def remove_sample(self, idx: int):
        self.sample_list.pop(idx)

    def extend(self, other: FrameDataset):
        if other.experiment.pixel_size != self.experiment.pixel_size:
            raise ValueError("other dataset has non-matching pixel size")
        if other.experiment.fps != self.experiment.fps:
            raise ValueError("other dataset has non-matching fps")

        self.sample_list.extend(other.sample_list)
