from __future__ import annotations
import numpy as np
import PIL.Image
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentMeta:
    fps: float
    orig_resolution: tuple[int, int]
    pixel_size: float
    comments: str


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
    frame: FrameMeta
    bboxes: np.ndarray = None
    keypoints: np.ndarray = None

    def __post_init__(self):
        have_bbox = self.bboxes is not None
        have_keypoints = self.keypoints is not None

        if have_bbox:
            assert self.bboxes.shape[-1] == 4
            assert 1 <= self.bboxes.ndim <= 2

        if have_keypoints:
            assert self.keypoints.dtype
            assert self.keypoints.shape[-1] == 4
            assert 1 <= self.keypoints.ndim <= 2

        if have_bbox and have_keypoints:
            assert self.bboxes.ndim == self.keypoints.ndim


@dataclass(frozen=True)
class FrameDataset:
    experiment: ExperimentMeta
    sample_list: list[Sample]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> Sample:
        return self.sample_list[idx]

    def __iter__(self):
        return self.sample_list.__iter__()

    def add_sample(self, sample: Sample):
        if sample.frame.pixel_size != self.experiment.pixel_size:
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
