from __future__ import annotations
import numpy as np
import PIL.Image
from dataclasses import dataclass

# TODO: FIX EVERYTHING HERE

@dataclass(frozen=True)
class FrameMeta:
    path: str
    size: tuple[int, int]
    pixel_size: float

    @staticmethod
    def from_file(path: str, pixel_size: float) -> FrameMeta:
        with PIL.Image.open(path) as image:
            return FrameMeta(path, image.size, pixel_size)


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


class FrameDataset:
    def __init__(self, sample_list: list[Sample] = []) -> None:
        self._sample_list: list[Sample] = sample_list

    def __len__(self):
        return len(self._sample_list)

    def __getitem__(self, idx: int) -> Sample:
        return self._sample_list[idx]

    def __iter__(self):
        return self._sample_list.__iter__()

    def add_sample(self, label: Sample):
        self._sample_list.append(label)

    def remove_sample(self, idx: int):
        self._sample_list.pop(idx)

    def extend(self, other: FrameDataset):
        self._sample_list.extend(other._sample_list)