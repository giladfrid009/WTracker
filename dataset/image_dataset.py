from __future__ import annotations
from collections.abc import Iterator
import numpy as np
import PIL.Image
from dataclasses import dataclass

from dataset.bbox_utils import BoxFormat

# Generic class for images
@dataclass(frozen=True)
class ImageMeta:
    path: str
    shape: tuple[int, int, int]  # [w, h, c]

    @staticmethod
    def from_file(full_path: str) -> ImageMeta:
        with PIL.Image.open(full_path) as image:
            w, h, c = *image.size, len(image.getbands())
            return ImageMeta(full_path, (w, h, c))


# Class for labled data
@dataclass(frozen=True)
class ImageSample:
    metadata: ImageMeta
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


# Dataset of labled data
@dataclass(frozen=True)
class ImageDataset:
    image_list: list[ImageSample]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx: int) -> ImageSample:
        return self.image_list[idx]

    def __iter__(self) -> Iterator[ImageSample]:
        return self.image_list.__iter__()

    def add_sample(self, sample: ImageSample):
        self.image_list.append(sample)

    def remove_sample(self, idx: int):
        self.image_list.pop(idx)

    def extend(self, other: ImageDataset):
        self.image_list.extend(other.image_list)
