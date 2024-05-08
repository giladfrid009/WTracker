from __future__ import annotations
from collections.abc import Iterator
import numpy as np
import PIL.Image
from dataclasses import dataclass

from dataset.bbox_utils import BoxFormat


@dataclass
class ImageMeta:
    """
    Represents metadata of an image.

    Attributes:
        path (str): The file path of the image.
        shape (tuple[int, int, int]): The shape of the image in the format [width, height, channels].
    """

    path: str
    shape: tuple[int, int, int]  # [w, h, c]

    @staticmethod
    def from_file(full_path: str) -> ImageMeta:
        """
        Creates an ImageMeta object from a file.

        Args:
            full_path (str): The full path of the image file.

        Returns:
            ImageMeta: The ImageMeta object representing the image metadata.
        """
        with PIL.Image.open(full_path) as image:
            w, h, c = *image.size, len(image.getbands())
            return ImageMeta(full_path, (w, h, c))


@dataclass
class ImageSample:
    """
    Represents an image sample with associated metadata, bounding boxes, and keypoints.

    Attributes:
        metadata (ImageMeta): The metadata associated with the image sample.
        bboxes (np.ndarray, optional): The bounding boxes for objects in the image. Shape: (num_samples, 4).
        bbox_format (BoxFormat, optional): The format of the bounding boxes.
        keypoints (np.ndarray, optional): The keypoints for objects in the image. Shape: (num_samples, num_keypoints, 2).
    """

    metadata: ImageMeta
    bboxes: np.ndarray | None = None
    bbox_format: BoxFormat | None = None
    keypoints: np.ndarray | None = None

    def __post_init__(self):
        if self.bboxes is not None:
            if self.bboxes.ndim == 1:
                self.bboxes = self.bboxes[np.newaxis, :]

            self.bboxes.astype(int, copy=False)

            assert self.bboxes.shape[-1] == 4
            assert self.bboxes.ndim == 2
            assert self.bbox_format is not None

        if self.keypoints is not None:
            if self.keypoints.ndim == 1:
                self.keypoints = self.keypoints[np.newaxis, np.newaxis, :]
            if self.keypoints.ndim == 2:
                self.keypoints = self.keypoints[np.newaxis, :]

            self.keypoints.astype(int, copy=False)

            assert self.keypoints.dtype
            assert self.keypoints.shape[-1] == 2
            assert self.keypoints.ndim == 3


class ImageDataset:
    """
    A class representing a dataset of image samples.

    Attributes:
        image_list (list[ImageSample]): A list of image samples in the dataset.
    """

    def __init__(self, image_list: list[ImageSample] | None = None):
        self.image_list = image_list or []

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> ImageSample:
        return self.image_list[idx]

    def __iter__(self) -> Iterator[ImageSample]:
        return self.image_list.__iter__()

    def add_sample(self, sample: ImageSample) -> None:
        """
        Adds a new image sample to the dataset.

        Args:
            sample (ImageSample): The image sample to be added.
        """
        self.image_list.append(sample)

    def remove_sample(self, idx: int) -> None:
        """
        Removes an image sample from the dataset at the specified index.

        Args:
            idx (int): The index of the image sample to be removed.
        """
        self.image_list.pop(idx)

    def extend(self, other: ImageDataset) -> None:
        """
        Extends the dataset by adding all image samples from another dataset.

        Args:
            other (ImageDataset): The dataset to be extended with.
        """
        self.image_list.extend(other.image_list)
