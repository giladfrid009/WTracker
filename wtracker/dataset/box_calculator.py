import cv2 as cv
import numpy as np
from typing import Collection
from tqdm.auto import tqdm
from tqdm.contrib import concurrent

from wtracker.utils.frame_reader import FrameReader
from wtracker.dataset.bg_extractor import BGExtractor
from wtracker.utils.threading_utils import adjust_num_workers


class BoxCalculator:
    """
    A class for calculating bounding boxes around an object for a sequence of frames.
    The bounding boxes are calculated by comparing the frames to a background image.
    The largest contour in the difference image between the frame and the background is used to calculate the bounding box.

    Args:
        frame_reader (FrameReader): The frame reader object holing the relevant frames.
        background (np.ndarray): The background image of the frames in the `frame_reader` argument.
        diff_thresh (int, optional): Threshold value for the detecting foreground objects.
            Pixels with difference value greater than this threshold are considered as foreground.
    """

    def __init__(
        self,
        frame_reader: FrameReader,
        background: np.ndarray,
        diff_thresh: int = 20,
    ) -> None:
        assert diff_thresh > 0, "Difference threshold must be greater than 0."
        assert frame_reader.frame_shape == background.shape, "Background shape must match frame shape."

        self._frame_reader = frame_reader
        self._background = background
        self._diff_thresh = diff_thresh

        self._all_bboxes = np.full((len(frame_reader), 4), -1, dtype=int)

    def all_bboxes(self) -> np.ndarray:
        """
        Returns all bounding boxes for all the frames.
        Note that if a bounding box has not been calculated for some frame, then the matching entry will be (-1, -1, -1, -1).

        Returns:
            np.ndarray: Array of bounding boxes, in shape (N, 4), where N is the number of frames.
            The bounding boxes are stored in the format (x, y, w, h).
        """
        return self._all_bboxes

    def get_bbox(self, frame_idx: int) -> np.ndarray:
        """
        Returns the bounding box for a given frame index.

        Args:
            frame_idx (int): The index of the frame from which to extract the bounding box.

        Returns:
            np.ndarray: The bounding box coordinates as a numpy array, in format (x, y, w, h).
        """

        bbox = self._all_bboxes[frame_idx]
        if bbox[0] == -1:
            # calculate bbox since it wasn't calculated before
            bbox = self._calc_bounding_box(frame_idx)
            self._all_bboxes[frame_idx] = bbox
        return bbox

    # TODO: I DONT THINK IT WORKS ON BGR IMAGES.
    # PROBABLY NEED TO CHECK WHETHER IMAGES ARE GRAYSCALE OR NOT, AND CONVERT TO GRAYSCALE IF NEEDED.
    # SIMILAR TO WHAT WE'RE DOING IN THE ErrorCalculator.calc_precise_error
    def _calc_bounding_box(self, frame_idx: int) -> np.ndarray:
        # get mask according to the threshold value
        frame = self._frame_reader[frame_idx]
        diff = cv.absdiff(frame, self._background)
        _, mask = cv.threshold(diff, self._diff_thresh, 255, cv.THRESH_BINARY)

        # apply morphological ops to the mask
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv.dilate(mask, np.ones((11, 11), np.uint8))

        # extract contours and bbox
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        if not contours:
            zero_bbox = np.array([0, 0, 0, 0])
            self._all_bboxes[frame_idx] = zero_bbox
            return zero_bbox

        largest_contour = max(contours, key=cv.contourArea)
        largest_bbox = cv.boundingRect(largest_contour)
        largest_bbox = np.asanyarray(largest_bbox, dtype=int)
        return largest_bbox

    def calc_specified_boxes(
        self,
        frame_indices: Collection[int],
        num_workers: int = None,
        chunk_size: int = 50,
    ) -> np.ndarray:
        """
        Calculate bounding boxes for the specified frame indices.

        Args:
            frame_indices (Iterable[int]): The indices of the frames for which to calculate the bboxes.
            num_workers (int, optional): Number of workers for parallel processing.
            If None is provided then number of workers is determined automatically.
            chunk_size (int, optional): Size of each chunk for parallel processing.

        Returns:
            np.ndarray: The calculated boxes for the specified frames.
        """
        num_workers = adjust_num_workers(len(frame_indices), chunk_size, num_workers)

        if num_workers > 0:
            bbox_list = concurrent.process_map(
                self.get_bbox,
                frame_indices,
                max_workers=num_workers,
                chunksize=chunk_size,
                desc="Extracting bboxes",
                unit="fr",
            )

            for idx, bbox in zip(frame_indices, bbox_list):
                self._all_bboxes[idx] = bbox

        else:
            for idx in tqdm(frame_indices, desc="Extracting bboxes", unit="fr"):
                self.get_bbox(idx)

        bboxes = self._all_bboxes[frame_indices, :]
        return bboxes

    def calc_all_boxes(
        self,
        num_workers: int = None,
        chunk_size: int = 50,
    ) -> np.ndarray:
        """
        Calculate bounding boxes for all frames.

        Args:
            num_workers (int, optional): Number of workers for parallel processing.
                If None is provided then number of workers is determined automatically.
            chunk_size (int, optional): Size of each chunk for parallel processing.

        Returns:
            np.ndarray: Array of bounding boxes for all frames.
        """

        indices = range(len(self._frame_reader))
        return self.calc_specified_boxes(indices, num_workers, chunk_size)
