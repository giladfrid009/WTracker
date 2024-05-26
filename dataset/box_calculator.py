import cv2 as cv
import numpy as np
from typing import Collection
from tqdm.auto import tqdm
from tqdm.contrib import concurrent
import multiprocessing

from utils.frame_reader import FrameReader
from dataset.bg_extractor import BGExtractor


class BoxCalculator:
    """
    A class for calculating bounding boxes for a sequence of frames.

    Methods:
        all_bboxes(): Returns all bounding boxes for all the frames.
        get_bbox(frame_idx): Returns the bounding box for a given frame index.
        get_background(): Returns the background image extracted from the frame reader frames.
        calc_specified_boxes(frame_indices, num_workers, chunk_size): Calculate bounding boxes for the specified frame indices.
        calc_all_boxes(num_workers, chunk_size): Calculate bounding boxes for all frames.
    """

    def __init__(
        self,
        frame_reader: FrameReader,
        bg_probes: int = 100,
        diff_thresh: int = 10,
    ) -> None:
        assert bg_probes > 0 and diff_thresh > 0

        self._frame_reader = frame_reader
        self._bg_probes = bg_probes
        self._diff_thresh = diff_thresh

        self._all_bboxes = np.full((len(frame_reader), 4), -1, dtype=int)
        self._background = None

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

    def get_background(self) -> np.ndarray:
        """
        Returns the background image extracted from the frame reader frames.

        Returns:
            np.ndarray: The background array.
        """

        if self._background is None:
            self._background = self._calc_background()
        return self._background

    def _calc_background(self) -> np.ndarray:
        bg_calc = BGExtractor(self._frame_reader)
        return bg_calc.calc_background(self._bg_probes, sampling="random", method="median")

    def _calc_bounding_box(self, frame_idx: int) -> np.ndarray:
        # get mask according to the threshold value
        frame = self._frame_reader[frame_idx]
        background = self.get_background()
        diff = cv.absdiff(frame, background)
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

    def _adjust_num_workers(self, num_tasks: int, chunk_size: int, num_workers: int) -> int:
        if num_workers is None:  # if None then choose automatically
            num_workers = min(multiprocessing.cpu_count() / 2, num_tasks / (2 * chunk_size))
            num_workers = round(num_workers)

        num_workers = min(num_workers, num_tasks // chunk_size)  # no point having workers without tasks
        num_workers = min(num_workers, multiprocessing.cpu_count())  # no point having more workers than cpus

        if num_workers < 0:  # make sure value is valid
            num_workers = 0

        return num_workers

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
            If None is provided then number of workers is determined automatically. Defaults to None.
            chunk_size (int, optional): Size of each chunk for parallel processing. Defaults to 50.

        Returns:
            np.ndarray: The calculated boxes for the specified frames.
        """

        self.get_background()

        num_workers = self._adjust_num_workers(len(frame_indices), chunk_size, num_workers)

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
                If None is provided then number of workers is determined automatically. Defaults to None.
            chunk_size (int, optional): Size of each chunk for parallel processing. Defaults to 50.

        Returns:
            np.ndarray: Array of bounding boxes for all frames.
        """

        indices = range(len(self._frame_reader))
        return self.calc_specified_boxes(indices, num_workers, chunk_size)
