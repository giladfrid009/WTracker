import cv2 as cv
import numpy as np
from tqdm import tqdm
from typing import Callable
from pathlib import Path

from data.file_helpers import create_directory
from data.frame_reader import FrameReader


class SampleExtractor:
    def __init__(self, frame_reader: FrameReader, frame_transform: Callable[[np.ndarray], np.ndarray] = None) -> None:
        self._frame_reader: FrameReader = frame_reader

        if frame_transform is None:
            frame_transform = lambda x: x
        self._transform = frame_transform

        # for caching
        self._video_bboxes: np.ndarray = None

    def calc_video_background(self, num_probes: int) -> np.ndarray:
        length = len(self._frame_reader)

        # randomly select frames
        frame_ids = np.random.choice(length, size=min(num_probes, length), replace=False)
        frame_ids = sorted(frame_ids)

        # get frames and apply transform
        extracted_list = []
        for i in tqdm(frame_ids, desc="extracting background", unit="fr"):
            frame = self._frame_reader[i]
            frame = self._transform(frame).astype(np.uint8)
            extracted_list.append(frame)

        # calculate the median along the time axis
        extracted = np.stack(extracted_list, axis=0)
        median = np.median(extracted, axis=0).astype(np.uint8)
        return median

    def calc_frame_bbox(self, frame: np.ndarray, background: np.ndarray, diff_thresh: int) -> np.ndarray:
        frame = self._transform(frame).astype(np.uint8)

        # Calculate difference between background and image
        diff = np.abs(frame.astype(np.int16) - background.astype(np.int16))
        diff = diff.astype(np.uint8)

        # Turn differences mask to black & white according to a threshold value
        _, mask = cv.threshold(diff, diff_thresh, 255, cv.THRESH_BINARY)

        # do some morphological magic to clean up noise from the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        # dilate to increase all object sizes in the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.dilate(mask, kernel, iterations=5)

        # find contours in the binary mask
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # find largest contour
        largest_contour = max(contours, key=lambda c: cv.contourArea(c))

        # Get matching bbox
        largest_bbox = cv.boundingRect(largest_contour)
        largest_bbox = np.asanyarray(largest_bbox, dtype=int)
        return largest_bbox

    def calc_video_bboxes(self, bg_probes: int = 100, fg_thresh: int = 10) -> np.ndarray:
        # first, we find the background, which we will use later
        background = self.calc_video_background(bg_probes)

        # extract bbox from each video frame
        bbox_list = []
        for frame in tqdm(self._frame_reader, desc="extracting bboxes", total=len(self._frame_reader)):
            bbox = self.calc_frame_bbox(frame, background, fg_thresh)
            bbox_list.append(bbox)

        # update cached variable
        self._video_bboxes = np.stack(bbox_list, axis=0)
        return self._video_bboxes

    def _analyze_sample_properties(
        self,
        bboxes: np.ndarray,
        start_index: int,
        slice_width: int,
        slice_height: int,
    ) -> tuple[tuple, tuple]:
        """
        Finds the index bounds and the coordinates of the longest video slice of the provided dimensions,
        for which the bounding boxes of the object are within slice size bounds.
        """
        # we care only after the `start_index` frame
        bboxes = bboxes[start_index:, :]

        # get bbox coordinates
        left, bottom, width, height = bboxes.T
        right = left + width
        top = bottom + height

        # The function calculates cumulative min up to the current index
        # We want to find the first index where the difference between the cumulative mins
        # and maxes is larger than the given thresholds
        min_left = np.minimum.accumulate(left)
        min_bottom = np.minimum.accumulate(bottom)
        max_right = np.maximum.accumulate(right)
        max_top = np.maximum.accumulate(top)

        # Find where bboxes are out of legal bounds
        illegal_width = max_right - min_left > slice_width
        illegal_height = max_top - min_bottom > slice_height
        is_illegal = illegal_width | illegal_height

        # Returns the last index where `False` can be inserted while maintaining order of the array
        # Note that `is_illegal` array is always sorted from `False` to `True`
        last_legal_idx = np.searchsorted(is_illegal, v=False, side="right")

        if last_legal_idx >= len(is_illegal):
            last_legal_idx = len(is_illegal) - 1

        # Return the exclusive last legal index
        slice_indices = (start_index, start_index + last_legal_idx + 1)

        # Get the bbox which matches the legal slice
        slice_bbox = (min_left[last_legal_idx], min_bottom[last_legal_idx], slice_width, slice_height)

        return slice_indices, slice_bbox

    def _create_and_save_sample(
        self,
        save_folder: str,
        trim_range: tuple[int, int],
        crop_dims: tuple[int, int, int, int],
    ):
        # create dir if doesn't exist
        create_directory(save_folder)

        x, y, w, h = crop_dims
        start, end = trim_range

        for i in range(start, end):
            # get frame and crop it
            frame = self._frame_reader[i]
            frame = frame[y : y + h, x : x + w]

            # save frame to sample path
            file_name = Path(self._frame_reader.files[i]).name
            full_path = Path(save_folder).joinpath(file_name).as_posix()
            cv.imwrite(full_path, frame)

    def generate_samples(
        self,
        count: int,
        width: int,
        height: int,
        save_folder: str,
    ):
        if self._video_bboxes is None:
            raise Exception(f"please run `{self.calc_video_bboxes.__name__}` first")

        # Randomly select frames
        frame_ids = np.random.choice(len(self._frame_reader), size=count, replace=False)
        frame_ids = sorted(frame_ids)

        for i, fid in tqdm(enumerate(frame_ids), desc="creating samples", total=count):
            # Find the properties of the video sample starting from `fid` frame
            trim_range, crop_dims = self._analyze_sample_properties(self._video_bboxes, fid, width, height)

            # format the saving path to match current sample
            sample_folder_path = save_folder.format(i)

            # create and save the sample
            self._create_and_save_sample(sample_folder_path, trim_range, crop_dims)

    def generate_all_samples(
        self,
        width: int,
        height: int,
        save_folder: str,
    ):
        if self._video_bboxes is None:
            raise Exception(f"please run `{self.calc_video_bboxes.__name__}` first")

        start_frame = 0
        iter = 0

        progress_bar = tqdm(desc="creating samples", total=len(self._frame_reader), unit="fr")
        while start_frame < len(self._frame_reader):
            # Find the properties of the video sample starting from start_frame
            trim_range, crop_dims = self._analyze_sample_properties(self._video_bboxes, start_frame, width, height)

            # format the saving path to match current sample
            sample_folder_path = save_folder.format(iter)

            # create and save the sample
            self._create_and_save_sample(sample_folder_path, trim_range, crop_dims)

            # updating progress bar and the next start_frame
            iter += 1
            trim_start, trim_end = trim_range
            start_frame = trim_end
            progress_bar.update(trim_end - trim_start)

        progress_bar.close()
