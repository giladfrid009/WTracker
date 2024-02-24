import cv2 as cv
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib import concurrent
from typing import Callable
from pathlib import Path

from data.file_utils import create_directory
from data.frame_reader import FrameReader
from dataset.bbox_utils import BoxFormat, BoxConverter


def identity_func(x):
    return x


class VideoExtractor:
    def __init__(
        self,
        frame_reader: FrameReader,
        bg_probes: int = 100,
        diff_thresh: int = 10,
        num_workers: int = 2,
        chunk_size: int = 50,
        frame_transform: Callable[[np.ndarray], np.ndarray] = None,
    ):
        self._frame_reader = frame_reader
        self._bg_probes = bg_probes
        self._diff_thresh = diff_thresh
        self._num_workers = num_workers
        self._chunk_size = chunk_size

        if frame_transform is None:
            frame_transform = identity_func
        self._transform = frame_transform

        # for caching
        self._cached_all_bboxes: np.ndarray = None
        self._cached_background: np.ndarray = None

    def background(self) -> np.ndarray:
        if self._cached_background is None:
            self._cached_background = self._calc_background()
        return self._cached_background

    def all_bboxes(self) -> np.ndarray:
        if self._cached_all_bboxes is None:
            self._cached_all_bboxes = self._calc_all_bboxes()
        return self._cached_all_bboxes

    def initialize(self, cache_bboxes: bool = False):
        """
        Initialize cached parameters for later use
        """
        self.background()
        if cache_bboxes:
            self.all_bboxes()

    def _calc_background(self) -> np.ndarray:
        length = len(self._frame_reader)
        size = min(self._bg_probes, length)
        # randomly select frames
        frame_ids = np.random.choice(length, size=size, replace=False)

        # get frames and apply transform
        extracted_list = []
        for frame_id in tqdm(frame_ids, desc="extracting background", unit="fr"):
            frame = self._frame_reader[frame_id]
            frame = self._transform(frame).astype(np.uint8, copy=False)
            extracted_list.append(frame)

        # calculate the median along the time axis
        extracted = np.stack(extracted_list, axis=0)
        median = np.median(extracted, axis=0).astype(np.uint8, copy=False)
        return median

    def _calc_bbox(self, frame: np.ndarray) -> np.ndarray:
        self._transform(frame)

        # Calculate difference between background and image
        diff = np.abs(frame.astype(np.int16) - self.background().astype(np.int16))
        diff = diff.astype(np.uint8)

        # Turn differences mask to black & white according to a threshold value
        _, mask = cv.threshold(diff, self._diff_thresh, 255, cv.THRESH_BINARY)

        # do some morphological magic to clean up noise from the mask
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # dilate to increase all object sizes in the mask
        mask = cv.dilate(mask, np.ones((3, 3), np.uint8), iterations=5)

        # find contours in the binary mask
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # find largest contour
        largest_contour = max(contours, key=lambda c: cv.contourArea(c))

        # Get matching bbox
        largest_bbox = cv.boundingRect(largest_contour)
        largest_bbox = np.asanyarray(largest_bbox, dtype=int)
        return largest_bbox

    def _calc_all_bboxes(self) -> np.ndarray:
        if self._num_workers > 0:
            bboxes = concurrent.process_map(
                self._calc_bbox,
                self._frame_reader,
                chunksize=self._chunk_size,
                desc="extracting bboxes",
                unit="fr",
            )
            return np.stack(bboxes, axis=0)

        bboxes = []
        for frame in tqdm(self._frame_reader, desc="extracting bboxes", unit="fr"):
            bbox = self._calc_bbox(frame)
            bboxes.append(bbox)
        return np.stack(bboxes, axis=0)

    def _calc_video_bounds_cached(
        self,
        start_frame: int,
        target_width: int,
        target_height: int,
    ) -> tuple[tuple, tuple]:
        """
        Finds the index bounds and the coordinates of the longest video slice of the provided dimensions,
        for which the bounding boxes of the object are within target size bounds.
        """
        # we care only after the `start_index` frame
        bboxes = self.all_bboxes()[start_frame:, :]

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
        illegal_width = max_right - min_left > target_width
        illegal_height = max_top - min_bottom > target_height
        is_illegal = illegal_width | illegal_height

        # Returns the last index where `False` can be inserted while maintaining order of the array
        # Note that `is_illegal` array is always sorted from `False` to `True`
        last_legal_idx = np.searchsorted(is_illegal, v=False, side="right")

        if last_legal_idx >= len(is_illegal):
            last_legal_idx = len(is_illegal) - 1

        # Return the exclusive last legal index
        slice_indices = (start_frame, start_frame + last_legal_idx + 1)

        # Get the bbox which matches the legal slice
        slice_bbox = (min_left[last_legal_idx], min_bottom[last_legal_idx], target_width, target_height)

        return slice_indices, slice_bbox

    def _calc_video_bounds_dynamic(
        self,
        start_frame: int,
        target_width: int,
        target_height: int,
        step_size: int = 1,
    ) -> tuple[tuple, tuple]:
        """
        Finds the index bounds and the coordinates of the longest video slice of the provided dimensions,
        for which the bounding boxes of the object are within target size bounds.
        """
        min_x, min_y = np.nan, np.nan
        max_x, max_y = np.nan, np.nan
        end_index = start_frame

        progress_bar = tqdm(desc="creating video sample", unit="fr")
        for cur_frame in range(start_frame, len(self._frame_reader), step_size):
            frame = self._frame_reader[cur_frame]
            bbox = VideoExtractor._calc_bbox(frame, self.background(), self._diff_thresh)

            x1, y1, x2, y2 = BoxConverter.change_format(bbox, BoxFormat.XYWH, BoxFormat.XYXY)

            new_min_x, new_min_y = np.nanmin([min_x, x1]), np.nanmin([min_y, y1])
            new_max_x, new_max_y = np.nanmax([max_x, x2]), np.nanmax([max_y, y2])

            # if the new max width or height is larger than the target then stop
            if new_max_x - new_min_x >= target_width or new_max_y - new_min_y >= target_height:
                break

            min_x, min_y = new_min_x, new_min_y
            max_x, max_y = new_max_x, new_max_y
            end_index = cur_frame
            progress_bar.update()

        progress_bar.close()

        # return the exclusive last legal index
        slice_indices = (start_frame, end_index + 1)

        # return the bbox which matches the legal slice
        slice_bbox = (int(min_x), int(min_y), target_width, target_height)

        return slice_indices, slice_bbox

    def _crop_and_save_video(
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
            frame = self._transform(frame).astype(np.uint8, copy=False)

            # save frame to sample path
            file_name = Path(self._frame_reader.files[i]).name
            full_path = Path(save_folder).joinpath(file_name).as_posix()
            cv.imwrite(full_path, frame)

    def _create_video(self, start_frame: int, width: int, height: int, folder_path: str) -> int:
        if self._cached_all_bboxes is None:
            # if no bbox cache then calculate bboxes frame by frame
            trim_range, crop_dims = self._calc_video_bounds_dynamic(start_frame, width, height)
        else:
            # otherwise, used numpy accelerated method on cached bboxes
            trim_range, crop_dims = self._calc_video_bounds_cached(self.all_bboxes(), start_frame, width, height)

        self._crop_and_save_video(folder_path, trim_range, crop_dims)

        return trim_range[1]

    def generate_videos(
        self,
        count: int,
        width: int,
        height: int,
        save_folder_format: str,
        step_size: int = 1,
    ):
        self.initialize(cache_bboxes=False)

        # Randomly select frames
        frame_ids = np.random.choice(len(self._frame_reader), size=count, replace=False)
        frame_ids = sorted(frame_ids)

        for iter, fid in tqdm(enumerate(frame_ids), desc="creating samples", total=count):
            # Find the properties of the video sample starting from `fid` frame

            if self._cached_all_bboxes is None:
                # if no bbox cache then calculate bboxes frame by frame
                trim_range, crop_dims = self._calc_video_bounds_dynamic(fid, width, height, step_size)
            else:
                # otherwise, used numpy accelerated method on cached bboxes
                trim_range, crop_dims = self._calc_video_bounds_cached(fid, width, height)

            # format the saving path to match current sample
            self._crop_and_save_video(save_folder_format.format(iter), trim_range, crop_dims)

    def generate_all_videos(
        self,
        width: int,
        height: int,
        save_folder_format: str,
    ):
        self.initialize(cache_bboxes=True)

        start_frame = 0
        iter = 0

        progress_bar = tqdm(desc="creating samples", total=len(self._frame_reader), unit="fr")
        while start_frame < len(self._frame_reader):
            # Find the properties of the video sample starting from start_frame
            trim_range, crop_dims = self._calc_video_bounds_cached(start_frame, width, height)

            # create and save the sample
            self._crop_and_save_video(save_folder_format.format(iter), trim_range, crop_dims)

            # updating progress bar and the next start_frame
            iter += 1
            trim_start, trim_end = trim_range

            start_frame = trim_end
            progress_bar.update(trim_end - trim_start)

        progress_bar.close()
