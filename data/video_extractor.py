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
        """
        :param frame_reader: the frame reader to extract the video samples from.
        :param bg_probes: number of random frames to take to calculate the background.
        :param diff_thresh: a pixel is considered as background if the diff between the background and pixel value is less than `diff_thresh`.
        :param num_workers: number of processes used for calculating all the bounding boxes for all frames in the `frame_reader`.
        :param chunk_size: number of frames sent at once for each process of the `num_workers` processes.
            For best performance set this value to `len(frame_reader) // num_workers`,
            but for progress tracking of frame extraction set this value to something significantly lower.
        :frame_transform: transformation function applied to each frame read from the `frame_reader`.
        """
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
        :param cache_bboxes: whether to calculate object bounding box for each frame. Note that this process might take very long time.
        """
        self.background()
        if cache_bboxes:
            self.all_bboxes()

    def _calc_background(self) -> np.ndarray:
        length = len(self._frame_reader)
        size = min(self._bg_probes, length)

        # get frames and apply transform
        frame_ids = np.random.choice(length, size=size, replace=False)
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

        # get mask according to the threshold value
        diff = np.abs(frame.astype(np.int16) - self.background().astype(np.int16))
        diff = diff.astype(np.uint8)
        _, mask = cv.threshold(diff, self._diff_thresh, 255, cv.THRESH_BINARY)

        # apply morphological ops to the mask
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv.dilate(mask, np.ones((1, 1), np.uint8))

        # extract contours and bbox
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=lambda c: cv.contourArea(c))
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
        start_index: int,
        target_size: tuple[int, int],
        max_length: int = None,
    ) -> tuple[tuple, tuple]:
        """
        Finds the index bounds and the coordinates of the longest video slice of the provided dimensions,
        for which the bounding boxes of the object are within target size bounds.
        """
        bboxes = self.all_bboxes()
        bboxes = bboxes[start_index:, :] if max_length is None else bboxes[start_index : start_index + max_length, :]

        # get bbox coordinates
        left, bottom, width, height = bboxes.T
        right = left + width
        top = bottom + height

        min_left = np.minimum.accumulate(left)
        min_bottom = np.minimum.accumulate(bottom)
        max_right = np.maximum.accumulate(right)
        max_top = np.maximum.accumulate(top)

        # Find where bboxes are out of legal bounds
        illegal_width = max_right - min_left > target_size[0]
        illegal_height = max_top - min_bottom > target_size[1]
        is_illegal = illegal_width | illegal_height

        last_legal_idx = np.searchsorted(is_illegal, v=False, side="right")
        if last_legal_idx >= len(is_illegal):
            last_legal_idx = len(is_illegal) - 1

        slice_indices = (start_index, start_index + last_legal_idx + 1)
        slice_bbox = (min_left[last_legal_idx], min_bottom[last_legal_idx], *target_size)
        return slice_indices, slice_bbox

    def _calc_video_bounds_dynamic(
        self,
        start_index: int,
        target_size: tuple[int, int],
        max_length: int = None,
        granularity: int = 1,
    ) -> tuple[tuple, tuple]:
        """
        Finds the index bounds and the coordinates of the longest video slice of the provided dimensions,
        for which the bounding boxes of the object are within target size bounds.
        """
        min_x, min_y = np.nan, np.nan
        max_x, max_y = np.nan, np.nan
        end_index = start_index

        max_length = len(self._frame_reader) if max_length is None else max_length
        last_index = min(len(self._frame_reader), start_index + max_length)

        for idx in tqdm(
            range(start_index, last_index, granularity),
            desc="creating video sample",
            unit="fr",
            total=None,
        ):
            frame = self._frame_reader[idx]
            bbox = self._calc_bbox(frame)

            x1, y1, x2, y2 = BoxConverter.change_format(bbox, BoxFormat.XYWH, BoxFormat.XYXY)

            new_min_x, new_min_y = np.nanmin([min_x, x1]), np.nanmin([min_y, y1])
            new_max_x, new_max_y = np.nanmax([max_x, x2]), np.nanmax([max_y, y2])

            # if the new max width or height is larger than the target then stop
            if new_max_x - new_min_x >= target_size[0] or new_max_y - new_min_y >= target_size[1]:
                break

            min_x, min_y = new_min_x, new_min_y
            max_x, max_y = new_max_x, new_max_y
            end_index = idx

        # slice indices - video frame range; slice_bbox - video crop coords
        slice_indices = (start_index, end_index + 1)
        slice_bbox = (int(min_x), int(min_y), *target_size)
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

    def generate_videos(
        self,
        count: int,
        frame_size: tuple[int, int],
        save_folder_format: str,
        max_length: int = None,
        granularity: int = 2,
    ):
        """
        Generate set amount of video samples. Each video sample starts at a random frame.
            Note, that resulting videos might overlap.
        :param count: number of video samples to generate.
        :param frame_size: resolution of the video sample, in format [w, h].
        :param save_folder_format: folder name format of a video sample.
            Must contain '{}' sequence in it's name, which will be replaced by the video sample number.
        :param max_length: the maximum number of frames in a single video sample.
        :param granularity: examine every `granularity` frames for object-out-of-frame condition.
            The computation is sped up by factor of `granularity`, but video sample length might be slightly imprecise.
        """
        self.initialize(cache_bboxes=False)

        # Randomly select frames
        rnd_fids = np.random.choice(len(self._frame_reader), size=count, replace=False)

        if self._cached_all_bboxes is None:
            for i, fid in tqdm(enumerate(rnd_fids), desc="creating samples", unit="vid", total=count):
                trim_range, crop_dims = self._calc_video_bounds_dynamic(fid, frame_size, max_length, granularity)
                self._crop_and_save_video(save_folder_format.format(i), trim_range, crop_dims)
        else:
            for i, fid in tqdm(enumerate(rnd_fids), desc="creating samples", unit="vid", total=count):
                trim_range, crop_dims = self._calc_video_bounds_cached(fid, frame_size, max_length)
                self._crop_and_save_video(save_folder_format.format(i), trim_range, crop_dims)

    def generate_all_videos(
        self,
        frame_size: tuple[int, int],
        save_folder_format: str,
        max_length: int = None,
    ):
        """
        Generate consecutive series of videos for all the frames stored in the `frame_reader` used for creating this class.
            Note, that the resulting videos do not overlap, and each video starts after the last frame of the previous video.
        :param count: number of video samples to generate.
        :param frame_size: resolution of the video sample, in format [w, h].
        :param save_folder_format: folder name format of a video sample.
            Must contain '{}' sequence in it's name, which will be replaced by the video sample number.
        :param max_length: the maximum number of frames in a single video sample.
        """
        self.initialize(cache_bboxes=True)

        progress_bar = tqdm(desc="creating samples", total=len(self._frame_reader), unit="fr")
        start_frame = 0
        i = 0

        while start_frame < len(self._frame_reader):
            (trim_start, trim_end), crop_dims = self._calc_video_bounds_cached(start_frame, frame_size, max_length)
            self._crop_and_save_video(save_folder_format.format(i), (trim_start, trim_end), crop_dims)

            # update loop params
            i += 1
            start_frame = trim_end
            progress_bar.update(trim_end - trim_start)

        progress_bar.close()
