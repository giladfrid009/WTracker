import cv2 as cv
import numpy as np
from view_controller import VideoStream
from tqdm import tqdm
from typing import Callable
import ffmpeg


class SampleExtractor:
    def __init__(
        self,
        video: VideoStream,
        frame_transform: Callable[[np.ndarray], np.ndarray] = None,
    ) -> None:
        self._video = video

        if frame_transform is None:
            frame_transform = lambda x: x
        self._transform = frame_transform

        # for caching
        self._video_bboxes = None

    # TODO: implement
    def _resize_boxes(self, bboxes: np.ndarray, resize_factor: float) -> np.ndarray:
        assert resize_factor > 0

        # Unpack columns
        x, y, w, h = bboxes.T

        return bboxes

    def calc_video_background(
        self,
        num_probes: int,
    ) -> np.ndarray:
        self._video.restart()
        length = len(self._video)
        width, height = self._video.frame_size()

        # Randomly select frames
        frame_ids = np.random.choice(length, size=num_probes, replace=False)
        frame_ids = sorted(frame_ids)

        # Store selected frames in an array
        frames = np.zeros(shape=(num_probes, height, width), dtype=np.uint8)

        # Extract frames
        for i, id in tqdm(enumerate(frame_ids), desc="calculating background", total=num_probes):
            self._video.seek(id)
            frame = self._video.get_frame()

            # Apply transform
            frame = self._transform(frame).astype(np.uint8)
            frames[i] = frame

        # Calculate the median along the time axis
        median = np.median(frames, axis=0).astype(np.uint8)
        return median

    def calc_image_bbox(
        self,
        image: np.ndarray,
        background: np.ndarray,
        diff_thresh: int,
    ) -> np.ndarray:

        image = self._transform(image).astype(np.uint8)

        # Calculate difference between background and image
        diff = np.abs(image.astype(np.int16) - background.astype(np.int16))
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

    def calc_video_bboxes(
        self,
        bg_probes: int = 100,
        fg_thresh: int = 10,
        size_factor: float = 1.0,
    ) -> np.ndarray:
        # first, we find the background, which we will use later
        background = self.calc_video_background(bg_probes)

        self._video.restart()
        length = len(self._video)
        bboxes = np.zeros(shape=(length, 4), dtype=int)

        # extract bbox from each video frame
        for i, frame in tqdm(enumerate(self._video), desc="extracting bboxes", total=len(self._video)):
            bbox = self.calc_image_bbox(frame, background, fg_thresh)
            bboxes[i] = bbox

        bboxes = self._resize_boxes(bboxes, size_factor)

        # update cached variable
        self._video_bboxes = bboxes

        return bboxes

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

    def _transform_and_save_sample(
        self,
        save_path: str,
        trim_range: tuple[int, int],
        crop_dims: tuple[int, int, int, int],
    ):
        from pathlib import Path

        # create parent dir if doesn't exist
        folder = Path(save_path).parent
        Path(folder).mkdir(parents=True, exist_ok=True)

        start, end = trim_range
        x, y, w, h = crop_dims

        # extract matching slice from the video
        stream = ffmpeg.input(self._video.path())
        stream = ffmpeg.crop(stream, x, y, w, h)
        stream = ffmpeg.trim(stream, start_frame=start, end_frame=end)
        stream = ffmpeg.output(stream, save_path)
        stream = ffmpeg.overwrite_output(stream)
        ffmpeg.run(stream, quiet=True)

    def generate_samples(
        self,
        count: int,
        width: int,
        height: int,
        save_path: str,
    ):
        if self._video_bboxes is None:
            raise Exception(f"please run `{self.calc_video_bboxes.__name__}` first")

        # Randomly select frames
        frame_ids = np.random.choice(len(self._video), size=count, replace=False)
        frame_ids = sorted(frame_ids)

        for i, fid in tqdm(enumerate(frame_ids), desc="creating samples", total=count):
            # Find the properties of the video sample starting from `fid` frame
            trim_range, crop_dims = self._analyze_sample_properties(self._video_bboxes, fid, width, height)

            # format the saving path to match current sample
            sample_path = save_path.format(i)

            # create and save the sample
            self._transform_and_save_sample(sample_path, trim_range, crop_dims)

    def generate_all_samples(
        self,
        width: int,
        height: int,
        save_path: str,
    ):
        if self._video_bboxes is None:
            raise Exception(f"please run `{self.calc_video_bboxes.__name__}` first")

        start_frame = 0
        iter = 0

        progress_bar = tqdm(desc="creating samples", total=len(self._video), unit="fr")
        while start_frame < len(self._video):
            # Find the properties of the video sample starting from start_frame
            trim_range, crop_dims = self._analyze_sample_properties(self._video_bboxes, start_frame, width, height)

            # format the saving path to match current sample
            sample_path = save_path.format(iter)

            # create and save the sample
            self._transform_and_save_sample(sample_path, trim_range, crop_dims)

            # updating progress bar and the next start_frame
            iter += 1
            trim_start, trim_end = trim_range
            start_frame = trim_end
            progress_bar.update(trim_end - trim_start)

        progress_bar.close()
