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

        # caching
        self._video_bboxes = None

    def extract_background(
        self,
        num_samples: int,
    ) -> np.ndarray:
        self._video.restart()
        length = len(self._video)
        width, height = self._video.frame_size()

        # Randomly select frames
        frame_ids = np.random.choice(length, size=num_samples, replace=False)
        frame_ids = sorted(frame_ids)

        # Store selected frames in an array
        frames = np.zeros(shape=(num_samples, height, width), dtype=np.uint8)

        # Extract frames
        for i, id in tqdm(enumerate(frame_ids), desc="calculating background", total=num_samples):
            self._video.seek(id)
            frame = self._video.get_frame()

            # Apply transform
            frame = self._transform(frame).astype(np.uint8)

            frames[i] = frame

        # Calculate the median along the time axis
        median = np.median(frames, axis=0).astype(np.uint8)
        return median

    def extract_frame_bbox(
        self,
        frame: np.ndarray,
        background: np.ndarray,
        diff_thresh: int,
    ) -> np.ndarray:

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

    def extract_video_bboxes(
        self,
        bg_samples: int = 100,
        fg_thresh: int = 10,
    ) -> np.ndarray:
        # first, we find the background, which we will use later
        background = self.extract_background(bg_samples)

        self._video.restart()
        length = len(self._video)
        bboxes = np.zeros(shape=(length, 4), dtype=int)

        # extract bbox from each video frame
        for i, frame in tqdm(enumerate(self._video), desc="extracting bboxes", total=len(self._video)):
            bbox = self.extract_frame_bbox(frame, background, fg_thresh)
            bboxes[i] = bbox

        # update cached variable
        self._video_bboxes = bboxes

        return bboxes

    def _find_longest_legal_slice(
        self,
        object_bboxes: np.ndarray,
        start_index: int,
        slice_width: int,
        slice_height: int,
    ) -> tuple:
        """
        Finds the index bounds and the coordinates of the longest video slice of the provided dimensions,
        for which the bounding boxes of the object are within slice size bounds.
        """
        left = object_bboxes[start_index:, 0]
        bottom = object_bboxes[start_index:, 1]
        right = object_bboxes[start_index:, 0] + object_bboxes[start_index:, 2]
        top = object_bboxes[start_index:, 1] + object_bboxes[start_index:, 3]

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

    def create_samples(self, count: int, sample_width: int, sample_height: int, dest_path: str):
        if self._video_bboxes is None:
            raise Exception("please run `extract_video_bboxes` first")

        # Randomly select frames
        start_frames = np.random.choice(len(self._video), size=count, replace=False)
        start_frames = sorted(start_frames)

        for i, start_frame_idx in tqdm(enumerate(start_frames), desc="creating samples", total=count):
            # Find the properties of the video slice for the current sample
            trim_range, crop_dims = self._find_longest_legal_slice(
                object_bboxes=self._video_bboxes,
                start_index=start_frame_idx,
                slice_width=sample_width,
                slice_height=sample_height,
            )

            out_path = dest_path.format(i)

            # Create and save a sample
            self.crop_video(out_path, trim_range, crop_dims)

    def crop_video(
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
