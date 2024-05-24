from typing import Collection
import numpy as np
import cv2 as cv


from frame_reader import FrameReader
from utils.bbox_utils import *


class ErrorCalculator:
    @staticmethod
    def _image_to_grayscale(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.uint8, copy=False)
        if image.ndim == 3:
            if image.shape[-1] == 3:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            elif image.shape[-1] == 1:
                image = image[:, :, 0]
            else:
                raise ValueError("Image must be either a gray or a color image.")
        return image

    @staticmethod
    def calculate_precise_single(
        bg_view: np.ndarray,
        worm_view: np.ndarray,
        worm_bbox: np.ndarray,
        micro_bbox: np.ndarray,
        diff_thresh: float,
    ) -> float:

        worm_view = ErrorCalculator._image_to_grayscale(worm_view)
        bg_view = ErrorCalculator._image_to_grayscale(bg_view)
        mask_worm = np.abs(worm_view.astype(int) - bg_view.astype(int)) > diff_thresh

        worm_bbox = BoxConverter.change_format(worm_bbox, BoxFormat.XYWH, BoxFormat.XYXY)
        micro_bbox = BoxConverter.change_format(micro_bbox, BoxFormat.XYWH, BoxFormat.XYXY)
        wrm_left, wrm_top, wrm_right, wrm_bottom = BoxUtils.unpack(worm_bbox)
        mic_left, mic_top, mic_right, mic_bottom = BoxUtils.unpack(micro_bbox)

        int_left = max(wrm_left, mic_left)
        int_top = max(wrm_top, mic_top)
        int_right = min(wrm_right, mic_right)
        int_bottom = min(wrm_bottom, mic_bottom)
        int_width = max(0, int_right - int_left)
        int_height = max(0, int_bottom - int_top)

        # shift the intersection to the worm view coordinates
        int_left -= wrm_left
        int_top -= wrm_top

        mask_mic = np.zeros(worm_view.shape[:2], dtype=bool)
        mask_mic[int_left : int_left + int_height, int_left : int_left + int_width] = True

        total = mask_worm.sum()
        if total == 0:
            return 0.0

        intersection = np.logical_and(mask_worm, mask_mic).sum()
        error = 1.0 - intersection / total

        return error

    # TODO: make implementation more efficient.
    # perhaps accept a list of worm_bboxes and micro_bbox and of worm_views and calculate all errors at once
    # note, that the function find_contour can't be vectorized to, and should process each worm_view separately
    @staticmethod
    def calculate_precise(
        background: np.ndarray,
        worm_bboxes: np.ndarray,
        mic_bboxes: np.ndarray,
        frame_nums: Collection[int],
        reader: FrameReader,
        diff_thresh: float = 10,
    ) -> np.ndarray:
        """
        Calculates error according to the precise difference between the worm and the microscope view.
        The error is calculated as the proportion of the worm that is not within the microscope view.
        The worm body is calculated by thresholding the difference between the worm view and the background.
        """
        assert len(frame_nums) == worm_bboxes.shape[0] == mic_bboxes.shape[0]

        worm_bboxes = BoxConverter.change_format(worm_bboxes, BoxFormat.XYWH, BoxFormat.XYXY)
        wrm_left, wrm_top, wrm_right, wrm_bottom = BoxUtils.unpack(worm_bboxes)

        h, w = reader.frame_size
        wrm_left = np.maximum(wrm_left, 0)
        wrm_top = np.maximum(wrm_top, 0)
        wrm_right = np.minimum(wrm_right, w)
        wrm_bottom = np.minimum(wrm_bottom, h)

        worm_bboxes = BoxUtils.pack(wrm_left, wrm_top, wrm_right, wrm_bottom)
        worm_bboxes = BoxConverter.change_format(worm_bboxes, BoxFormat.XYXY, BoxFormat.XYWH)

        errors = np.zeros(len(frame_nums), dtype=float)

        for i, frame_num in enumerate(frame_nums):
            wx, wy, ww, wh = worm_bboxes[i]
            worm_view = reader[frame_num][wy : wy + wh, wx : wx + ww]
            bg_view = background[wy : wy + wh, wx : wx + ww]

            errors[i] = ErrorCalculator.calculate_precise_single(
                bg_view=bg_view,
                worm_view=worm_view,
                worm_bbox=worm_bboxes[i],
                micro_bbox=mic_bboxes[i],
                diff_thresh=diff_thresh,
            )

        errors = np.nan_to_num(errors, nan=0, neginf=0.0, posinf=0.0, copy=False)

        return errors

    @staticmethod
    def calculate_approx(worm_bboxes: np.ndarray, mic_bboxes: np.ndarray) -> np.ndarray:
        """
        Calculates error according to the bounding box difference between the worm and the microscope view.
        The error is calculated as the proportion of the worm bounding box that is not within the microscope view.
        """

        wrm_left, wrm_top, wrm_width, wrm_height = BoxUtils.unpack(worm_bboxes)
        mic_left, mic_top, mic_width, mic_height = BoxUtils.unpack(mic_bboxes)
        wrm_right, wrm_bottom = wrm_left + wrm_width, wrm_top + wrm_height
        mic_right, mic_bottom = mic_left + mic_width, mic_top + mic_height

        int_left = np.maximum(wrm_left, mic_left)
        int_top = np.maximum(wrm_top, mic_top)
        int_right = np.minimum(wrm_right, mic_right)
        int_bottom = np.minimum(wrm_bottom, mic_bottom)

        int_width = np.maximum(0, int_right - int_left)
        int_height = np.maximum(0, int_bottom - int_top)

        intersection = int_width * int_height
        total = wrm_width * wrm_height

        errors = 1.0 - intersection / total

        errors = np.nan_to_num(errors, nan=0, neginf=0.0, posinf=0.0, copy=False)

        return errors
