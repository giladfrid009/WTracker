from typing import Collection
import numpy as np
import cv2 as cv
from tqdm.auto import tqdm

from utils.frame_reader import FrameReader
from utils.bbox_utils import *


class ErrorCalculator:
    # TODO: FIX. DOES NOT VWORK CORRECTLY.
    # TODO: CURRENT PERF: 60 FPS
    # TODO: TEST IMPLEMENTATION
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

        worm_bboxes = np.round(worm_bboxes, decimals=0).astype(int, copy=False)
        mic_bboxes = np.round(mic_bboxes, decimals=0).astype(int, copy=False)

        worm_bboxes = BoxConverter.change_format(worm_bboxes, BoxFormat.XYWH, BoxFormat.XYXY)
        mic_bboxes = BoxConverter.change_format(mic_bboxes, BoxFormat.XYWH, BoxFormat.XYXY)

        wrm_left, wrm_top, wrm_right, wrm_bottom = BoxUtils.unpack(worm_bboxes)
        mic_left, mic_top, mic_right, mic_bottom = BoxUtils.unpack(mic_bboxes)

        # clip worm bounding boxes to the frame size
        h, w = reader.frame_size
        wrm_left = np.maximum(wrm_left, 0)
        wrm_top = np.maximum(wrm_top, 0)
        wrm_right = np.minimum(wrm_right, w)
        wrm_bottom = np.minimum(wrm_bottom, h)

        worm_bboxes = BoxUtils.pack(wrm_left, wrm_top, wrm_right, wrm_bottom)

        # calculate intersection of worm and microscope bounding boxes
        int_left = np.maximum(wrm_left, mic_left)
        int_top = np.maximum(wrm_top, mic_top)
        int_right = np.minimum(wrm_right, mic_right)
        int_bottom = np.minimum(wrm_bottom, mic_bottom)

        int_width = np.maximum(0, int_right - int_left)
        int_height = np.maximum(0, int_bottom - int_top)

        # shift the intersection to the worm view coordinates
        int_left -= wrm_left
        int_top -= wrm_top

        int_bboxes = BoxUtils.pack(int_left, int_top, int_width, int_height)
        int_bboxes = BoxConverter.change_format(int_bboxes, BoxFormat.XYWH, BoxFormat.XYXY)

        errors = np.zeros(len(frame_nums), dtype=float)

        for i, frame_num in tqdm(enumerate(frame_nums), desc="Calculating Error", unit="fr"):
            wrm_bbox = worm_bboxes[i]
            int_bbox = int_bboxes[i]

            if np.isfinite(wrm_bbox).all() == False:
                errors[i] = np.nan
                continue

            frame = reader[frame_num]

            worm_view = frame[wrm_bbox[1] : wrm_bbox[3], wrm_bbox[0] : wrm_bbox[2]]
            bg_view = background[wrm_bbox[1] : wrm_bbox[3], wrm_bbox[0] : wrm_bbox[2]]

            diff = np.abs(worm_view.astype(int) - bg_view.astype(int))

            # if images are color, convert to grayscale
            # conversion is correct if it's BGR
            if diff.ndim == 3 and diff.shape[2] == 3:
                diff = 0.114 * diff[:, :, 0] + 0.587 * diff[:, :, 1] + 0.299 * diff[:, :, 2]

            if diff.ndim != 2:
                raise ValueError("Image must be either a gray or a color image.")

            mask_wrm = diff > diff_thresh

            mask_mic = np.zeros_like(mask_wrm, dtype=bool)
            mask_mic[int_bbox[1] : int_bbox[3], int_bbox[0] : int_bbox[2]] = True

            total = mask_wrm.sum()
            if total == 0:
                errors[i] = 0.0
                continue

            intersection = np.logical_and(mask_wrm, mask_mic).sum()
            error = 1.0 - intersection / total
            errors[i] = error

        return errors

    @staticmethod
    def calculate_bbox_error(worm_bboxes: np.ndarray, mic_bboxes: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def calculate_mse_error(worm_bboxes: np.ndarray, mic_bboxes: np.ndarray) -> np.ndarray:
        """
        Calculates error according to the mean squared error between the centers of the worm and the microscope view.
        """
        worm_centers = BoxUtils.center(worm_bboxes)
        mic_centers = BoxUtils.center(mic_bboxes)
        errors = np.linalg.norm(worm_centers - mic_centers, axis=1)
        return errors
