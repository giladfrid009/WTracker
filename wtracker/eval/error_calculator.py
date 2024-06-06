from typing import Collection
import numpy as np
import cv2 as cv
from tqdm.auto import tqdm
from typing import Callable

from wtracker.utils.frame_reader import FrameReader
from wtracker.utils.bbox_utils import BoxUtils, BoxConverter, BoxFormat


class ErrorCalculator:
    """
    The ErrorCalculator class provides methods to calculate different types of errors based on worm position and the microscope view.
    """

    # TODO: TEST IMPLEMENTATIONs
    probe_hook: Callable[[np.ndarray, np.ndarray], None] = None  # takes mask and view for testing

    @staticmethod
    def calculate_segmentation(
        bbox: np.ndarray,
        image: np.ndarray,
        background: np.ndarray,
        diff_thresh: float,
    ) -> np.ndarray:
        """
        Calculates the segmentation error between a view and background image.

        Args:
            bbox (np.ndarray): The bounding box of the image, in the format (x, y, x, y).
            image (np.ndarray): The image to calculate segmentation from.
            background (np.ndarray): The background image.
            diff_thresh (float): The difference threshold to distinguish foreground and background objects from.

        Returns:
            np.ndarray: The segmentation mask.

        Raises:
            ValueError: If the image is not grayscale or color.
        """
        assert image.shape[0] == bbox[3] - bbox[1]
        assert image.shape[1] == bbox[2] - bbox[0]

        bg_view = background[bbox[1] : bbox[3], bbox[0] : bbox[2]]

        diff = np.abs(image.astype(int) - bg_view.astype(int))

        # if images are color, convert to grayscale
        # conversion is correct if it's BGR
        if diff.ndim == 3 and diff.shape[2] == 3:
            diff = 0.114 * diff[:, :, 0] + 0.587 * diff[:, :, 1] + 0.299 * diff[:, :, 2]

        if diff.ndim != 2:
            raise ValueError("Image must be either a gray or a color image.")

        mask_wrm = diff > diff_thresh

        return mask_wrm

    @staticmethod
    def calculate_precise(
        background: np.ndarray,
        worm_bboxes: np.ndarray,
        mic_bboxes: np.ndarray,
        frame_nums: Collection[int],
        worm_reader: FrameReader,
        diff_thresh: float = 10,
    ) -> np.ndarray:
        """
        Calculates the precise error for each frame in the given sequence.
        This error is based on precise segmentation of the worm object from the frame, and
        determining the exact proportion of worm's body outside the microscope view.

        Args:
            background (np.ndarray): The background image.
            worm_bboxes: A numpy array of shape (N, 4) representing the bounding boxes of worms. The bounding boxes should be in the format (x, y, w, h).
            mic_bboxes: A numpy array of shape (N, 4) representing the bounding boxes of the microscope. The bounding boxes should be in the format (x, y, w, h).
            frame_nums (Collection[int]): The frame numbers to calculate the error for.
            worm_reader (FrameReader): A frame reader containing segmented worm images for each frame. These worm images should match the shape of the worm bounding boxes.
                Frames passed in frame_nums are read from this reader by index.
            diff_thresh (float, optional): The difference threshold to distinguish foreground and background objects from. Defaults to 10.
                A foreground object is detected if the pixel value difference with the background is greater than this threshold.

        Returns:
            np.ndarray: Array of errors of shape (N,) representing the precise segmentation error for each frame.

        Raises:
            AssertionError: If the length of frame_nums, worm_bboxes, and mic_bboxes do not match.

        """
        assert len(frame_nums) == worm_bboxes.shape[0] == mic_bboxes.shape[0]

        worm_bboxes = BoxConverter.change_format(worm_bboxes, BoxFormat.XYWH, BoxFormat.XYXY)
        mic_bboxes = BoxConverter.change_format(mic_bboxes, BoxFormat.XYWH, BoxFormat.XYXY)

        worm_bboxes = BoxUtils.round(worm_bboxes, BoxFormat.XYXY)
        mic_bboxes = BoxUtils.round(mic_bboxes, BoxFormat.XYXY)

        wrm_left, wrm_top, wrm_right, wrm_bottom = BoxUtils.unpack(worm_bboxes)
        mic_left, mic_top, mic_right, mic_bottom = BoxUtils.unpack(mic_bboxes)

        # clip worm bounding boxes to the frame size
        H, W = background.shape[:2]
        wrm_left = np.clip(wrm_left, a_min=0, a_max=W - 1)
        wrm_top = np.clip(wrm_top, a_min=0, a_max=H - 1)
        wrm_right = np.clip(wrm_right, a_min=0, a_max=W - 1)
        wrm_bottom = np.clip(wrm_bottom, a_min=0, a_max=H - 1)

        worm_bboxes = BoxUtils.pack(wrm_left, wrm_top, wrm_right, wrm_bottom)

        # calculate intersection of worm and microscope bounding boxes
        int_left = np.clip(wrm_left, a_min=mic_left, a_max=mic_right)
        int_top = np.clip(wrm_top, a_min=mic_top, a_max=mic_bottom)
        int_right = np.clip(wrm_right, a_min=mic_left, a_max=mic_right)
        int_bottom = np.clip(wrm_bottom, a_min=mic_top, a_max=mic_bottom)

        int_width = np.maximum(0, int_right - int_left)
        int_height = np.maximum(0, int_bottom - int_top)

        # shift the intersection to the worm view coordinates
        int_left -= wrm_left
        int_top -= wrm_top

        int_bboxes = BoxUtils.pack(int_left, int_top, int_width, int_height)
        int_bboxes = BoxConverter.change_format(int_bboxes, BoxFormat.XYWH, BoxFormat.XYXY)

        errors = np.zeros(len(frame_nums), dtype=float)

        for i, frame_num in tqdm(enumerate(frame_nums), total=len(frame_nums), desc="Calculating Error", unit="fr"):
            wrm_bbox = worm_bboxes[i]
            int_bbox = int_bboxes[i]

            if np.isfinite(wrm_bbox).all() == False:
                errors[i] = 1
                continue

            worm_view = worm_reader[frame_num]

            mask_wrm = ErrorCalculator.calculate_segmentation(
                bbox=wrm_bbox,
                image=worm_view,
                background=background,
                diff_thresh=diff_thresh,
            )

            if ErrorCalculator.probe_hook is not None:
                ErrorCalculator.probe_hook(worm_view, mask_wrm)

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
        Calculate the bounding box error between worm bounding boxes and microscope bounding boxes.
        This error calculates the proportion of the worm bounding box that is outside the microscope bounding box.

        Args:
            worm_bboxes: A numpy array of shape (N, 4) representing the bounding boxes of worms. The bounding boxes should be in the format (x, y, w, h).
            mic_bboxes: A numpy array of shape (N, 4) representing the bounding boxes of the microscope. The bounding boxes should be in the format (x, y, w, h).

        Returns:
            np.ndarray: Array of errors of shape (N,) representing the bounding box error for each pair of worm and microscope bounding boxes.
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
        errors[total == 0] = 0.0

        return errors

    @staticmethod
    def calculate_mse_error(worm_bboxes: np.ndarray, mic_bboxes: np.ndarray) -> np.ndarray:
        """
        Calculates the Mean Squared Error (MSE) error between the centers of worm bounding boxes and microscope bounding boxes.

        Args:
            worm_bboxes: A numpy array of shape (N, 4) representing the bounding boxes of worms. The bounding boxes should be in the format (x, y, w, h).
            mic_bboxes: A numpy array of shape (N, 4) representing the bounding boxes of the microscope. The bounding boxes should be in the format (x, y, w, h).

        Returns:
            np.ndarray: Array of errors of shape (N,) representing the MSE error for each pair of worm and microscope bounding boxes.
        """
        worm_centers = BoxUtils.center(worm_bboxes)
        mic_centers = BoxUtils.center(mic_bboxes)
        errors = np.mean((worm_centers - mic_centers) ** 2, axis=1)
        return errors
