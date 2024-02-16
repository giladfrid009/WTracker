import numpy as np
from typing import Callable, Tuple

from dataset.bbox_utils import BoxUtils


# TODO: IS THIS FILE EVEN NEEDED ANYMORE?
# PERHAPS IT CAN BE USEFUL FOR THE FUTURE TO CHECK IF MULTIPLE WORMS ARE WITHIN A SINGLE IMAGE,
# OR EVEN IF THERE ARE ANY WORMS IN AN IMAGE.

# TODO: BORROW SOME METHODS FROM HERE BEFORE DELETING THIS FILE.


class BoxExtractor:
    @staticmethod
    def resize_boxes(bboxes: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
        if bboxes is None:
            return None
        assert BoxUtils.is_bbox(bboxes)

        # Unpack columns
        x, y, w, h = bboxes.T

        # Calc center of bbox
        x_mid = x + w // 2
        y_mid = y + h // 2

        # Calc upper left corner
        new_x = x_mid - new_width // 2
        new_y = y_mid - new_height // 2

        # Join the columns back
        new_bboxes = np.column_stack((new_x, new_y, new_width, new_height))
        return new_bboxes

    @staticmethod
    def sanitize_boxes(bboxes: np.ndarray, image_shape: tuple[int]) -> np.ndarray:
        if bboxes is None:
            return None

        # Validate bboxes arg
        assert BoxUtils.is_bbox(bboxes)

        # Get max possible dimensions
        max_height = image_shape[0] - 1
        max_width = image_shape[1] - 1

        # Unpack columns
        x, y, w, h = bboxes.T

        # Make sure bboxes have valid dimensions and within image shape
        good_mask = (x >= 0) & (y >= 0) & (w > 0) & (h > 0)
        good_mask = good_mask & (x + w <= max_width) & (y + h <= max_height)

        # Select only good bboxes
        bboxes = bboxes[good_mask]
        return bboxes

    @staticmethod
    def remove_overlapping_boxes(bboxes: np.ndarray, bboxes_other: np.ndarray = None) -> np.ndarray:
        """
        If bboxes_other is not None then overlap is checked against these bboxes
        """
        if bboxes is None:
            return None

        # Validate args
        assert BoxUtils.is_bbox(bboxes)
        assert bboxes_other is None or (BoxUtils.is_bbox(bboxes_other) and bboxes.shape == bboxes_other.shape)

        # Get num elements
        num_boxes = bboxes.shape[0]

        # Calculate left, right, top, bottom limits
        left = np.expand_dims(bboxes[:, 0], axis=1)
        right = np.expand_dims(bboxes[:, 0] + bboxes[:, 2], axis=1)
        top = np.expand_dims(bboxes[:, 1], axis=1)
        bottom = np.expand_dims(bboxes[:, 1] + bboxes[:, 3], axis=1)

        # Calculate left, right, top, bottom limits of other
        if bboxes_other is None:
            left_other = left
            right_other = right
            top_other = top
            bottom_other = bottom
        else:
            left_other = np.expand_dims(bboxes_other[:, 0], axis=1)
            right_other = np.expand_dims(bboxes_other[:, 0] + bboxes_other[:, 2], axis=1)
            top_other = np.expand_dims(bboxes_other[:, 1], axis=1)
            bottom_other = np.expand_dims(bboxes_other[:, 1] + bboxes_other[:, 3], axis=1)

        # Check for left limit intrusions, right limit intrusions, ...
        check_l = (left <= left_other.T) & (left_other.T <= right)
        check_r = (left <= right_other.T) & (right_other.T <= right)
        check_t = (top <= top_other.T) & (top_other.T <= bottom)
        check_b = (top <= bottom_other.T) & (bottom_other.T <= bottom)

        # Check for combinations of left-top intrusions, left-bottom intrusions, ...
        check_lt = check_l & check_t
        check_lb = check_l & check_b
        check_rt = check_r & check_t
        check_rb = check_r & check_b

        # Get all combinations; get rid of self identical matches
        check = check_lt | check_lb | check_rt | check_rb
        check = np.fill_diagonal(check, False)
        check = np.argwhere(check)

        # Get unique indices of bad bboxes
        bad_indices = np.unique(check)

        # Get indices of good bboxes
        good_indices = np.arange(num_boxes)
        good_indices = good_indices[np.in1d(good_indices, bad_indices, invert=True)]

        # Take only the good bboxes
        good_bboxes = np.take(bboxes, good_indices, axis=0)
        return good_bboxes

    @staticmethod
    def extract_slices(image: np.ndarray, bboxes: np.ndarray) -> list[np.ndarray]:
        if bboxes is None:
            return None

        # Validate args
        assert BoxUtils.is_bbox(bboxes)

        slices = []
        for bbox in bboxes:
            # Deconstruct bbox
            x, y, w, h = bbox

            # Extract a slice and add to slice list
            slice = image[y : y + h, x : x + w]
            slices.append(slice)

        return slices

    def __init__(
        self,
        find_boxes: Callable[[np.ndarray], np.ndarray],
        slice_size: int = 400,
        object_size: int = 150,
    ) -> None:
        self._find_boxes = find_boxes
        self._slice_size = slice_size
        self._object_size = object_size

    def run(self, image: np.ndarray) -> Tuple[np.ndarray, list[np.ndarray]]:
        # Find bboxes according to given params
        bboxes = self._find_boxes(image)

        # Calc slice-sized bboxes
        slice_bboxes = BoxExtractor.resize_boxes(bboxes, new_width=self._slice_size, new_height=self._slice_size)

        # Calc object-sized bboxes
        object_bboxes = BoxExtractor.resize_boxes(slice_bboxes, self._object_size, self._object_size)

        # Remove overlapping bboxes between slice and object bboxes
        slice_bboxes = BoxExtractor.remove_overlapping_boxes(slice_bboxes, object_bboxes)

        # Remove bboxes which are out of bounds
        slice_bboxes = BoxExtractor.sanitize_boxes(slice_bboxes, image.shape)

        # Get corresponding image slices to the camera-bboxes
        image_slices = BoxExtractor.extract_slices(image, slice_bboxes)

        return slice_bboxes, image_slices
