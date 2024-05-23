import numpy as np
import cv2 as cv

# TODO: test implementation on real images. we're using numpy directly since it's faster than opencv
class ErrorCalc:
    def __init__(self, background: np.ndarray, diff_thresh: float):
        self._diff_thresh = diff_thresh
        self._background = self._to_grayscale(background)

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.uint8, copy=False)
        if image.ndim == 3:
            if image.shape[-1] == 3:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            elif image.shape[-1] == 1:
                image = image[:, :, 0]
            else:
                raise ValueError("Image must be either a gray or a color image.")
        return image

    def _compute_mask(self, image: np.ndarray, image_bbox: tuple[int, int, int, int]) -> np.ndarray:
        image = self._to_grayscale(image)

        x, y, w, h = image_bbox
        assert (h, w) == image.shape[:2]

        bg = self._background[y : y + h, x : x + w]
        diff = np.abs(image.astype(int) - bg.astype(int))
        mask = diff > self._diff_thresh
        return mask

    # TODO: make implementation more efficient.
    # perhaps accept a list of worm_bboxes and micro_bbox and of worm_views and calculate all errors at once
    # note, that the function find_contour can't be vectorized to, and should process each worm_view separately
    def calculate(
        self,
        worm_view: np.ndarray,
        worm_bbox: tuple[float, float, float, float],
        micro_bbox: tuple[int, int, int, int],
    ) -> float:
        # calculates the error given a micro view and it's bbox, and also given a bbox of the worm.
        # the error is calculated as the proportion of the body of the worm that is within the microscope view

        wrm_mask = self._compute_mask(worm_view, worm_bbox)

        x_wrm, y_wrm, w_wrm, h_wrm = worm_bbox
        x_mic, y_mic, w_mic, h_mic = micro_bbox

        # make micro view relative to the worm bbox
        x_mic = x_mic - x_wrm
        y_mic = y_mic - y_wrm

        if x_mic < 0:
            w_mic += x_mic
            x_mic = 0

        if y_mic < 0:
            h_mic += y_mic
            y_mic = 0

        w_mic = max(0, w_mic)
        h_mic = max(0, h_mic)

        mic_mask = np.zeros((h_wrm, w_wrm), dtype=bool)
        mic_mask[y_mic : y_mic + h_mic, x_mic : x_mic + w_mic] = True

        total = wrm_mask.sum()
        intersection = np.logical_and(wrm_mask, mic_mask).sum()
        error = 1.0 - intersection / total
        return error
