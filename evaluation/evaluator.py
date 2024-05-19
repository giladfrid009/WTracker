import numpy as np
import cv2 as cv

class Evaluator:
    def __init__(self, background: np.ndarray, diff_thresh: float):
        self._background = background
        self._diff_thresh = diff_thresh

    def find_contour(self, worm_view: np.ndarray, worm_bbox: tuple[int, int, int, int]) -> np.ndarray:
        worm_view = worm_view.astype(np.uint8, copy=False)

        # get mask according to the threshold value
        x, y, w, h = worm_bbox

        assert (w, h) == worm_view.shape[:2]

        bg = self._background[y : y + h, x : x + w]

        diff = cv.absdiff(worm_view.astype(bg.dtype, copy=False), bg)
        _, mask = cv.threshold(diff, self._diff_thresh, 255, cv.THRESH_BINARY)

        # apply morphological ops to the mask
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv.dilate(mask, np.ones((11, 11), np.uint8))

        # extract contours
        contours, _ = cv.findContours(mask[:, :, 0], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        largest_contour = max(contours, key=cv.contourArea)

        img = np.zeros_like(worm_view, dtype=np.uint8)
        cv.drawContours(img, largest_contour, -1, color=(1, 1, 1), thickness=cv.FILLED)
        contour_mask = img.astype(bool, copy=False).any(axis=-1)

        return contour_mask

    # calculates the error given a micro view and it's bbox, and also given a bbox of the worm.
    # the error is calculated as the proportion of the body of the worm that is within the microscope view
    def calc_error(
        self,
        worm_view: np.ndarray,
        worm_bbox: tuple[float, float, float, float],
        micro_bbox: tuple[int, int, int, int],
    ) -> float:
        wrm_mask = self.find_contour(worm_view, worm_bbox)

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
        error = intersection / total
        return error
