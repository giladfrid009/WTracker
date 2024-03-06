import cv2 as cv
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib import concurrent
import multiprocessing

from data.frame_reader import FrameReader


class BoxCalculator:
    def __init__(self, frame_reader: FrameReader, bg_probes: int = 100, diff_thresh: int = 10):
        assert bg_probes > 0 and diff_thresh > 0

        self._frame_reader = frame_reader
        self.bg_probes = bg_probes
        self.diff_thresh = diff_thresh

        self._all_bboxes = np.full(len(frame_reader), np.nan, dtype=int)
        self._background = None

    def __len__(self) -> int:
        return len(self._all_bboxes)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.get_bbox(idx)

    def all_bboxes(self) -> np.ndarray:
        return self._all_bboxes

    def get_bbox(self, frame_idx: int) -> np.ndarray:
        bbox = self._all_bboxes[frame_idx]
        if np.isnan(bbox[0]):
            bbox = self.calc_single_bounding_box(frame_idx)
            self._all_bboxes[frame_idx] = bbox
        return bbox

    def get_background(self) -> np.ndarray:
        bg = self._background
        if bg is None:
            bg = self.calc_background()
            self._background = bg
        return bg

    def calc_background(self) -> np.ndarray:
        length = len(self._frame_reader)
        size = min(self.bg_probes, length)

        # get frames
        frame_ids = np.random.choice(length, size=size, replace=False)
        extracted_list = []
        for frame_id in tqdm(frame_ids, desc="Extracting background frames", unit="fr"):
            frame = self._frame_reader[frame_id]
            extracted_list.append(frame)

        # calculate the median along the time axis
        extracted = np.stack(extracted_list, axis=0)
        median = np.median(extracted, axis=0).astype(np.uint8, copy=False)
        return median

    def calc_single_bounding_box(self, frame_idx: int) -> np.ndarray:
        # get mask according to the threshold value
        frame = self._frame_reader[frame_idx]
        background = self.calc_background()
        diff = np.abs(frame.astype(np.int16, copy=False) - background.astype(np.int16, copy=False))
        _, mask = cv.threshold(diff.astype(np.uint8, copy=False), self.diff_thresh, 255, cv.THRESH_BINARY)

        # apply morphological ops to the mask
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv.dilate(mask, np.ones((11, 11), np.uint8))

        # extract contours and bbox
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            zero_bbox = np.array([0, 0, 0, 0])
            self._all_bboxes[frame_idx] = zero_bbox
            return zero_bbox

        largest_contour = max(contours, key=cv.contourArea)
        largest_bbox = cv.boundingRect(largest_contour)
        largest_bbox = np.asanyarray(largest_bbox, dtype=int)
        return largest_bbox

    def calc_bounding_boxes(
        self,
        start: int = 0,
        end: int = None,
        step: int = 1,
        num_workers: int = multiprocessing.cpu_count(),
        chunk_size: int = 50,
    ) -> np.ndarray:

        if end is None:
            end = len(self._frame_reader)

        frame_indices = range(start, end, step)
        self.get_background()

        if num_workers > 0:
            bbox_list = concurrent.process_map(
                self.get_bbox,
                frame_indices,
                max_workers=num_workers,
                chunksize=chunk_size,
                desc="Extracting bboxes",
                unit="fr",
            )

            for idx, bbox in zip(frame_indices, bbox_list):
                self._all_bboxes[idx] = bbox

        else:
            for idx in tqdm(frame_indices, desc="Extracting bboxes", unit="fr"):
                self.get_bbox(idx)

        return self._all_bboxes
