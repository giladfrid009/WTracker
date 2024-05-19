import numpy as np
from tqdm.auto import tqdm

from frame_reader import FrameReader


class BGExtractor:
    def __init__(self, reader: FrameReader):
        self.reader = reader

    def calc_background(self, num_probes: int, spacing: str = "random", method: str = "median") -> np.ndarray:
        assert spacing in ["random", "uniform"]
        assert method in ["median", "mean"]

        length = len(self.reader)
        size = min(num_probes, length)

        if spacing == "random":
            frame_ids = np.random.choice(length, size=size, replace=False)
        elif spacing == "uniform":
            frame_ids = np.linspace(0, length - 1, num=size)
            frame_ids = np.unique(frame_ids.astype(int, copy=False))

        if method == "median":
            bg = self._calc_background_median(frame_ids)
        elif method == "mean":
            bg = self._calc_background_mean(frame_ids)

        return bg

    def _calc_background_mean(self, frame_ids: np.ndarray) -> np.ndarray:
        sum = np.zeros(self.reader.frame_shape, dtype=np.float64)

        # read frames
        for frame_id in tqdm(frame_ids, desc="Extracting background frames", unit="fr"):
            frame = self.reader[frame_id]
            sum += frame

        mean = sum / len(frame_ids)
        return mean.astype(np.uint8, copy=False)

    def _calc_background_median(self, frame_ids: np.ndarray) -> np.ndarray:
        # get frames
        extracted_list = []
        for frame_id in tqdm(frame_ids, desc="Extracting background frames", unit="fr"):
            frame = self.reader[frame_id]
            extracted_list.append(frame)

        # calculate the median along the time axis
        extracted = np.stack(extracted_list, axis=0)
        median = np.median(extracted, axis=0).astype(np.uint8, copy=False)

        return median
