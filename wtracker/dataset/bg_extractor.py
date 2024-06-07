import numpy as np
from tqdm.auto import tqdm

from wtracker.utils.frame_reader import FrameReader


class BGExtractor:
    """
    A class for extracting the background from a given sequence of frames, provided by a FrameReader.
    """

    def __init__(self, reader: FrameReader):
        self.reader = reader

    def calc_background(self, num_probes: int, sampling: str = "random", method: str = "median") -> np.ndarray:
        """
        Calculate the background of the dataset.

        Args:
            num_probes (int): The number of probes to sample for background calculation.
            sampling (str, optional): The sampling method for selecting probes. Can be "random" or "uniform".
                "uniform" will select frames uniformly spaced from the FrameReader.
                "random" will select frames randomly from the FrameReader.
            method (str, optional): The method for calculating the background. Can be "median" or "mean".
                The background is calculated by either taking the median or mean of the sampled frames.

        Returns:
            np.ndarray: The calculated background as a numpy array.
        """

        assert sampling in ["random", "uniform"]
        assert method in ["median", "mean"]

        length = len(self.reader)
        size = min(num_probes, length)

        if sampling == "random":
            frame_ids = np.random.choice(length, size=size, replace=False)
        elif sampling == "uniform":
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
