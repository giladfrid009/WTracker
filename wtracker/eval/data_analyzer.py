from __future__ import annotations
import pandas as pd
import numpy as np
import tqdm.contrib.concurrent as concurrent

from wtracker.sim.config import TimingConfig
from wtracker.eval.error_calculator import ErrorCalculator
from wtracker.utils.frame_reader import FrameReader
from wtracker.utils.threading_utils import adjust_num_workers


class DataAnalyzer:
    """
    A class for analyzing simulation log.

    Args:
        time_config (TimingConfig): The timing configuration.
        log_path (pd.DataFrame): Dataframe containing the simulation log data.
    """

    def __init__(
        self,
        time_config: TimingConfig,
        log_data: pd.DataFrame,
    ):
        self.time_config = time_config
        self.data = log_data.copy()
        self._orig_data = log_data
        self._unit = "frame"

    @property
    def unit(self) -> str:
        return self._unit

    def save(self, path: str) -> None:
        """
        Save the full analyzed data to a csv file.
        """
        self._orig_data.to_csv(path, index=False)

    @staticmethod
    def load(time_config: TimingConfig, csv_path: str) -> DataAnalyzer:
        """
        Create a DataAnalyzer object from a csv file containing experiment data,
        regardless whether if it's analyzed or not.

        Args:
            time_config (TimingConfig): The timing configuration.
            csv_path (str): Path to the csv file containing the experiment data.
        """
        data = pd.read_csv(csv_path)
        return DataAnalyzer(time_config, data)

    def initialize(self, period: int = 10):
        """
        Initializes the data analyzer.
        It's essential to call this function if the class was created from a non-analyzed log data.

        Args:
            period (int): The period for calculating speed in frames.
                The speed is calculated by measuring the distance between current frame and period frames before.
        """
        data = self._orig_data

        data["time"] = data["frame"]
        data["cycle_step"] = data["frame"] % self.time_config.cycle_frame_num

        data = DataAnalyzer._calc_centers(data)
        data = DataAnalyzer._calc_speed(data, period)
        data = DataAnalyzer._calc_worm_deviation(data)
        data = DataAnalyzer._calc_errors(data)
        data = data.round(5)

        self._orig_data = data
        self.data = self._orig_data.copy()

    @staticmethod
    def _calc_centers(data: pd.DataFrame) -> pd.DataFrame:
        data["wrm_center_x"] = data["wrm_x"] + data["wrm_w"] / 2
        data["wrm_center_y"] = data["wrm_y"] + data["wrm_h"] / 2
        data["mic_center_x"] = data["mic_x"] + data["mic_w"] / 2
        data["mic_center_y"] = data["mic_y"] + data["mic_h"] / 2
        return data

    @staticmethod
    def _calc_speed(data: pd.DataFrame, n: int) -> pd.DataFrame:
        diff = data["time"].diff(n).to_numpy()
        data["wrm_speed_x"] = data["wrm_center_x"].diff(n) / diff
        data["wrm_speed_y"] = data["wrm_center_y"].diff(n) / diff
        data["wrm_speed"] = np.sqrt(data["wrm_speed_x"] ** 2 + data["wrm_speed_y"] ** 2)
        return data

    @staticmethod
    def _calc_worm_deviation(data: pd.DataFrame) -> pd.DataFrame:
        data["worm_deviation_x"] = data["wrm_center_x"] - data["mic_center_x"]
        data["worm_deviation_y"] = data["wrm_center_y"] - data["mic_center_y"]
        data["worm_deviation"] = np.sqrt(data["worm_deviation_x"] ** 2 + data["worm_deviation_y"] ** 2)
        return data

    @staticmethod
    def _calc_errors(data: pd.DataFrame) -> pd.DataFrame:
        wrm_bboxes = data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].to_numpy()
        mic_bboxes = data[["mic_x", "mic_y", "mic_w", "mic_h"]].to_numpy()
        bbox_error = ErrorCalculator.calculate_bbox_error(wrm_bboxes, mic_bboxes)
        data["bbox_error"] = bbox_error
        data["precise_error"] = 1.0
        return data

    def remove_cycle(self, cycles: int | list[int]):
        """
        Remove the specified cycles from the data.

        Args:
            cycles (int | list[int]): The cycle(s) to remove from the data.
        """
        if isinstance(cycles, int):
            cycles = [cycles]
        mask = self.data["cycle"].isin(cycles)
        self.data = self.data[~mask]

    def clean(
        self,
        trim_cycles: bool = False,
        imaging_only: bool = False,
        bounds: tuple[float, float, float, float] = None,
    ) -> None:
        """
        Clean the data by the provided parameters.

        Args:
            trim_cycles (bool): whether to remove the first and the last cycles from the data.
            imaging_only (bool): Flag indicating whether to include only imaging phases in the analysis.
            legal_bounds (tuple[float, float, float, float]): The legal bounds for worm movement.
        """
        data = self.data

        if imaging_only:
            mask = data["phase"] == "imaging"
            data = data[mask]

        if bounds is not None:
            has_pred = np.isfinite(data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].to_numpy()).all(axis=1)

            mask_wrm = has_pred  # if there is a prediction for a frame then look at worm bbox
            mask_wrm &= (data["wrm_x"] >= bounds[0]) & (data["wrm_x"] + data["wrm_w"] <= bounds[2])
            mask_wrm &= (data["wrm_y"] >= bounds[1]) & (data["wrm_y"] + data["wrm_h"] <= bounds[3])

            mask_mic = ~has_pred  # if there is no prediction for a frame then look at micro bbox
            mask_mic &= (data["mic_x"] >= bounds[0]) & (data["mic_x"] + data["mic_w"] <= bounds[2])
            mask_mic &= (data["mic_y"] >= bounds[1]) & (data["mic_y"] + data["mic_h"] <= bounds[3])

            data = data[mask_wrm | mask_mic]

        if trim_cycles:
            mask = data["cycle"] != 0
            mask &= data["cycle"] != data["cycle"].max()
            data = data[mask]

        self.data = data

    def reset_changes(self):
        """
        Reset the data to its original state.
        Note, that this method will not reset the unit of time and distance.
        """
        self.data = self._orig_data.copy()
        self._unit = "frame"

    def column_names(self) -> list[str]:
        """
        Returns a list of all column names in the analyzed data.

        Returns:
            list[str]: A list of column names.
        """
        return self.data.columns.to_list()

    def change_unit(self, unit: str):
        """
        Changes the unit of time and distance in the data.

        Args:
            unit (str, optional): The new unit of time to convert into. Can be "frame" or "sec".
                If "sec" is chosen, the time will be converted to seconds, and the distance metric is micrometer.
                If "frame" is chosen, the time will be in frames, and the distance metric is pixels.
        """
        assert unit in ["frame", "sec"]

        if self._unit == unit:
            return

        data = self.data

        if unit == "sec":  # frame -> sec
            dist_factor = self.time_config.mm_per_px * 1000
            time_factor = self.time_config.ms_per_frame / 1000

        if unit == "frame":  # sec -> frame
            dist_factor = self.time_config.px_per_mm / 1000
            time_factor = self.time_config.frames_per_sec

        data["time"] *= time_factor
        data[["plt_x", "plt_y"]] *= dist_factor
        data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]] *= dist_factor
        data[["mic_x", "mic_y", "mic_w", "mic_h"]] *= dist_factor
        data[["cam_x", "cam_y", "cam_w", "cam_h"]] *= dist_factor
        data[["wrm_center_x", "wrm_center_y"]] *= dist_factor
        data[["mic_center_x", "mic_center_y"]] *= dist_factor
        data[["worm_deviation_x", "worm_deviation_y", "worm_deviation"]] *= dist_factor
        data[["wrm_speed_x", "wrm_speed_y", "wrm_speed"]] *= dist_factor / time_factor

        self._unit = unit
        self.data = data

    # TODO: TEST
    def calc_precise_error_experimental(
        self,
        worm_reader: FrameReader,
        background: np.ndarray,
        diff_thresh=20,
        num_workers: int = None,
        chunk_size: int = 2000,
    ) -> None:
        """
        Calculate the precise error between the worm and the microscope view.
        This error is segmentation based, and measures the proportion of worm's head that is
        outside of the view of the microscope. Note that this calculation might take a while.

        Args:
            worm_reader (FrameReader): Images of the worm at each frame, cropped to the size of the bounding box
                which was detected around the worm.
            background (np.ndarray): The background image of the entire experiment.
            diff_thresh (int): Difference threshold to differentiate between the background and foreground.
                A foreground object is detected if the pixel value difference with the background is greater than this threshold.
            num_workers (int, optional): The number of workers to use for parallel processing.
                If None, the number of workers is determined automatically.
            chunk_size (int, optional): The size of each processing chunk.
        """
        frames = self._orig_data["frame"].to_numpy().astype(int, copy=False)
        wrm_bboxes = self._orig_data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].to_numpy()
        mic_bboxes = self._orig_data[["mic_x", "mic_y", "mic_w", "mic_h"]].to_numpy()

        errors = np.ones_like(frames, dtype=float)
        mask = np.isfinite(wrm_bboxes).all(axis=1)

        wrm_bboxes = wrm_bboxes[mask]
        mic_bboxes = mic_bboxes[mask]
        frames = frames[mask]

        num_sections = len(frames) // chunk_size
        wrm_bboxes_list = np.array_split(wrm_bboxes, num_sections, axis=0)
        mic_bboxes_list = np.array_split(mic_bboxes, num_sections, axis=0)
        frames_list = np.array_split(frames, num_sections)

        # TODO: add non-multithreaded case whenever num_workers=0

        num_workers = adjust_num_workers(len(frames), chunk_size, num_workers)

        def calc_error(idx: int) -> np.ndarray:
            return ErrorCalculator.calculate_precise(
                background=background,
                worm_bboxes=wrm_bboxes_list[idx],
                mic_bboxes=mic_bboxes_list[idx],
                frame_nums=frames_list[idx],
                worm_reader=worm_reader,
                diff_thresh=diff_thresh,
            )

        results = concurrent.thread_map(
            calc_error,
            list(range(len(wrm_bboxes_list))),
            max_workers=num_workers,
            chunksize=1,
            desc="Extracting bboxes",
            unit="fr",
            leave=False,
        )

        # set the error in the original data
        errors[mask] = np.concatenate(results)
        self._orig_data["precise_error"] = errors

        # copy relevant error entries into the work data
        idx = self.data["frame"].to_numpy(dtype=int, copy=False)
        self.data["precise_error"] = errors[idx]

    def calc_precise_error(
        self,
        worm_reader: FrameReader,
        background: np.ndarray,
        diff_thresh=20,
    ) -> None:
        """
        Calculate the precise error between the worm and the microscope view.
        This error is segmentation based, and measures the proportion of worm's head that is
        outside of the view of the microscope. Note that this calculation might take a while.

        Args:
            worm_reader (FrameReader): Images of the worm at each frame, cropped to the size of the bounding box
                which was detected around the worm.
            background (np.ndarray): The background image of the entire experiment.
            diff_thresh (int): Difference threshold to differentiate between the background and foreground.
                A foreground object is detected if the pixel value difference with the background is greater than this threshold.
        """
        frames = self._orig_data["frame"].to_numpy().astype(np.int32, copy=False)
        wrm_bboxes = self._orig_data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].to_numpy()
        mic_bboxes = self._orig_data[["mic_x", "mic_y", "mic_w", "mic_h"]].to_numpy()

        indices = list(range(len(frames)))

        errors = ErrorCalculator.calculate_precise(
            background=background,
            worm_bboxes=wrm_bboxes,
            mic_bboxes=mic_bboxes,
            frame_nums=frames,
            worm_reader=worm_reader,
            diff_thresh=diff_thresh,
        )

        self._orig_data["precise_error"] = errors

        # copy relevant error entries into the work data
        idx = self.data["frame"].to_numpy(dtype=int, copy=False)
        self.data["precise_error"] = errors[idx]

    def calc_anomalies(
        self,
        no_preds: bool = True,
        min_bbox_error: float = np.inf,
        min_dist_error: float = np.inf,
        min_speed: float = np.inf,
        min_size: float = np.inf,
        remove_anomalies: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate anomalies in the data based on specified criteria.

        Args:
            no_preds (bool, optional): Flag indicating whether to consider instances with missing predictions.
            min_bbox_error (float, optional): Minimum bounding box error threshold to consider as anomaly.
            min_dist_error (float, optional): Minimum distance error threshold to consider as anomaly.
            min_speed (float, optional): Minimum speed threshold to consider as anomaly.
            min_size (float, optional): Minimum size threshold to consider as anomaly.
            remove_anomalies (bool, optional): Flag indicating whether to remove the anomalies from the data.

        Returns:
            pd.DataFrame: DataFrame containing the anomalies found in the data.
        """

        data = self.data

        mask_speed = data["wrm_speed"] >= min_speed
        mask_bbox_error = data["bbox_error"] >= min_bbox_error
        mask_dist_error = data["worm_deviation"] >= min_dist_error
        mask_worm_width = data["wrm_w"] >= min_size
        mask_worm_height = data["wrm_h"] >= min_size

        mask_no_preds = np.isfinite(data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].to_numpy()).all(axis=1) == False
        mask_no_preds = no_preds & mask_no_preds

        mask = mask_speed | mask_bbox_error | mask_dist_error | mask_worm_width | mask_worm_height | mask_no_preds

        anomalies = data[mask].copy()
        anomalies["speed_anomaly"] = mask_speed[mask]
        anomalies["bbox_error_anomaly"] = mask_bbox_error[mask]
        anomalies["dist_error_anomaly"] = mask_dist_error[mask]
        anomalies["width_anomaly"] = mask_worm_width[mask]
        anomalies["height_anomaly"] = mask_worm_height[mask]
        anomalies["no_pred_anomaly"] = mask_no_preds[mask]

        if remove_anomalies:
            self.data = self.data[~mask]

        return anomalies

    def describe(self, columns: list[str] = None, num: int = 3, percentiles: list[float] = None) -> pd.DataFrame:
        """
        Generate descriptive statistics of the specified columns in the table containing the data.

        Args:
            columns (list[str], optional): List of column names to include in the analysis. If None, all columns will be included.
            num (int, optional): Number of evenly spaced percentiles to include in the analysis. If percentiles is not None, this parameter is ignored.
            percentiles (list[float], optional): List of specific percentiles to include in the analysis. If None, evenly spaced percentiles will be used.

        Returns:
            pd.DataFrame: A DataFrame containing the descriptive statistics of the specified columns.
        """
        if columns is None:
            columns = self.column_names()

        if percentiles is None:
            percentiles = np.linspace(start=0, stop=1.0, num=num + 2)[1:-1]

        return self.data[columns].describe(percentiles)

    def print_stats(self) -> None:
        """
        Prints various statistics related to the data.

        This method calculates and prints the following statistics:
        - Count of Removed Frames: The number of frames that were removed from the original data.
        - Total Count of No Pred Frames: The number of frames where the predictions are missing.
        - Total Num of Cycles: The number of unique cycles in the data.
        - Non Perfect Predictions: The percentage of predictions that are not perfect.
        """
        num_removed = len(self._orig_data.index) - len(self.data.index)
        print(f"Count of Removed Frames: {num_removed} ({round(100 * num_removed / len(self._orig_data.index), 3)}%)")

        no_preds = self.data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].isna().any(axis=1).sum()
        print(f"Count of No-Pred Frames: {no_preds} ({round(100 * no_preds / len(self.data.index), 3)}%)")

        num_cycles = self.data["cycle"].nunique()
        print(f"Total Num of Cycles: {num_cycles}")

        non_perfect = (self.data["bbox_error"] > 1e-7).sum() / len(self.data.index)
        print(f"Non Perfect Predictions: {round(100 * non_perfect, 3)}%")
