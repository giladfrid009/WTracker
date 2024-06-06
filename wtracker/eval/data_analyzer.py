from __future__ import annotations
import pandas as pd
import numpy as np

from wtracker.sim.config import TimingConfig
from wtracker.eval.error_calculator import ErrorCalculator
from wtracker.utils.frame_reader import FrameReader
from wtracker.utils.io_utils import pickle_save_object, pickle_load_object


class DataAnalyzer:
    def __init__(
        self,
        time_config: TimingConfig = None,
        log_path: str = None,
        unit: str = "frame",
    ):

        assert unit in ["frame", "sec"]

        self.unit = unit
        self.time_config = time_config
        self.log_path = log_path
        self.table = pd.read_csv(log_path)

    def save(self, path: str) -> None:
        pickle_save_object(self, path)

    @staticmethod
    def load(path: str) -> DataAnalyzer:
        return pickle_load_object(path)

    def initialize(
        self,
        period: int = 10,
        imaging_only: bool = True,
        legal_bounds: tuple[float, float, float, float] = None,
    ) -> None:
        """
        Initialize the data for analysis.

        Args:
            period (int, optional): The number of frames to calculate speed over. Defaults to 10.
            imaging_only (bool, optional): Whether to include only imaging phases. Defaults to True.
            legal_bounds (tuple[float, float, float, float], optional): The legal bounds for the worm head location, in frames. Defaults to None. Bounds format is (x_min, y_min, x_max, y_max).
        """

        data = self.table
        data["time"] = data["frame"]
        data["cycle_step"] = data["frame"] % self.time_config.cycle_frame_num

        if self.unit == "sec":
            data = self._convert_frames_to_secs(data)

        data = self._calc_centers(data)
        data = self._calc_speed(data, period)

        if legal_bounds is not None:
            if self.unit == "sec":
                legal_bounds = tuple(x * self.time_config.mm_per_px * 1000 for x in legal_bounds)

            data = self._remove_out_of_bounds(data, legal_bounds)

        data = data[data["cycle"] != 0]
        data = data[data["cycle"] != data["cycle"].max()]

        if imaging_only:
            data = data[data["phase"] == "imaging"]

        data = self._calc_worm_deviation(data)
        data = self._calc_errors(data)
        data = data.round(5)

        self.table = data

    def _convert_frames_to_secs(self, data: pd.DataFrame) -> pd.DataFrame:
        frame_to_secs = self.time_config.ms_per_frame / 1000
        px_to_micrometer = self.time_config.mm_per_px * 1000
        data["time"] = data["time"] * frame_to_secs
        data[["plt_x", "plt_y"]] = data[["plt_x", "plt_y"]] * px_to_micrometer
        data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]] = data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]] * px_to_micrometer
        data[["mic_x", "mic_y", "mic_w", "mic_h"]] = data[["mic_x", "mic_y", "mic_w", "mic_h"]] * px_to_micrometer
        data[["cam_x", "cam_y", "cam_w", "cam_h"]] = data[["cam_x", "cam_y", "cam_w", "cam_h"]] * px_to_micrometer
        return data

    def _calc_centers(self, data: pd.DataFrame) -> pd.DataFrame:
        data["wrm_center_x"] = data["wrm_x"] + data["wrm_w"] / 2
        data["wrm_center_y"] = data["wrm_y"] + data["wrm_h"] / 2
        data["mic_center_x"] = data["mic_x"] + data["mic_w"] / 2
        data["mic_center_y"] = data["mic_y"] + data["mic_h"] / 2
        return data

    def _calc_speed(self, data: pd.DataFrame, n: int) -> pd.DataFrame:
        diff = data["time"].diff(n).to_numpy()
        data["wrm_speed_x"] = data["wrm_center_x"].diff(n) / diff
        data["wrm_speed_y"] = data["wrm_center_y"].diff(n) / diff
        data["wrm_speed"] = np.sqrt(data["wrm_speed_x"] ** 2 + data["wrm_speed_y"] ** 2)
        return data

    def _calc_worm_deviation(self, data: pd.DataFrame) -> pd.DataFrame:
        data["worm_deviation_x"] = data["wrm_center_x"] - data["mic_center_x"]
        data["worm_deviation_y"] = data["wrm_center_y"] - data["mic_center_y"]
        data["worm_deviation"] = np.sqrt(data["worm_deviation_x"] ** 2 + data["worm_deviation_y"] ** 2)
        return data

    def _remove_out_of_bounds(self, data: pd.DataFrame, bounds: tuple[float, float, float, float]) -> pd.DataFrame:
        mask = (data["wrm_x"] >= bounds[0]) & (data["wrm_x"] + data["wrm_w"] <= bounds[2])
        mask &= (data["wrm_y"] >= bounds[1]) & (data["wrm_y"] + data["wrm_h"] <= bounds[3])
        return data[mask]

    def _calc_errors(self, data: pd.DataFrame) -> pd.DataFrame:
        wrm_bboxes = data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].to_numpy()
        mic_bboxes = data[["mic_x", "mic_y", "mic_w", "mic_h"]].to_numpy()
        bbox_error = ErrorCalculator.calculate_bbox_error(wrm_bboxes, mic_bboxes)
        mse_error = ErrorCalculator.calculate_mse_error(wrm_bboxes, mic_bboxes)
        data["bbox_error"] = bbox_error
        data["mse_error"] = mse_error
        data["precise_error"] = 1.0

        return data

    def calc_precise_error(self, worm_reader: FrameReader, background: np.ndarray, diff_thresh=20) -> None:
        assert len(worm_reader) == len(self.table)

        frames = self.table["frame"].to_numpy().astype(int, copy=False)
        wrm_bboxes = self.table[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].to_numpy()
        mic_bboxes = self.table[["mic_x", "mic_y", "mic_w", "mic_h"]].to_numpy()

        # TODO: TEST IF WORKS WHEN unit is "sec"
        if self.unit == "sec":
            wrm_bboxes = wrm_bboxes * self.time_config.px_per_mm * 1000
            mic_bboxes = mic_bboxes * self.time_config.px_per_mm * 1000

        errors = np.ones_like(frames, np.nan, dtype=float)
        mask = np.isfinite(wrm_bboxes).all(axis=1)

        wrm_bboxes = wrm_bboxes[mask]
        mic_bboxes = mic_bboxes[mask]
        frames = frames[mask]

        results = ErrorCalculator.calculate_precise(
            background,
            wrm_bboxes,
            mic_bboxes,
            worm_reader=worm_reader,
            frame_nums=frames,
            diff_thresh=diff_thresh,
        )

        errors[mask] = results
        self.table["precise_error"] = errors

    def column_names(self) -> list[str]:
        return self.table.columns.to_list()

    def find_anomalies(
        self,
        no_preds: bool = True,
        min_bbox_error: float = np.inf,
        min_dist_error: float = np.inf,
        min_speed: float = np.inf,
        min_size: float = np.inf,
    ) -> pd.DataFrame:

        mask_speed = self.table["wrm_speed"] >= min_speed
        mask_bbox_error = self.table["bbox_error"] >= min_bbox_error
        mask_dist_error = self.table["worm_deviation"] >= min_dist_error
        mask_worm_width = self.table["wrm_w"] >= min_size
        mask_worm_height = self.table["wrm_h"] >= min_size
        mask_no_preds = self.table[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].isna().any(axis=1) & no_preds

        mask = mask_speed | mask_bbox_error | mask_dist_error | mask_worm_width | mask_worm_height | mask_no_preds

        mask_speed = mask_speed[mask]
        mask_bbox_error = mask_bbox_error[mask]
        mask_dist_error = mask_dist_error[mask]
        mask_worm_width = mask_worm_width[mask]
        mask_worm_height = mask_worm_height[mask]
        mask_no_preds = mask_no_preds[mask]

        anomalies = self.table[mask].copy()

        anomalies["speed_anomaly"] = mask_speed
        anomalies["bbox_error_anomaly"] = mask_bbox_error
        anomalies["dist_error_anomaly"] = mask_dist_error
        anomalies["width_anomaly"] = mask_worm_width
        anomalies["height_anomaly"] = mask_worm_height
        anomalies["no_pred_anomaly"] = mask_no_preds
        return anomalies

    def describe(self, columns: list[str] = None, num: int = 3, percentiles: list[float] = None) -> pd.DataFrame:
        if columns is None:
            columns = self.column_names()

        if percentiles is None:
            percentiles = np.linspace(start=0, stop=1.0, num=num + 2)[1:-1]
            
        return self.table[columns].describe(percentiles)

    def print_stats(self) -> None:
        no_preds = self.table[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].isna().any(axis=1).sum()
        print(f"Total Count of No Pred Frames: {no_preds} ({round(100 * no_preds / len(self.table.index), 3)}%)")

        num_cycles = self.table["cycle"].nunique()
        print(f"Total Num of Cycles: {num_cycles}")

        non_perfect = (self.table["bbox_error"] > 1e-7).sum() / len(self.table.index)
        print(f"Non Perfect Predictions: {round(100 * non_perfect, 3)}%")
