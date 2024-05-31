from matplotlib.axes import Axes
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Callable
from functools import partial

from sim.config import TimingConfig
from utils.gui_utils import UserPrompt
from eval.error_calculator import ErrorCalculator


# TODO: review which additional functions are needed to import from the original
# plotter to this one.

# Notice that Plotter2 class is able to switch between units of time
# Also, plotter2 first needs to be explicitly initialized

#TODO: add exp_num/exp_name column
#TODO: add ability for joint plot and histogram to create plot for each exp seperatly
#TODO: replace histplot with distplot

class Plotter2:
    def __init__(
        self,
        time_config: TimingConfig = None,
        log_paths: list[str] = None,
    ) -> None:
        if time_config is None:
            time_config = TimingConfig.load_json()

        if log_paths is None:
            log_paths = UserPrompt.open_file("Select log files", [("csv", ".csv")], multiple=True)
            log_paths = [str(path) for path in log_paths]

        self.time_config = time_config
        self.log_paths = log_paths

        self.data_list = Plotter2.load_logs(self.log_paths)
        self.all_data: pd.DataFrame = None

    @staticmethod
    def load_logs(log_paths: list[str]) -> list[pd.DataFrame]:
        logs: list[pd.DataFrame] = []
        for file in log_paths:
            log = pd.read_csv(file)
            logs.append(log)

        columns = logs[0].columns
        for log in logs:
            assert log.columns.equals(columns), "Columns do not match"

        return logs

    def initialize(
        self,
        n: int = 10,
        imaging_only: bool = True,
        unit: str = "frame",
    ) -> None:
        """
        Initialize the data for analysis.

        Args:
            n (int, optional): The number of frames to calculate speed over. Defaults to 10.
            imaging_only (bool, optional): Whether to include only imaging phases. Defaults to True.
            unit (str, optional): The unit for time. Can be "frame" or "sec". Defaults to "frame".
        """

        assert unit in ["frame", "sec"]

        self.apply_foreach(Plotter2._add_time_column)
        self.apply_foreach(Plotter2._calc_cycle_steps, time_config=self.time_config)

        if unit == "sec":
            self.apply_foreach(Plotter2._frame_to_secs, time_config=self.time_config)

        self.apply_foreach(Plotter2._calc_centers)
        self.apply_foreach(Plotter2._calc_speed, n=n)
        self.apply_foreach(Plotter2._clean)

        if imaging_only:
            self.apply_foreach(Plotter2._remove_phase, phase="moving")

        self.apply_foreach(Plotter2._worm_deviation)
        self.apply_foreach(Plotter2._calc_bbox_error)

        self.all_data = self.concat_all()

    def apply_foreach(self, func: Callable[[pd.DataFrame, dict], pd.DataFrame], **kwargs) -> None:
        partial_func = partial(func, **kwargs)
        for i, data in enumerate(self.data_list):
            self.data_list[i] = partial_func(data)

    def concat_all(self) -> pd.DataFrame:
        return pd.concat(self.data_list, ignore_index=True)

    def column_names(self) -> list[str]:
        return self.data_list[0].columns.to_list()

    def describe(self, columns: list[str] = None, num: int = 3) -> pd.DataFrame:
        if columns is None:
            columns = self.column_names()

        percentiles = np.linspace(start=0, stop=1.0, num=num + 2)[1:-1]
        return self.all_data[columns].describe(percentiles).round(3)

    def print_stats(self):
        num_cycles = sum([len(log["cycle"].unique()) for log in self.data_list])
        print(f"Num of Cycles: {num_cycles}")

        non_perfect = (self.all_data["bbox_area_diff"] > 1e-7).sum() / len(self.all_data.index)
        print(f"Non Perfect Predictions: {round(100 * non_perfect, 3)}%")

    @staticmethod
    def _frame_to_secs(data: pd.DataFrame, time_config: TimingConfig) -> pd.DataFrame:
        data["time"] = data["time"] * time_config.ms_per_frame / 1000
        data[["plt_x", "plt_y"]] = data[["plt_x", "plt_y"]] * time_config.mm_per_px * 1000
        data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]] = (
            data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]] * time_config.mm_per_px * 1000
        )
        data[["mic_x", "mic_y", "mic_w", "mic_h"]] = (
            data[["mic_x", "mic_y", "mic_w", "mic_h"]] * time_config.mm_per_px * 1000
        )
        data[["cam_x", "cam_y", "cam_w", "cam_h"]] = (
            data[["cam_x", "cam_y", "cam_w", "cam_h"]] * time_config.mm_per_px * 1000
        )
        return data

    @staticmethod
    def _add_time_column(data: pd.DataFrame) -> pd.DataFrame:
        data["time"] = data["frame"]
        return data

    @staticmethod
    def _calc_cycle_steps(data: pd.DataFrame, time_config: TimingConfig) -> pd.DataFrame:
        data["cycle_step"] = data["frame"] % time_config.cycle_frame_num
        return data

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
    def _worm_deviation(data: pd.DataFrame) -> pd.DataFrame:
        data["worm_deviation_x"] = data["wrm_center_x"] - data["mic_center_x"]
        data["worm_deviation_y"] = data["wrm_center_y"] - data["mic_center_y"]
        data["worm_deviation"] = np.sqrt(data["worm_deviation_x"] ** 2 + data["worm_deviation_y"] ** 2)
        return data

    @staticmethod
    def _clean(data: pd.DataFrame) -> pd.DataFrame:
        data = Plotter2._remove_nopreds(data)
        data = Plotter2._remove_cycle(data, 0)
        data = Plotter2._remove_cycle(data, data["cycle"].max())
        return data

    @staticmethod
    def _remove_nopreds(data: pd.DataFrame) -> pd.DataFrame:
        data.dropna(inplace=True)
        return data

    @staticmethod
    def _remove_cycle(data: pd.DataFrame, cycle: int) -> pd.DataFrame:
        return data[data["cycle"] != cycle]

    @staticmethod
    def _remove_phase(data: pd.DataFrame, phase: str) -> pd.DataFrame:
        return data[data["phase"] != phase]

    @staticmethod
    def _calc_bbox_error(data: pd.DataFrame) -> pd.DataFrame:
        wrm_bboxes = data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].to_numpy()
        mic_bboxes = data[["mic_x", "mic_y", "mic_w", "mic_h"]].to_numpy()
        area_diff = ErrorCalculator.calculate_approx(wrm_bboxes, mic_bboxes)
        data["bbox_area_diff"] = area_diff
        return data

    def plot_speed_vs_error(self, error_kind: str = "bbox", **kwargs) -> sns.JointGrid:
        if error_kind == "bbox":
            error_col = "bbox_area_diff"
        elif error_kind == "dist":
            error_col = "worm_deviation"
        else:
            raise ValueError(f"Invalid error kind: {error_kind}")

        plot = self.create_jointplot("wrm_speed", error_col, kind="scatter", **kwargs)
        plot.figure.suptitle(f"Speed vs Error")
        plot.set_axis_labels("speed", "Error")
        return plot

    def plot_speed(self) -> Axes:
        plot = self.create_histogram("wrm_speed", stat="density", bins=100)
        plot.figure.suptitle("Worm Speed Distribution")
        plot.set_xlabel("speed")
        plot.set_ylabel("density")
        return plot

    def plot_deviation(self) -> sns.JointGrid:
        plot = self.create_jointplot("cycle_step", "worm_deviation", kind="hist", stat="count")
        plt.title("Distance between worm and microscope centers as a function of cycle step")
        plot.set_axis_labels("cycle step", "distance")
        return plot

    def create_histogram(
        self,
        x_col: str,
        bins: int = None,
        hue_col: str = None,
        condition: Callable = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] = None,
        **kwargs,
    ) -> Axes:
        data = self.all_data
        if transform is not None:
            data = transform(data)

        if condition is not None:
            data = data[condition(data)]

        if bins is None:
            bins = "auto"

        plot = sns.histplot(data=data, x=x_col, hue=hue_col, bins=bins, **kwargs)
        return plot

    def create_jointplot(
        self,
        x_col: str,
        y_col: str,
        kind: str = "scatter",
        hue_col: str = None,
        condition: Callable = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] = None,
        **kwargs,
    ) -> sns.JointGrid:

        data = self.all_data
        if transform is not None:
            data = transform(data)

        if condition is not None:
            data = data[condition(data)]

        plot = sns.jointplot(data=data, x=x_col, y=y_col, hue=hue_col, kind=kind, **kwargs)
        return plot
