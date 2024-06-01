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
from utils.frame_reader import FrameReader
from dataset.bg_extractor import BGExtractor

# TODO: should we add a find_anomalies function that finds all the rows in the data with anomalous values?
# for example, when the worm is moving too fast, or when the error is too high


class Plotter2:
    def __init__(
        self,
        time_config: TimingConfig = None,
        log_paths: list[str] = None,
        plot_height: int = 7,
    ) -> None:
        self.plot_height = plot_height

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
        unit: str = "frame",
        n: int = 10,
        imaging_only: bool = True,
        legal_bounds: tuple[float, float, float, float] = None,
    ) -> None:
        """
        Initialize the data for analysis.

        Args:
            unit (str, optional): The unit for time. Can be "frame" or "sec". Defaults to "frame".
            n (int, optional): The number of frames to calculate speed over. Defaults to 10.
            imaging_only (bool, optional): Whether to include only imaging phases. Defaults to True.
            legal_bounds (tuple[float, float, float, float], optional): The legal bounds for the worm head location, in frames. Defaults to None. Bounds format is (x_min, y_min, x_max, y_max).
        """

        assert unit in ["frame", "sec"]

        for i, data in enumerate(self.data_list):
            self.data_list[i] = Plotter2._add_log_num_column(data, i)

        self.apply_foreach(Plotter2._add_time_column)
        self.apply_foreach(Plotter2._calc_cycle_steps, time_config=self.time_config)

        if unit == "sec":
            self.apply_foreach(Plotter2._frame_to_secs, time_config=self.time_config)

        self.apply_foreach(Plotter2._calc_centers)
        self.apply_foreach(Plotter2._calc_speed, n=n)

        if legal_bounds is not None:
            if unit == "sec":
                legal_bounds = tuple(x * self.time_config.mm_per_px * 1000 for x in legal_bounds)

            self.apply_foreach(Plotter2._remove_out_of_bounds, bounds=legal_bounds)

        # TODO: do we want to fill nopreds or remove them? when calculating speed, filling would be better.

        # self.apply_foreach(Plotter2._fill_nopreds)
        self.apply_foreach(Plotter2._remove_nopreds)
        self.apply_foreach(Plotter2._remove_first_cycle)
        self.apply_foreach(Plotter2._remove_first_cycle)

        if imaging_only:
            self.apply_foreach(Plotter2._remove_phase, phase="moving")

        self.apply_foreach(Plotter2._calc_worm_deviation)
        self.apply_foreach(Plotter2._calc_errors)

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

        non_perfect = (self.all_data["bbox_error"] > 1e-7).sum() / len(self.all_data.index)
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
    def _add_log_num_column(data: pd.DataFrame, num: int) -> pd.DataFrame:
        data["log_num"] = num
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
    def _calc_worm_deviation(data: pd.DataFrame) -> pd.DataFrame:
        data["worm_deviation_x"] = data["wrm_center_x"] - data["mic_center_x"]
        data["worm_deviation_y"] = data["wrm_center_y"] - data["mic_center_y"]
        data["worm_deviation"] = np.sqrt(data["worm_deviation_x"] ** 2 + data["worm_deviation_y"] ** 2)
        return data

    @staticmethod
    def _remove_nopreds(data: pd.DataFrame) -> pd.DataFrame:
        data = data.dropna()
        return data

    @staticmethod
    def _fill_nopreds(data: pd.DataFrame) -> pd.DataFrame:
        data = data.ffill()
        return data

    @staticmethod
    def _remove_out_of_bounds(data: pd.DataFrame, bounds: tuple[float, float, float, float]) -> pd.DataFrame:
        mask = (data["wrm_x"] >= bounds[0]) & (data["wrm_x"] + data["wrm_w"] <= bounds[2])
        mask &= (data["wrm_y"] >= bounds[1]) & (data["wrm_y"] + data["wrm_h"] <= bounds[3])
        return data[mask]

    @staticmethod
    def _remove_first_cycle(data: pd.DataFrame) -> pd.DataFrame:
        return data[data["cycle"] != 0]

    @staticmethod
    def _remove_last_cycle(data: pd.DataFrame) -> pd.DataFrame:
        return data[data["cycle"] != data["cycle"].max()]

    @staticmethod
    def _remove_phase(data: pd.DataFrame, phase: str) -> pd.DataFrame:
        return data[data["phase"] != phase]

    @staticmethod
    def _calc_errors(data: pd.DataFrame) -> pd.DataFrame:
        wrm_bboxes = data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].to_numpy()
        mic_bboxes = data[["mic_x", "mic_y", "mic_w", "mic_h"]].to_numpy()
        bbox_error = ErrorCalculator.calculate_bbox_error(wrm_bboxes, mic_bboxes)
        data["bbox_error"] = bbox_error

        mse_error = ErrorCalculator.calculate_mse_error(wrm_bboxes, mic_bboxes)
        data["mse_error"] = mse_error

        return data

    # TODO: is this really a good place for this function?
    def calc_precise_error(self, frames: list[FrameReader], bg_probes=1000, diff_thresh=20):
        for i, data in enumerate(self.data_list):
            reader = frames[i]
            background = BGExtractor(reader).calc_background(
                num_probes=bg_probes,
                sampling="uniform",
                method="median",
            )

            worm_bboxes = data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].to_numpy()
            mic_bboxes = data[["mic_x", "mic_y", "mic_w", "mic_h"]].to_numpy()

            precise_error = ErrorCalculator.calculate_precise(
                background,
                worm_bboxes,
                mic_bboxes,
                reader=reader,
                frame_nums=data["frame"].values,
                diff_thresh=diff_thresh,
            )

            data["precise_error"] = precise_error

        self.all_data["precise_error"] = pd.concat([log["precise_error"] for log in self.data_list])

    def plot_speed_vs_error(
        self,
        error_kind: str = "bbox",
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        **kwargs,
    ) -> sns.JointGrid:
        if error_kind == "bbox":
            error_col = "bbox_error"
        elif error_kind == "dist":
            error_col = "worm_deviation"
        elif error_kind == "mse":
            error_col = "mse_error"
        elif error_kind == "precise":
            if "precise_error" not in self.all_data.columns:
                raise ValueError("Precise error have not been calculated")
            error_col = "precise_error"
        else:
            raise ValueError(f"Invalid error kind: {error_kind}")

        plot = self.create_jointplot(
            x_col="wrm_speed",
            y_col=error_col,
            kind="reg",
            x_label="speed",
            y_label="error",
            title=f"Speed vs {error_kind} Error",
            condition=condition,
            scatter_kws=dict(linewidths=0.5, edgecolor="black"),
            line_kws=dict(color="r"),
            **kwargs,
        )

        return plot

    def plot_speed(
        self,
        log_wise: bool = False,
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        **kwargs,
    ) -> sns.FacetGrid:

        plot = self.create_distplot(
            x_col="wrm_speed",
            kind="hist",
            x_label="speed",
            title="Worm Speed Distribution",
            log_wise=log_wise,
            condition=condition,
            kde=True,
            **kwargs,
        )

        return plot

    def plot_error(
        self,
        error_kind: str = "bbox",
        log_wise: bool = False,
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        **kwargs,
    ) -> sns.FacetGrid:
        if error_kind == "bbox":
            error_col = "bbox_error"
        elif error_kind == "dist":
            error_col = "worm_deviation"
        elif error_kind == "mse":
            error_col = "mse_error"
        elif error_kind == "precise":
            if "precise_error" not in self.all_data.columns:
                raise ValueError("Precise error have not been calculated")
            error_col = "precise_error"
        else:
            raise ValueError(f"Invalid error kind: {error_kind}")

        plot = self.create_distplot(
            x_col=error_col,
            kind="hist",
            x_label="error",
            title=f"{error_kind} Error Distribution",
            log_wise=log_wise,
            condition=condition,
            kde=True,
            **kwargs,
        )
        return plot

    def plot_trajectory(
        self,
        hue_col="log_num",
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        **kwargs,
    ) -> sns.JointGrid:

        plot = self.create_jointplot(
            x_col="wrm_center_x",
            y_col="wrm_center_y",
            x_label="X",
            y_label="Y",
            title="Worm Trajectory",
            hue_col=hue_col,
            kind="scatter",
            alpha=1,
            linewidth=0,
            condition=condition,
            **kwargs,
        )

        plot.ax_marg_x.remove()
        plot.ax_marg_y.remove()
        plot.ax_joint.invert_yaxis()

        return plot

    def plot_head_size(
        self,
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        **kwargs,
    ) -> sns.JointGrid:

        plot = self.create_jointplot(
            x_col="wrm_w",
            y_col="wrm_h",
            x_label="width",
            y_label="height",
            title="Worm Head Size",
            kind="hex",
            condition=condition,
            **kwargs,
        )

        return plot

    def plot_deviation(
        self,
        percentile: float = 0.999,
        log_wise: bool = False,
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        **kwargs,
    ) -> sns.JointGrid:

        q = self.all_data["worm_deviation"].quantile(percentile)

        if condition is not None:
            cond_func = lambda x: condition(x) & (x["worm_deviation"] < q)
        else:
            cond_func = lambda x: x["worm_deviation"] < q

        plot = self.create_catplot(
            x_col="cycle_step",
            y_col="worm_deviation",
            x_label="cycle step",
            y_label="distance",
            kind="violin",
            title="Distance between worm and microscope centers as function of cycle step",
            log_wise=log_wise,
            condition=cond_func,
            **kwargs,
        )

        return plot

    def create_distplot(
        self,
        x_col: str,
        y_col: str = None,
        hue_col: str = None,
        log_wise: bool = False,
        kind: str = "hist",
        x_label: str = "",
        y_label: str = "",
        title: str | None = None,
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] = None,
        **kwargs,
    ) -> sns.FacetGrid:

        assert kind in ["hist", "kde", "ecdf"]

        data = self.all_data
        if transform is not None:
            data = transform(data)

        if condition is not None:
            data = data[condition(data)]

        plot = sns.displot(
            data=data,
            x=x_col,
            y=y_col,
            hue=hue_col,
            col="log_num" if log_wise else None,
            kind=kind,
            height=self.plot_height,
            **kwargs,
        )

        plot.set_xlabels(x_label.capitalize())
        plot.set_ylabels(y_label.capitalize())

        if title is not None:
            if log_wise:
                title = f"Log {{col_name}} :: {title.title()}"
                plot.set_titles(title)
            else:
                plot.figure.suptitle(title.title())

        plot.tight_layout()

        return plot

    def create_catplot(
        self,
        x_col: str,
        y_col: str = None,
        hue_col: str = None,
        log_wise: bool = False,
        kind: str = "strip",
        x_label: str = "",
        y_label: str = "",
        title: str | None = None,
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] = None,
        **kwargs,
    ) -> sns.FacetGrid:

        assert kind in ["strip", "box", "violin", "boxen", "bar", "count"]

        data = self.all_data
        if transform is not None:
            data = transform(data)

        if condition is not None:
            data = data[condition(data)]

        plot = sns.catplot(
            data=data,
            x=x_col,
            y=y_col,
            hue=hue_col,
            col="log_num" if log_wise else None,
            kind=kind,
            height=self.plot_height,
            **kwargs,
        )

        plot.set_xlabels(x_label.capitalize())
        plot.set_ylabels(y_label.capitalize())

        if title is not None:
            if log_wise:
                title = f"Log {{col_name}} :: {title.title()}"
                plot.set_titles(title)
            else:
                plot.figure.suptitle(title.title())

        plot.tight_layout()

        return plot

    def create_jointplot(
        self,
        x_col: str,
        y_col: str,
        hue_col: str = None,
        kind: str = "scatter",
        x_label: str = "",
        y_label: str = "",
        title: str = "",
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] = None,
        **kwargs,
    ) -> sns.JointGrid:

        assert kind in ["scatter", "kde", "hist", "hex", "reg", "resid"]

        data = self.all_data

        if transform is not None:
            data = transform(data)

        if condition is not None:
            data = data[condition(data)]

        plot = sns.jointplot(
            data=data,
            x=x_col,
            y=y_col,
            hue=hue_col,
            kind=kind,
            height=self.plot_height,
            **kwargs,
        )

        plot.set_axis_labels(x_label.capitalize(), y_label.capitalize())
        plot.figure.suptitle(title.title())
        plot.figure.tight_layout()

        return plot
