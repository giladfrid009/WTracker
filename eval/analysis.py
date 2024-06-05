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


class Plotter:
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

        self.data_list = Plotter.load_logs(self.log_paths)
        self.all_data: pd.DataFrame = None
        self.unit = None

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
        self.unit = unit

        for i, data in enumerate(self.data_list):
            self.data_list[i] = Plotter._add_log_num_column(data, i)

        self.apply_foreach(Plotter._add_time_column)
        self.apply_foreach(Plotter._calc_cycle_steps, time_config=self.time_config)

        if unit == "sec":
            self.apply_foreach(Plotter._frame_to_secs, time_config=self.time_config)

        self.apply_foreach(Plotter._calc_centers)
        self.apply_foreach(Plotter._calc_speed, n=n)

        if legal_bounds is not None:
            if unit == "sec":
                legal_bounds = tuple(x * self.time_config.mm_per_px * 1000 for x in legal_bounds)

            self.apply_foreach(Plotter._remove_out_of_bounds, bounds=legal_bounds)

        self.apply_foreach(lambda df: df[df["cycle"] != 0])
        self.apply_foreach(lambda df: df[df["cycle"] != df["cycle"].max()])

        if imaging_only:
            self.apply_foreach(lambda df: df[df["phase"] == "imaging"])

        self.apply_foreach(Plotter._calc_worm_deviation)
        self.apply_foreach(Plotter._calc_errors)

        self.apply_foreach(lambda df: df.round(5))

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
        no_preds = self.all_data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].isna().any(axis=1).sum()
        print(f"Total Count of No Pred Frames: {no_preds} ({round(100 * no_preds / len(self.all_data.index), 3)}%)")

        num_cycles = sum([len(log["cycle"].unique()) for log in self.data_list])
        print(f"Total Num of Cycles: {num_cycles}")

        non_perfect = (self.all_data["bbox_error"] > 1e-7).sum() / len(self.all_data.index)
        print(f"Non Perfect Predictions: {round(100 * non_perfect, 3)}%")

    def find_anomalies(
        self,
        no_preds: bool = True,
        min_bbox_error: float = np.inf,
        min_dist_error: float = np.inf,
        min_speed: float = np.inf,
        min_size: float = np.inf,
    ) -> pd.DataFrame:

        mask_speed = self.all_data["wrm_speed"] >= min_speed
        mask_bbox_error = self.all_data["bbox_error"] >= min_bbox_error
        mask_dist_error = self.all_data["worm_deviation"] >= min_dist_error
        mask_worm_width = self.all_data["wrm_w"] >= min_size
        mask_worm_height = self.all_data["wrm_h"] >= min_size
        mask_no_preds = self.all_data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].isna().any(axis=1) & no_preds

        mask = mask_speed | mask_bbox_error | mask_dist_error | mask_worm_width | mask_worm_height | mask_no_preds

        mask_speed = mask_speed[mask]
        mask_bbox_error = mask_bbox_error[mask]
        mask_dist_error = mask_dist_error[mask]
        mask_worm_width = mask_worm_width[mask]
        mask_worm_height = mask_worm_height[mask]
        mask_no_preds = mask_no_preds[mask]

        anomalies = self.all_data[mask].copy()

        anomalies["speed_anomaly"] = mask_speed
        anomalies["bbox_error_anomaly"] = mask_bbox_error
        anomalies["dist_error_anomaly"] = mask_dist_error
        anomalies["width_anomaly"] = mask_worm_width
        anomalies["height_anomaly"] = mask_worm_height
        anomalies["no_pred_anomaly"] = mask_no_preds

        return anomalies

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
    def _remove_out_of_bounds(data: pd.DataFrame, bounds: tuple[float, float, float, float]) -> pd.DataFrame:
        mask = (data["wrm_x"] >= bounds[0]) & (data["wrm_x"] + data["wrm_w"] <= bounds[2])
        mask &= (data["wrm_y"] >= bounds[1]) & (data["wrm_y"] + data["wrm_h"] <= bounds[3])
        return data[mask]

    @staticmethod
    def _calc_errors(data: pd.DataFrame) -> pd.DataFrame:
        wrm_bboxes = data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].to_numpy()
        mic_bboxes = data[["mic_x", "mic_y", "mic_w", "mic_h"]].to_numpy()
        bbox_error = ErrorCalculator.calculate_bbox_error(wrm_bboxes, mic_bboxes)
        data["bbox_error"] = bbox_error

        mse_error = ErrorCalculator.calculate_mse_error(wrm_bboxes, mic_bboxes)
        data["mse_error"] = mse_error

        return data

    def calc_precise_error(self, worm_image_paths: list[str], backgrounds: list[np.ndarray], diff_thresh=20) -> None:
        assert len(worm_image_paths) == len(backgrounds) == len(self.data_list)

        for i, df in enumerate(self.data_list):
            worm_reader = FrameReader.create_from_directory(worm_image_paths[i])
            background = backgrounds[i]

            frames = df["frame"].to_numpy().astype(int, copy=False)
            wrm_bboxes = df[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].to_numpy()
            mic_bboxes = df[["mic_x", "mic_y", "mic_w", "mic_h"]].to_numpy()

            # TODO: TEST IF FIXES - DOESNT WORK IF unit is "sec"
            if self.unit == "sec":
                wrm_bboxes = wrm_bboxes * self.time_config.px_per_mm * 1000
                mic_bboxes = mic_bboxes * self.time_config.px_per_mm * 1000

            errors = np.full_like(frames, np.nan, dtype=float)
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
            df["precise_error"] = errors

        self.all_data["precise_error"] = pd.concat([log["precise_error"] for log in self.data_list])

    # TODO: HERE WE DISPLAY THE ERROR PER FRAME, WE NEED TO DISPLAY ERROR PER CYCLE.
    # I.E. ARGMAX OF ERROR PER CYCLE

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
            kind="scatter",
            x_label="speed",
            y_label=f"{error_kind} Error",
            title=f"Speed vs {error_kind} Error",
            condition=condition,
            #            scatter_kws=dict(linewidths=0.5, edgecolor="black"),
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
            data=data.dropna(),
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
            data=data.dropna(),
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
            data=data.dropna(),
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
