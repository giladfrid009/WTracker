from __future__ import annotations
import pandas as pd
import seaborn as sns
from typing import Callable
import itertools

from wtracker.eval.data_analyzer import DataAnalyzer


class Plotter:
    def __init__(
        self,
        data_list: list[DataAnalyzer],
        plot_height: int = 7,
    ) -> None:
        self._plot_height = plot_height
        self._data_list = data_list

        serials = [d.data["log_num"].unique().tolist() for d in data_list]
        serials = itertools.chain(*serials)
        assert len(set(serials)) == len(data_list), "Serial numbers must be unique"

        self._all_data = pd.concat([d.data for d in data_list], ignore_index=True)

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
            if "precise_error" not in self._all_data.columns:
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
            if "precise_error" not in self._all_data.columns:
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

        q = self._all_data["worm_deviation"].quantile(percentile)

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

        data = self._all_data
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
            height=self._plot_height,
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

        data = self._all_data
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
            height=self._plot_height,
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

        data = self._all_data

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
            height=self._plot_height,
            **kwargs,
        )

        plot.set_axis_labels(x_label.capitalize(), y_label.capitalize())
        plot.figure.suptitle(title.title())
        plot.figure.tight_layout()

        return plot
