from __future__ import annotations
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Callable


class Plotter:
    """
    A class for plotting experiment log data.
    The experiment data was previously analyzed by the DataAnalyzer class.
    Supports analysis of multiple logs at once.

    Args:
        data_list (list[pd.DataFrame]): A list of dataframes, each holding the data of a single experiment log.
        plot_height (int, optional): The height of the plot.
        palette (str, optional): The color palette to use for the plots.
        title_size (int, optional): The size of the title text.
        label_size (int, optional): The size of the axis labels and values.
        units (tuple[str, str], optional): The units of the data in format (time, dist). If None, the units are automatically detected.
    """

    def __init__(
        self,
        data_list: list[pd.DataFrame],
        plot_height: int = 7,
        palette: str = "viridis",
        title_size: int = 16,
        label_size: int = 13,
        units: tuple[str, str] = None,
    ) -> None:
        self.plot_height = plot_height
        self.palette = palette
        self.title_size = title_size
        self.label_size = label_size

        for i, data in enumerate(data_list):
            data["log_num"] = i

        self.data = pd.concat([d for d in data_list], ignore_index=True)

        if units is None:
            units = self._detect_units()

        self.time_unit, self.dist_unit = units

    def _detect_units(self) -> tuple[str, str]:
        """
        Detect the time and distance units of the data.

        Returns:
            tuple[str, str]: The units of the data.
        """

        data = self.data.iloc[:5]

        if np.array_equal(data["frame"].round(3), data["time"].round(3)):
            return ("fr", "px")

        else:
            return ("sec", "μm")

    def _get_error_column(self, error_kind: str) -> str:
        if error_kind == "bbox":
            return "bbox_error"
        elif error_kind == "dist":
            return "worm_deviation"
        elif error_kind == "precise":
            return "precise_error"
        else:
            raise ValueError(f"Invalid error kind: {error_kind}")

    def _get_error_label(self, error_kind: str) -> str:
        if error_kind == "dist":
            return f"{error_kind} error ({self.dist_unit})"
        elif error_kind in ["bbox", "precise"]:
            return f"{error_kind} error (% outside)"
        else:
            raise ValueError(f"Invalid error kind: {error_kind}")

    def _clean_data(self, data: pd.DataFrame, *col_names: str) -> pd.DataFrame:
        """
        Takes a subset of the data at the provided columns, and afterwards removes all rows which contain nan values within them.
        The purpose of this function is to clean the data for plotting.

        Args:
            data (pd.DataFrame): The input data
            *col_names (str): The columns of the data to extract and clean

        Returns:
            pd.DataFrame: The input data only at the specified columns, after removing all rows which contain nan.
        """

        col_names = [col for col in col_names if col is not None and col != ""]

        if len(col_names) == 0:
            raise RuntimeError("No valid columns were passed")

        if len(data) == 0:
            raise RuntimeError("The input data was empty")

        data = data[col_names]
        data = data.dropna()

        if len(data) == 0:
            raise RuntimeError(
                "There are no valid rows within the data. Perhaps one of the columns contains only np.nan values"
            )

        return data

    def plot_speed(
        self,
        log_wise: bool = False,
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        plot_kind: str = "hist",
        **kwargs,
    ) -> sns.FacetGrid:
        """
        Plot the speed distribution of the worm.

        Args:
            log_wise (bool, optional): Whether to plot each log separately.
            condition (Callable[[pd.DataFrame], pd.DataFrame], optional): A function to filter the data.
            plot_kind (str, optional): The kind of plot to create. Can be "hist", "kde", or "ecdf".
            **kwargs: Additional keyword arguments to pass the `Plotter.create_distplot` function.
        """

        return self.create_distplot(
            x_col="wrm_speed",
            x_label=f"speed ({self.dist_unit}/{self.time_unit})",
            title="Worm Speed Distribution",
            plot_kind=plot_kind,
            log_wise=log_wise,
            condition=condition,
            kde=True,
            **kwargs,
        )

    def plot_error(
        self,
        error_kind: str = "bbox",
        log_wise: bool = False,
        cycle_wise: bool = False,
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        plot_kind: str = "hist",
        **kwargs,
    ) -> sns.FacetGrid:
        """
        Plot the error distribution.

        Args:
            error_kind (str, optional): The kind of error to plot. Can be "bbox", "dist", or "precise".
            log_wise (bool, optional): Whether to plot each log separately.
            cycle_wise (bool, optional): Whether to calculate single value per cycle.
            condition (Callable[[pd.DataFrame], pd.DataFrame], optional): A function to filter the data.
            plot_kind (str, optional): The kind of plot to create. Can be "hist", "kde", or "ecdf".
            **kwargs: Additional keyword arguments to pass the `Plotter.create_distplot` function.

        Returns:
            sns.FacetGrid: The plot object.
        """

        error_col = self._get_error_column(error_kind)

        data = self.data
        if cycle_wise:
            data = self.data.groupby(["log_num", "cycle"])[error_col].max().reset_index()

        return self.create_distplot(
            x_col=error_col,
            x_label=self._get_error_label(error_kind),
            title=f"{error_kind} Error Distribution",
            plot_kind=plot_kind,
            log_wise=log_wise,
            condition=condition,
            data=data,
            **kwargs,
        )

    def plot_cycle_error(
        self,
        error_kind: str = "bbox",
        log_wise: bool = False,
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        plot_kind: str = "boxen",
        **kwargs,
    ) -> sns.JointGrid:
        """
        Plot the error as a function of the cycle step.

        Args:
            error_kind (str, optional): The kind of error to plot. Can be "bbox", "dist", or "precise".
            log_wise (bool, optional): Whether to plot each log separately.
            condition (Callable[[pd.DataFrame], pd.DataFrame], optional): A function to filter the data.
            plot_kind (str, optional): The kind of plot to create. Can be "strip", "box", "violin", "boxen", "bar", or "count".
            **kwargs: Additional keyword arguments to pass the `Plotter.create_catplot` function.

        Returns:
            sns.JointGrid: The plot object.
        """

        return self.create_catplot(
            x_col="cycle_step",
            y_col=self._get_error_column(error_kind),
            x_label="cycle step (fr)",
            y_label=self._get_error_label(error_kind),
            title=f"{error_kind} error as function of cycle step",
            plot_kind=plot_kind,
            log_wise=log_wise,
            condition=condition,
            **kwargs,
        )

    def plot_speed_vs_error(
        self,
        error_kind: str = "bbox",
        cycle_wise: bool = False,
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        kind: str = "hist",
        **kwargs,
    ) -> sns.JointGrid:
        """
        Plot the speed of the worm vs the error.

        Args:
            error_kind (str, optional): The kind of error to plot. Can be "bbox", "dist", or "precise".
            cycle_wise (bool, optional): Whether to calculate single value per cycle.
            condition (Callable[[pd.DataFrame], pd.DataFrame], optional): A function to filter the data.
            kind (str, optional): The kind of plot to create. Can be "scatter", "kde", "hist", "hex", "reg", or "resid".
            **kwargs: Additional keyword arguments to pass the `Plotter.create_jointplot` function.

        Returns:
            sns.JointGrid: The plot object.
        """

        error_col = self._get_error_column(error_kind)

        data = self.data
        if cycle_wise:
            data = (
                self.data.groupby(["log_num", "cycle"])[[error_col, "wrm_speed"]]
                .aggregate({error_col: "max", "wrm_speed": "mean"})
                .reset_index()
            )

        return self.create_jointplot(
            x_col="wrm_speed",
            y_col=error_col,
            plot_kind=kind,
            x_label=f"speed ({self.dist_unit}/{self.time_unit})",
            y_label=self._get_error_label(error_kind),
            title=f"Speed vs {error_kind} Error",
            condition=condition,
            data=data,
            **kwargs,
        )

    def plot_trajectory(
        self,
        hue_col="log_num",
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        **kwargs,
    ) -> sns.JointGrid:
        """
        Plot the trajectory of the worm.

        Args:
            hue_col (str, optional): The column to use for coloring the plot.
            condition (Callable[[pd.DataFrame], pd.DataFrame], optional): A function to filter the data.
            **kwargs: Additional keyword arguments to pass the `Plotter.create_jointplot` function.

        Returns:
            sns.JointGrid: The plot object.
        """

        plot = self.create_jointplot(
            x_col="wrm_center_x",
            y_col="wrm_center_y",
            x_label=f"X ({self.dist_unit})",
            y_label=f"Y" f" ({self.dist_unit})",
            title=f"Worm Trajectory",
            hue_col=hue_col,
            plot_kind="scatter",
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
        plot_kind: str = "hist",
        **kwargs,
    ) -> sns.JointGrid:
        """
        Plot the size of the worm head.

        Args:
            condition (Callable[[pd.DataFrame], pd.DataFrame], optional): A function to filter the data.
            plot_kind (str, optional): The kind of plot to create. Can be "scatter", "kde", "hist", "hex", "reg", or "resid".
            **kwargs: Additional keyword arguments to pass the `Plotter.create_jointplot` function.

        Returns:
            sns.JointGrid: The plot object.
        """

        return self.create_jointplot(
            x_col="wrm_w",
            y_col="wrm_h",
            x_label=f"width ({self.dist_unit})",
            y_label=f"height ({self.dist_unit})",
            title="Worm Head Size",
            plot_kind=plot_kind,
            condition=condition,
            **kwargs,
        )

    def create_distplot(
        self,
        x_col: str,
        y_col: str = None,
        hue_col: str = None,
        log_wise: bool = False,
        plot_kind: str = "hist",
        x_label: str = "",
        y_label: str = "",
        title: str | None = None,
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] = None,
        data: pd.DataFrame = None,
        **kwargs,
    ) -> sns.FacetGrid:
        """
        Create a distribution plot from the data.

        Args:
            x_col (str): The column to plot on the x-axis.
            y_col (str, optional): The column to plot on the y-axis.
            hue_col (str, optional): The column to use for coloring the plot.
            log_wise (bool, optional): Whether to plot each log separately.
            plot_kind (str, optional): The kind of plot to create. Can be "hist", "kde", or "ecdf".
            x_label (str, optional): The x-axis label.
            y_label (str, optional): The y-axis label.
            title (str, optional): The title of the plot.
            condition (Callable[[pd.DataFrame], pd.DataFrame], optional): A function to filter the data.
            transform (Callable[[pd.DataFrame], pd.DataFrame], optional): A function to transform the data.
            data (pd.DataFrame, optional): Custom data to plot from. If None, the data passed to the constructor of the class is used.
            **kwargs: Additional keyword arguments to pass to the `seaborn.displot` function.

        Returns:
            sns.FacetGrid: The plot object.
        """

        assert plot_kind in ["hist", "kde", "ecdf"]

        if data is None:
            data = self.data

        if transform is not None:
            data = transform(data)

        if condition is not None:
            data = data[condition(data)]

        data = self._clean_data(data, x_col, y_col, hue_col, "log_num")

        palette = self.palette if hue_col is not None else None

        plot = sns.displot(
            data=data,
            x=x_col,
            y=y_col,
            hue=hue_col,
            col="log_num" if log_wise else None,
            kind=plot_kind,
            height=self.plot_height,
            palette=palette,
            **kwargs,
        )

        plot.set_xlabels(x_label.capitalize(), fontsize=self.label_size, fontweight="semibold")
        plot.set_ylabels(y_label.capitalize(), fontsize=self.label_size, fontweight="semibold")
        plot.tick_params(axis="both", labelsize=self.label_size)

        if title is not None:
            if log_wise:
                title = f"Log {{col_name}} :: {title.title()}"
                plot.set_titles(title, size=self.title_size, fontweight="bold")
            else:
                plot.figure.suptitle(title.title(), fontsize=self.title_size, fontweight="bold")

        plot.tight_layout()

        return plot

    def create_catplot(
        self,
        x_col: str,
        y_col: str = None,
        hue_col: str = None,
        log_wise: bool = False,
        plot_kind: str = "strip",
        x_label: str = "",
        y_label: str = "",
        title: str | None = None,
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] = None,
        data: pd.DataFrame = None,
        **kwargs,
    ) -> sns.FacetGrid:
        """
        Create a categorical plot from the data.

        Args:
            x_col (str): The column to plot on the x-axis.
            y_col (str, optional): The column to plot on the y-axis.
            hue_col (str, optional): The column to use for coloring the plot.
            log_wise (bool, optional): Whether to plot each log separately.
            plot_kind (str, optional): The kind of plot to create. Can be "strip", "box", "violin", "boxen", "bar", or "count".
            x_label (str, optional): The x-axis label.
            y_label (str, optional): The y-axis label.
            title (str, optional): The title of the plot.
            condition (Callable[[pd.DataFrame], pd.DataFrame], optional): A function to filter the data.
            transform (Callable[[pd.DataFrame], pd.DataFrame], optional): A function to transform the data.
            data (pd.DataFrame, optional): Custom data to plot from. If None, the data passed to the constructor of the class is used.
            **kwargs: Additional keyword arguments to pass to the `seaborn.catplot` function.

        Returns:
            sns.FacetGrid: The plot object.
        """

        assert plot_kind in ["strip", "box", "violin", "boxen", "bar", "count"]

        if data is None:
            data = self.data

        if transform is not None:
            data = transform(data)

        if condition is not None:
            data = data[condition(data)]

        data = self._clean_data(data, x_col, y_col, hue_col, "log_num")

        palette = self.palette if hue_col is not None else None

        plot = sns.catplot(
            data=data,
            x=x_col,
            y=y_col,
            hue=hue_col,
            col="log_num" if log_wise else None,
            kind=plot_kind,
            height=self.plot_height,
            palette=palette,
            **kwargs,
        )

        plot.set_xlabels(x_label.capitalize(), fontsize=self.label_size, fontweight="semibold")
        plot.set_ylabels(y_label.capitalize(), fontsize=self.label_size, fontweight="semibold")
        plot.tick_params(axis="both", labelsize=self.label_size)

        if title is not None:
            if log_wise:
                title = f"Log {{col_name}} :: {title.title()}"
                plot.set_titles(title, size=self.title_size, fontweight="bold")
            else:
                plot.figure.suptitle(title.title(), fontsize=self.title_size, fontweight="bold")

        plot.tight_layout()

        return plot

    def create_jointplot(
        self,
        x_col: str,
        y_col: str,
        hue_col: str = None,
        plot_kind: str = "scatter",
        x_label: str = "",
        y_label: str = "",
        title: str = "",
        condition: Callable[[pd.DataFrame], pd.DataFrame] = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] = None,
        data: pd.DataFrame = None,
        **kwargs,
    ) -> sns.JointGrid:
        """
        Create a joint plot from the data.

        Args:
            x_col (str): The column to plot on the x-axis.
            y_col (str): The column to plot on the y-axis.
            hue_col (str, optional): The column to use for coloring the plot.
            plot_kind (str, optional): The kind of plot to create. Can be "scatter", "kde", "hist", "hex", "reg", or "resid".
            x_label (str, optional): The x-axis label.
            y_label (str, optional): The y-axis label.
            title (str, optional): The title of the plot.
            condition (Callable[[pd.DataFrame], pd.DataFrame], optional): A function to filter the data.
            transform (Callable[[pd.DataFrame], pd.DataFrame], optional): A function to transform the data.
            data (pd.DataFrame, optional): Custom data to plot from. If None, the data passed to the constructor of the class is used.
            **kwargs: Additional keyword arguments to pass to the `seaborn.jointplot` function.

        Returns:
            sns.JointGrid: The plot object.
        """

        assert plot_kind in ["scatter", "kde", "hist", "hex", "reg", "resid"]

        if data is None:
            data = self.data

        if transform is not None:
            data = transform(data)

        if condition is not None:
            data = data[condition(data)]

        data = self._clean_data(data, x_col, y_col, hue_col)

        palette = self.palette if hue_col is not None else None

        plot = sns.jointplot(
            data=data,
            x=x_col,
            y=y_col,
            hue=hue_col,
            kind=plot_kind,
            height=self.plot_height,
            palette=palette,
            marginal_kws=dict(palette=palette),
            **kwargs,
        )

        plot.set_axis_labels(
            x_label.capitalize(), y_label.capitalize(), fontsize=self.label_size, fontweight="semibold"
        )
        plot.figure.suptitle(title.title(), fontsize=self.title_size, fontweight="bold")
        plot.ax_joint.tick_params(axis="both", labelsize=self.label_size)
        plot.figure.tight_layout()

        return plot
