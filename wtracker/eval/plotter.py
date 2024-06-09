from __future__ import annotations
import pandas as pd
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
    """

    def __init__(
        self,
        data_list: list[pd.DataFrame],
        plot_height: int = 7,
        palette: str = "viridis",
    ) -> None:
        self.plot_height = plot_height
        self.palette = palette

        for i, data in enumerate(data_list):
            data["log_num"] = i

        self.data = pd.concat([d for d in data_list], ignore_index=True)

    def _get_error_column(self, error_kind: str) -> str:
        if error_kind == "bbox":
            return "bbox_error"
        elif error_kind == "dist":
            return "worm_deviation"
        elif error_kind == "precise":
            return "precise_error"
        else:
            raise ValueError(f"Invalid error kind: {error_kind}")

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
            x_label="speed",
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
            cycle_wise (bool, optional): Whether to plot each cycle separately.
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
            x_label=f"{error_kind} error",
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

        error_col = self._get_error_column(error_kind)

        return self.create_catplot(
            x_col="cycle_step",
            y_col=error_col,
            x_label="cycle step",
            y_label=f"{error_kind} error",
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
            cycle_wise (bool, optional): Whether to plot each cycle separately.
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
            x_label="speed",
            y_label=f"{error_kind} error",
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
            x_label="X",
            y_label="Y",
            title="Worm Trajectory",
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
            x_label="width",
            y_label="height",
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

        palette = self.palette if hue_col is not None else None

        plot = sns.displot(
            data=data.dropna(),
            x=x_col,
            y=y_col,
            hue=hue_col,
            col="log_num" if log_wise else None,
            kind=plot_kind,
            height=self.plot_height,
            palette=palette,
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

        palette = self.palette if hue_col is not None else None

        plot = sns.catplot(
            data=data.dropna(),
            x=x_col,
            y=y_col,
            hue=hue_col,
            col="log_num" if log_wise else None,
            kind=plot_kind,
            height=self.plot_height,
            palette=palette,
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

        palette = self.palette if hue_col is not None else None

        plot = sns.jointplot(
            data=data.dropna(),
            x=x_col,
            y=y_col,
            hue=hue_col,
            kind=plot_kind,
            height=self.plot_height,
            palette=palette,
            marginal_kws=dict(palette=palette),
            **kwargs,
        )

        plot.set_axis_labels(x_label.capitalize(), y_label.capitalize())
        plot.figure.suptitle(title.title())
        plot.figure.tight_layout()

        return plot
