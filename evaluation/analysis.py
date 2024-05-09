import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.simulator import TimingConfig


class Plotter:
    def __init__(self, log_path: str, timing_config: TimingConfig):
        self._data = pd.read_csv(log_path)
        self._header = self._data.columns
        self.timing_config = timing_config

    def data_prep_frames(self, n: int = 1) -> pd.DataFrame:
        data = self._data.copy()

        data = Plotter.concat(data, Plotter.calc_centers(data, "wrm"))
        data = Plotter.concat(data, Plotter.calc_centers(data, "mic"))
        data = Plotter.remove_no_pred_rows(data)
        data = Plotter.concat(data, Plotter.calc_speed(data, n=n))
        data = Plotter.concat(data, Plotter.calc_area_diff(data), Plotter.calc_max_edge_diff(data))
        data = Plotter.remove_cycle(data, 0)
        return data

    def data_prep_cycles(self, n: int = 15) -> pd.DataFrame:
        data = self._data.copy()
        data = Plotter.data_prep_frames(data, n=1)
        data = Plotter.concat(data, Plotter.calc_speed(data, n=n))
        data = Plotter.concat(data, Plotter.calc_area_diff(data), Plotter.calc_max_edge_diff(data))
        data = Plotter.remove_cycle(data, 0)
        return data

    def print_statistics(self) -> pd.DataFrame:
        print("##################### No Preds #####################")
        no_pred_mask = np.ma.mask_or(self._data["wrm_x"].isna(), self._data["wrm_x"] < 0)
        no_pred_frames = (self._data[no_pred_mask])["frame"]
        print(f"Num of No Preds: {len(no_pred_frames)}")
        print(f"No prediction in frames: {no_pred_frames.to_list()}")
        print(f"corresponding cycle steps: {(no_pred_frames%self.timing_config.cycle_length).to_list()}")
        print("##################### Cycles #####################")
        print(f"Num of cycles: {self._data['cycle'].nunique()}")
        print("##################### Area Diff #####################")
        
        data = self.data_prep_frames()
        non_perfect_pred_ratio = (data["bbox_area_diff"] > 1e-7).sum() / len(data.index)
        print(f"Non Perfect Predictions: {round(100 * non_perfect_pred_ratio, 3)}%")
        print("##################### General #####################")
        display(data.describe())

        return data

    @staticmethod
    def concat(*dataframes: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(list(dataframes), axis=1, ignore_index=False, copy=False)

    @staticmethod
    def remove_no_pred_rows(data: pd.DataFrame) -> pd.DataFrame:
        return data.dropna(inplace=False)

    @staticmethod
    def remove_cycle(data: pd.DataFrame, cycle: int) -> pd.DataFrame:
        indexes = data.loc[data["cycle"] == cycle].index
        return data.drop(index=indexes)

    @staticmethod
    def remove_no_pred_cycles(data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove cycles with negative bounding box coordinates or dimensions, which indicate that no worm was detected.
        """
        # Check if any row within each cycle has negative values in 'worm' columns
        has_negative = data.groupby("cycle")[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].transform("min").lt(0).any(axis=1)

        # Filter the dataframe to keep only cycles without negative values
        filtered_df = data[~has_negative].copy()

        return filtered_df

    @staticmethod
    def remove_phase(data: pd.DataFrame, phase: str) -> pd.DataFrame:
        """
        Remove rows with the specified phase.
        """
        return data[data["phase"] != phase]

    @staticmethod
    def calc_max_edge_diff(data: pd.DataFrame) -> np.ndarray:
        """
        Calculate the length difference between the edges of the worm's bounding boxes and the microscope's bounding boxes.

        Returns:
            numpy.ndarray: Array of length differences between the edges of the bounding boxes
        """

        worm_boxes = data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].values
        mic_boxes = data[["mic_x", "mic_y", "mic_w", "mic_h"]].values

        worm_left, worm_right = worm_boxes[:, 0], worm_boxes[:, 0] + worm_boxes[:, 2]
        worm_top, worm_bottom = worm_boxes[:, 1], worm_boxes[:, 1] + worm_boxes[:, 3]
        mic_left, mic_right = mic_boxes[:, 0], mic_boxes[:, 0] + mic_boxes[:, 2]
        mic_top, mic_bottom = mic_boxes[:, 1], mic_boxes[:, 1] + mic_boxes[:, 3]

        # Calculate the length difference between the edges of the bounding boxes
        x_diff = np.maximum(0, np.maximum(worm_right - mic_right, mic_left - worm_left))
        y_diff = np.maximum(0, np.maximum(worm_bottom - mic_bottom, mic_top - worm_top))

        return pd.DataFrame({"bbox_edge_diff": np.maximum(x_diff, y_diff)}, index=data.index)

    @staticmethod
    def calc_area_diff(data: pd.DataFrame) -> np.ndarray:
        """
        Calculate the area difference between the worms bounding boxes and the microscope's bounding boxes.
        The area difference is the area by which the worm's bounding box exceeds the microscope's bounding box.

        Returns:
            numpy.ndarray: Array of length differences between the edges of the bounding boxes
        """

        worm_boxes = data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].values
        mic_boxes = data[["mic_x", "mic_y", "mic_w", "mic_h"]].values

        worm_left, worm_right = worm_boxes[:, 0], worm_boxes[:, 0] + worm_boxes[:, 2]
        worm_top, worm_bottom = worm_boxes[:, 1], worm_boxes[:, 1] + worm_boxes[:, 3]
        mic_left, mic_right = mic_boxes[:, 0], mic_boxes[:, 0] + mic_boxes[:, 2]
        mic_top, mic_bottom = mic_boxes[:, 1], mic_boxes[:, 1] + mic_boxes[:, 3]

        left = np.maximum(worm_left, mic_left)
        right = np.minimum(worm_right, mic_right)
        top = np.maximum(worm_top, mic_top)
        bottom = np.minimum(worm_bottom, mic_bottom)

        inter_width = np.maximum(0, right - left)
        inter_height = np.maximum(0, bottom - top)

        inter_area = inter_width * inter_height
        worm_area = worm_boxes[:, 2] * worm_boxes[:, 3]

        area_diff = (worm_area - inter_area) / worm_area

        return pd.DataFrame({"bbox_area_diff": area_diff}, index=data.index)

    @staticmethod
    def calc_centers(data: pd.DataFrame, field_name: str = "wrm") -> pd.DataFrame:
        """
        Calculate the centers of the bounding boxes and add them to the data.
        """
        center_x = data[f"{field_name}_x"] + data[f"{field_name}_w"] / 2
        center_y = data[f"{field_name}_y"] + data[f"{field_name}_h"] / 2

        centers = pd.DataFrame(
            {f"{field_name}_center_x": center_x, f"{field_name}_center_y": center_y}, index=data.index
        )
        return centers

    @staticmethod
    def axis_movement(data: pd.DataFrame, n: int = 1) -> pd.DataFrame:
        frame_diff = data["frame"].diff(n).to_numpy()
        vec_x = data["wrm_center_x"].diff(n) / frame_diff
        vec_y = data["wrm_center_y"].diff(n) / frame_diff

        return pd.DataFrame({"vec_x": vec_x, "vec_y": vec_y}, index=data.index)

    @staticmethod
    def calc_speed(data: pd.DataFrame, n: int = 1) -> pd.DataFrame:
        """
        Calculate the worm speed and add it to the data.
        """
        frame_diff = data["frame"].diff(n).to_numpy()
        wrm_speed = np.sqrt(data["wrm_center_x"].diff(n) ** 2 + data["wrm_center_y"].diff(n) ** 2) / frame_diff

        return pd.DataFrame({"wrm_speed": wrm_speed}, index=data.index)

    @staticmethod
    def get_cycle_stats(data: pd.DataFrame, transforms: dict) -> pd.DataFrame:
        return data.groupby("cycle")[list(transforms.keys())].agg(transforms).reset_index()

    @staticmethod
    def bbox_area(data: pd.DataFrame, width_col: str, height_col: str) -> np.ndarray:
        mask = data[width_col] < 0
        data.loc[mask, width_col] = 0

        mask = data[height_col] < 0
        data.loc[mask, height_col] = 0

        return data[width_col] * data[height_col]

    @staticmethod
    def rolling_average(data: pd.DataFrame, window_size: int, column: str) -> pd.DataFrame:
        rolling_avg = data[column].rolling(window=window_size, center=False).mean()
        return rolling_avg

    @staticmethod
    def scatter_data(
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str = "",
        x_label: str = None,
        y_label: str = None,
    ):
        if x_label is None:
            x_label = x_col

        if y_label is None:
            y_label = y_col

        x_data = data[x_col]
        y_data = data[y_col]
        corr = np.corrcoef(x_data, y_data)[0, 1]
        slope = np.polyfit(x_data, y_data, 1)[0]

        print("Correlation Coefficient: {:.2f}".format(corr))
        print("Correlation Slope: {:.2f}".format(slope))

        data = data[[x_col, y_col]]
        display(data.round(5).describe(percentiles=np.linspace(0.1, 0.9, 9, endpoint=True)))

        plt.subplots(figsize=(8, 6))
        plt.scatter(x_data, y_data, s=20, alpha=0.5)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_area_vs_speed(self, min_speed: float = None, min_diff: float = None):
        data = self.data_prep_frames()

        avg_speed = data.groupby("cycle")["wrm_speed"].mean()

        data = Plotter.remove_phase(data, "moving")

        max_area_diff = data.groupby("cycle")["bbox_area_diff"].max()

        data = pd.DataFrame({"wrm_speed": avg_speed, "bbox_area_diff": max_area_diff})

        if min_speed is not None:
            data = data[data["wrm_speed"] >= min_speed]

        if min_diff is not None:
            data = data[data["bbox_area_diff"] >= min_diff]

        Plotter.scatter_data(
            data,
            "wrm_speed",
            "bbox_area_diff",
            "Area Diff vs. Avg Speed",
        )

    def plot_area_vs_dist(self, min_dist: float = None, min_diff: float = None):
        data = self.data_prep_frames()

        dx = data.groupby("cycle")["wrm_center_x"].apply(lambda x: x.iloc[-1] - x.iloc[0])
        dy = data.groupby("cycle")["wrm_center_y"].apply(lambda y: y.iloc[-1] - y.iloc[0])
        dist = np.sqrt(dx**2 + dy**2)

        data = Plotter.remove_phase(data, "moving")

        max_area_diff = data.groupby("cycle")["bbox_area_diff"].max()

        data = pd.DataFrame({"dist": dist, "max_area_diff": max_area_diff})

        if min_dist is not None:
            data = data[data["dist"] >= min_dist]

        if min_diff is not None:
            data = data[data["max_area_diff"] >= min_diff]

        Plotter.scatter_data(
            data,
            "dist",
            "max_area_diff",
            "Max Area Diff vs. Distance",
        )

    def plot_deviation(self, n:int=1, condition=None) -> plt.Figure:
        data = self.data_prep_frames(n=n)
        data["worm_center_dist"] = np.sqrt(
            (data["wrm_center_x"] - data["mic_center_x"]) ** 2 + (data["wrm_center_y"] - data["mic_center_y"]) ** 2
        )
        data["cycle_step"] = data["frame"] % self.timing_config.cycle_length
        data['angle'] = np.arctan2(data['wrm_w'], data['wrm_h'])
        if condition is not None:
            data = data[condition(data)]

        g = sns.jointplot(data=data, x="cycle_step", y="worm_center_dist", kind="hist")
        g.figure.suptitle(f"distance between worm and microscope centers as a function of cycle step")
        g.set_axis_labels("cycle step", "distance")
        plt.show()
        return g.figure

    def plot_2d_deviation(self, n:int=1, hue="cycle_step", condition=None) -> plt.Figure:
        data = self.data_prep_frames(n=n)
        data["worm_center_dist_x"] = data["wrm_center_x"] - data["mic_center_x"]
        data["worm_center_dist_y"] = data["wrm_center_y"] - data["mic_center_y"]

        data["cycle_step"] = data["frame"] % self.timing_config.cycle_length
        if condition is not None:
            data = data[condition(data)]
        g = sns.jointplot(data=data, x="worm_center_dist_x", y="worm_center_dist_y", kind="scatter", hue=hue, alpha=0.6)
        g.set_axis_labels("distance x", "distance y")
        g.figure.suptitle(f"distance between worm and microscope centers in each axis")

        return g.figure

    def plot_area_vs_speed_guy(self, n: int = 1, window_size: int = 15, hue=None, condition=None) -> plt.Figure:
        data = self.data_prep_frames(n=n)
        data["wrm_speed_avg"] = Plotter.rolling_average(data, window_size=window_size, column="wrm_speed")
        # fig, ax = plt.subplots()
        mask = data["bbox_area_diff"] > 1e-3
        if condition is not None:
            mask = condition(data) & mask
        g = sns.jointplot(
            data=data[mask], x="wrm_speed_avg", y="bbox_area_diff", hue=hue, kind="scatter", xlim=(0, 3), dropna=True
        )
        g.figure.suptitle(f"n = {n}, rolling window = {window_size}")
        return g.figure
    
    def plot_area_vs_time(self, n: int = 1, window_size: int = 15, hue=None, condition=None) -> plt.Figure:
        data = self.data_prep_frames(n=n)
        data["wrm_speed_avg"] = Plotter.rolling_average(data, window_size=window_size, column="wrm_speed")
        # fig, ax = plt.subplots()
        mask = data["bbox_area_diff"] > 1e-3
        if condition is not None:
            mask = condition(data) & mask
        g = sns.jointplot(
            data=data[mask], x="wrm_speed_avg", y="bbox_area_diff", hue=hue, kind="scatter", xlim=(0, 3), dropna=True
        )
        g.figure.suptitle(f"n = {n}, rolling window = {window_size}")
        return g.figure

    def plot_cycle_step_vs_speed(self, n: int = 1, hue=None, condition=None) -> plt.Figure:
        data = self.data_prep_frames(n=n)
        data["cycle_step"] = data["frame"] % self.timing_config.cycle_length

        if condition is not None:
            data = data[condition(data)]
        g = sns.catplot(data=data, x="cycle_step", y="wrm_speed", kind="violin")
        g.set_axis_labels("cycle step", "worm speed")
        return g.figure

    def plot_trajectory(self, hue:str='frame', condition=None):
        data = self.data_prep_frames(n=1)
        if condition is not None:
            data = data[condition(data)]

        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x='wrm_center_x', y='wrm_center_y', hue=hue, ax=ax, alpha=0.3, linewidth=0.2)
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.invert_yaxis()
        fig.tight_layout()
        plt.show()