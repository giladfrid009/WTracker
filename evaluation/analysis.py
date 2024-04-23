import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, log_path: str):
        self._data = pd.read_csv(log_path)
        self._header = self._data.columns

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

    @staticmethod
    def concat(*dataframes: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(list(dataframes), axis=1, ignore_index=False, copy=False)

    @staticmethod
    def remove_no_pred_rows(data: pd.DataFrame) -> pd.DataFrame:
        mask = (data["wrm_x"] == -1) & (data["wrm_y"] == -1)
        return data.loc[~mask]

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
        data[f"{field_name}_center_x"] = data[f"{field_name}_x"] + data[f"{field_name}_w"] / 2
        data[f"{field_name}_center_y"] = data[f"{field_name}_y"] + data[f"{field_name}_h"] / 2

        center_x = data[f"{field_name}_x"] + data[f"{field_name}_w"] / 2
        center_y = data[f"{field_name}_y"] + data[f"{field_name}_h"] / 2

        centers = pd.DataFrame({"center_x": center_x, "center_y": center_y}, index=data.index)
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

        plt.subplots(figsize=(8, 6))
        plt.scatter(x_data, y_data, s=20, alpha=0.5)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_area_vs_speed(self):
        data = self.data_prep_frames()

        avg_speed = data.groupby("cycle")["wrm_speed"].mean()

        data = Plotter.remove_phase(data, "moving")

        max_area_diff = data.groupby("cycle")["bbox_area_diff"].max()

        data = pd.DataFrame({"wrm_speed": avg_speed, "bbox_area_diff": max_area_diff})

        display(data.describe())

        Plotter.scatter_data(
            data,
            "wrm_speed",
            "bbox_area_diff",
            "Area Diff vs. Avg Speed",
        )

    def plot_area_vs_dist(self):
        data = self.data_prep_frames()

        dx = data.groupby("cycle")["wrm_center_x"].apply(lambda x: x.iloc[-1] - x.iloc[0])
        dy = data.groupby("cycle")["wrm_center_y"].apply(lambda y: y.iloc[-1] - y.iloc[0])
        dist = np.sqrt(dx**2 + dy**2)

        data = Plotter.remove_phase(data, "moving")

        max_area_diff = data.groupby("cycle")["bbox_area_diff"].max()

        data = pd.DataFrame({"dist": dist, "max_area_diff": max_area_diff})

        Plotter.scatter_data(
            data,
            "dist",
            "max_area_diff",
            "Max Area Diff vs. Distance",
        )
