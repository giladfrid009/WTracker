import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, log_path: str):
        self._data = pd.read_csv(log_path)
        self._header = self._data.columns

    @staticmethod
    def calc_max_edge_diff(worm_boxes: np.ndarray, mic_boxes: np.ndarray) -> np.ndarray:
        """
        Calculate the length difference between the edges of the worm's bounding boxes and the microscope's bounding boxes.

        Args:
            worm_boxes (numpy.ndarray): Array of (x, y, width, height) tuples for the worm's bounding boxes
            mic_boxes (numpy.ndarray): Array of (x, y, width, height) tuples for the microscope's bounding boxes

        Returns:
            numpy.ndarray: Array of length differences between the edges of the bounding boxes
        """
        worm_left, worm_right = worm_boxes[:, 0], worm_boxes[:, 0] + worm_boxes[:, 2]
        worm_top, worm_bottom = worm_boxes[:, 1], worm_boxes[:, 1] + worm_boxes[:, 3]
        mic_left, mic_right = mic_boxes[:, 0], mic_boxes[:, 0] + mic_boxes[:, 2]
        mic_top, mic_bottom = mic_boxes[:, 1], mic_boxes[:, 1] + mic_boxes[:, 3]

        # Calculate the length difference between the edges of the bounding boxes
        x_diff = np.maximum(0, np.maximum(worm_right - mic_right, mic_left - worm_left))
        y_diff = np.maximum(0, np.maximum(worm_bottom - mic_bottom, mic_top - worm_top))

        return np.maximum(x_diff, y_diff)

    @staticmethod
    def calc_area_diff(worm_boxes: np.ndarray, mic_boxes: np.ndarray) -> np.ndarray:
        """
        Calculate the area difference between the worms bounding boxes and the microscope's bounding boxes.
        The area difference is the area by which the worm's bounding box exceeds the microscope's bounding box.

        Args:
            worm_boxes (numpy.ndarray): Array of (x, y, width, height) tuples for the worm's bounding boxes
            mic_boxes (numpy.ndarray): Array of (x, y, width, height) tuples for the microscope's bounding boxes

        Returns:
            numpy.ndarray: Array of length differences between the edges of the bounding boxes
        """
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

        return np.abs(worm_area - inter_area)

    @staticmethod
    def remove_phase(data: pd.DataFrame, phase: str) -> pd.DataFrame:
        """
        Remove rows with the specified phase.
        """
        return data[data["phase"] != phase]

    @staticmethod
    def remove_invalid_cycles(data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove cycles with negative bounding box coordinates or dimensions, which indicate that no worm was detected.
        """
        # Check if any row within each cycle has negative values in 'worm' columns
        has_negative = data.groupby("cycle")[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].transform("min").lt(0).any(axis=1)

        # Filter the dataframe to keep only cycles without negative values
        filtered_df = data[~has_negative].copy()

        return filtered_df

    @staticmethod
    def remove_invalid(data: pd.DataFrame) -> pd.DataFrame:
        data = Plotter.remove_phase(data, "moving")
        return Plotter.remove_invalid_cycles(data)

    @staticmethod
    def calc_centers(data: pd.DataFrame, field_name: str = "wrm") -> pd.DataFrame:
        """
        Calculate the centers of the bounding boxes and add them to the data.
        """
        data[f"{field_name}_center_x"] = data[f"{field_name}_x"] + data[f"{field_name}_w"] / 2
        data[f"{field_name}_center_y"] = data[f"{field_name}_y"] + data[f"{field_name}_h"] / 2
        return data

    @staticmethod
    def calc_speed(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the worm speed and add it to the data.
        """
        if "wrm_center_x" not in data.columns:
            data = Plotter.calc_centers(data, "wrm")

        if "mic_center_x" not in data.columns:
            data = Plotter.calc_centers(data, "mic")

        data["wrm_vec_x"] = data["wrm_center_x"].diff()
        data["wrm_vec_y"] = data["wrm_center_y"].diff()
        data["wrm_speed"] = np.sqrt(data["wrm_center_x"].diff() ** 2 + data["wrm_center_y"].diff() ** 2)
        return data

    @staticmethod
    def get_cycle_stats(data: pd.DataFrame, x_col: str, x_func: str, y_col: str, y_func: str) -> pd.DataFrame:
        return data.groupby("cycle")[[x_col, y_col]].agg({x_col: x_func, y_col: y_func}).reset_index()

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

        plt.figure(figsize=(8, 6))
        plt.scatter(data[x_col], data[y_col], s=20, alpha=0.5)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_v1(self):
        data = self._data
        data = Plotter.calc_speed(data)
        data = Plotter.remove_invalid(data)

        worm_bboxes = data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].values
        mic_bboxes = data[["mic_x", "mic_y", "mic_w", "mic_h"]].values

        print(worm_bboxes.shape)
        print(mic_bboxes.shape)

        data["bbox_area_diff"] = Plotter.calc_area_diff(worm_bboxes, mic_bboxes)
        data["bbox_edge_diff"] = Plotter.calc_max_edge_diff(worm_bboxes, mic_bboxes)

        data.fillna(0, inplace=True)

        cycle_data1 = Plotter.get_cycle_stats(data, "bbox_edge_diff", "max", "wrm_speed", "mean")
        Plotter.scatter_data(cycle_data1, "wrm_speed", "bbox_edge_diff", "Max Edge Diff vs. Mean Speed")

        cycle_data2 = Plotter.get_cycle_stats(data, "bbox_area_diff", "max", "wrm_speed", "mean")
        Plotter.scatter_data(cycle_data2, "wrm_speed", "bbox_area_diff", "Max Area Diff vs. Mean Speed")
