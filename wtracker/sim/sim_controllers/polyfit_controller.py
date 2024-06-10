import numpy as np
import pandas as pd
from dataclasses import dataclass
import numpy.polynomial.polynomial as poly

from wtracker.sim.config import TimingConfig
from wtracker.sim.simulator import Simulator
from wtracker.sim.sim_controllers.csv_controller import CsvController
from wtracker.utils.bbox_utils import BoxUtils, BoxConverter, BoxFormat
from wtracker.utils.config_base import ConfigBase


@dataclass
class PolyfitConfig(ConfigBase):
    degree: int
    """The degree of the polynomial, which will be fitted to the worm movement."""

    sample_times: list[int]
    """Times at which the worm position is be sampled for the polynomial fit.
    Time 0 denotes the beginning of the current cycle. Negative values are allowed."""

    weights: list[float] = None
    """Weights for each position sample for the polynomial fit. If None, all weights are set to 1.0.
    If the weights are not uniform, weighted polynomial fit is performed, 
    where the residuals of samples with higher weights are more important for the fitting."""

    def __post_init__(self):
        self.sample_times = sorted(self.sample_times)
        if self.weights is None:
            self.weights = [1.0 for _ in self.sample_times]

        assert len(self.sample_times) == len(self.weights)


class PolyfitController(CsvController):
    def __init__(
        self,
        timing_config: TimingConfig,
        polyfit_config: PolyfitConfig,
        csv_path: str,
    ) -> None:
        """
        Args:
            timing_config (TimingConfig): The timing configuration of the simulation.
            csv_path (str): The path to the csv file with the worm data.
            polyfit_config (PolyfitConfig): The configuration for the polynomial fit.
        """
        super().__init__(timing_config, csv_path)

        self.polyfit_config = polyfit_config
        self._sample_times = np.asanyarray(polyfit_config.sample_times, dtype=int)
        self._weights = np.asanyarray(polyfit_config.weights, dtype=float)

    def provide_movement_vector(self, sim: Simulator) -> tuple[int, int]:
        timing = self.timing_config
        config = self.polyfit_config

        bboxes = self.predict(sim.cycle_number * timing.cycle_frame_num + self._sample_times, relative=False)

        # make all bboxes relative to current camera view
        camera_bbox = sim.view.camera_position
        bboxes[:, 0] -= camera_bbox[0]
        bboxes[:, 1] -= camera_bbox[1]

        positions = BoxUtils.center(bboxes)
        mask = np.isfinite(positions).all(axis=1)
        time = self._sample_times[mask]
        positions = positions[mask]
        weights = self._weights[mask]

        if len(time) == 0:
            return 0, 0

        # predict future x and future y based on the fitted polynomial
        coeffs = poly.polyfit(time, positions, deg=config.degree, w=weights)
        x_pred, y_pred = poly.polyval(timing.cycle_frame_num + timing.imaging_frame_num // 2, coeffs)

        # calculate camera correction based on the speed of the worm and current worm position
        camera_mid = sim.view.camera_size[0] / 2, sim.view.camera_size[1] / 2

        dx = round(x_pred - camera_mid[0])
        dy = round(y_pred - camera_mid[1])

        return dx, dy


# TODO: ADD DOCS
# TODO: ACCEPT MULTIPLE CSVS
class WeightEvaluator:
    def __init__(
        self,
        csv_path: str,
        timing_config: TimingConfig,
        cycle_start_times: np.ndarray,
        input_time_offsets: np.ndarray,
        pred_time_offset: int,
        min_speed: float = 0,
    ):
        self.timing_config = timing_config
        self.pred_time_offset = pred_time_offset
        self.min_speed = min_speed

        bboxes = pd.read_csv(csv_path, usecols=["wrm_x", "wrm_y", "wrm_w", "wrm_h"]).to_numpy(dtype=float)

        self.positions = np.empty((len(bboxes), 2), dtype=float)
        self.positions[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2
        self.positions[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2

        self.input_time_offsets = np.sort(input_time_offsets)
        self.cycle_start_times = cycle_start_times

        self._initialize()

    def _initialize(self):
        N = self.input_time_offsets.shape[0]

        # x are times, y are positions

        # create input and target arrays for the times
        x_input = np.repeat(self.cycle_start_times, repeats=N) + np.tile(
            self.input_time_offsets, reps=self.cycle_start_times.shape[0]
        )
        x_input = x_input.reshape(-1, N)
        x_target = self.cycle_start_times + self.pred_time_offset

        # remove input and target cycles with invalid time
        # i.e. when input time is negative or target time is out of bounds
        mask = (x_input >= 0).all(axis=1) & (x_target < len(self.positions))
        x_input = x_input[mask, :]
        x_target = x_target[mask]

        # get input and target positions for each cycle
        y_input = self.positions[x_input.flatten(), :]
        y_input = y_input.reshape(-1, N, 2)
        y_target = self.positions[x_target.flatten(), :]
        y_target = y_target.reshape(-1, 2)

        # remove all cycles with invalid positions
        input_mask = np.isfinite(y_input).all(axis=(1, 2))
        target_mask = np.isfinite(y_target).all(axis=1)
        mask = input_mask & target_mask
        y_input = y_input[mask, :, :]
        y_target = y_target[mask, :]

        # remove cycles with average speed below threshold
        dist = np.sqrt((y_target[:, 1] - y_input[:, 0, 1]) ** 2 + (y_target[:, 0] - y_input[:, 0, 0]) ** 2)
        time = self.pred_time_offset - self.input_time_offsets[0]
        speed_mask = dist / time >= self.min_speed
        y_input = y_input[speed_mask, :, :]
        y_target = y_target[speed_mask, :]

        # set attributes
        self.x_input = self.input_time_offsets.reshape(N)
        self.y_input = y_input.swapaxes(0, 1).reshape(N, -1)
        self.y_target = y_target.reshape(-1)
        self.x_target = np.full_like(self.y_target, self.pred_time_offset)

        # print stats
        init_num_cycles = len(self.cycle_start_times)
        final_num_cycles = len(self.y_target) // 2
        removed_percent = round((init_num_cycles - final_num_cycles) / init_num_cycles * 100, 1)
        print(f"Number of evaluation cycles: {final_num_cycles}")
        print(f"Number of cycles removed: {init_num_cycles - final_num_cycles} ({removed_percent} %)")

    def _polyval(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluate a polynomial at given values.

        Args:
            coeffs (np.ndarray): Coefficients of the polynomial. Coefficients at increasing order. Should have shape [deg+1, N].
            x (np.ndarray): Values at which to evaluate the polynomial. Should have shape [N].

        Returns:
            np.ndarray: The result of evaluating the polynomial at the given values. Shape is [N].

        """
        coeffs = coeffs.swapaxes(0, 1)
        van = np.vander(x, N=coeffs.shape[1], increasing=True)
        return np.sum(van * coeffs, axis=-1)

    def eval(self, weights: np.ndarray, deg: int = 2) -> float:
        """
        Evaluate the mean absolute error (MAE) of the polynomial fit.

        Args:
            weights (np.ndarray): The weights used for the polynomial fit. Should have shape [N].
            deg (int, optional): The degree of the polynomial fit.

        Returns:
            float: The mean absolute error (MAE) of the polynomial fit.
        """
        coeffs = poly.polyfit(self.x_input, self.y_input, deg=deg, w=weights)
        y_pred = self._polyval(coeffs, self.x_target)
        mae = np.mean(np.abs(self.y_target - y_pred))
        return mae
