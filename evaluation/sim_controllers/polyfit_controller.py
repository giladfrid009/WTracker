import numpy as np
import pandas as pd

from dataset.bbox_utils import *
from evaluation.simulator import *
from evaluation.sim_controllers import CsvController
from dataset.bbox_utils import *


class PolyfitController(CsvController):
    def __init__(
        self,
        timing_config: TimingConfig,
        csv_path: str,
        degree: int = 2,
        weights: np.ndarray | None = None,
        sample_times: np.ndarray | None = None,
    ) -> None:
        """
        Args:
            timing_config (TimingConfig): The timing configuration of the simulation.
            csv_path (str): The path to the csv file with the worm data.
            degree (int, optional): The degree of the polynomial to fit. Defaults to 2.
            weights (np.ndarray, optional): The weights for the polynomial fitting. If None then all weights are 1.
            sample_times (np.ndarray, optional): The sample times to use for the polynomial fitting.
                If None then all samples are gathered from the current cycle, in intervals of pred_frame_num.
                Value 0 marks the begging of the current cycle, negative values are allowed.
        """
        super().__init__(timing_config, csv_path)
        self.degree = degree

        timing = self.timing_config

        if sample_times is None:
            sample_times = np.arange(timing.imaging_frame_num - timing.pred_frame_num, -1, -timing.pred_frame_num)
            sample_times = np.flip(sample_times)

        sample_times = np.sort(sample_times)
        self.sample_times = sample_times

        if weights is None:
            weights = np.ones(sample_times.shape[0])

        self.weights = weights

        assert weights.shape[0] == sample_times.shape[0]

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        timing = self.timing_config

        bboxes = self.predict(sim.cycle_number * timing.cycle_length + self.sample_times, relative=False)

        # make all bboxes relative to current camera view
        camera_bbox = sim.camera._calc_view_bbox(*sim.camera.camera_size)
        bboxes[:, 0] -= camera_bbox[0]
        bboxes[:, 1] -= camera_bbox[1]

        positions = BoxUtils.center(bboxes)
        mask = ~np.isnan(positions).any(axis=1)
        time = self.sample_times[mask]
        positions = positions[mask]
        weights = self.weights[mask]

        if len(time) == 0:
            return 0, 0

        try:    
            coeffs = np.polyfit(time, positions, deg=self.degree, w=weights)

            # predict future x and future y based on the fitted polynomial
            x_pred, y_pred = np.polyval(coeffs, timing.cycle_length + timing.imaging_frame_num // 2)
        except:
            x_pred, y_pred = positions[-1, :]

        # calculate camera correction based on the speed of the worm and current worm position
        camera_mid = sim.camera.camera_size[0] / 2, sim.camera.camera_size[1] / 2

        dx = round(x_pred - camera_mid[0])
        dy = round(y_pred - camera_mid[1])

        return dx, dy


class WeightEvaluator:
    def __init__(
        self,
        csv_path: str,
        timing_config: TimingConfig,
        input_offsets: np.ndarray,
        start_times: np.ndarray,
        eval_offset: int,
        min_speed: float = 0,
    ):
        self.timing_config = timing_config
        self.eval_offset = eval_offset
        self.min_speed = min_speed

        bboxes = pd.read_csv(csv_path, usecols=["wrm_x", "wrm_y", "wrm_w", "wrm_h"]).to_numpy(dtype=float)

        self.positions = np.empty((len(bboxes), 2), dtype=float)
        self.positions[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2
        self.positions[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2

        self.input_offsets = np.sort(input_offsets)
        self.start_times = start_times

        self._initialize()

    # TODO: CLEANUP CODE, MAYBE SPLIT INTO SEVERAL FUNCTIONS
    def _initialize(self):
        N = self.input_offsets.shape[0]

        # remove cycles with invalid time
        start_times = self.start_times + self.input_offsets[0]
        eval_times = self.start_times + self.eval_offset
        time_mask = (start_times >= 0) & (eval_times < len(self.positions))
        start_times = self.start_times[time_mask]

        # create data arrays
        input_times = np.repeat(start_times, repeats=N) + np.tile(self.input_offsets, reps=start_times.shape[0]) 
        dst_times = start_times + self.eval_offset
        input_pos = self.positions[input_times, :]  
        dst_pos = self.positions[dst_times, :]

        # remove invalid positions according to dst
        dst_mask = np.all(np.isfinite(dst_pos), axis=-1)
        mask = np.repeat(dst_mask, repeats=N, axis=0)
        input_pos = input_pos[mask, :]
        dst_pos = dst_pos[dst_mask, :]

        # remove invalid positions according to input
        input_mask = np.isfinite(input_pos).reshape(-1, N, 2)
        input_mask = np.all(np.all(input_mask, axis=-1), axis=-1)

        input_pos = input_pos.reshape(-1, N, 2)
        input_pos = input_pos[input_mask, :, :]
        dst_pos = dst_pos[input_mask, :]

        # calculate average speed of each cycle
        dist = np.sqrt((dst_pos[:, 0] - input_pos[:, 0, 0]) ** 2 + (dst_pos[:, 1] - input_pos[:, 0, 1]) ** 2)
        time = self.eval_offset - self.input_offsets[0]
        speed_mask = dist / time >= self.min_speed
        input_pos = input_pos[speed_mask, :, :]
        dst_pos = dst_pos[speed_mask, :]

        # set attributes
        self.x_input = self.input_offsets.reshape(N)
        self.y_input = input_pos.swapaxes(0, 1).reshape(N, -1)
        self.x_target = np.asanyarray([self.eval_offset])
        self.y_target = dst_pos.reshape(-1)

        # print stats
        init_num_cycles = len(self.start_times)
        final_num_cycles = len(dst_pos)
        removed_percent = round((init_num_cycles - final_num_cycles) / init_num_cycles * 100, 1)
        print(f"Number of evaluation cycles: {final_num_cycles}")
        print(f"Number of cycles removed: {init_num_cycles - final_num_cycles} ({removed_percent} %)")

    def _polyval(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluate a polynomial at given values.

        Args:
            coeffs (np.ndarray): Coefficients of the polynomial. Coefficients at decreasing order. Should have shape [deg+1, N].
            x (np.ndarray): Values at which to evaluate the polynomial. Should have shape [N].

        Returns:
            np.ndarray: The result of evaluating the polynomial at the given values. Shape is [N].

        """
        coeffs = coeffs.swapaxes(0, 1)
        van = np.vander(x, N=coeffs.shape[1], increasing=False)
        return np.sum(van * coeffs, axis=-1)

    def eval(self, weights: np.ndarray, deg: int = 2) -> float:
        """
        Evaluate the mean absolute error (MAE) of the polynomial fit.

        Args:
            weights (np.ndarray): The weights used for the polynomial fit. Should have shape [N].
            deg (int, optional): The degree of the polynomial fit. Defaults to 2.

        Returns:
            float: The mean squared error (MSE) of the polynomial fit.
        """
        coeffs = np.polyfit(self.x_input, self.y_input, deg=deg, w=weights)
        x_target = np.broadcast_to(self.x_target, shape=(coeffs.shape[1],))
        y_hat = self._polyval(coeffs, x_target)
        mae = np.mean(np.abs(self.y_target - y_hat))
        return mae