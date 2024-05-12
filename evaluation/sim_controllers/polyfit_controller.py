import numpy as np
from numpy.polynomial import polynomial

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

    # TODO: FIX
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
        position = positions[mask]

        if len(time) == 0:
            return 0, 0

        coeffs = np.polyfit(time, position, deg=self.degree, w=self.weights)

        # predict future x and future y based on the fitted polynomial
        x_pred, y_pred = np.polyval(coeffs, timing.cycle_length + timing.imaging_frame_num // 2)

        # calculate camera correction based on the speed of the worm and current worm position
        camera_mid = sim.camera.camera_size[0] / 2, sim.camera.camera_size[1] / 2

        dx = round(x_pred - camera_mid[0])
        dy = round(y_pred - camera_mid[1])

        return dx, dy
