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
        super().__init__(timing_config, csv_path)
        self.degree = degree

        timing = self.timing_config

        if sample_times is None:
            sample_times = np.arange(timing.imaging_frame_num - timing.pred_frame_num, -1, -timing.pred_frame_num)
            sample_times = np.flip(sample_times)

        sample_times = np.sort(sample_times)
        self.sample_times = sample_times

        if weights is None:
            weights = np.exp(-(sample_times[-1] - sample_times) / timing.imaging_frame_num)
            weights[-1] = 1000

        self.weights = weights

        assert weights.shape[0] == sample_times.shape[0]

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        timing = self.timing_config

        bboxes = self.predict(sim.cycle_number * timing.cycle_length + self.sample_times)

        # calculate mid coords
        bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2

        valid_mask = ~np.isnan(bboxes).any(axis=1)

        time = self.sample_times[valid_mask]
        position = bboxes[:, 0:2][valid_mask]

        if len(time) == 0:
            return 0, 0

        coeffs = polynomial.polyfit(time, position, deg=self.degree, w=self.weights)

        future_x, future_y = polynomial.polyval(timing.cycle_length + timing.imaging_frame_num / 4, c=coeffs)

        # calculate camera correction based on the speed of the worm and current worm position
        camera_mid = sim.camera.camera_size[0] / 2, sim.camera.camera_size[1] / 2

        dx = round(future_x - camera_mid[0])
        dy = round(future_y - camera_mid[1])

        return dx, dy
