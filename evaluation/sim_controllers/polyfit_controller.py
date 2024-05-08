import numpy as np

from dataset.bbox_utils import *
from evaluation.simulator import *
from evaluation.sim_controllers import CsvController
from dataset.bbox_utils import *


class PolyfitController(CsvController):
    def __init__(self, timing_config: TimingConfig, csv_path: str, degree: int = 1):
        super().__init__(timing_config, csv_path)
        self.degree = degree

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        timing = self.timing_config

        pred_times = np.flip(np.arange(sim.frame_number - timing.pred_frame_num, -1, -timing.pred_frame_num))

        bboxes = self.predict(*pred_times)

        # bboxes[:, 0] and bboxes[:, 1] are the mid coords
        bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2

        valid_mask = ~np.isnan(bboxes).any(axis=1)

        time = pred_times[valid_mask] - sim.cycle_number * timing.cycle_length
        pos = bboxes[:, 0:2][valid_mask]

        if len(time) == 0:
            return 0, 0

        # weights = np.ones_like(time)
        weights = np.exp(-np.abs(time - time[-1]) / timing.cycle_length)
        weights[-1] = 1000  # forces the polynomial to go through the last point
        poly = np.polyfit(time, pos, deg=self.degree, w=weights)

        future_x, future_y = np.polyval(poly, timing.cycle_length + timing.imaging_frame_num / 4)

        # calculate camera correction based on the speed of the worm and current worm position
        camera_mid = sim.camera.camera_size[0] / 2, sim.camera.camera_size[1] / 2

        dx = round(future_x - camera_mid[0])
        dy = round(future_y - camera_mid[1])

        return dx, dy
