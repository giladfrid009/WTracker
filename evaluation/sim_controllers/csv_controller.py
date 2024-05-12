from collections import deque
from typing import Collection
import pandas as pd

from evaluation.simulator import *
from dataset.bbox_utils import *


class CsvController(SimController):
    def __init__(self, timing_config: TimingConfig, csv_path: str):
        super().__init__(timing_config)

        self.csv_path = csv_path
        self._csv_data = pd.read_csv(self.csv_path, usecols=["wrm_x", "wrm_y", "wrm_w", "wrm_h"]).to_numpy(dtype=float)
        self._camera_bboxes = deque(maxlen=timing_config.cycle_length)

    def on_sim_start(self, sim: Simulator):
        self._camera_bboxes.clear()

    def on_camera_frame(self, sim: Simulator):
        self._camera_bboxes.append(sim.camera._calc_view_bbox(*sim.camera.camera_size))

    def predict(self, frame_nums: Collection[int]) -> np.ndarray:
        assert len(frame_nums) > 0

        frame_nums = np.asanyarray(frame_nums, dtype=int)
        worm_bboxes = self._csv_data[frame_nums, :]

        cam_bboxes = [self._camera_bboxes[n % self.timing_config.cycle_length] for n in frame_nums]
        cam_bboxes = np.asanyarray(cam_bboxes, dtype=float)

        # make bbox relative to camera view
        worm_bboxes[:, 0] -= cam_bboxes[:, 0]
        worm_bboxes[:, 1] -= cam_bboxes[:, 1]

        return worm_bboxes

    def _cycle_predict_all(self, sim: Simulator) -> np.ndarray:
        start = (sim.cycle_number - 1) * self.timing_config.cycle_length
        end = start + self.timing_config.cycle_length
        end = min(end, len(self._csv_data))
        return self.predict(np.arange(start, end))

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        bbox = self.predict([sim.frame_number - self.timing_config.pred_frame_num])
        bbox = bbox[0, :]

        if np.isnan(bbox).any():  
            return 0, 0

        # calculate the speed of the worm based on both predictions
        bbox_mid = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2

        # calculate camera correction based on the speed of the worm and current worm position
        camera_mid = sim.camera.camera_size[0] / 2, sim.camera.camera_size[1] / 2

        dx = round(bbox_mid[0] - camera_mid[0])
        dy = round(bbox_mid[1] - camera_mid[1])

        return dx, dy
