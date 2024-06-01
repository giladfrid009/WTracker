from collections import deque
from typing import Collection
import pandas as pd

from sim.config import TimingConfig
from sim.simulator import SimController, Simulator
from utils.bbox_utils import *


class CsvController(SimController):
    def __init__(self, timing_config: TimingConfig, csv_path: str):
        super().__init__(timing_config)

        self.csv_path = csv_path
        self._csv_data = pd.read_csv(self.csv_path, usecols=["wrm_x", "wrm_y", "wrm_w", "wrm_h"]).to_numpy(dtype=float)
        self._camera_bboxes = deque(maxlen=timing_config.cycle_frame_num)

    def on_sim_start(self, sim: Simulator):
        self._camera_bboxes.clear()

    def on_camera_frame(self, sim: Simulator):
        self._camera_bboxes.append(sim.view.camera_position)

    # TODO: if relative = True then this function works only if frame number if within the last cycle.
    # maybe fix that.
    def predict(self, frame_nums: Collection[int], relative: bool = True) -> np.ndarray:
        assert len(frame_nums) > 0

        frame_nums = np.asanyarray(frame_nums, dtype=int)

        valid_mask = (frame_nums >= 0) & (frame_nums < self._csv_data.shape[0])

        worm_bboxes = np.full((frame_nums.shape[0], 4), np.nan)

        worm_bboxes[valid_mask] = self._csv_data[frame_nums[valid_mask], :]

        if not relative:
            return worm_bboxes

        cam_bboxes = [self._camera_bboxes[n % self.timing_config.cycle_frame_num] for n in frame_nums]
        cam_bboxes = np.asanyarray(cam_bboxes, dtype=float)

        # make bbox relative to camera view
        worm_bboxes[:, 0] -= cam_bboxes[:, 0]
        worm_bboxes[:, 1] -= cam_bboxes[:, 1]

        return worm_bboxes

    def begin_movement_prediction(self, sim: Simulator) -> None:
        pass

    def provide_movement_vector(self, sim: Simulator) -> tuple[int, int]:
        bbox = self.predict([sim.frame_number - self.timing_config.pred_frame_num])
        bbox = bbox[0, :]

        if not np.isfinite(bbox).all():
            return 0, 0

        center = BoxUtils.center(bbox)
        cam_center = sim.view.camera_size[0] / 2, sim.view.camera_size[1] / 2

        dx = round(center[0] - cam_center[0])
        dy = round(center[1] - cam_center[1])

        return dx, dy

    def _cycle_predict_all(self, sim: Simulator) -> np.ndarray:
        start = (sim.cycle_number - 1) * self.timing_config.cycle_frame_num
        end = start + self.timing_config.cycle_frame_num
        end = min(end, len(self._csv_data))
        return self.predict(np.arange(start, end))
