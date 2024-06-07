from typing import Collection
import numpy as np
from collections import deque
from torch import Tensor

from wtracker.sim.config import TimingConfig
from wtracker.sim.simulator import Simulator
from wtracker.sim.sim_controllers.csv_controller import CsvController
from wtracker.utils.bbox_utils import BoxUtils, BoxConverter, BoxFormat
from wtracker.neural.mlp import WormPredictor
from wtracker.neural.config import IOConfig


class MLPController(CsvController):
    """
    MLPController class represents a controller that uses a WormPredictor model to provide movement vectors for a simulation.

    Args:
        timing_config (TimingConfig): The timing configuration for the simulation.
        csv_path (str): The path to the CSV file containing the simulation data.
        model (WormPredictor): The WormPredictor model used for predicting worm movement.
        max_speed (float): max speed of the worm in mm/s, predictions above this will be clipped.
    """

    def __init__(self, timing_config: TimingConfig, csv_path: str, model: WormPredictor, max_speed: float = 0.9):
        super().__init__(timing_config, csv_path)
        self.model: WormPredictor = model
        self.io_config: IOConfig = model.io_config
        self.model.eval()

        px_per_mm = self.timing_config.px_per_mm
        fps = self.timing_config.frames_per_sec
        max_speed_px_frame = max_speed * (px_per_mm / fps)
        self.max_dist_per_pred = max_speed_px_frame * (self.io_config.pred_frames[0])

    def provide_movement_vector(self, sim: Simulator) -> tuple[int, int]:
        # frames for prediction (input to the model)
        frames_for_pred = np.asanyarray(self.io_config.input_frames, dtype=int)
        frames_for_pred += sim.frame_number - self.timing_config.pred_frame_num

        cam_center = BoxUtils.center(np.asanyarray(sim.view.camera_position))
        worm_bboxes = self.predict(frames_for_pred, relative=False).reshape(1, -1)
        if not np.isfinite(worm_bboxes).all():
            return 0, 0

        # relative position of the worm to the camera center, we use the worm x,y instead of center because of how the model and dataset are built
        rel_x, rel_y = worm_bboxes[0, 0] - cam_center[0], worm_bboxes[0, 1] - cam_center[1]

        # make coordinates relative to first bbox
        x = worm_bboxes[0, 0]
        x_mask = np.arange(0, worm_bboxes.shape[1]) % 4 == 0
        y = worm_bboxes[0, 1]
        y_mask = np.arange(0, worm_bboxes.shape[1]) % 4 == 1

        worm_bboxes[:, x_mask] -= x
        worm_bboxes[:, y_mask] -= y

        # predict the movement of the worm via the model
        pred = self.model.forward(Tensor(worm_bboxes)).flatten().detach().numpy()

        # make sure the prediction is within the limits and apply post-proccessing steps

        pred = np.clip(pred, -self.max_dist_per_pred, self.max_dist_per_pred)
        dx = round(pred[0].item() + rel_x)
        dy = round(pred[1].item() + rel_y)
        # dx = np.clip(dx, -self.max_dist_per_pred, self.max_dist_per_pred)
        # dy = np.clip(dy, -self.max_dist_per_pred, self.max_dist_per_pred)
        return (dx, dy)

    def print_model(self):
        print(self.model)
