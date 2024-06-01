from typing import Collection
from numpy import ndarray
from collections import deque
from torch import Tensor

from sim.config import TimingConfig
from sim.simulator import Simulator
from sim.sim_controllers.csv_controller import CsvController
from utils.bbox_utils import *
from neural.mlp import *
from neural.config import IOConfig


# TODO: CLEAN UP this FILE
# TODO: fix crash of MLPController when reaching the end of the csv file
# I believe this crash happens when we provide a framereader as input, and the simulator reaches the last cycle.


class MLPController(CsvController):
    def __init__(self, timing_config: TimingConfig, csv_path: str, model: nn.Module, io_config: IOConfig):
        super().__init__(timing_config, csv_path)
        self.io_config = io_config
        self.model = model
        self.model.eval()

    def provide_movement_vector(self, sim: Simulator) -> tuple[int, int]:
        # frames for prediction (input to the model)
        frames_for_pred = np.asanyarray(self.io_config.input_frames, dtype=int)
        frames_for_pred += sim.frame_number - self.timing_config.pred_frame_num

        cam_center = BoxUtils.center(np.asanyarray(sim.view.camera_position))
        worm_bboxes = self._csv_data[frames_for_pred, :].reshape(1, -1)
        if not np.isfinite(worm_bboxes).all():
            return 0, 0

        # worm_center = BoxUtils.center(worm_bboxes[0, :4]) # center of first bbox (should be the box of the pred frame (0))
        # print(f"boxes= {worm_bboxes[0, :4]} :: center= {worm_center}")

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
        pred = np.clip(pred, -20, 20)
        dx = round(pred[0].item() + rel_x)
        dy = round(pred[1].item() + rel_y)

        # print(f"pred = {(dx, dy)}")
        return (dx, dy)

    def print_model(self):
        print(self.model)
