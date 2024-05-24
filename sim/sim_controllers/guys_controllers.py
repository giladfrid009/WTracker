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
# TODO: add MLPConfig class
# TODO: fix crash of MLPController when reaching the end of the csv file


class MLPController(CsvController):
    def __init__(self, timing_config: TimingConfig, csv_path: str, model: nn.Module, io_config: IOConfig):
        super().__init__(timing_config, csv_path)
        self.io_config = io_config
        self.model = model
        self.model.eval()

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
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


class Controller2(CsvController):
    def __init__(self, timing_config: TimingConfig, csv_path: str, past_frames_list=[0]):
        super().__init__(timing_config, csv_path)
        # self._camera_bboxes = deque(maxlen=timing_config.cycle_length*3)
        self.pred_data = deque(maxlen=3)

    def on_camera_frame(self, sim: Simulator):
        super().on_camera_frame(sim)

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        # how many frames to go back from the last prediction to estimate the speed
        # assert self.timing_config.imaging_frame_num - self.timing_config.pred_frame_num - past_frames_num >= 0
        print("############## provide_moving_vector ##############")
        delta_t = np.array([[3], [15]])
        pred_frames_ahead = self.timing_config.pred_frame_num + self.timing_config.imaging_frame_num // 4

        bbox_list = self.predict(
            sim,
            sim.frame_number - self.timing_config.pred_frame_num,
            sim.frame_number - self.timing_config.pred_frame_num - 3,
        )
        for box in bbox_list:
            if box is None:
                print("Warning::Controller1::No Pred")
                return 0, 0

        cam_x, cam_y, cam_w, cam_h = sim.view.camera_position
        bbox_list[0] = (bbox_list[0][0] + cam_x, bbox_list[0][1] + cam_y, bbox_list[0][2], bbox_list[0][3])
        bbox_list[1] = (bbox_list[1][0] + cam_x, bbox_list[1][1] + cam_y, bbox_list[1][2], bbox_list[1][3])
        self.pred_data.append(bbox_list[0])

        if len(self.pred_data) < 1:
            # print("Warning::Controller1::No Pred")
            return 0, 0

        bbox_old = self.pred_data.popleft()
        bbox_list.append(bbox_old)
        bboxes = np.array(bbox_list)

        boxes_center = BoxUtils.center(bboxes)

        # calculate the speed of the worm based on both predictions
        speeds = -np.diff(boxes_center, axis=0) / delta_t
        wrm_width = 5.59
        angle1 = np.arctan2(speeds[1, 1], speeds[1, 0])  # angle of speed of last cycle
        x_sign, y_sign = np.sign(speeds[0, 0]), np.sign(speeds[0, 1])
        angle2 = np.arctan2(
            (bboxes[0, 3] - wrm_width) * y_sign, (bboxes[0, 2] - wrm_width) * x_sign
        )  # angle of speed of current cycle

        angle_diff = np.abs(angle2 - angle1) / (2 * np.pi)
        speed_x = speeds[1, 0] * (1 - angle_diff)
        speed_y = speeds[1, 1] * (1 - angle_diff)
        direc_x = speeds[0, 0] * angle_diff
        direc_y = speeds[0, 1] * angle_diff

        # calculate camera correction based on the speed of the worm and current worm position
        camera_mid = cam_x + cam_w / 2, cam_y + cam_h / 2

        # print(f"boxes_center: {boxes_center}")
        # print(f"camera_mid: {camera_mid}")
        # print(f"angle1: {angle1} :: angle2: {angle2}")
        print(f"angle_diff: {angle_diff}")
        print(f"speeds: {speeds}")

        dx = round((boxes_center[0, 0] - camera_mid[0] + (speed_x) * pred_frames_ahead))
        dy = round(
            # bbox_new_mid[1]
            # camera_mid[1]
            (boxes_center[0, 1] - camera_mid[1] + (speed_y) * pred_frames_ahead)
        )
        print(f"dx: {dx} :: dy: {dy}")
        return dx, dy


class Controller1(CsvController):
    def __init__(self, timing_config: TimingConfig, csv_path: str, past_frames_list=[0]):
        super().__init__(timing_config, csv_path)
        self.pred_data = deque(maxlen=3)

    def on_camera_frame(self, sim: Simulator):
        super().on_camera_frame(sim)

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        # how many frames to go back from the last prediction to estimate the speed
        # assert self.timing_config.imaging_frame_num - self.timing_config.pred_frame_num - past_frames_num >= 0
        delta_t = self.timing_config.cycle_length
        pred_frames_ahead = self.timing_config.pred_frame_num + self.timing_config.imaging_frame_num // 2

        bbox_new = self.predict(
            sim,
            sim.frame_number - self.timing_config.pred_frame_num,
            sim.frame_number - self.timing_config.pred_frame_num - 3,
        )

        if bbox_new is None:
            print("Warning::Controller1::No Pred")
            return 0, 0

        cam_x, cam_y, cam_w, cam_h = sim.view.camera_position
        bbox_new = (bbox_new[0] + cam_x, bbox_new[1] + cam_y, bbox_new[2], bbox_new[3])
        self.pred_data.append(bbox_new)

        if len(self.pred_data) < 2:
            # print("Warning::Controller1::No Pred")
            return 0, 0

        bbox_old = self.pred_data.popleft()

        # calculate the speed of the worm based on both predictions
        bbox_old_mid = bbox_old[0] + bbox_old[2] / 2, bbox_old[1] + bbox_old[3] / 2
        bbox_new_mid = bbox_new[0] + bbox_new[2] / 2, bbox_new[1] + bbox_new[3] / 2
        movement = bbox_new_mid[0] - bbox_old_mid[0], bbox_new_mid[1] - bbox_old_mid[1]
        speed_per_frame = movement[0] / delta_t, movement[1] / delta_t

        # calculate camera correction based on the speed of the worm and current worm position
        camera_mid = sim.view.camera_size[0] / 2, sim.view.camera_size[1] / 2

        # TODO: TUNE HEURISTIC OF DX AND DY CALCULATION

        dx = round(
            # bbox_new_mid[0]
            # camera_mid[0]
            speed_per_frame[0]
            * pred_frames_ahead
        )
        dy = round(
            # bbox_new_mid[1]
            # camera_mid[1]
            speed_per_frame[1]
            * pred_frames_ahead
        )

        return dx, dy
