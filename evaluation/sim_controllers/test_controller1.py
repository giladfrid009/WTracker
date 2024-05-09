from evaluation.simulator import *
from evaluation.sim_controllers import CsvController
from dataset.bbox_utils import *


class TestController1(CsvController):
    def __init__(self, timing_config: TimingConfig, csv_path: str):
        super().__init__(timing_config, csv_path)

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        # how many frames to go back from the last prediction to estimate the speed
        past_frames_num = self.timing_config.pred_frame_num
        assert self.timing_config.imaging_frame_num - self.timing_config.pred_frame_num - past_frames_num >= 0

        bboxes = self.predict(
            [
                sim.frame_number - self.timing_config.pred_frame_num - past_frames_num,
                sim.frame_number - self.timing_config.pred_frame_num,
            ]
        )

        bbox_old = bboxes[0, :]
        bbox_new = bboxes[1, :]

        old_empty = np.isnan(bbox_old).any()
        new_empty = np.isnan(bbox_new).any()

        if old_empty and new_empty:
            return 0, 0

        if new_empty:
            bbox_new = bbox_old

        if old_empty:
            bbox_old = bbox_new

        # calculate the speed of the worm based on both predictions
        bbox_old_mid = bbox_old[0] + bbox_old[2] / 2, bbox_old[1] + bbox_old[3] / 2
        bbox_new_mid = bbox_new[0] + bbox_new[2] / 2, bbox_new[1] + bbox_new[3] / 2
        movement = bbox_new_mid[0] - bbox_old_mid[0], bbox_new_mid[1] - bbox_old_mid[1]
        speed_per_frame = movement[0] / past_frames_num, movement[1] / past_frames_num

        # calculate camera correction based on the speed of the worm and current worm position
        camera_mid = sim.camera.camera_size[0] / 2, sim.camera.camera_size[1] / 2

        # TODO: TUNE HEURISTIC OF future_x AND future_y CALCULATION

        future_x = bbox_new_mid[0] + speed_per_frame[0] * (
            self.timing_config.moving_frame_num
            + self.timing_config.pred_frame_num
            + self.timing_config.imaging_frame_num / 4
        )

        future_y = bbox_new_mid[1] + speed_per_frame[1] * (
            self.timing_config.moving_frame_num
            + self.timing_config.pred_frame_num
            + self.timing_config.imaging_frame_num / 4
        )

        dx = round(future_x - camera_mid[0])
        dy = round(future_y - camera_mid[1])

        return dx, dy
