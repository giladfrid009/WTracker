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

        bbox_old, bbox_new = self.predict(
            sim.frame_number - self.timing_config.pred_frame_num - past_frames_num,
            sim.frame_number - self.timing_config.pred_frame_num,
        )

        if bbox_old is None and bbox_new is None:
            return 0, 0

        if bbox_new is None:
            bbox_new = bbox_old

        if bbox_old is None:
            bbox_old = bbox_new

        # calculate the speed of the worm based on both predictions
        bbox_old_mid = bbox_old[0] + bbox_old[2] / 2, bbox_old[1] + bbox_old[3] / 2
        bbox_new_mid = bbox_new[0] + bbox_new[2] / 2, bbox_new[1] + bbox_new[3] / 2
        movement = bbox_new_mid[0] - bbox_old_mid[0], bbox_new_mid[1] - bbox_old_mid[1]
        speed_per_frame = movement[0] / past_frames_num, movement[1] / past_frames_num

        # calculate camera correction based on the speed of the worm and current worm position
        camera_mid = sim.camera.camera_size[0] / 2, sim.camera.camera_size[1] / 2

        # TODO: TUNE HEURISTIC OF DX AND DY CALCULATION

        dx = round(
            bbox_new_mid[0]
            - camera_mid[0]
            + speed_per_frame[0]
            * (
                self.timing_config.moving_frame_num
                + self.timing_config.pred_frame_num
                + self.timing_config.imaging_frame_num / 4
            )
        )
        dy = round(
            bbox_new_mid[1]
            - camera_mid[1]
            + speed_per_frame[1]
            * (
                self.timing_config.moving_frame_num
                + self.timing_config.pred_frame_num
                + self.timing_config.imaging_frame_num / 4
            )
        )

        return dx, dy
