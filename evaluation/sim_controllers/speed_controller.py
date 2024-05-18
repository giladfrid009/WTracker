from evaluation.simulator import *
from evaluation.sim_controllers import CsvController
from dataset.bbox_utils import *


# TODO: FIX
class SpeedController(CsvController):
    def __init__(self, timing_config: TimingConfig, csv_path: str, prev_sample_offset: int = None):
        if prev_sample_offset is None:
            prev_sample_offset = timing_config.imaging_frame_num - timing_config.pred_frame_num - 1

        assert timing_config.imaging_frame_num - timing_config.pred_frame_num - prev_sample_offset > 0
        
        super().__init__(timing_config, csv_path)
        self.prev_sample_offset = prev_sample_offset


    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        bboxes = self.predict(
            [
                sim.frame_number - self.timing_config.pred_frame_num - self.prev_sample_offset,
                sim.frame_number - self.timing_config.pred_frame_num,
            ]
        )

        bbox_old = bboxes[0, :]
        bbox_new = bboxes[1, :]

        old_empty = not np.isfinite(bbox_old).all()
        new_empty = not np.isfinite(bbox_new).all()

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
        speed_per_frame = movement[0] / self.prev_sample_offset, movement[1] / self.prev_sample_offset

        # calculate camera correction based on the speed of the worm and current worm position
        camera_mid = sim.camera.camera_size[0] / 2, sim.camera.camera_size[1] / 2

        print(sim.position)

        future_x = bbox_new_mid[0] + speed_per_frame[0] * (
            self.timing_config.moving_frame_num
            + self.timing_config.pred_frame_num
            + self.timing_config.imaging_frame_num / 2
        )

        future_y = bbox_new_mid[1] + speed_per_frame[1] * (
            self.timing_config.moving_frame_num
            + self.timing_config.pred_frame_num
            + self.timing_config.imaging_frame_num / 2
        )

        dx = round(future_x - camera_mid[0])
        dy = round(future_y - camera_mid[1])

        return dx, dy
