from utils.path_utils import *
from evaluation.simulator import *
from evaluation.sim_controllers.csv_controller import CsvController

class OptimalController(CsvController):
    def __init__(self, timing_config: TimingConfig, csv_path: str):
        super().__init__(timing_config, csv_path)

        self._data["ctr_wrm_x"] = self._data["wrm_x"] + self._data["wrm_w"] / 2
        self._data["ctr_wrm_y"] = self._data["wrm_y"] + self._data["wrm_h"] / 2

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        next_imaging_start = (sim.cycle_number + 1) * self.timing_config.cycle_length
        next_imaging_end = next_imaging_start + self.timing_config.imaging_frame_num

        # extract next imaging phase data
        next_mask = (self._data.index >= next_imaging_start) & (self._data.index < next_imaging_end)
        next_df = self._data[next_mask]
        next_df = next_df.dropna(inplace=False)

        if len(next_df) == 0:
            0, 0

        x_mid = next_df["ctr_wrm_x"].median()
        y_mid = next_df["ctr_wrm_y"].median()

        # normalize worm position to camera view
        cam_bbox = sim.camera._calc_view_bbox(*sim.camera.camera_size)
        x_mid = x_mid - cam_bbox[0]
        y_mid = y_mid - cam_bbox[1]

        bbox_mid = x_mid, y_mid
        cam_mid = sim.camera.camera_size[0] / 2, sim.camera.camera_size[1] / 2

        return round(bbox_mid[0] - cam_mid[0]), round(bbox_mid[1] - cam_mid[1])

