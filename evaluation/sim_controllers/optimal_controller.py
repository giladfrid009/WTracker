from utils.path_utils import *
from evaluation.simulator import *
from evaluation.sim_controllers.csv_controller import CsvController


class OptimalController(CsvController):
    def __init__(self, timing_config: TimingConfig, csv_path: str):
        super().__init__(timing_config, csv_path)

        self._csv_centers = np.empty((len(self._csv_data), 2), dtype=self._csv_data.dtype)
        self._csv_centers[:, 0] = self._csv_data[:, 0] + self._csv_data[:, 2] / 2
        self._csv_centers[:, 1] = self._csv_data[:, 1] + self._csv_data[:, 3] / 2

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        # extract portion matching next imaging phase
        next_imaging_start = (sim.cycle_number + 1) * self.timing_config.cycle_length
        next_imaging_end = next_imaging_start + self.timing_config.imaging_frame_num
        
        next_imaging = self._csv_centers[next_imaging_start:next_imaging_end, :]
        next_imaging = next_imaging[~np.isnan(next_imaging).any(axis=1)]

        if len(next_imaging) == 0:
            return 0, 0

        x_mid, y_mid = np.median(next_imaging, axis=0)

        # normalize worm position to camera view
        cam_bbox = sim.camera._calc_view_bbox(*sim.camera.camera_size)
        x_mid = x_mid - cam_bbox[0]
        y_mid = y_mid - cam_bbox[1]

        bbox_mid = x_mid, y_mid
        cam_mid = sim.camera.camera_size[0] / 2, sim.camera.camera_size[1] / 2

        return round(bbox_mid[0] - cam_mid[0]), round(bbox_mid[1] - cam_mid[1])
