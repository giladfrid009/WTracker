import numpy as np

from wtracker.sim.config import TimingConfig
from wtracker.sim.simulator import Simulator
from wtracker.sim.sim_controllers.csv_controller import CsvController


class OptimalController(CsvController):
    def __init__(self, timing_config: TimingConfig, csv_path: str):
        super().__init__(timing_config, csv_path)

        self._csv_centers = np.empty((len(self._csv_data), 2), dtype=self._csv_data.dtype)
        self._csv_centers[:, 0] = self._csv_data[:, 0] + self._csv_data[:, 2] / 2
        self._csv_centers[:, 1] = self._csv_data[:, 1] + self._csv_data[:, 3] / 2

    def provide_movement_vector(self, sim: Simulator) -> tuple[int, int]:
        # extract portion matching next imaging phase
        next_imaging_start = (sim.cycle_number + 1) * self.timing_config.cycle_frame_num
        next_imaging_end = next_imaging_start + self.timing_config.imaging_frame_num

        next_imaging = self._csv_centers[next_imaging_start:next_imaging_end, :]
        next_imaging = next_imaging[np.isfinite(next_imaging).all(axis=1)]

        if len(next_imaging) == 0:
            return 0, 0

        x_next, y_next = np.median(next_imaging, axis=0)

        cam_x, cam_y, cam_w, cam_h = sim.view.camera_position
        cam_mid = cam_x + cam_w / 2, cam_y + cam_h / 2

        return round(x_next - cam_mid[0]), round(y_next - cam_mid[1])
