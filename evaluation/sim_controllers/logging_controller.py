from dataclasses import dataclass, field
from collections import deque

from utils.path_utils import *
from utils.io_utils import ImageSaver
from utils.config_base import ConfigBase
from utils.log_utils import CSVLogger
from evaluation.simulator import *


@dataclass
class LogConfig(ConfigBase):
    root_folder: str
    save_mic_view: bool = False
    save_cam_view: bool = False
    save_err_view: bool = True
    mic_folder_name: str = "micro"
    cam_folder_name: str = "camera"
    err_folder_name: str = "errors"
    bbox_file_name: str = "bboxes.csv"
    mic_file_name: str = "mic_{:09d}.png"
    cam_file_name: str = "cam_{:09d}.png"

    mic_file_path: str = field(init=False)
    cam_file_path: str = field(init=False)
    err_file_path: str = field(init=False)
    bbox_file_path: str = field(init=False)

    def __post_init__(self):
        self.mic_file_path = join_paths(self.root_folder, self.mic_folder_name, self.mic_file_name)
        self.cam_file_path = join_paths(self.root_folder, self.cam_folder_name, self.cam_file_name)
        self.err_file_path = join_paths(self.root_folder, self.err_folder_name, self.cam_file_name)
        self.bbox_file_path = join_paths(self.root_folder, self.bbox_file_name)

    def create_dirs(self):
        create_parent_directory(self.mic_file_path)
        create_parent_directory(self.cam_file_path)
        create_parent_directory(self.err_file_path)
        create_parent_directory(self.bbox_file_path)


class LoggingController(SimController):
    def __init__(
        self,
        sim_controller: SimController,
        log_config: LogConfig,
    ):
        super().__init__(sim_controller.timing_config)

        self.sim_controller = sim_controller
        self.log_config = log_config

        self._camera_frames = deque(maxlen=self.timing_config.cycle_length*2)
        self._platform_positions = deque(maxlen=self.timing_config.cycle_length*2)
        self._camera_bboxes = deque(maxlen=self.timing_config.cycle_length*2)
        self._micro_bboxes = deque(maxlen=self.timing_config.cycle_length*2)
        self._has_logged_cycle = False

    def on_sim_start(self, sim: Simulator):
        self.sim_controller.on_sim_start(sim)

        self._has_logged_cycle = False
        self._camera_frames.clear()
        self._platform_positions.clear()
        self._camera_bboxes.clear()
        self._micro_bboxes.clear()
        self.log_config.create_dirs()

        self._image_saver = ImageSaver(tqdm=False)
        self._image_saver.start()

        self._bbox_logger = CSVLogger(
            self.log_config.bbox_file_path,
            col_names=[
                "frame",
                "cycle",
                "phase",
                "plt_x",
                "plt_y",
                "cam_x",
                "cam_y",
                "cam_w",
                "cam_h",
                "mic_x",
                "mic_y",
                "mic_w",
                "mic_h",
                "wrm_x",
                "wrm_y",
                "wrm_w",
                "wrm_h",
            ],
        )

    def on_cycle_start(self, sim: Simulator):
        self.sim_controller.on_cycle_start(sim)
        self._has_logged_cycle = False

    def on_camera_frame(self, sim: Simulator):
        self.sim_controller.on_camera_frame(sim)

        # log everything
        self._platform_positions.append(sim.position)
        self._camera_bboxes.append(sim.camera._calc_view_bbox(*sim.camera.camera_size))
        self._micro_bboxes.append(sim.camera._calc_view_bbox(*sim.camera.micro_size))

        cam_view = sim.camera_view() if self.log_config.save_err_view or self.log_config.save_cam_view else None

        if self.log_config.save_err_view:
            self._camera_frames.append(cam_view)

        if self.log_config.save_cam_view:
            # save camera view
            path = self.log_config.cam_file_path.format(sim.frame_number)
            self._image_saver.schedule_save(cam_view, path)

        if self.log_config.save_mic_view:
            # save micro view
            mic_view = sim.camera.micro_view()
            path = self.log_config.mic_file_path.format(sim.frame_number)
            self._image_saver.schedule_save(mic_view, path)

    def _log_cycle(self, sim: Simulator):
        if self._has_logged_cycle:
            return

        worm_bboxes = self.sim_controller._cycle_predict_all(sim)

        cycle_first_frame = (sim.cycle_number - 1) * self.timing_config.cycle_length

        for i, worm_bbox in enumerate(worm_bboxes):
            csv_row = {}
            csv_row["plt_x"], csv_row["plt_y"] = self._platform_positions[i]
            csv_row["cam_x"], csv_row["cam_y"], csv_row["cam_w"], csv_row["cam_h"] = self._camera_bboxes[i]
            csv_row["mic_x"], csv_row["mic_y"], csv_row["mic_w"], csv_row["mic_h"] = self._micro_bboxes[i]

            frame_number = cycle_first_frame + i
            phase = (
                "imaging" if i % self.timing_config.cycle_length < self.timing_config.imaging_frame_num else "moving"
            )

            csv_row["frame"] = frame_number
            csv_row["cycle"] = sim.cycle_number - 1
            csv_row["phase"] = phase

            if worm_bbox is not None:
                # format bbox to be have absolute position
                cam_bbox = self._camera_bboxes[i]
                worm_bbox = (worm_bbox[0] + cam_bbox[0], worm_bbox[1] + cam_bbox[1], worm_bbox[2], worm_bbox[3])
            else:
                if self.log_config.save_err_view:
                    # save prediction error
                    err_view = self._camera_frames[i]
                    path = self.log_config.err_file_path.format(frame_number)
                    self._image_saver.schedule_save(err_view, path)

                worm_bbox = (None, None, None, None)

            csv_row["wrm_x"], csv_row["wrm_y"], csv_row["wrm_w"], csv_row["wrm_h"] = worm_bbox

            self._bbox_logger.write(csv_row)

        self._bbox_logger.flush()

        self._has_logged_cycle = True

    def on_cycle_end(self, sim: Simulator):
        self._log_cycle(sim)
        self.sim_controller.on_cycle_end(sim)

        self._camera_frames.clear()
        self._platform_positions.clear()
        self._camera_bboxes.clear()
        self._micro_bboxes.clear()

    def on_sim_end(self, sim: Simulator):
        if self._has_logged_cycle is False:
            self._log_cycle(sim)

        self.sim_controller.on_sim_end(sim)
        self._image_saver.close()
        self._bbox_logger.close()

    def on_imaging_start(self, sim: Simulator):
        self.sim_controller.on_imaging_start(sim)

    def on_micro_frame(self, sim: Simulator):
        self.sim_controller.on_micro_frame(sim)

    def on_imaging_end(self, sim: Simulator):
        self.sim_controller.on_imaging_end(sim)

    def on_movement_start(self, sim: Simulator):
        self.sim_controller.on_movement_start(sim)

    def on_movement_end(self, sim: Simulator):
        self.sim_controller.on_movement_end(sim)

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        return self.sim_controller.provide_moving_vector(sim)

    def _cycle_predict_all(self, sim: Simulator) -> list[tuple[float, float, float, float]]:
        return self.sim_controller._cycle_predict_all(sim)
