from collections import deque
import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy

from wtracker.sim.simulator import Simulator, SimController
from wtracker.utils.io_utils import ImageSaver, FrameSaver
from wtracker.utils.log_utils import CSVLogger
from wtracker.utils.config_base import ConfigBase
from wtracker.utils.path_utils import join_paths, create_parent_directory
from wtracker.utils.bbox_utils import BoxUtils, BoxFormat


@dataclass
class LogConfig(ConfigBase):
    root_folder: str
    """The directory where the logs will be saved into."""

    save_mic_view: bool = False
    """Whether to save the microscope view of each frame."""

    save_cam_view: bool = False
    """Whether to save the camera view of each frame."""

    save_err_view: bool = True
    """Whether to camera view of frames in which no prediction was made."""

    save_wrm_view: bool = False
    """whether to save the detected worm head of each frame."""

    mic_folder_name: str = "micro"
    cam_folder_name: str = "camera"
    err_folder_name: str = "errors"
    wrm_folder_name: str = "worms"

    # TODO: WHY DO WE SAVE IN PNG FORMAT AND NOT BMP?

    bbox_file_name: str = "bboxes.csv"
    mic_file_name: str = "mic_{:09d}.png"
    cam_file_name: str = "cam_{:09d}.png"
    wrm_file_name: str = "wrm_{:09d}.png"

    mic_file_path: str = field(init=False)
    cam_file_path: str = field(init=False)
    err_file_path: str = field(init=False)
    wrm_file_path: str = field(init=False)
    bbox_file_path: str = field(init=False)

    def __post_init__(self):
        self.mic_file_path = join_paths(self.root_folder, self.mic_folder_name, self.mic_file_name)
        self.cam_file_path = join_paths(self.root_folder, self.cam_folder_name, self.cam_file_name)
        self.err_file_path = join_paths(self.root_folder, self.err_folder_name, self.cam_file_name)
        self.wrm_file_path = join_paths(self.root_folder, self.wrm_folder_name, self.wrm_file_name)
        self.bbox_file_path = join_paths(self.root_folder, self.bbox_file_name)

    def create_dirs(self) -> None:
        create_parent_directory(self.bbox_file_path)
        create_parent_directory(self.mic_file_path)
        create_parent_directory(self.cam_file_path)
        create_parent_directory(self.err_file_path)
        create_parent_directory(self.wrm_file_path)


class LoggingController(SimController):
    def __init__(
        self,
        sim_controller: SimController,
        log_config: LogConfig,
    ):
        super().__init__(sim_controller.timing_config)

        self.sim_controller = sim_controller
        self.log_config = log_config

        self._camera_frames = deque(maxlen=self.timing_config.cycle_frame_num)
        self._platform_positions = deque(maxlen=self.timing_config.cycle_frame_num)
        self._camera_bboxes = deque(maxlen=self.timing_config.cycle_frame_num)
        self._micro_bboxes = deque(maxlen=self.timing_config.cycle_frame_num)

    def on_sim_start(self, sim: Simulator):
        self.sim_controller.on_sim_start(sim)

        self._camera_frames.clear()
        self._platform_positions.clear()
        self._camera_bboxes.clear()
        self._micro_bboxes.clear()
        self.log_config.create_dirs()

        self._image_saver = ImageSaver(tqdm=True)
        self._image_saver.start()

        self._frame_saver = FrameSaver(deepcopy(sim.view._frame_reader), tqdm=True)
        self._frame_saver.start()

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

    def on_camera_frame(self, sim: Simulator):
        self.sim_controller.on_camera_frame(sim)

        # log everything
        self._platform_positions.append(sim.position)
        self._camera_bboxes.append(sim.view.camera_position)
        self._micro_bboxes.append(sim.view.micro_position)

        if self.log_config.save_err_view:
            cam_view = sim.camera_view()
            self._camera_frames.append(cam_view)

        if self.log_config.save_cam_view:
            # save camera view
            cam_view = sim.camera_view()
            path = self.log_config.cam_file_path.format(sim.frame_number)
            self._image_saver.schedule_save(cam_view, path)

        if self.log_config.save_mic_view:
            # save micro view
            mic_view = sim.view.micro_view()
            path = self.log_config.mic_file_path.format(sim.frame_number)
            self._image_saver.schedule_save(mic_view, path)

    def _log_cycle(self, sim: Simulator):
        cycle_number = sim.cycle_number - 1
        frame_offset = cycle_number * self.timing_config.cycle_frame_num

        worm_bboxes = self.sim_controller._cycle_predict_all(sim)
        cam_bboxes = np.asanyarray(list(self._camera_bboxes))

        # make worm bboxes coordinate absolute
        worm_bboxes[:, 0] += cam_bboxes[:, 0]
        worm_bboxes[:, 1] += cam_bboxes[:, 1]

        # calc the crop dims to get the worm view from the original frame
        (H, W) = sim.experiment_config.orig_resolution
        crop_dims, is_crop_legal = BoxUtils.discretize(worm_bboxes, (H, W), BoxFormat.XYWH)

        for i, worm_bbox in enumerate(worm_bboxes):
            frame_number = frame_offset + i

            # if no prediction and we're saving error frames
            if not np.isfinite(worm_bbox).all() and self.log_config.save_err_view:
                err_view = self._camera_frames[i]
                path = self.log_config.err_file_path.format(frame_number)
                self._image_saver.schedule_save(img=err_view, img_name=path)

            # save cropped worm view if crop is legal
            if self.log_config.save_wrm_view and is_crop_legal[i]:
                crop_dim = crop_dims[i]
                path = self.log_config.wrm_file_path.format(frame_number)
                self._frame_saver.schedule_save(img_index=frame_number, crop_dims=crop_dim, img_name=path)

            csv_row = {}
            csv_row["plt_x"], csv_row["plt_y"] = self._platform_positions[i]
            csv_row["cam_x"], csv_row["cam_y"], csv_row["cam_w"], csv_row["cam_h"] = self._camera_bboxes[i]
            csv_row["mic_x"], csv_row["mic_y"], csv_row["mic_w"], csv_row["mic_h"] = self._micro_bboxes[i]
            csv_row["cycle"] = cycle_number
            csv_row["frame"] = frame_number
            csv_row["phase"] = "imaging" if i < self.timing_config.imaging_frame_num else "moving"
            csv_row["wrm_x"], csv_row["wrm_y"], csv_row["wrm_w"], csv_row["wrm_h"] = worm_bbox
            self._bbox_logger.write(csv_row)

        self._bbox_logger.flush()

    def on_cycle_end(self, sim: Simulator):
        self._log_cycle(sim)
        self.sim_controller.on_cycle_end(sim)

        self._camera_frames.clear()
        self._platform_positions.clear()
        self._camera_bboxes.clear()
        self._micro_bboxes.clear()

    def on_sim_end(self, sim: Simulator):
        self.sim_controller.on_sim_end(sim)
        self._image_saver.close()
        self._frame_saver.close()
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

    def begin_movement_prediction(self, sim: Simulator) -> None:
        return self.sim_controller.begin_movement_prediction(sim)

    def provide_movement_vector(self, sim: Simulator) -> tuple[int, int]:
        return self.sim_controller.provide_movement_vector(sim)

    def _cycle_predict_all(self, sim: Simulator) -> np.ndarray:
        return self.sim_controller._cycle_predict_all(sim)
