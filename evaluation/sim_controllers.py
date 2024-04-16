import numpy as np
from ultralytics import YOLO
import cv2 as cv
import csv
from dataclasses import dataclass, field
from collections import deque

from utils.path_utils import *
from utils.io_utils import ImageSaver
from utils.config_base import ConfigBase
from utils.log_utils import CSVLogger
from evaluation.simulator import *
from evaluation.simulator import Simulator


class YoloController(SimController):
    def __init__(self, config: TimingConfig, model: YOLO, device: str = "cpu", **pred_kwargs: dict):
        super().__init__(config)
        self._camera_frames = deque(maxlen=config.imaging_frame_num)
        self._model = model.to(device)
        self.device = device

        self.pred_kwargs = self._format_kwargs(**pred_kwargs)

    def _format_kwargs(self, **pred_kwargs) -> dict:
        default_args = {
            "imgsz": 416,
            "conf": 0.1,
            "verbose": False,
        }

        for key, val in default_args.items():
            pred_kwargs.setdefault(key, val)

        return pred_kwargs

    def on_sim_start(self, sim: Simulator):
        self._camera_frames.clear()

    def on_camera_frame(self, cycle_step: int, sim: Simulator, cam_view: np.ndarray):
        self._camera_frames.append(cam_view)

    def predict(self, *frames: np.ndarray) -> tuple[int, int, int, int] | list[tuple[int, int, int, int]]:
        if len(frames) == 0:
            return []

        # convert grayscale images to BGR because YOLO expects 3-channel images
        if frames[0].ndim == 2 or frames[0].shape[-1] == 1:
            frames = [cv.cvtColor(frame, cv.COLOR_GRAY2BGR) for frame in frames]

        # predict bounding boxes and format results
        results = self._model.predict(frames, device=self.device, max_det=1, **self.pred_kwargs)
        results = [res.boxes.xywh[0].to(int).tolist() if len(res.boxes) > 0 else None for res in results]

        if len(results) == 1:
            return results[0]

        return results

    def provide_moving_vector_simple(self, sim: Simulator) -> tuple[int, int]:
        bbox = self.predict(self._camera_frames[-self.config.pred_frame_num])
        if bbox is None:
            return 0, 0

        bbox_mid = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        camera_mid = sim.camera.camera_size[0] / 2, sim.camera.camera_size[1] / 2

        return round(bbox_mid[0] - camera_mid[0]), round(bbox_mid[1] - camera_mid[1])

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        assert len(self._camera_frames) >= 2 * self.config.pred_frame_num

        # predict the bounding boxes twice during microscope imaging
        frames = self._camera_frames[-2 * self.config.pred_frame_num], self._camera_frames[-self.config.pred_frame_num]
        bbox_old, bbox_new = self.predict(*frames)

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
        speed_per_frame = movement[0] / self.config.pred_frame_num, movement[1] / self.config.pred_frame_num

        # calculate camera correction based on the speed of the worm and current worm position
        camera_mid = sim.camera.camera_size[0] / 2, sim.camera.camera_size[1] / 2
        dx = round(bbox_new_mid[0] - camera_mid[0] + speed_per_frame[0] * self.config.moving_frame_num)
        dy = round(bbox_new_mid[1] - camera_mid[1] + speed_per_frame[1] * self.config.moving_frame_num)

        return dx, dy


@dataclass
class LogConfig(ConfigBase):
    root_folder: str
    save_mic_view: bool = False
    save_cam_view: bool = False
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


# TODO: save all configs
class LoggingController(YoloController):
    def __init__(
        self,
        log_config: LogConfig,
        timing_config: TimingConfig,
        model: YOLO,
        device: str = "cpu",
        **pred_kwargs: dict,
    ):
        super().__init__(timing_config, model, device, **pred_kwargs)

        self.log_config = log_config
        self._frame_number = -1
        self._cycle_number = -1

    def on_sim_start(self, sim: Simulator):
        self._frame_number = -1
        self._cycle_number = -1
        self._camera_frames.clear()
        self.log_config.create_dirs()

        self._image_saver = ImageSaver(tqdm=False)
        self._image_saver.start()

        self._bbox_logger = CSVLogger(
            self.log_config.bbox_file_path,
            col_names=[
                "frame",
                "cycle",
                "cam_x",
                "cam_y",
                "cam_w",
                "cam_h",
                "mic_x",
                "mic_y",
                "mic_w",
                "mic_h",
                "worm_x",
                "worm_y",
                "worm_w",
                "worm_h",
            ],
        )

    def on_cycle_start(self, sim: Simulator):
        self._cycle_number += 1

    def on_sim_end(self, sim: Simulator):
        self._image_saver.close()
        self._bbox_logger.close()

    def on_camera_frame(self, cycle_step: int, sim: Simulator, cam_view: np.ndarray):
        self._frame_number += 1
        self._camera_frames.append(cam_view)

        if self.log_config.save_cam_view:
            # save camera view
            path = self.log_config.cam_file_path.format(self._frame_number)
            self._image_saver.schedule_save(cam_view, path)

    def on_micro_frame(self, sim: Simulator, micro_view: np.ndarray):
        if self.log_config.save_mic_view:
            # save micro view
            path = self.log_config.mic_file_path.format(self._frame_number)
            self._image_saver.schedule_save(micro_view, path)

    def on_imaging_end(self, sim: Simulator):
        # Note that these coords include the padding size
        cam_pos = sim.camera._calc_view_bbox(*sim.camera.camera_size)
        mic_pos = sim.camera._calc_view_bbox(*sim.camera.micro_size)

        bboxes = self.predict(*self._camera_frames)

        for i, bbox in enumerate(bboxes):
            frame_number = self._frame_number - len(self._camera_frames) + i + 1

            if bbox is not None:
                # format bbox to be have position
                bbox = (bbox[0] + cam_pos[0], bbox[1] + cam_pos[1], bbox[2], bbox[3])
            else:
                # log prediction error
                path = self.log_config.err_file_path.format(frame_number)
                self._image_saver.schedule_save(self._camera_frames[i], path)
                bbox = (-1, -1, -1, -1)

            self._bbox_logger.write((frame_number, self._cycle_number) + cam_pos + mic_pos + bbox)

        self._bbox_logger.flush()

    def on_cycle_end(self, sim: Simulator):
        # TODO: do all the yolo predictions here
        pass
