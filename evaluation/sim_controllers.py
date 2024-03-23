import numpy as np
from ultralytics import YOLO
import cv2 as cv
import csv
from dataclasses import dataclass, field
from collections import deque

from utils.path_utils import join_paths, create_directory
from utils.io_utils import ImageSaver
from evaluation.simulator import *
from evaluation.simulator import Simulator


class YoloController(SimController):
    def __init__(self, config: TimingConfig, model: YOLO, device: str = "cpu", imgsz: int = 416):
        super().__init__(config)
        self._camera_frames = deque(maxlen=1)
        self._model = model.to(device)
        self.device = device
        self.imgsz = imgsz

    def on_sim_start(self, sim: Simulator):
        self._camera_frames.clear()

    def on_camera_frame(self, sim: Simulator, cam_view: np.ndarray):
        self._camera_frames.append(cam_view)

    def predict(self, *frames: np.ndarray) -> list[tuple[int, int, int, int]]:
        if len(frames) == 0:
            return []

        if frames[0].ndim == 2 or frames[0].shape[-1] == 1:
            frames = [cv.cvtColor(frame, cv.COLOR_GRAY2BGR) for frame in frames]

        results = self._model.predict(frames, conf=0.1, device=self.device, imgsz=self.imgsz, max_det=1, verbose=False)
        results = [res.boxes.xywh[0].to(int).tolist() if len(res.boxes) > 0 else None for res in results]
        return results

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        bboxes = self.predict(self._camera_frames[0])
        bbox = bboxes[0]
        if bbox is None:
            return 0, 0

        bbox_center = int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)

        view_center = sim.camera.camera_size[0] // 2, sim.camera.camera_size[1] // 2

        return bbox_center[0] - view_center[0], bbox_center[1] - view_center[1]


@dataclass
class LogConfig:
    root_folder: str
    mic_folder_name: str = "micro"
    cam_folder_name: str = "camera"
    err_folder_name: str = "errors"
    bbox_file_name: str = "bboxes.csv"
    mic_folder_path: str = field(init=False)
    cam_folder_path: str = field(init=False)
    err_folder_path: str = field(init=False)
    bbox_file_path: str = field(init=False)
    save_mic_view: bool = False
    save_cam_view: bool = False

    def __post_init__(self):
        self.mic_folder_path = join_paths(self.root_folder, self.mic_folder_name)
        self.cam_folder_path = join_paths(self.root_folder, self.cam_folder_name)
        self.err_folder_path = join_paths(self.root_folder, self.err_folder_name)
        self.bbox_file_path = join_paths(self.root_folder, self.bbox_file_name)

    def create_dirs(self):
        create_directory(self.root_folder)
        create_directory(self.err_folder_path)
        create_directory(self.mic_folder_path)
        create_directory(self.cam_folder_path)


class LoggingController(YoloController):
    def __init__(self, log_config: LogConfig, timing_config: TimingConfig, model: YOLO, device: str = "cpu", imgsz: int = 416):
        super().__init__(timing_config, model, device, imgsz)

        self.log_config = log_config
        self._camera_frames = deque(maxlen=self.config.imaging_frame_num)
        self._frame_number = -1
        self._cycle_number = -1

    def on_sim_start(self, sim: Simulator):
        self._frame_number = -1
        self._cycle_number = -1
        self._camera_frames.clear()
        self.log_config.create_dirs()

        self._image_saver = ImageSaver(self.log_config.root_folder, tqdm=False)
        self._image_saver.start()

        self._bbox_file = open(self.log_config.bbox_file_path, "w")
        self._bbox_logger = csv.writer(self._bbox_file)
        self._bbox_logger.writerow(
            [
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
            ]
        )

    def on_cycle_start(self, sim: Simulator):
        self._cycle_number += 1

    def on_sim_end(self, sim: Simulator):
        self._image_saver.close()
        self._bbox_file.flush()
        self._bbox_file.close()

    def on_camera_frame(self, sim: Simulator, cam_view: np.ndarray):
        self._frame_number += 1
        self._camera_frames.append(cam_view)

        if self.log_config.save_cam_view:
            path = join_paths(self.log_config.cam_folder_name, f"cam_{self._frame_number:09d}.png")
            self._image_saver.schedule(cam_view, path)

    def on_micro_frame(self, sim: Simulator, micro_view: np.ndarray):
        if self.log_config.save_mic_view:
            path = join_paths(self.log_config.mic_folder_name, f"mic_{self._frame_number:09d}.png")
            self._image_saver.schedule(micro_view, path)

    def on_imaging_end(self, sim: Simulator):
        # Note that these coords include the padding size
        cam_pos = sim.camera._calc_view_bbox(*sim.camera.camera_size)
        mic_pos = sim.camera._calc_view_bbox(*sim.camera.micro_size)

        bboxes = self.predict(*self._camera_frames)

        for i, bbox in enumerate(bboxes):
            frame_number = self._frame_number - len(self._camera_frames) + i + 1
            if bbox is not None:
                bbox = (bbox[0] + cam_pos[0], bbox[1] + cam_pos[1], bbox[2], bbox[3])
            else:
                path = join_paths(self.log_config.err_folder_name, f"cam_{frame_number:09d}.png")
                self._image_saver.schedule(self._camera_frames[i], path)
                bbox = (-1, -1, -1, -1)

            self._bbox_logger.writerow((frame_number, self._cycle_number) + cam_pos + mic_pos + bbox)

        self._bbox_file.flush()
