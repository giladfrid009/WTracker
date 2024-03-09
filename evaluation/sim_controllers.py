import numpy as np
from ultralytics import YOLO
import cv2 as cv
from typing import Iterable
import csv
from dataclasses import dataclass, field

from utils.file_utils import join_paths, create_directory
from evaluation.simulator import *
from evaluation.simulator import Simulator


class YoloController(SimController):
    def __init__(self, config: TimingConfig, model: YOLO, device: str = "cpu"):
        super().__init__(config)
        self.camera_frames = deque(maxlen=1)
        self.model = model.to(device)
        self.device = device

    def on_sim_start(self, sim: Simulator):
        self.camera_frames.clear()

    def on_frame(self, sim: Simulator, cam_view: np.ndarray):
        self.camera_frames.append(cam_view)

    def predict(self, *frames: Iterable[np.ndarray]):
        results = self.model.predict(frames, conf=0.1, device=self.device, imgsz=416)
        results = [res.boxes.xywh[0].to(int).tolist() if len(res.boxes) > 0 else None for res in results]

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        bboxes = self.predict(self.camera_frames[0])
        bbox = bboxes[0]
        if bbox is None:
            return 0, 0

        pred_center = int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)

        middle = sim.camera.camera_size[0] // 2, sim.camera.camera_size[1] // 2

        dx, dy = pred_center[0] - middle[0], pred_center[1] - middle[1]

        return dx, dy


@dataclass
class LogConfig:
    log_folder: str
    micro_view_folder: str = field(init=False)
    camera_view_folder: str = field(init=False)
    bboxes_file_path: str = field(init=False)
    save_micro_view: bool = False
    save_camera_view: bool = False

    def __post_init__(self):
        self.micro_view_folder = join_paths(self.log_folder, "micro")
        self.camera_view_folder = join_paths(self.log_folder, "camera")
        self.bboxes_file_path = join_paths(self.log_folder, "bboxes.csv")

    def create_dirs(self):
        create_directory(self.log_folder)
        if self.save_micro_view:
            create_directory(self.micro_view_folder)
        if self.save_camera_view:
            create_directory(self.camera_view_folder)


class LoggingController(YoloController):
    def __init__(self, log_config: LogConfig, timing_config: TimingConfig, model: YOLO, device: str = "cpu"):
        super().__init__(timing_config, model, device)

        self.log_config = log_config
        self.camera_frames = deque(maxlen=self.config.imaging_frame_num)
        self.micro_frames = deque(maxlen=self.config.imaging_frame_num)
        self.frame_number = 0

    def on_frame(self, sim: Simulator, cam_view: np.ndarray):
        self.camera_frames.append(cam_view)

    def on_imaging_frame(self, sim: Simulator, micro_view: np.ndarray):
        self.micro_frames.append(micro_view)

    def on_sim_start(self, sim: Simulator):
        self.camera_frames.clear()
        self.micro_frames.clear()

        self.log_config.create_dirs()
        self.bbox_file = open(self.log_config.bboxes_file_path, "w")
        self.bbox_logger = csv.writer(self.bbox_file)

        self.bbox_logger.writerow(
            [
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

    def on_sim_end(self, sim: Simulator):
        self.bbox_file.flush()
        self.bbox_file.close()

    def on_imaging_end(self, sim: Simulator):
        # Note that these coords include the padding size
        cam_pos = sim.camera._calc_view_bbox(*sim.camera.camera_size)
        mic_pos = sim.camera._calc_view_bbox(*sim.camera.micro_size)

        bboxes = self.predict(*self.camera_frames)

        for i, bbox in enumerate(bboxes):
            if bbox is not None:
                bboxes[i] = [bbox[0] + cam_pos[0], bbox[1] + cam_pos[1], bbox[2], bbox[3]]

        self.log_bboxes(cam_pos, mic_pos, bboxes)

        for cam_view, mic_view in zip(self.camera_frames, self.micro_frames):
            self.log_views(cam_view, mic_view)

    def log_bboxes(self, cam_pos, mic_pos, worm_bbox):
        self.bbox_logger.writerows([cam_pos + mic_pos + worm_bbox])

    def log_views(self, cam_view, mic_view):
        if self.log_config.save_camera_view:
            file_path = join_paths(self.log_config.camera_view_folder, f"cam_{self.frame_number:09d}.png")
            cv.imwrite(file_path, cam_view)
        if self.log_config.save_micro_view:
            file_path = join_paths(self.log_config.micro_view_folder, f"mic_{self.frame_number:09d}.png")
            cv.imwrite(file_path, mic_view)
