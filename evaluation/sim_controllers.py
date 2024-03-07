import numpy as np
from ultralytics import YOLO
import cv2 as cv

from evaluation.simulator import *
from evaluation.simulator import Simulator


class YoloController(SimController):
    def __init__(self, config: TimingConfig, model: YOLO, device: str = "cpu"):
        super().__init__(config, frame_mem=1)

        self.model = model
        self.device = device

    def on_frame(self, sim: Simulator, cam_view: np.ndarray):
        super().on_frame(sim, cam_view)
        sim.camera.visualize_world()
    
    def on_imaging_frame(self, sim: Simulator, micro_view: np.ndarray):
        super().on_imaging_frame(sim, micro_view)
        cv.imshow("Micro View", micro_view)
        cv.waitKey(0)

    def predict(self, frame: np.ndarray):
        res = self.model.predict([frame], conf=0.1, device=self.device, imgsz=416)
        if len(res) == 0 or res[0].boxes.xywh.shape[0] == 0:
            return None
        return res[0].boxes.xyxy[0]

    def get_moving_coords(self, sim: Simulator) -> tuple[int, int]:
        return 25, 25
        
        bbox = self.predict(self.last_frames[0])
        if bbox is None:
            return 0, 0

        pred_center = (
            ((bbox[0] + bbox[2]) / 2).to(int).item(),
            ((bbox[1] + bbox[3]) / 2).to(int).item(),
        )

        middle = sim.camera.camera_size[0] // 2, sim.camera.camera_size[1] // 2

        dx, dy = (
            pred_center[0] - middle[0],
            pred_center[1] - middle[1],
        )

        return dx, dy


class LoggingController(SimController):
    def __init__(self, config: TimingConfig, model: YOLO, device: str = "cpu"):
        super().__init__(config, frame_mem=1)

        self.model = model
        self.device = device

    def on_frame(self, sim: Simulator, cam_view: np.ndarray):
        super().on_frame(sim, cam_view)

        # TODO: PERFORM LOGGING
        if self.is_imaging:
            pass

    def predict(self, frame: np.ndarray):
        res = self.model.predict([frame], conf=0.1, device=self.device, imgsz=416)
        if len(res) == 0 or res[0].boxes.xywh.shape[0] == 0:
            return None
        return res[0].boxes.xyxy[0]

    def get_moving_coords(self, sim: Simulator) -> tuple[int, int]:
        bbox = self.predict(self.last_frames[0])
        if bbox is None:
            return 0, 0

        pred_center = (
            ((bbox[0] + bbox[2]) / 2).to(int).item(),
            ((bbox[1] + bbox[3]) / 2).to(int).item(),
        )

        center = sim.camera.camera_size[0] // 2, sim.camera.camera_size[1] // 2

        dx, dy = (
            pred_center[0] - center[0],
            pred_center[1] - center[1],
        )

        return dx, dy
