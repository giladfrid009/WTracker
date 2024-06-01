from typing import Collection, Any
from dataclasses import dataclass, field
import numpy as np
import cv2 as cv
from collections import deque
from ultralytics import YOLO


from sim.simulator import Simulator, SimController
from utils.bbox_utils import *
from sim.config import TimingConfig
from utils.config_base import ConfigBase


@dataclass
class YoloConfig(ConfigBase):
    model_path: str
    """The path to the pretrained YOLO weights file."""

    device: str = "cpu"
    """Inference device for YOLO. Can be either 'cpu' or 'cuda'."""

    verbose: bool = False
    """Whether to print verbose output during YOLO inference."""

    pred_kwargs: dict = field(
        default_factory=lambda: {
            "imgsz": 384,
            "conf": 0.1,
        }
    )
    """Additional keyword arguments for the YOLO prediction method."""

    model: YOLO = field(default=None, init=False, repr=False)
    """The YOLO model object."""

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["model"]  # we dont want to serialize the model
        return state

    def load_model(self) -> YOLO:
        if self.model is None:
            self.model = YOLO(self.model_path, task="detect", verbose=self.verbose)
        return self.model


class YoloController(SimController):
    def __init__(self, timing_config: TimingConfig, yolo_config: YoloConfig):
        super().__init__(timing_config)
        self.yolo_config = yolo_config
        self._camera_frames = deque(maxlen=timing_config.cycle_frame_num)
        self._model = yolo_config.load_model()

    def on_sim_start(self, sim: Simulator):
        self._camera_frames.clear()

    def on_camera_frame(self, sim: Simulator):
        self._camera_frames.append(sim.camera_view())

    def on_cycle_end(self, sim: Simulator):
        self._camera_frames.clear()

    def predict(self, frames: Collection[np.ndarray]) -> np.ndarray:
        assert len(frames) > 0

        # convert grayscale images to BGR because YOLO expects 3-channel images
        if frames[0].ndim == 2 or frames[0].shape[-1] == 1:
            frames = [cv.cvtColor(frame, cv.COLOR_GRAY2BGR) for frame in frames]

        # predict bounding boxes and format results
        results = self._model.predict(
            source=frames,
            device=self.yolo_config.device,
            max_det=1,
            verbose=self.yolo_config.verbose,
            **self.yolo_config.pred_kwargs,
        )

        results = [res.numpy() for res in results]

        bboxes = []
        for res in results:
            if len(res.boxes.xyxy) == 0:
                bboxes.append(np.full([4], np.nan))
            else:
                bbox = BoxConverter.to_xywh(res.boxes.xyxy[0], BoxFormat.XYXY)
                bboxes.append(bbox)

        return np.stack(bboxes, axis=0)

    def begin_movement_prediction(self, sim: Simulator) -> None:
        pass

    def provide_movement_vector(self, sim: Simulator) -> tuple[int, int]:
        frame = self._camera_frames[-self.timing_config.pred_frame_num]
        bbox = self.predict([frame])
        bbox = bbox[0]

        if not np.isfinite(bbox).all():
            return 0, 0

        bbox_mid = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        camera_mid = sim.view.camera_size[0] / 2, sim.view.camera_size[1] / 2

        return round(bbox_mid[0] - camera_mid[0]), round(bbox_mid[1] - camera_mid[1])

    def _cycle_predict_all(self, sim: Simulator) -> np.ndarray:
        return self.predict(self._camera_frames)
