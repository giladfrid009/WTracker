import numpy as np
import cv2 as cv
from collections import deque
from typing import overload

from dataset.bbox_utils import BoxConverter, BoxFormat
from utils.path_utils import *
from evaluation.simulator import *
from dataset.bbox_utils import *
from evaluation.config import *


class YoloController(SimController):
    def __init__(self, timing_config: TimingConfig, yolo_config: YoloConfig):
        super().__init__(timing_config)
        self.yolo_config = yolo_config
        self._camera_frames = deque(maxlen=timing_config.cycle_length)
        self._model = yolo_config.load_model()

    def on_sim_start(self, sim: Simulator):
        self._camera_frames.clear()

    def on_camera_frame(self, sim: Simulator):
        self._camera_frames.append(sim.camera_view())

    @overload
    def predict(self, frames: np.ndarray) -> tuple[float, float, float, float]: ...

    @overload
    def predict(self, *frames: np.ndarray) -> list[tuple[float, float, float, float]]: ...

    def predict(
        self, *frames: np.ndarray
    ) -> tuple[float, float, float, float] | list[tuple[float, float, float, float]]:
        if len(frames) == 0:
            return []

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
                bboxes.append(None)
            else:
                bbox = BoxConverter.to_xywh(res.boxes.xyxy[0], BoxFormat.XYXY)
                bbox = bbox.tolist()
                bboxes.append(bbox)

        if len(bboxes) == 1:
            return bboxes[0]

        return bboxes

    def _cycle_predict_all(self, sim: Simulator) -> list[tuple[float, float, float, float]]:
        return self.predict(*self._camera_frames)

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        frame = self._camera_frames[-self.timing_config.pred_frame_num]
        bbox = self.predict(frame)
        if bbox is None:
            return 0, 0

        bbox_mid = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        camera_mid = sim.camera.camera_size[0] / 2, sim.camera.camera_size[1] / 2

        return round(bbox_mid[0] - camera_mid[0]), round(bbox_mid[1] - camera_mid[1])

    def on_cycle_end(self, sim: Simulator):
        self._camera_frames.clear()
