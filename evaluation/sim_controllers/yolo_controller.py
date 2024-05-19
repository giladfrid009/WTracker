from typing import Collection
import numpy as np
import cv2 as cv
from collections import deque

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

    def _cycle_predict_all(self, sim: Simulator) -> np.ndarray:
        return self.predict(self._camera_frames)

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        frame = self._camera_frames[-self.timing_config.pred_frame_num]
        bbox = self.predict([frame])
        bbox = bbox[0]
        
        if not np.isfinite(bbox).all():
            return 0, 0

        bbox_mid = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        camera_mid = sim.view.camera_size[0] / 2, sim.view.camera_size[1] / 2

        return round(bbox_mid[0] - camera_mid[0]), round(bbox_mid[1] - camera_mid[1])

    def on_cycle_end(self, sim: Simulator):
        self._camera_frames.clear()
