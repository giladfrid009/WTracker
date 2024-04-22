import numpy as np
from ultralytics import YOLO
import cv2 as cv
from dataclasses import dataclass, field
from collections import deque
import pandas as pd

from dataset.bbox_utils import BoxConverter, BoxFormat
from utils.path_utils import *
from utils.io_utils import ImageSaver
from utils.config_base import ConfigBase
from utils.log_utils import CSVLogger
from evaluation.simulator import *
from evaluation.simulator import Simulator, TimingConfig
from dataset.bbox_utils import *


@dataclass
class YoloConfig(ConfigBase):
    model_path: str
    device: str = "cpu"
    task: str = "detect"
    verbose: bool = False
    pred_kwargs: dict = field(
        default_factory=lambda: {
            "imgsz": 416,
            "conf": 0.1,
        }
    )

    model: YOLO = field(default=None, init=False, repr=False)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["model"]  # we dont want to serialize the model
        return state

    def load_model(self) -> YOLO:
        if self.model is None:
            self.model = YOLO(self.model_path, task=self.task, verbose=self.verbose)
        return self.model


class YoloController(SimController):
    def __init__(self, timing_config: TimingConfig, yolo_config: YoloConfig):
        super().__init__(timing_config)
        self.yolo_config = yolo_config
        self._camera_frames = deque(maxlen=timing_config.imaging_frame_num)
        self._model = yolo_config.load_model()

    def on_sim_start(self, sim: Simulator):
        self._camera_frames.clear()

    def on_camera_frame(self, sim: Simulator, cam_view: np.ndarray):
        self._camera_frames.append(cam_view)

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
                bboxes.append(bbox.tolist())

        if len(bboxes) == 1:
            return bboxes[0]

        return bboxes

    def _cycle_predict_all(self, sim: Simulator) -> list[tuple[float, float, float, float]]:
        return self.predict(*self._camera_frames)

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        bbox = self.predict(self._camera_frames[-self.timing_config.pred_frame_num])
        if bbox is None:
            return 0, 0

        bbox_mid = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        camera_mid = sim.camera.camera_size[0] / 2, sim.camera.camera_size[1] / 2

        return round(bbox_mid[0] - camera_mid[0]), round(bbox_mid[1] - camera_mid[1])


class CsvController(SimController):
    def __init__(self, timing_config: TimingConfig, csv_path: str):
        super().__init__(timing_config)

        self.csv_path = csv_path
        self._log_df = pd.read_csv(self.csv_path, index_col="frame")

    def predict(self, *frame_nums: int) -> tuple[float, float, float, float] | list[tuple[float, float, float, float]]:
        if len(frame_nums) == 0:
            return []

        bboxes = []
        for frame_num in frame_nums:
            # get the absolute positions of predicted bbox and of camera
            row = self._log_df.loc[frame_num]
            worm_bbox = row[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].to_list()
            cam_bbox = row[["cam_x", "cam_y", "cam_w", "cam_h"]].to_list()

            # make bbox relative to camera view
            worm_bbox[0] = worm_bbox[0] - cam_bbox[0]
            worm_bbox[1] = worm_bbox[1] - cam_bbox[1]

            if any(c == -1 for c in worm_bbox):
                bboxes.append(None)
            else:
                bboxes.append(worm_bbox)

        if len(bboxes) == 1:
            return bboxes[0]

        return bboxes

    def _cycle_predict_all(self, sim: Simulator) -> list[tuple[float, float, float, float]]:
        start = sim.cycle_number * self.timing_config.cycle_length
        end = start + self.timing_config.cycle_length
        return self.predict(*range(start, end))

    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        bbox_old, bbox_new = self.predict(
            sim.frame_number - 2 * self.timing_config.pred_frame_num,
            sim.frame_number - self.timing_config.pred_frame_num,
        )

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
        speed_per_frame = (
            movement[0] / self.timing_config.pred_frame_num,
            movement[1] / self.timing_config.pred_frame_num,
        )

        # calculate camera correction based on the speed of the worm and current worm position
        camera_mid = sim.camera.camera_size[0] / 2, sim.camera.camera_size[1] / 2
        dx = round(
            bbox_new_mid[0]
            - camera_mid[0]
            + speed_per_frame[0] * (self.timing_config.moving_frame_num + self.timing_config.pred_frame_num)
        )
        dy = round(
            bbox_new_mid[1]
            - camera_mid[1]
            + speed_per_frame[1] * (self.timing_config.moving_frame_num + self.timing_config.pred_frame_num)
        )

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


class LoggingController(SimController):
    def __init__(
        self,
        sim_controller: SimController,
        log_config: LogConfig,
    ):
        super().__init__(sim_controller.timing_config)

        self.sim_controller = sim_controller
        self.log_config = log_config

        self._camera_frames = deque(maxlen=self.timing_config.cycle_length)
        self._platform_positions = deque(maxlen=self.timing_config.cycle_length)
        self._camera_bboxes = deque(maxlen=self.timing_config.cycle_length)
        self._micro_bboxes = deque(maxlen=self.timing_config.cycle_length)

    def on_sim_start(self, sim: Simulator):
        self.sim_controller.on_sim_start(sim)

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

    def on_sim_end(self, sim: Simulator):
        self.sim_controller.on_sim_end(sim)

        self._image_saver.close()
        self._bbox_logger.close()

    def on_cycle_start(self, sim: Simulator):
        self.sim_controller.on_cycle_start(sim)

    def on_cycle_end(self, sim: Simulator):
        self.sim_controller.on_cycle_end(sim)

        worm_bboxes = self._cycle_predict_all(sim)

        for i, worm_bbox in enumerate(worm_bboxes):
            csv_row = {}
            csv_row["plt_x"], csv_row["plt_y"] = self._platform_positions[i]
            csv_row["cam_x"], csv_row["cam_y"], csv_row["cam_w"], csv_row["cam_h"] = self._camera_bboxes[i]
            csv_row["mic_x"], csv_row["mic_y"], csv_row["mic_w"], csv_row["mic_h"] = self._micro_bboxes[i]

            frame_number = sim.frame_number - len(self._camera_frames) + i + 1
            phase = (
                "imaging" if i % self.timing_config.cycle_length <= self.timing_config.imaging_frame_num else "moving"
            )

            csv_row["frame"] = frame_number
            csv_row["cycle"] = sim.cycle_number
            csv_row["phase"] = phase

            if worm_bbox is not None:
                # format bbox to be have absolute position
                cam_bbox = self._camera_bboxes[i]
                worm_bbox = (worm_bbox[0] + cam_bbox[0], worm_bbox[1] + cam_bbox[1], worm_bbox[2], worm_bbox[3])
            else:
                # log prediction error
                path = self.log_config.err_file_path.format(frame_number)
                self._image_saver.schedule_save(self._camera_frames[i], path)
                worm_bbox = (-1, -1, -1, -1)

            csv_row["wrm_x"], csv_row["wrm_y"], csv_row["wrm_w"], csv_row["wrm_h"] = worm_bbox

            self._bbox_logger.write(csv_row)

        self._bbox_logger.flush()

    def on_camera_frame(self, sim: Simulator, cam_view: np.ndarray):
        self.sim_controller.on_camera_frame(sim, cam_view)

        # log everything
        self._camera_frames.append(cam_view)
        self._platform_positions.append(sim.camera.position)
        self._camera_bboxes.append(sim.camera._calc_view_bbox(*sim.camera.camera_size))
        self._micro_bboxes.append(sim.camera._calc_view_bbox(*sim.camera.micro_size))

        if self.log_config.save_cam_view:
            # save camera view
            path = self.log_config.cam_file_path.format(sim.frame_number)
            self._image_saver.schedule_save(cam_view, path)

        if self.log_config.save_mic_view:
            # save micro view
            mic_view = sim.camera.micro_view()
            path = self.log_config.mic_file_path.format(sim.frame_number)
            self._image_saver.schedule_save(mic_view, path)

    def on_imaging_start(self, sim: Simulator):
        self.sim_controller.on_imaging_start(sim)

    def on_micro_frame(self, sim: Simulator, micro_view: np.ndarray):
        self.sim_controller.on_micro_frame(sim, micro_view)

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
