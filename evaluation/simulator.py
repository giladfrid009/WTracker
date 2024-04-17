from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import abc
import cv2 as cv
from tqdm.auto import tqdm

from frame_reader import FrameReader
from evaluation.view_controller import ViewController
from utils.config_base import ConfigBase
from utils.display_utils import ImageDisplay, HotKey


@dataclass
class TimingConfig(ConfigBase):
    frames_per_sec: int
    secs_per_frame: float = field(init=False)

    imaging_time_ms: float
    imaging_frame_num: int = field(init=False)

    pred_time_ms: float
    pred_frame_num: int = field(init=False)

    moving_time_ms: float
    moving_frame_num: int = field(init=False)

    px_per_mm: int
    mm_per_px: float = field(init=False)

    camera_size_mm: tuple[float, float]
    camera_size_px: tuple[int, int] = field(init=False)

    micro_size_mm: tuple[float, float]
    micro_size_px: tuple[int, int] = field(init=False)

    init_position: tuple[int, int]

    frame_padding_value: tuple[int, int, int] = field(default_factory=lambda: (255, 255, 255))

    def __post_init__(self):
        self.secs_per_frame = 1000 / self.frames_per_sec
        self.imaging_frame_num = math.ceil(self.imaging_time_ms / self.secs_per_frame)
        self.pred_frame_num = math.ceil(self.pred_time_ms / self.secs_per_frame)
        self.moving_frame_num = math.ceil(self.moving_time_ms / self.secs_per_frame)
        self.mm_per_px = 1 / self.px_per_mm

        self.camera_size_px = (
            round(self.px_per_mm * self.camera_size_mm[0]),
            round(self.px_per_mm * self.camera_size_mm[1]),
        )

        self.micro_size_px = (
            round(self.px_per_mm * self.micro_size_mm[0]),
            round(self.px_per_mm * self.micro_size_mm[1]),
        )


class Simulator:
    def __init__(self, config: TimingConfig, reader: FrameReader, controller: SimController) -> None:
        self._config = config
        self._controller = controller

        self._camera = ViewController(
            frame_reader=reader,
            camera_size=config.camera_size_px,
            micro_size=config.micro_size_px,
            init_position=config.init_position,
            padding_value=config.frame_padding_value,
        )

    @property
    def camera(self) -> ViewController:
        return self._camera

    @property
    def position(self) -> tuple[int, int]:
        return self._camera.position

    def reset(self):
        self._camera.reset()

    def run_current(self):
        config = self._config
        cycle_length = config.imaging_frame_num + config.moving_frame_num
        cycle_step = self.camera.index % cycle_length

        if cycle_step == 0:
            self._controller.on_cycle_start(self)

        self._controller.on_camera_frame(self, cycle_step, self._camera.camera_view())

        if cycle_step == 0:
            self._controller.on_imaging_start(self)

        if cycle_step < config.imaging_frame_num:
            self._controller.on_micro_frame(self, self._camera.micro_view())

        if cycle_step == config.imaging_frame_num - 1:
            self._controller.on_imaging_end(self)

        if cycle_step == config.imaging_frame_num:
            dx, dy = self._controller.provide_moving_vector(self)
            self._camera.move_position(dx, dy)
            self._controller.on_movement_start(self)

        if cycle_step == config.imaging_frame_num + config.moving_frame_num - 1:
            self._controller.on_movement_end(self)
            self._controller.on_cycle_end(self)

    def run_all(self, visualize: bool = False):
        self.reset()
        config = self._config

        cycle_length = config.imaging_frame_num + config.moving_frame_num

        total_cycles = len(self._camera) // cycle_length
        pbar = tqdm(total=total_cycles, desc="Simulation Progress", unit="cycle")
        self._controller.on_sim_start(self)

        while self._camera.progress():
            self.run_current()

            if visualize:
                self.camera.visualize_world(timeout=1)

            if self.camera.index % cycle_length == cycle_length - 1:
                pbar.update(1)

        self._controller.on_sim_end(self)
        pbar.close()

    def run_manual(self, start_frame: int = 0):
        self.reset()

        self.camera.seek(start_frame)
        self._controller.on_sim_start(self)

        break_loop = False

        def next(key: str):
            self.camera.progress()
            self.run_current()
            self.camera.visualize_world(timeout=1)
            self.camera.image_display.set_title(f"Frame: {self.camera.index}")

        def prev(key: str):
            self.camera.progress(-1)
            self.run_current()
            self.camera.visualize_world(timeout=1)
            self.camera.image_display.set_title(f"Frame: {self.camera.index}")

        def quit(key: str):
            nonlocal break_loop
            break_loop = True

        self.camera.image_display.register_hotkey(HotKey("d", next))
        self.camera.image_display.register_hotkey(HotKey("a", prev))
        self.camera.image_display.register_hotkey(HotKey("q", quit))

        while not break_loop and self.camera.can_read():
            self.camera.image_display.wait_key()

        self._controller.on_sim_end(self)


class SimController(abc.ABC):
    def __init__(self, timing_config: TimingConfig):
        self.timing_config = timing_config

    def on_sim_start(self, sim: Simulator):
        pass

    def on_sim_end(self, sim: Simulator):
        pass

    def on_cycle_start(self, sim: Simulator):
        pass

    def on_cycle_end(self, sim: Simulator):
        pass

    def on_camera_frame(self, cycle_step: int, sim: Simulator, cam_view: np.ndarray):
        pass

    def on_imaging_start(self, sim: Simulator):
        pass

    def on_micro_frame(self, sim: Simulator, micro_view: np.ndarray):
        pass

    def on_imaging_end(self, sim: Simulator):
        pass

    def on_movement_start(self, sim: Simulator):
        pass

    def on_movement_end(self, sim: Simulator):
        pass

    @abc.abstractmethod
    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        pass
