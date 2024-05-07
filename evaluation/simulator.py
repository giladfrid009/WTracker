from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import abc
from tqdm.auto import tqdm

from dataset.raw_dataset import ExperimentConfig
from frame_reader import FrameReader
from evaluation.view_controller import ViewController
from utils.config_base import ConfigBase


@dataclass
class TimingConfig(ConfigBase):
    frames_per_sec: int
    ms_per_frame: float = field(init=False)

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

    frame_padding_value: tuple[int, int, int] = field(default_factory=lambda: (255, 255, 255))

    def __post_init__(self):
        self.ms_per_frame = 1000 / self.frames_per_sec
        self.imaging_frame_num = math.ceil(self.imaging_time_ms / self.ms_per_frame)
        self.pred_frame_num = math.ceil(self.pred_time_ms / self.ms_per_frame)
        self.moving_frame_num = math.ceil(self.moving_time_ms / self.ms_per_frame)
        self.mm_per_px = 1 / self.px_per_mm

        self.camera_size_px = (
            round(self.px_per_mm * self.camera_size_mm[0]),
            round(self.px_per_mm * self.camera_size_mm[1]),
        )

        self.micro_size_px = (
            round(self.px_per_mm * self.micro_size_mm[0]),
            round(self.px_per_mm * self.micro_size_mm[1]),
        )

    @property
    def cycle_length(self) -> int:
        return self.imaging_frame_num + self.moving_frame_num

    @property
    def cycle_time_ms(self) -> float:
        return self.cycle_length * self.ms_per_frame


class Simulator:
    def __init__(
        self,
        timing_config: TimingConfig,
        experiment_config: ExperimentConfig,
        reader: FrameReader,
        controller: SimController,
        motor_controller: MovementController,
    ) -> None:
        self._timing_config = timing_config
        self._experiment_config = experiment_config
        self._controller = controller
        self._motor_controller = motor_controller

        self._camera = ViewController(
            frame_reader=reader,
            camera_size=timing_config.camera_size_px,
            micro_size=timing_config.micro_size_px,
            init_position=experiment_config.init_position,
            padding_value=timing_config.frame_padding_value,
        )

    @property
    def camera(self) -> ViewController:
        return self._camera

    @property
    def position(self) -> tuple[int, int]:
        return self._camera.position

    @property
    def cycle_number(self) -> int:
        return self._camera.index // self._timing_config.cycle_length

    @property
    def frame_number(self) -> int:
        return self._camera.index

    @property
    def cycle_step(self) -> int:
        return self._camera.index % self._timing_config.cycle_length

    def camera_view(self) -> np.ndarray:
        return self._camera.camera_view()

    def micro_view(self) -> np.ndarray:
        return self._camera.micro_view()

    def _reset(self):
        self.camera.reset()
        self.camera.set_position(*self._experiment_config.init_position)

    def run(self, visualize: bool = False, wait_key: bool = False):
        config = self._timing_config

        total_cycles = len(self._camera) // config.cycle_length
        pbar = tqdm(total=total_cycles, desc="Simulation Progress", unit="cycle")

        self._reset()
        self._controller.on_sim_start(self)

        while self._camera.progress():
            if self.cycle_step == 0:
                if self.cycle_number > 0:
                    self._controller.on_movement_end(self)
                    self._controller.on_cycle_end(self)
                
                self._controller.on_cycle_start(self)

            self._controller.on_camera_frame(self)

            if self.cycle_step == 0:
                self._controller.on_imaging_start(self)

            if self.cycle_step < config.imaging_frame_num:
                self._controller.on_micro_frame(self)

            if self.cycle_step == config.imaging_frame_num:
                self._controller.on_imaging_end(self)
                self._controller.on_movement_start(self)
                dx, dy = self._controller.provide_moving_vector(self)
                self._motor_controller.register_move(dx, dy)

            if config.imaging_frame_num <= self.cycle_step < config.imaging_frame_num + config.moving_frame_num:
                dx, dy = self._motor_controller.step()
                self._camera.move_position(dx, dy)

            if self.cycle_step == config.cycle_length - 1:
                pbar.update(1)

            if visualize:
                self._camera.visualize_world(timeout=0 if wait_key else 1)

        self._controller.on_sim_end(self)

        pbar.close()


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

    def on_camera_frame(self, sim: Simulator):
        pass

    def on_imaging_start(self, sim: Simulator):
        pass

    def on_micro_frame(self, sim: Simulator):
        pass

    def on_imaging_end(self, sim: Simulator):
        pass

    def on_movement_start(self, sim: Simulator):
        pass

    def on_movement_end(self, sim: Simulator):
        pass

    @abc.abstractmethod
    def provide_moving_vector(self, sim: Simulator) -> tuple[int, int]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _cycle_predict_all(self, sim: Simulator) -> list[tuple[float, float, float, float]]:
        """
        Returns a list of bbox predictions of the worm, for each frame of the current cycle.
        If a prediction is not available, return None for that frame.
        Must work even if the current cycle is not finished yet.
        Used internally for logging.
        """
        raise NotImplementedError()


class MovementController(abc.ABC):
    def __init__(self, timing_config: TimingConfig):
        self.timing_config = timing_config
        self.movement_steps = self.timing_config.moving_frame_num

    @abc.abstractmethod
    def register_move(dx: int, dy: int):
        pass

    @abc.abstractmethod
    def step(self) -> tuple[int, int]:
        pass
