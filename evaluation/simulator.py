from __future__ import annotations
import numpy as np
import abc
from tqdm.auto import tqdm

from frame_reader import FrameReader
from dummy_reader import DummyReader
from evaluation.view_controller import ViewController
from evaluation.config import *
from evaluation.motor_controllers import MotorController, SimpleMotorController

class Simulator:
    def __init__(
        self,
        timing_config: TimingConfig,
        experiment_config: ExperimentConfig,
        sim_controller: SimController,
        reader: FrameReader = None,
        motor_controller: MotorController = None,
    ) -> None:
        self._timing_config = timing_config
        self._experiment_config = experiment_config
        self._sim_controller = sim_controller

        if reader is None:
            reader = DummyReader(experiment_config)

        if motor_controller is None:
            motor_controller = SimpleMotorController(timing_config, move_after_ratio=0.5)
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
        self._sim_controller.on_sim_start(self)

        while self._camera.progress():
            if self.cycle_step == 0:
                if self.cycle_number > 0:
                    self._sim_controller.on_movement_end(self)
                    self._sim_controller.on_cycle_end(self)
                
                self._sim_controller.on_cycle_start(self)

            self._sim_controller.on_camera_frame(self)

            if self.cycle_step == 0:
                self._sim_controller.on_imaging_start(self)

            if self.cycle_step < config.imaging_frame_num:
                self._sim_controller.on_micro_frame(self)

            if self.cycle_step == config.imaging_frame_num:
                self._sim_controller.on_imaging_end(self)
                self._sim_controller.on_movement_start(self)
                dx, dy = self._sim_controller.provide_moving_vector(self)
                self._motor_controller.register_move(dx, dy)

            if config.imaging_frame_num <= self.cycle_step < config.imaging_frame_num + config.moving_frame_num:
                dx, dy = self._motor_controller.step()
                self._camera.move_position(dx, dy)

            if self.cycle_step == config.cycle_length - 1:
                pbar.update(1)

            if visualize:
                self._camera.visualize_world(timeout=0 if wait_key else 1)

        self._sim_controller.on_sim_end(self)

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

