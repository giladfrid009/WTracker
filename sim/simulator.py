from __future__ import annotations
import numpy as np
import abc
from tqdm.auto import tqdm

from utils.frame_reader import FrameReader, DummyReader
from sim.view_controller import ViewController
from sim.config import *
from sim.motor_controllers import MotorController, SineMotorController


class Simulator:
    """
    A class representing a simulator for a biological experiment.

    Args:
        timing_config (TimingConfig): The timing configuration for the experiment.
        experiment_config (ExperimentConfig): The experiment configuration.
        sim_controller (SimController): The simulation controller.
        reader (FrameReader, optional): The frame reader. Defaults to None.
        motor_controller (MotorController, optional): The motor controller. Defaults to None.

    Attributes:
        _timing_config (TimingConfig): The timing configuration for the experiment.
        _experiment_config (ExperimentConfig): The experiment configuration.
        _sim_controller (SimController): The simulation controller.
        _motor_controller (MotorController): The motor controller.
        _view (ViewController): The view controller.

    """

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
            num_frames = experiment_config.num_frames
            padding_size = (timing_config.camera_size_px[0] // 2 * 2, timing_config.camera_size_px[1] // 2 * 2)
            resolution = tuple([sum(x) for x in zip(experiment_config.orig_resolution, padding_size)])
            reader = DummyReader(num_frames, resolution, colored=True)

        if motor_controller is None:
            motor_controller = SineMotorController(timing_config)
        self._motor_controller = motor_controller

        self._view = ViewController(
            frame_reader=reader,
            camera_size=timing_config.camera_size_px,
            micro_size=timing_config.micro_size_px,
            init_position=experiment_config.init_position,
        )

    @property
    def view(self) -> ViewController:
        """
        Get the view controller.

        Returns:
            ViewController: The view controller.

        """
        return self._view

    @property
    def position(self) -> tuple[int, int]:
        """
        Get the current position.

        Returns:
            tuple[int, int]: The current position.

        """
        return self._view.position

    @property
    def cycle_number(self) -> int:
        """
        Get the current cycle number.

        Returns:
            int: The current cycle number.

        """
        return self._view.index // self._timing_config.cycle_length

    @property
    def frame_number(self) -> int:
        """
        Get the current frame number.

        Returns:
            int: The current frame number.

        """
        return self._view.index

    @property
    def cycle_step(self) -> int:
        """
        Get the current cycle step.

        Returns:
            int: The current cycle step.

        """
        return self._view.index % self._timing_config.cycle_length

    def camera_view(self) -> np.ndarray:
        """
        Get the view that the camera sees.

        Returns:
            np.ndarray: The camera view.

        """
        return self._view.camera_view()

    def micro_view(self) -> np.ndarray:
        """
        Get the view that the microscope sees.

        Returns:
            np.ndarray: The micro view.

        """
        return self._view.micro_view()

    def _reset(self):
        """
        Reset the simulator.

        """
        self.view.reset()
        self.view.set_position(*self._experiment_config.init_position)

    def run(self, visualize: bool = False, wait_key: bool = False):
        """
        Run the simulation.

        Args:
            visualize (bool, optional): Whether to visualize the simulation. Defaults to False.
            wait_key (bool, optional): Whether to wait for a key press to advance the simulation during visualization. Defaults to False.

        """
        config = self._timing_config

        total_cycles = len(self._view) // config.cycle_length
        pbar = tqdm(total=total_cycles, desc="Simulation Progress", unit="cycle")

        self._reset()
        self._sim_controller.on_sim_start(self)

        while self._view.progress():
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

            if self.cycle_step == config.imaging_frame_num - config.pred_frame_num:
                self._sim_controller.begin_movement_prediction(self)

            if self.cycle_step == config.imaging_frame_num:
                self._sim_controller.on_imaging_end(self)
                dx, dy = self._sim_controller.provide_movement_vector(self)
                self._sim_controller.on_movement_start(self)
                self._motor_controller.register_move(dx, dy)

            if config.imaging_frame_num <= self.cycle_step < config.imaging_frame_num + config.moving_frame_num:
                dx, dy = self._motor_controller.step()
                self._view.move_position(dx, dy)

            if self.cycle_step == config.cycle_length - 1:
                pbar.update(1)

            if visualize:
                self._view.visualize_world(timeout=0 if wait_key else 1)

        self._sim_controller.on_sim_end(self)

        pbar.close()


class SimController(abc.ABC):
    """
    Abstract base class for simulator controllers.

    Attributes:
        timing_config (TimingConfig): The timing configuration for the simulator.

    Methods:
        on_sim_start(sim: Simulator): Called when the simulation starts.
        on_sim_end(sim: Simulator): Called when the simulation ends.
        on_cycle_start(sim: Simulator): Called when a new cycle starts.
        on_cycle_end(sim: Simulator): Called when a cycle ends.
        on_camera_frame(sim: Simulator): Called when a camera frame is captured.
        on_imaging_start(sim: Simulator): Called when imaging phase starts.
        on_micro_frame(sim: Simulator): Called when a micro frame is captured.
        on_imaging_end(sim: Simulator): Called when imaging phase ends.
        on_movement_start(sim: Simulator): Called when movement phase starts.
        on_movement_end(sim: Simulator): Called when movement phase ends.
        begin_movement_prediction(sim: Simulator) -> None: Begins the movement prediction.
        provide_movement_vector(sim: Simulator) -> tuple[int, int]: Provides the movement vector for the simulator. The platform is moved by the provided vector.
        _cycle_predict_all(sim: Simulator) -> np.ndarray: Returns a list of bbox predictions of the worm for each frame of the current cycle.
    """

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
    def begin_movement_prediction(self, sim: Simulator) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def provide_movement_vector(self, sim: Simulator) -> tuple[int, int]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _cycle_predict_all(self, sim: Simulator) -> np.ndarray:
        """
        Returns a list of bbox predictions of the worm, for each frame of the current cycle.
        If a prediction is not available, return None for that frame.
        Used internally for logging.
        """
        raise NotImplementedError()
