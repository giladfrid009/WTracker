import abc
import numpy as np

from wtracker.sim.config import TimingConfig


class MotorController(abc.ABC):
    """
    Abstract base class for motor controllers used in the Simulator class.
    This motor controls the movement of the simulated platform.

    Args:
        timing_config (TimingConfig): The timing configuration of the simulation.

    Attributes:
        timing_config (TimingConfig): The timing configuration for the motor controller.
        movement_steps (int): The number of movement steps (in units of frames) based on the timing configuration.
    """

    def __init__(self, timing_config: TimingConfig):
        self.timing_config = timing_config
        self.movement_steps = self.timing_config.moving_frame_num

    @abc.abstractmethod
    def register_move(self, dx: int, dy: int):
        pass

    @abc.abstractmethod
    def step(self) -> tuple[int, int]:
        pass


class StepMotorController(MotorController):
    """
    A simple motor controller that manages the movement of a motor.
    The motor moved the entire distance in one step, the movement happens after 'move_after_ratio' percent of 'movement_steps' have passed.

    Args:
        timing_config (TimingConfig): The timing configuration of the simulation.
        move_after_ratio (float, optional): The ratio of movement steps after which the motor should move. Defaults to 0.5.
    """

    def __init__(self, timing_config: TimingConfig, move_after_ratio: float = 0.5):
        assert 0 <= move_after_ratio <= 1
        super().__init__(timing_config)
        self.queue: list = []
        self.move_at_step = round(self.movement_steps * move_after_ratio)

    def register_move(self, dx: int, dy: int):
        for _ in range(self.movement_steps - 1):
            self.queue.append((0, 0))
        self.queue.insert(self.move_at_step, (dx, dy))

    def step(self) -> tuple[int, int]:
        return self.queue.pop(0)


class SineMotorController(MotorController):
    """
    A motor controller that generates sinusoidal movements.

    Args:
        timing_config (TimingConfig): The timing configuration of the simulation.
    """

    def __init__(self, timing_config: TimingConfig):
        super().__init__(timing_config)
        self.queue: list = []

    def register_move(self, dx: int, dy: int) -> None:
        assert len(self.queue) == 0

        for i in range(self.movement_steps):
            step_size = (
                np.cos((i * np.pi) / self.movement_steps) - np.cos(((i + 1) * np.pi) / self.movement_steps)
            ) / 2
            step = (step_size * dx, step_size * dy)
            self.queue.append(step)

    def step(self) -> tuple[int, int]:
        dx, dy = self.queue.pop(0)
        rdx, rdy = (round(dx), round(dy))
        resid_x, resid_y = dx - rdx, dy - rdy

        if self.queue:
            self.queue[0] = (self.queue[0][0] + resid_x, self.queue[0][1] + resid_y)

        return (rdx, rdy)
