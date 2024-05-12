import abc
import numpy as np

from evaluation.config import TimingConfig


class MotorController(abc.ABC):
    def __init__(self, timing_config: TimingConfig):
        self.timing_config = timing_config
        self.movement_steps = self.timing_config.moving_frame_num

    @abc.abstractmethod
    def register_move(self, dx: int, dy: int):
        pass

    @abc.abstractmethod
    def step(self) -> tuple[int, int]:
        pass


class SimpleMotorController(MotorController):
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
    def __init__(self, timing_config: TimingConfig):
        super().__init__(timing_config)
        self.queue: list = []

    def _reset(self):
        self.queue.clear()

    def register_move(self, dx: int, dy: int) -> None:
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

        return (rdx, rdy)  # TODO: remove round for more accurate movement
