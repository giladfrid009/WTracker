from collections import queue
from math import ceil

from evaluation.simulator import MovementController, TimingConfig


class SimpleMovementController(MovementController):
    def __init__(self, timing_config: TimingConfig, move_after_ratio: float = 0.5):
        super().__init__(timing_config)
        self.queue: list = []
        assert 0 <= move_after_ratio <= 1
        self.move_at_step = ceil(self.movement_steps * move_after_ratio)

    def register_move(self, dx: int, dy: int):
        for _ in range(self.movement_steps - 1):
            self.queue.append((0, 0))
        self.queue.insert(self.move_at_step, (dx, dy))

    def step(self) -> tuple[int, int]:
        return self.queue.pop(0)
