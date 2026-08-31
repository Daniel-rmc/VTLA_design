"""Learning-rate schedules owned by the VTLA LeRobot plugin."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from lerobot.optim.schedulers import LRSchedulerConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@LRSchedulerConfig.register_subclass("warmup_stable_cosine_decay")
@dataclass
class WarmupStableCosineDecaySchedulerConfig(LRSchedulerConfig):
    """Warm up, hold the peak LR, then cosine-decay near the end.

    The phase boundaries are expressed against ``reference_training_steps`` and
    are scaled to the actual run length.  This keeps smoke tests and short
    resume checks usable without changing the production schedule.
    """

    num_warmup_steps: int = 1_000
    reference_training_steps: int = 30_000
    decay_start_step: int = 27_000
    peak_lr: float = 1e-5
    decay_lr: float = 1e-6

    def __post_init__(self) -> None:
        if self.reference_training_steps <= 0:
            raise ValueError("reference_training_steps must be positive")
        if not 0 <= self.num_warmup_steps <= self.decay_start_step:
            raise ValueError("num_warmup_steps must be in [0, decay_start_step]")
        if not self.decay_start_step < self.reference_training_steps:
            raise ValueError("decay_start_step must be smaller than reference_training_steps")
        if self.peak_lr <= 0:
            raise ValueError("peak_lr must be positive")
        if not 0 <= self.decay_lr <= self.peak_lr:
            raise ValueError("decay_lr must be in [0, peak_lr]")

    def scaled_phase_steps(self, num_training_steps: int) -> tuple[int, int]:
        """Return warmup end and decay start for an arbitrary run length."""
        if num_training_steps <= 0:
            raise ValueError("num_training_steps must be positive")

        scale = num_training_steps / self.reference_training_steps
        warmup_steps = int(self.num_warmup_steps * scale)
        decay_start = int(self.decay_start_step * scale)

        # Leave at least one scheduler interval for the final decay.  A one-step
        # smoke test therefore uses peak LR for its only update and reaches the
        # configured floor when the scheduler advances past that update.
        decay_start = min(decay_start, num_training_steps - 1)
        warmup_steps = min(warmup_steps, decay_start)
        return warmup_steps, decay_start

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        warmup_steps, decay_start = self.scaled_phase_steps(num_training_steps)
        if num_training_steps != self.reference_training_steps:
            logging.info(
                "Auto-scaling VTLA LR scheduler from %d to %d total steps: "
                "warmup %d -> %d, decay start %d -> %d",
                self.reference_training_steps,
                num_training_steps,
                self.num_warmup_steps,
                warmup_steps,
                self.decay_start_step,
                decay_start,
            )

        alpha = self.decay_lr / self.peak_lr

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                # Match the non-zero linear warmup convention used by LeRobot's
                # Pi0/Pi0.5 scheduler.
                if current_step <= 0:
                    return 1.0 / (warmup_steps + 1)
                fraction_remaining = 1.0 - current_step / warmup_steps
                return (1.0 / (warmup_steps + 1) - 1.0) * fraction_remaining + 1.0

            if current_step <= decay_start:
                return 1.0

            decay_duration = max(1, num_training_steps - decay_start)
            progress = (min(current_step, num_training_steps) - decay_start) / decay_duration
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (1.0 - alpha) * cosine + alpha

        return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

