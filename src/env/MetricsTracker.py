import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class MetricsTracker:
    """

    Used to track:
    - per-timestep metrics in buffer (cleared when MetricsTracker.flush() is called)
    - per-training metrics like the sliding window of episode return (not cleared when MetricsTracker.flush() is called)
    - misc. metrics like WR against various bots (store in special storage?)
    - additional metrics that are calculated (like elapsed time)

    """
    def __init__(self, log_dir: str | None = None, use_tensorboard: bool = False, return_window_size: int = 128, hyperparameters: dict[str, Any] = {}):
        self.buffers: dict[str, list[float]] = defaultdict(list)
        self.episode_returns: Deque[float] = deque(maxlen=return_window_size)
        self.start_time = time.time()
        self.flat_statistics: set[str] = set()

        self.writer = None
        if use_tensorboard:
            self.writer = SummaryWriter(f"{PROJECT_ROOT}/experiments/{log_dir}/logs")

        self.hyperparameters = hyperparameters

    def register_flat_statistic(self, name: str):
        """
        Register a statistic as flat (meaning no aggregation is performed)
        """
        self.flat_statistics.add(name)

    def record(self, key: str, value: float) -> None:
        self.buffers[key].append(value)

    def record_episode_return(self, episode_return: float):
        self.episode_returns.append(episode_return)

    def flush(self, timestep: int):
        """
        Resets buffer data, flushes it to tensorboard if applicable, and prints a text summary
        """
        summary: dict[str, float] = {}

        for key, values in self.buffers.items():
            if not values:
                continue
            arr = np.asarray(values)
            if key in self.flat_statistics:
                summary[key] = arr.mean()
            else:
                summary[f"{key}/mean"] = arr.mean()
                summary[f"{key}/std"] = arr.std()
                summary[f"{key}/min"] = arr.min()
                summary[f"{key}/max"] = arr.max()

        if self.episode_returns:
            summary["episode_return/mean"] = np.mean(self.episode_returns)
            summary["episode_return/std"] = np.std(self.episode_returns)

        summary["time/elapsed_seconds"] = time.time() - self.start_time

        if self.writer is not None:
            for key, value in summary.items():
                self.writer.add_scalar(key, value, timestep)

        self.buffers.clear()
        self.printout_summary(timestep, summary)

    def printout_summary(self, timestep: int, summary: dict[str, float]):
        parts = [f"iter={timestep}"]
        for key, value in summary.items():
            parts.append(f"{key}={value:.2f}")
        print(" | ".join(parts))

    def close(self, metrics: dict[str, Any]):
        if self.writer is not None:
            self.writer.close()
            hyperparam_writer = SummaryWriter(f"{PROJECT_ROOT}/experiments/{self.writer.log_dir}/logs")
            hyperparam_writer.add_hparams(
                hparam_dict=self.hyperparameters,
                metric_dict=metrics,
                run_name="."
            )
            hyperparam_writer.close()





