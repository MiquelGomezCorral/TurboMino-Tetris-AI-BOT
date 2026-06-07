from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps: int, n_steps: int, n_envs: int = 1):
        super().__init__(verbose=0)
        self.total = total_timesteps
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.last_update = 0
        self._start_time = None

    def _on_training_start(self):
        self._start_time = time.time()
        self.pbar = tqdm(
            total=self.total,
            unit="st",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

    def _on_step(self):
        self.pbar.update(self.n_envs)
        n = self.num_timesteps
        if n - self.last_update >= self.n_steps:
            self.last_update = n
            self._update_postfix()
        return True

    def _on_rollout_end(self):
        it = getattr(self.model, "_n_updates", 0) + 1
        self.pbar.set_description(f"Iter {it}")
        self._update_postfix()

    def _update_postfix(self):
        log = self.logger.name_to_value if hasattr(self.logger, "name_to_value") else {}
        buf = self.model.ep_info_buffer if self.model.ep_info_buffer else []

        rew = np.mean([e["r"] for e in buf]) if buf else 0.0
        length = np.mean([e["l"] for e in buf]) if buf else 0.0

        elapsed = time.time() - self._start_time if self._start_time else 1.0
        fps = int(self.num_timesteps / elapsed) if elapsed > 0 else 0

        pc = {
            "rew": f"{rew:.1f}",
            "len": f"{length:.1f}",
            "loss": f"{log.get('train/loss', 0):.1f}",
            "kl": f"{log.get('train/approx_kl', 0):.4f}",
            "fps": f"{fps}",
        }
        if "val/mean_score" in log:
            pc["val"] = f"{log['val/mean_score']:.1f}"
        self.pbar.set_postfix(**pc)

    def _on_training_end(self):
        self.pbar.close()


class EntropyAnnealCallback(BaseCallback):
    """Linearly anneals the entropy coefficient from start to end over training."""

    def __init__(self, start: float, end: float, total_timesteps: int):
        super().__init__(verbose=0)
        self.start = start
        self.end = end
        self.total = total_timesteps

    def _on_step(self) -> bool:
        progress = min(1.0, self.model.num_timesteps / self.total)
        self.model.ent_coef = self.end + (self.start - self.end) * (1.0 - progress)
        return True
