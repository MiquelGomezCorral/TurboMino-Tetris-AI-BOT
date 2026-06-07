from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps: int, n_steps: int, n_envs: int = 1):
        super().__init__(verbose=0)
        self.total = total_timesteps
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.last_update = 0

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total,
            unit="st",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

    def _on_step(self):
        self.pbar.update(self.n_envs)
        n = self.num_timesteps
        # Update postfix at rollout boundaries (after train() ran)
        if n - self.last_update >= self.n_steps:
            self.last_update = n
            self._postfix_from_logger()
        return True

    def _on_rollout_end(self):
        it = getattr(self.model, "_n_updates", 0) + 1
        self.pbar.set_description(f"Iter {it}")
        self._postfix_from_logger()

    def _postfix_from_logger(self):
        log = self.logger.name_to_value if hasattr(self.logger, "name_to_value") else {}
        if not log:
            return
        pc = {
            "rew": f"{log.get('rollout/ep_rew_mean', 0):.1f}",
            "len": f"{log.get('rollout/ep_len_mean', 0):.1f}",
            "loss": f"{log.get('train/loss', 0):.1f}",
            "kl": f"{log.get('train/approx_kl', 0):.4f}",
            "fps": f"{log.get('time/fps', 0):.0f}",
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
