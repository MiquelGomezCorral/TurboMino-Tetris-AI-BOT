from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time
import pytorch_lightning as pl

from .test import test_on_game

# ==========================================
# Callbacks for Stable Baselines3 training
# ==========================================

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



class TetrisValidationCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int = 10000, n_eval_episodes: int = 5, max_pieces: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.max_pieces = max_pieces
        self.best_mean_score = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            scores, lines, pieces, all_clears, tetrises = test_on_game(
                n_eval_episodes=self.n_eval_episodes,
                max_pieces=self.max_pieces,
                eval_env=self.eval_env,
                model=self.model,
            )

            mean_score = np.mean(scores)
            self.logger.record("val/mean_score", mean_score)
            self.logger.record("val/mean_lines_cleared", np.mean(lines))
            self.logger.record("val/mean_pieces_placed", np.mean(pieces))
            self.logger.record("val/mean_all_clears", np.mean(all_clears))
            self.logger.record("val/mean_tetrises", np.mean(tetrises))

            if mean_score > self.best_mean_score:
                self.best_mean_score = mean_score
                save_path = f"{self.model.tensorboard_log}/best_model"
                self.model.save(save_path)
                if self.verbose > 0:
                    print(f"\n - New best validation score: {mean_score:.2f}! Model saved.")

        return True



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
    

# ==========================================
# Callbacks for PyTorch Lightning training 
# ==========================================

class TetrisEvalCallback(pl.Callback):
    def __init__(self, eval_env, eval_freq: int = 1, n_eval_episodes: int = 5, max_pieces: int = 100):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.max_pieces = max_pieces
        self.best_mean_score = -np.inf

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch % self.eval_freq != 0:
            return

        # Extract the encoder and run actual game episodes
        scores, lines, pieces, all_clears, tetrises = test_on_game(
            n_eval_episodes=self.n_eval_episodes,
            max_pieces=self.max_pieces,
            eval_env=self.eval_env,
            model=pl_module,       # your TurboMinoModule
        )

        mean_score = np.mean(scores)
        pl_module.log("game/mean_score",         mean_score,       prog_bar=True)
        pl_module.log("game/mean_lines_cleared",  np.mean(lines))
        pl_module.log("game/mean_pieces_placed",  np.mean(pieces))
        pl_module.log("game/mean_all_clears",  np.mean(all_clears))
        pl_module.log("game/mean_tetrises",  np.mean(tetrises))

        if mean_score > self.best_mean_score:
            self.best_mean_score = mean_score
            ckpt_path = f"{trainer.logger.log_dir}/best_game_model.ckpt"
            trainer.save_checkpoint(ckpt_path)
            print(f"\n - New best game score: {mean_score:.2f}! Saved to {ckpt_path}")


