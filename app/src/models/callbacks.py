from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time

from .test import test_on_game

# ==========================================
# Callbacks for Stable Baselines3 training
# ==========================================

def _curriculum_gate_passed(scores, pieces, max_pieces: int, learned_ratio: float, min_score: float):
    score_ratio = np.mean([score >= min_score for score in scores])
    survival_ratio = np.mean([piece >= max_pieces for piece in pieces])
    return score_ratio >= learned_ratio and survival_ratio >= learned_ratio, score_ratio, survival_ratio


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps: int, rollout_steps: int, n_envs: int = 1):
        super().__init__(verbose=0)
        self.total = total_timesteps
        self.rollout_steps = rollout_steps
        self.n_envs = n_envs
        self.last_update = 0
        self._start_time = None
        self._start_timesteps = 0

    def _on_training_start(self):
        self._start_time = time.time()
        self._start_timesteps = self.num_timesteps
        self.last_update = self.num_timesteps
        self.pbar = tqdm(
            total=self.total,
            unit="st",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

    def _on_step(self):
        self.pbar.update(self.n_envs)
        n = self.num_timesteps
        if n - self.last_update >= self.rollout_steps * self.n_envs:
            self.last_update = n
            self._update_postfix()
        return True

    def _on_rollout_end(self):
        rollout = self.num_timesteps // (self.rollout_steps * self.n_envs)
        self.pbar.set_description(f"Rollout {rollout}")
        self._update_postfix()

    def _update_postfix(self):
        log = self.logger.name_to_value if hasattr(self.logger, "name_to_value") else {}
        buf = self.model.ep_info_buffer if self.model.ep_info_buffer else []

        rew = np.mean([e["r"] for e in buf]) if buf else 0.0
        length = np.mean([e["l"] for e in buf]) if buf else 0.0

        elapsed = time.time() - self._start_time if self._start_time else 1.0
        stage_steps = self.num_timesteps - self._start_timesteps
        fps = int(stage_steps / elapsed) if elapsed > 0 else 0

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
    def __init__(
        self, eval_env, eval_freq: int = 10000, n_eval_episodes: int = 5,
        max_pieces: int = 100, learned_ratio: float | None = None,
        min_score: float = 0.0, eval_seed: int | None = None,
        best_model_path: str | None = None, verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.max_pieces = max_pieces
        self.learned_ratio = learned_ratio
        self.min_score = min_score
        self.eval_seed = eval_seed
        self.best_model_path = best_model_path
        self.saved_best_model_path = None
        self.best_key = None
        self.learned = False

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            rewards, scores, lines, pieces, all_clears, tetrises = test_on_game(
                n_eval_episodes=self.n_eval_episodes,
                max_pieces=self.max_pieces,
                eval_env=self.eval_env,
                model=self.model,
                seed=self.eval_seed,
            )

            mean_score = np.mean(scores)
            self.logger.record("val/mean_reward", np.mean(rewards))
            self.logger.record("val/min_score", np.min(scores))
            self.logger.record("val/mean_score", mean_score)
            self.logger.record("val/max_score", np.max(scores))
            self.logger.record("val/mean_lines_cleared", np.mean(lines))
            self.logger.record("val/min_pieces_placed", np.min(pieces))
            self.logger.record("val/mean_pieces_placed", np.mean(pieces))
            self.logger.record("val/max_pieces_placed", np.max(pieces))
            self.logger.record("val/mean_all_clears", np.mean(all_clears))
            self.logger.record("val/mean_tetrises", np.mean(tetrises))

            summary = (
                f"Validation @ {self.num_timesteps:_}: "
                f"score min/avg/max={min(scores):.0f}/{mean_score:.1f}/{max(scores):.0f} | "
                f"pieces min/avg/max={min(pieces)}/{np.mean(pieces):.1f}/{max(pieces)}"
            )
            if self.learned_ratio is not None:
                self.learned, score_ratio, survival_ratio = _curriculum_gate_passed(
                    scores, pieces, self.max_pieces, self.learned_ratio, self.min_score,
                )
                self.logger.record("curriculum/score_ratio", score_ratio)
                self.logger.record("curriculum/survival_ratio", survival_ratio)
                self.logger.record("curriculum/gate_passed", self.learned)
                best_key = (
                    (1, mean_score) if self.learned
                    else (0, min(score_ratio, survival_ratio), mean_score)
                )
                summary += f" | score pass={score_ratio:.0%}, survival={survival_ratio:.0%}"
            else:
                best_key = (mean_score,)

            tqdm.write(summary)

            if self.best_key is None or best_key > self.best_key:
                self.best_key = best_key
                self.best_model_path = self.best_model_path or f"{self.model.tensorboard_log}/best_model.zip"
                self.model.save(self.best_model_path)
                self.saved_best_model_path = self.best_model_path
                if self.verbose > 0:
                    print(f"\n - New best validation score: {mean_score:.2f}! Model saved.")

            if self.learned:
                tqdm.write("Curriculum gate passed.")
                return False

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
