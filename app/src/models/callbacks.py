from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from .test import test_on_game

# ==========================================
# Callbacks for Stable Baselines3 training
# ==========================================

def _curriculum_gate_passed(rewards, pieces, max_pieces: int, learned_ratio: float, min_reward: float):
    reward_ratio = np.mean([reward >= min_reward for reward in rewards])
    survival_ratio = np.mean([piece >= max_pieces for piece in pieces])
    return reward_ratio >= learned_ratio and survival_ratio >= learned_ratio, reward_ratio, survival_ratio


class PPOProgressCallback(BaseCallback):
    """Write reliable sample and optimization progress lines to the log."""

    def __init__(self, total_timesteps: int):
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self._rollout_buffer = None
        self._original_get = None
        self._update_index = 0
        self._epoch = 0

    def _on_training_start(self):
        self._start_timesteps = self.model.num_timesteps
        self._wrap_rollout_buffer()
        self._print_samples()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        self._update_index += 1
        self._epoch = 0
        self._print_samples()
        self._print_epochs()

    def _wrap_rollout_buffer(self):
        self._rollout_buffer = self.model.rollout_buffer
        self._original_get = self._rollout_buffer.get

        def get(batch_size=None):
            for rollout_data in self._original_get(batch_size):
                yield rollout_data
            else:
                self._epoch += 1
                self._print_epochs()

        self._rollout_buffer.get = get

    def _print_samples(self):
        completed = max(0, self.num_timesteps - self._start_timesteps)
        shown = min(self.total_timesteps, completed)
        ratio = shown / self.total_timesteps if self.total_timesteps else 1.0
        bar = _progress_bar(ratio)
        print(
            f"[ppo samples] |{bar}| {shown:,}/{self.total_timesteps:,} "
            f"({ratio:.1%})",
            flush=True,
        )

    def _print_epochs(self, partial=False):
        total_epochs = self.model.n_epochs
        ratio = self._epoch / total_epochs if total_epochs else 1.0
        suffix = " partial" if partial else ""
        print(
            f"[ppo epochs] update={self._update_index} |{_progress_bar(ratio)}| "
            f"{self._epoch}/{total_epochs}{suffix}",
            flush=True,
        )

    def _on_training_end(self):
        if self._update_index and self._epoch < self.model.n_epochs:
            self._print_epochs(partial=True)
        self._print_samples()
        if self._rollout_buffer is not None and self._original_get is not None:
            self._rollout_buffer.get = self._original_get
            self._original_get = None


def _progress_bar(ratio: float, width: int = 24) -> str:
    filled = round(max(0.0, min(1.0, ratio)) * width)
    return "#" * filled + "-" * (width - filled)


class TetrisValidationCallback(BaseCallback):
    def __init__(
        self, eval_env, best_model_path: str, eval_freq: int = 10000, n_eval_episodes: int = 5,
        max_pieces: int = 100, learned_ratio: float | None = None,
        min_reward: float = 0.0, eval_seed: int | None = None,
        model_path_template: str | None = None,
        on_best_model=None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.max_pieces = max_pieces
        self.learned_ratio = learned_ratio
        self.min_reward = min_reward
        self.eval_seed = eval_seed
        self.best_model_path = best_model_path
        self.model_path_template = model_path_template
        self.on_best_model = on_best_model
        self.best_key = None
        self.learned = False
        self._stop_training = False


    def _on_training_start(self):
        self._stop_training = not self._run_evaluation()

    def _on_step(self) -> bool:
        if self._stop_training:
            return False
        if self.n_calls > 0 and self.n_calls % self.eval_freq == 0:
            return self._run_evaluation()

        return True

    def _run_evaluation(self) -> bool:
        model_path = None
        if self.model_path_template:
            model_path = self.model_path_template.format(
                num_timesteps=self.num_timesteps
            )
            # Reuse of a timestep path must not load an older architecture from a previous run.
            self.model.save(model_path)
        rewards, game_scores, lines, pieces, all_clears, tetrises = test_on_game(
            CONFIG=self.eval_env.CONFIG,
            T_CONFIG=self.eval_env.T_CONFIG,
            eval_episodes=self.n_eval_episodes,
            max_pieces=self.max_pieces,
            model_path=model_path,
            eval_seed=self.eval_seed,
        )

        mean_reward = np.mean(rewards)
        mean_game_score = np.mean(game_scores)
        self.logger.record("val/mean_reward", mean_reward)
        self.logger.record("val/min_reward", np.min(rewards))
        self.logger.record("val/max_reward", np.max(rewards))
        self.logger.record("val/mean_game_score", mean_game_score)
        self.logger.record("val/min_game_score", np.min(game_scores))
        self.logger.record("val/max_game_score", np.max(game_scores))
        self.logger.record("val/mean_lines_cleared", np.mean(lines))
        self.logger.record("val/min_pieces_placed", np.min(pieces))
        self.logger.record("val/mean_pieces_placed", np.mean(pieces))
        self.logger.record("val/max_pieces_placed", np.max(pieces))
        self.logger.record("val/mean_all_clears", np.mean(all_clears))
        self.logger.record("val/mean_tetrises", np.mean(tetrises))

        summary = (
            f"Validation @ {self.num_timesteps:_}: "
            f"reward min/avg/max={min(rewards):.1f}/{mean_reward:.1f}/{max(rewards):.1f} | "
            f"game score min/avg/max={min(game_scores):.0f}/{mean_game_score:.1f}/{max(game_scores):.0f} | "
            f"pieces min/avg/max={min(pieces)}/{np.mean(pieces):.1f}/{max(pieces)}"
        )
        if self.learned_ratio is not None:
            self.learned, reward_ratio, survival_ratio = _curriculum_gate_passed(
                rewards, pieces, self.max_pieces, self.learned_ratio, self.min_reward,
            )
            self.logger.record("curriculum/reward_ratio", reward_ratio)
            self.logger.record("curriculum/survival_ratio", survival_ratio)
            self.logger.record("curriculum/gate_passed", self.learned)
            best_key = (
                (1, mean_reward) if self.learned
                else (0, min(reward_ratio, survival_ratio), mean_reward)
            )
            summary += f" | reward pass={reward_ratio:.0%}, survival={survival_ratio:.0%}"
        else:
            best_key = (mean_reward,)

        tqdm.write(summary)

        if self.best_key is None or best_key > self.best_key:
            self.best_key = best_key
            self.model.save(self.best_model_path)
            if self.on_best_model:
                self.on_best_model(self.best_model_path, self.num_timesteps, self.learned)
            if self.verbose > 0:
                print(f"\n - New best validation reward: {mean_reward:.2f}! Model saved.")

        if self.learned:
            tqdm.write("Curriculum gate passed.")

        # SB3 callback results mean "continue training". A curriculum stage
        # should continue until its evaluation gate passes.
        return self.learned_ratio is None or not self.learned
