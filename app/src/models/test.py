from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
import torch
import pytorch_lightning as pl

from maikol_utils.print_utils import print_separator

from src.data import load_tetrio_loader
from src.config import Configuration
from src.tetris import TetrisConfiguration
from stable_baselines3.common.utils import obs_as_tensor

from .gym_env import make_eval_env
from .utils import load_model


_eval_env = None
_eval_model = None
_eval_config = None
_eval_seed = None
_eval_max_pieces = None


def _init_game_evaluator(
    CONFIG: Configuration,
    T_CONFIG: TetrisConfiguration,
    model_path: str | None,
    eval_seed: int | None,
    max_pieces: int,
):
    global _eval_config, _eval_env, _eval_model, _eval_seed, _eval_max_pieces
    _eval_config = CONFIG
    _eval_seed = eval_seed
    _eval_max_pieces = max_pieces
    _eval_env = make_eval_env(CONFIG, T_CONFIG)
    _eval_model = load_model(
        CONFIG, T_CONFIG, env=_eval_env, model_path=model_path
    )


def _evaluate_episode(episode: int):
    episode_seed = _eval_seed + episode if _eval_seed is not None else None
    obs, _ = _eval_env.reset(seed=episode_seed)
    done = False
    pieces_placed = 0
    total_reward = 0.0

    while not done and pieces_placed < _eval_max_pieces:
        action_masks = _eval_env.unwrapped.valid_action_mask()
        action, _ = _eval_model.predict(
            obs, action_masks=action_masks, deterministic=True
        )
        obs, reward, terminated, truncated, info = _eval_env.step(action)
        total_reward += reward
        pieces_placed += 1
        done = terminated or truncated

    game = _eval_env.unwrapped.game
    return (
        total_reward,
        game.get_score(),
        game.get_lines_cleared(),
        pieces_placed,
        game.get_total_all_clears(),
        game.get_total_tetrises(),
    )


def test_model(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    print_separator("Test Classifier on Game", sep_type="LONG")
    pl.seed_everything(CONFIG.seed, workers=True)

    # print_separator("Test Classifier on Tetrio", sep_type="LONG")
    # results = test_tetrio(CONFIG, T_CONFIG, model=model)
    # print(results)

    print_separator("Test Classifier on Game")
    rewards, scores, lines, pieces, all_clears, tetrises = test_on_game(
        CONFIG=CONFIG,
        T_CONFIG=T_CONFIG,
        eval_seed=CONFIG.eval_seed,
    )
    print(f" - Average Reward: {sum(rewards) / len(rewards):.2f}")
    print(f" - Average Score: {sum(scores) / len(scores):.2f}")
    print(f" - Average Lines Cleared: {sum(lines) / len(lines):.2f}")
    print(f" - Pieces Placed: min={min(pieces)}, avg={sum(pieces) / len(pieces):.2f}, max={max(pieces)}")
    print(f" - Average All Clears: {sum(all_clears) / len(all_clears):.2f}")
    print(f" - Average Tetrises: {sum(tetrises) / len(tetrises):.2f}")
    

def test_tetrio(CONFIG: Configuration, T_CONFIG: TetrisConfiguration, model=None):
    pl.seed_everything(CONFIG.seed, workers=True)

    test_loader = load_tetrio_loader(CONFIG, T_CONFIG, CONFIG.tetrio_test)

    if isinstance(model, pl.LightningModule):
        trainer = pl.Trainer(deterministic="warn")
        results = trainer.test(model=model, dataloaders=test_loader)
        return results

    # --- sb3 MaskablePPO path ---
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    for obs, target in tqdm(test_loader, desc="Testing"):
        action_masks_np = obs['placement_mask'].numpy()
        actions, _ = model.predict(obs, action_masks=action_masks_np, deterministic=True)
        top1 = (actions == target.numpy()).sum()
        correct_top1 += top1
        total += len(target)

        try:
            tensor_obs = obs_as_tensor(obs, model.policy.device)
            features = model.policy.extract_features(tensor_obs)
            latent_pi, _ = model.policy.mlp_extractor(features)
            distribution = model.policy._get_action_dist_from_latent(latent_pi)
            mask_tensor = torch.tensor(action_masks_np, device=model.policy.device)
            distribution.apply_masking(mask_tensor)
            logits = distribution.distribution.logits
            top3 = logits.topk(3, dim=1).indices == target.to(model.policy.device).unsqueeze(1)
            correct_top3 += top3.any(dim=1).sum().item()
        except Exception:
            correct_top3 = correct_top1

    acc_top1 = correct_top1 / total
    acc_top3 = correct_top3 / total
    return [{'test/acc_top1': acc_top1, 'test/acc_top3': acc_top3}]


def test_on_game(
    CONFIG: Configuration,
    T_CONFIG: TetrisConfiguration,
    model_path: str | None = None,
    eval_seed: int | None = None,
    eval_episodes: int | None = None,
    max_pieces: int | None = None,
):
    eval_episodes = CONFIG.eval_episodes if eval_episodes is None else eval_episodes
    max_pieces = CONFIG.max_eval_pieces if max_pieces is None else max_pieces
    with ProcessPoolExecutor(
        max_workers=CONFIG.num_workers,
        mp_context=get_context("spawn"),
        initializer=_init_game_evaluator,
        initargs=(CONFIG, T_CONFIG, model_path, eval_seed, max_pieces),
    ) as executor:
        results = list(tqdm(
            executor.map(_evaluate_episode, range(eval_episodes)),
            total=eval_episodes,
            desc="Evaluating",
        ))

    return tuple(map(list, zip(*results)))
