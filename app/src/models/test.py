from tqdm import tqdm
import torch
import pytorch_lightning as pl

from maikol_utils.print_utils import print_separator

from src.data import load_tetrio_loader
from src.config import Configuration
from src.tetris import TetrisConfiguration
from stable_baselines3.common.utils import obs_as_tensor

from .gym_env import TetrisEnv, make_eval_env
from .utils import load_model


def test_model(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    print_separator("Test Classifier on Game", sep_type="LONG")
    pl.seed_everything(CONFIG.seed, workers=True)

    model = load_model(CONFIG, T_CONFIG)
    

    # print_separator("Test Classifier on Tetrio", sep_type="LONG")
    # results = test_tetrio(CONFIG, T_CONFIG, model=model)
    # print(results)

    print_separator("Test Classifier on Game")
    eval_env = make_eval_env(CONFIG, T_CONFIG)
    scores, lines, pieces, all_clears, tetrises = test_on_game(
        n_eval_episodes=CONFIG.eval_episodes,
        max_pieces=CONFIG.max_eval_pieces,
        eval_env=eval_env,
        model=model,
    )
    print(f" - Average Score: {sum(scores) / len(scores):.2f}")
    print(f" - Average Lines Cleared: {sum(lines) / len(lines):.2f}")
    print(f" - Average Pieces Placed: {sum(pieces) / len(pieces):.2f}")
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


def test_on_game(n_eval_episodes: int, max_pieces: int, eval_env: TetrisEnv, model):

    scores = []
    lines = []
    pieces = []
    all_clears = []
    tetrises = []

    for _ in tqdm(range(n_eval_episodes), desc="Evaluating"):
        obs, _ = eval_env.reset()
        done = False
        pieces_placed = 0

        while not done and pieces_placed < max_pieces:
            action_masks = eval_env.unwrapped.valid_action_mask()
            action, _ = model.predict(
                obs, action_masks=action_masks, deterministic=True
            )
            obs, reward, terminated, truncated, info = eval_env.step(action)
            pieces_placed += 1
            done = terminated or truncated

        game = eval_env.unwrapped.game
        scores.append(game.get_score())
        lines.append(game.get_lines_cleared())
        pieces.append(pieces_placed)
        all_clears.append(game.get_total_all_clears())
        tetrises.append(game.get_total_tetrises())

    return scores, lines, pieces, all_clears, tetrises