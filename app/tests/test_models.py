import copy
import os
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from src.config import Configuration
from src.models.TurboMino import BoardEncoder, TurboMinoEncoder, TurboMinoModule
from src.models.gym_env import TetrisEnv
from src.models.train_ppo import _stage_progress
from src.models.train_ppo_utils import (
    _load_resume_state,
    _save_resume_state,
    _stage_index_from_checkpoint,
)
from src.tetris import TetrisConfiguration


class ConfigurationTests(unittest.TestCase):
    def test_configuration_does_not_create_output_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Configuration(
                DATA_PATH=os.path.join(directory, "data"),
                MODELS_PATH=os.path.join(directory, "models"),
                LOGS_PATH=os.path.join(directory, "logs"),
                CONFIGS_PATH=os.path.join(directory, "configs"),
                exp_name="test",
            )

            self.assertFalse(os.path.exists(config.DATA_PATH))
            self.assertFalse(os.path.exists(config.MODELS_PATH))
            self.assertFalse(os.path.exists(config.LOGS_PATH))
            self.assertFalse(os.path.exists(config.checkpoint_dir))
            self.assertFalse(os.path.exists(config.log_dir))
            self.assertEqual(
                config.best_model_path,
                os.path.join(config.checkpoint_dir, "best_model.zip"),
            )

    def test_eval_frequency_is_derived_from_rollout_count(self):
        config = Configuration(
            rollout_samples=10_000,
            n_envs=12,
            eval_every_rollouts=4,
        )

        self.assertEqual(config.n_epochs, 10)
        self.assertEqual(config.tetrio_epochs, 100)
        self.assertEqual(config.rollout_steps(), 834)
        self.assertEqual(config.eval_steps(), 3_336)

    def test_random_width_rejects_out_of_bounds_values(self):
        with self.assertRaises(ValueError):
            Configuration(max_board_size_w=10, random_width={11: 1.0})
        with self.assertRaises(ValueError):
            Configuration(random_width={4.5: 1.0})


class TetrisEnvTests(unittest.TestCase):
    def test_board_observations_use_uint8_storage(self):
        env = TetrisEnv(Configuration(), TetrisConfiguration())

        self.assertEqual(env.observation_space["boards"].dtype, np.dtype(np.uint8))

    def test_reset_uses_configured_vanish_zone(self):
        config = Configuration(max_board_size_h=8, max_board_size_w=4)
        tetris_config = TetrisConfiguration(board_w=4, board_h=8, vanish_zone=2)
        env = TetrisEnv(config, tetris_config)

        env.reset(seed=1)

        self.assertEqual(env.game.board.vanish_zone, tetris_config.vanish_zone)


class BoardEncoderTests(unittest.TestCase):
    def test_forward_preserves_batch_and_placement_dimensions(self):
        encoder = BoardEncoder(height=25, width=10, d_model=156)

        output = encoder(torch.zeros(2, 3, 25, 10, dtype=torch.uint8))

        self.assertEqual(output.shape, (2, 3, 156))

    def test_all_invalid_uint8_boards_return_float_tokens(self):
        encoder = BoardEncoder(height=8, width=8, d_model=4, ch=2, k=1)

        output = encoder(
            torch.zeros(1, 2, 8, 8, dtype=torch.uint8),
            torch.tensor([[False, False]]),
        )

        self.assertEqual(output.dtype, torch.float32)

    def test_wide_k_configures_residual_width(self):
        config = Configuration()
        config.wide_k = 3
        tetris_config = TetrisConfiguration(board_w=4)
        observation_space = TetrisEnv(config, tetris_config).observation_space

        encoder = TurboMinoEncoder(observation_space, tetris_config, config)

        self.assertEqual(encoder.board_encoder.stem[0].out_channels, 32)
        self.assertEqual(encoder.board_encoder.res_1.net[1].out_channels, 32 * config.wide_k)

    def test_invalid_boards_do_not_affect_valid_tokens_or_batch_norm(self):
        encoder = BoardEncoder(height=8, width=8, d_model=4, ch=2, k=1).train()
        changed_encoder = copy.deepcopy(encoder)
        boards = torch.randn(1, 3, 8, 8)
        changed_boards = boards.clone()
        changed_boards[:, 1] = 100.0
        valid_mask = torch.tensor([[True, False, True]])

        torch.manual_seed(7)
        tokens = encoder(boards, valid_mask)
        torch.manual_seed(7)
        changed_tokens = changed_encoder(changed_boards, valid_mask)

        torch.testing.assert_close(tokens[valid_mask], changed_tokens[valid_mask])
        for original, changed in zip(encoder.modules(), changed_encoder.modules()):
            if isinstance(original, nn.BatchNorm2d):
                torch.testing.assert_close(original.running_mean, changed.running_mean)
                torch.testing.assert_close(original.running_var, changed.running_var)

    def test_encoder_zeros_invalid_placement_features(self):
        config = Configuration(
            max_placements=4,
            max_board_size_h=8,
            max_board_size_w=4,
            d_model=8,
            n_heads=2,
            head_hidden=8,
            channels=2,
            wide_k=1,
            features_per_placement=2,
        )
        tetris_config = TetrisConfiguration(board_w=4, board_h=8, vanish_zone=0)
        observation_space = TetrisEnv(config, tetris_config).observation_space
        encoder = TurboMinoEncoder(observation_space, tetris_config, config).eval()
        observations = {
            "boards": torch.randn(1, 4, 8, 4),
            "queues": torch.zeros(1, 2, tetris_config.max_pieces_in_view, tetris_config.num_piece_categories),
            "queue_idx": torch.tensor([[0, 1, 0, 1]]),
            "game_state": torch.zeros(1, 4),
            "placement_mask": torch.tensor([[True, False, False, True]]),
        }

        features = encoder(observations).view(1, config.max_placements, config.features_per_placement)

        torch.testing.assert_close(features[:, 1:3], torch.zeros_like(features[:, 1:3]))

        module = TurboMinoModule(config, tetris_config, observation_space).eval()
        logits = module(observations)
        self.assertTrue(torch.all(logits[:, 1:3] == TurboMinoEncoder.MASK_VALUE))


class TurboMinoModuleTests(unittest.TestCase):
    def test_validation_logs_top_k_accuracies(self):
        config = Configuration(
            max_placements=4,
            max_board_size_h=8,
            max_board_size_w=4,
            d_model=8,
            n_heads=2,
            head_hidden=8,
            channels=2,
            wide_k=1,
            features_per_placement=2,
        )
        tetris_config = TetrisConfiguration(board_w=4, board_h=8, vanish_zone=0)
        observation_space = TetrisEnv(config, tetris_config).observation_space
        module = TurboMinoModule(config, tetris_config, observation_space)
        module.forward = lambda _: torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])

        with patch.object(module, "log") as log:
            module._shared_step(({}, torch.tensor([0, 3])), "val")

        metrics = {call.args[0]: call.args[1].item() for call in log.call_args_list}
        self.assertEqual(set(metrics), {
            "val/loss", "val/acc_top1", "val/acc_top3", "val/acc_top5", "val/acc_top10",
        })
        self.assertEqual(metrics["val/acc_top1"], 0.5)
        self.assertEqual(metrics["val/acc_top3"], 0.5)
        self.assertEqual(metrics["val/acc_top5"], 1.0)
        self.assertEqual(metrics["val/acc_top10"], 1.0)
        self.assertTrue(all(call.kwargs["prog_bar"] for call in log.call_args_list))


class ResumeStateTests(unittest.TestCase):
    def test_stage_progress_uses_saved_start_when_resuming_same_width(self):
        config = Configuration(
            curriculum={4: 1000, 8: 2000},
            resume_model_path="models/w8/model.zip",
        )
        model = SimpleNamespace(num_timesteps=1_500)
        resume_state = {
            "curriculum": {
                "board_width": 8,
                "stage_start_global_steps": 1_000,
                "stage_complete": False,
            }
        }

        start, completed = _stage_progress(
            model, config, stage_idx=1, board_w=8, stage_time=2_000,
            stage_start_global_steps=1_000, resume_state=resume_state,
            resume_stage_index=1,
        )

        self.assertEqual((start, completed), (1_000, 500))

        resume_state["curriculum"]["stage_complete"] = True
        _, completed = _stage_progress(
            model, config, stage_idx=1, board_w=8, stage_time=2_000,
            stage_start_global_steps=1_000, resume_state=resume_state,
            resume_stage_index=1,
        )
        self.assertEqual(completed, 2_000)

    def test_resume_state_preserves_curriculum_progress_and_config(self):
        config = Configuration(curriculum={4: 1000, 8: 2000})
        tetris_config = TetrisConfiguration(board_w=8)

        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.zip")
            _save_resume_state(
                model_path,
                config,
                tetris_config,
                stage_index=1,
                stage_start_global_steps=1000,
                stage_target_steps=2000,
                stage_completed_steps=500,
                stage_complete=np.bool_(False),
            )

            state = _load_resume_state(model_path)

        self.assertEqual(state["config"]["curriculum"], {4: 1000, 8: 2000})
        self.assertEqual(state["curriculum"]["stage_completed_steps"], 500)
        self.assertEqual(state["curriculum"]["global_steps"], 1500)
        self.assertEqual(_stage_index_from_checkpoint("models/w8/model.zip", [(4, 1000), (8, 2000)]), 1)
        self.assertEqual(_stage_index_from_checkpoint("models/w-1/model.zip", [(4, 1000), (-1, 2000)]), 1)


if __name__ == "__main__":
    unittest.main()
