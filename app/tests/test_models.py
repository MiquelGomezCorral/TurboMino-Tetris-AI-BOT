import copy
import unittest

import torch
from torch import nn

from src.config import Configuration
from src.models.TurboMino import BoardEncoder, TurboMinoEncoder, TurboMinoModule
from src.models.gym_env import TetrisEnv
from src.tetris import TetrisConfiguration


class BoardEncoderTests(unittest.TestCase):
    def test_forward_preserves_batch_and_placement_dimensions(self):
        encoder = BoardEncoder(height=25, width=10, d_model=156)

        output = encoder(torch.zeros(2, 3, 25, 10))

        self.assertEqual(output.shape, (2, 3, 156))

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


if __name__ == "__main__":
    unittest.main()
