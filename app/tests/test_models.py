import unittest

import torch

from src.config import Configuration
from src.models.TurboMino import BoardEncoder, TurboMinoEncoder
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


if __name__ == "__main__":
    unittest.main()
