import unittest

import torch

from src.models.TurboMino import BoardEncoder


class BoardEncoderTests(unittest.TestCase):
    def test_forward_preserves_batch_and_placement_dimensions(self):
        encoder = BoardEncoder(height=25, width=10, d_model=156)

        output = encoder(torch.zeros(2, 3, 25, 10))

        self.assertEqual(output.shape, (2, 3, 156))


if __name__ == "__main__":
    unittest.main()
