import unittest

import pygame

from src.tetris import PieceEnum, TetrisConfiguration, draw_garbage_bar


class VisualizationTests(unittest.TestCase):
    def test_garbage_bar_groups_rounds_and_highlights_immediate_lines(self):
        config = TetrisConfiguration(board_w=4, board_h=4, cell_size=4)
        surface = pygame.Surface((config.screen_width, config.screen_height))
        incoming_garbage = [(2, 3, 0), (1, 3, 1), (2, 1, 2)]

        draw_garbage_bar(config, surface, incoming_garbage, board_height=5, visible_height=4)

        x = config.garbage_bar_offset_x * config.cell_size + 2
        self.assertEqual(surface.get_at((x, 5 * 4 - 1 * 4 + 2))[:3], config.game_over_color)
        self.assertEqual(surface.get_at((x, 5 * 4 - 2 * 4 + 2))[:3], config.game_over_color)
        self.assertEqual(surface.get_at((x, 5 * 4 - 3 * 4 + 2))[:3], config.colors[PieceEnum.G])
        self.assertEqual(surface.get_at((x, 5 * 4 - 4 * 4 + 2))[:3], config.colors[PieceEnum.G])
        self.assertEqual(surface.get_at((x, 5 * 4 - 5 * 4 + 2))[:3], (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
