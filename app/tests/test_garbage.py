import unittest
from collections import deque

import numpy as np

from src.tetris import ActionEnum, Board, PieceEnum, ScoringSystem, SpinType, Tetris


class GarbageTests(unittest.TestCase):
    def test_board_adds_garbage_with_one_hole(self):
        board = Board(width=4, height=4, vanish_zone=1, color_map=True)
        board.b_rows[0] = 1
        board.c_rows[0, 0] = PieceEnum.I.value

        self.assertFalse(board.add_garbage(2, hole=1))
        np.testing.assert_array_equal(board.b_rows[:3], [13, 13, 1])
        np.testing.assert_array_equal(board.c_rows[0], [8, 0, 8, 8])
        self.assertEqual(board.c_rows[2, 0], PieceEnum.I.value)

        board.b_rows[-1] = 1
        self.assertTrue(board.add_garbage(1, hole=0))

    def test_weighted_garbage_waits_for_configured_delay(self):
        game = Tetris(
            width=4,
            height=6,
            vanish_zone=0,
            garbage_prob=1,
            garbage_delay=3,
            garbage_probs=[0, 1, 0, 0, 0, 0, 0, 0],
        )
        game.randomly_add_garbage()

        lines, turns, hole = game.incoming_garbage[0]
        self.assertEqual((lines, turns), (2, 3))
        self.assertEqual(game.get_incoming_garbage(), 2)
        self.assertEqual(game.get_immediate_garbage(), 0)

        game.garbage_prob = 0
        game.manage_garbage(0, 0)
        game.manage_garbage(0, 0)
        self.assertEqual(game.get_immediate_garbage(), 2)
        game.manage_garbage(0, 0)

        self.assertEqual(game.get_incoming_garbage(), 0)
        self.assertEqual(int(game.board.b_rows[0]), 15 & ~(1 << hole))

    def test_attack_cancels_fifo_and_cap_keeps_remainder(self):
        game = Tetris(width=4, height=8, vanish_zone=0, garbage_prob=0, garbage_cap=3)
        game.incoming_garbage = deque([(5, 1, 0)])

        game.manage_garbage(cleared_lines=2, attack=1)
        self.assertEqual(game.get_incoming_garbage(), 4)
        self.assertEqual(game.get_immediate_garbage(), 3)
        self.assertFalse(np.any(game.board.b_rows))

        game.manage_garbage(cleared_lines=0, attack=0)
        self.assertEqual(game.get_incoming_garbage(), 1)
        np.testing.assert_array_equal(game.board.b_rows[:3], [14, 14, 14])

    def test_scoring_returns_tetrio_attack(self):
        scoring = ScoringSystem()

        self.assertEqual(scoring.evaluate_drop(4, SpinType.NONE, False, 0, True), 4)
        self.assertEqual(scoring.evaluate_drop(4, SpinType.NONE, False, 0, True), 6)
        self.assertEqual(scoring.get_b2b_streak(), 2)

        spins = ScoringSystem()
        for expected in (2, 3, 4, 5):
            self.assertEqual(spins.evaluate_drop(1, SpinType.REGULAR, False, 0, True), expected)

        perfect_clear = ScoringSystem()
        self.assertEqual(perfect_clear.evaluate_drop(1, SpinType.NONE, True, 0, True), 5)
        self.assertEqual(perfect_clear.get_b2b_streak(), 1)

    def test_b2b_survives_empty_drops_and_releases_surge(self):
        scoring = ScoringSystem()
        for _ in range(4):
            scoring.evaluate_drop(4, SpinType.NONE, False, 0, True)

        scoring.evaluate_drop(0, SpinType.NONE, False, 0, True)
        self.assertEqual(scoring.get_b2b_streak(), 4)
        self.assertEqual(scoring.evaluate_drop(1, SpinType.NONE, False, 0, True), 4)
        self.assertEqual(scoring.get_b2b_streak(), 0)

    def test_garbage_is_generated_only_after_drop(self):
        game = Tetris(garbage_prob=1, garbage_probs=[1, 0, 0, 0, 0, 0, 0, 0])

        game.move_active_piece(ActionEnum.LEFT)
        self.assertEqual(game.get_incoming_garbage(), 0)
        game.move_active_piece(ActionEnum.DROP)
        self.assertEqual(game.get_incoming_garbage(), 1)


if __name__ == "__main__":
    unittest.main()
