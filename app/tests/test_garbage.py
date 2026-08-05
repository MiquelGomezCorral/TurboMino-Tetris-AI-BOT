import unittest
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from src.config import Configuration
from src.data.tetrio import PrecomputedTetrioDataset, flip_rows
from src.tetris import ActionEnum, Board, MoveSearcher, PieceEnum, ScoringSystem, SpinType, Tetris, TetrisConfiguration


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
            garbage_lines_probs=[0, 1, 0, 0, 0, 0, 0, 0],
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

    def test_combo_uses_tetrio_values(self):
        scoring = ScoringSystem()

        self.assertEqual(scoring.get_combo(), 0)
        self.assertEqual(scoring.evaluate_drop(1, SpinType.NONE, False, 0, True), 0)
        self.assertEqual(scoring.get_combo(), 1)
        self.assertEqual(scoring.score, 100)
        self.assertEqual(scoring.evaluate_drop(1, SpinType.NONE, False, 0, True), 0)
        self.assertEqual(scoring.get_combo(), 2)
        self.assertEqual(scoring.score, 250)
        self.assertEqual(scoring.evaluate_drop(1, SpinType.NONE, False, 0, True), 1)
        self.assertEqual(scoring.get_combo(), 3)
        scoring.evaluate_drop(0, SpinType.NONE, False, 0, True)
        self.assertEqual(scoring.get_combo(), 0)

    def test_b2b_survives_empty_drops_and_releases_surge(self):
        scoring = ScoringSystem()
        for _ in range(4):
            scoring.evaluate_drop(4, SpinType.NONE, False, 0, True)

        scoring.evaluate_drop(0, SpinType.NONE, False, 0, True)
        self.assertEqual(scoring.get_b2b_streak(), 4)
        self.assertEqual(scoring.evaluate_drop(1, SpinType.NONE, False, 0, True), 4)
        self.assertEqual(scoring.get_b2b_streak(), 0)

    def test_garbage_is_generated_only_after_drop(self):
        game = Tetris(garbage_prob=1, garbage_lines_probs=[1, 0, 0, 0, 0, 0, 0, 0])

        game.move_active_piece(ActionEnum.LEFT)
        self.assertEqual(game.get_incoming_garbage(), 0)
        game.move_active_piece(ActionEnum.DROP)
        self.assertEqual(game.get_incoming_garbage(), 1)

    def test_move_searcher_accepts_game_state_override(self):
        game = Tetris(garbage_prob=0)
        searcher = MoveSearcher(game, Configuration(), TetrisConfiguration())

        _, features = searcher.get_all_features([2, 3, 4, 5])

        np.testing.assert_array_equal(features["game_state"], np.array([2, 3, 4, 5], dtype=np.float32))


class TetrioDataTests(unittest.TestCase):
    def test_flip_rows_mirrors_fields_and_piece_types(self):
        row = {
            "playfield": "INSGJLTN",
            "playfield_next": "GNNNNTGG",
            "real_current": "S",
            "placed": "J",
            "real_hold": "Z",
            "hold": "L",
            "next": "ISZJLOT",
        }

        flipped = flip_rows(row, width=4)

        self.assertEqual(flipped["playfield"], "GSNINTLJ")
        self.assertEqual(flipped["playfield_next"], "NNNGGGTN")
        self.assertEqual(flipped["real_current"], "Z")
        self.assertEqual(flipped["placed"], "L")
        self.assertEqual(flipped["real_hold"], "S")
        self.assertEqual(flipped["hold"], "J")
        self.assertEqual(flipped["next"], "IZSLJOT")
        self.assertEqual(row["placed"], "J")

    def test_precomputed_dataset_applies_augmentation(self):
        with TemporaryDirectory() as data_dir:
            bucket = Path(data_dir) / "0000"
            bucket.mkdir()
            queues = np.zeros((2, 1, 8), dtype=np.float32)
            queues[0, 0, PieceEnum.S.value] = 1
            queues[1, 0, PieceEnum.J.value] = 1
            np.savez(
                bucket / "000000.npz",
                boards=np.array([
                    [[0, 1, 0, 0]],
                    [[1, 1, 0, 0]],
                ], dtype=np.float32),
                queues=queues,
                queue_idx=np.array([0, 1], dtype=np.int64),
                game_state=np.array([1, 2, 3, 4], dtype=np.float32),
                target=np.int64(1),
            )
            dataset = PrecomputedTetrioDataset(
                data_dir,
                SimpleNamespace(max_placements=3, aug_prob=1),
            )

            obs, target = dataset[0]

        np.testing.assert_array_equal(obs["boards"].numpy(), [
            [[0, 0, 1, 0]],
            [[0, 0, 1, 1]],
            [[0, 0, 0, 0]],
        ])
        self.assertEqual(obs["queues"][0, 0, PieceEnum.Z.value], 1)
        self.assertEqual(obs["queues"][1, 0, PieceEnum.L.value], 1)
        np.testing.assert_array_equal(obs["placement_mask"].numpy(), [True, True, False])
        self.assertEqual(target.item(), 1)
if __name__ == "__main__":
    unittest.main()
