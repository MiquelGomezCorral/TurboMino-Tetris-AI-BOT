import unittest
from types import SimpleNamespace

from src.config import Configuration
from src.models.gym_env import TetrisEnv
from src.tetris import TetrisConfiguration
from src.tetris.scoring import PlacementEvent, ScoringSystem, SpinType


class FakeGame:
    def __init__(self, terminated, event):
        self.board = object()
        self.terminated = terminated
        self.event = event

    def move_active_piece(self, action):
        pass

    def is_game_over(self):
        return self.terminated

    def get_last_placement_event(self):
        return self.event


class FakeEvaluator:
    def __init__(self, values):
        self.values = iter(values)

    def evaluate(self, board):
        return SimpleNamespace(compute_total=lambda: next(self.values))


class RewardTests(unittest.TestCase):
    def make_env(self, config, game):
        env = TetrisEnv(config, TetrisConfiguration())
        env.game = game
        env.all_placements = [({"sequence": ()}, None)]
        env._get_obs = lambda: {}
        return env

    def test_scoring_records_placement_event(self):
        scoring = ScoringSystem()

        scoring.evaluate_drop(2, SpinType.REGULAR, False, 10, True)

        self.assertEqual(scoring.last_placement_event, PlacementEvent(2, False, True))
        self.assertEqual(scoring.score, 1220)

    def test_game_reward_uses_placement_event_not_player_score(self):
        config = Configuration()
        config.use_survival_rewards = False
        config.use_game_rewards = True
        game = FakeGame(False, PlacementEvent(4, True, True))

        _, reward, terminated, _, _ = self.make_env(config, game).step(0)

        self.assertFalse(terminated)
        self.assertAlmostEqual(reward, 1.0)

    def test_terminal_reward_excludes_game_and_heuristic_rewards(self):
        config = Configuration()
        config.use_game_rewards = True
        config.use_heuristic_rewards = True
        game = FakeGame(True, PlacementEvent(4, True, True))
        env = self.make_env(config, game)
        env.evaluator = FakeEvaluator((0.0,))

        _, reward, terminated, _, _ = env.step(0)

        self.assertTrue(terminated)
        self.assertEqual(reward, -5.0)

    def test_heuristic_reward_is_scaled_and_capped(self):
        config = Configuration()
        config.use_survival_rewards = False
        config.use_heuristic_rewards = True
        game = FakeGame(False, PlacementEvent(0, False, False))
        env = self.make_env(config, game)
        env.evaluator = FakeEvaluator((0.0, 100.0))

        _, reward, terminated, _, _ = env.step(0)

        self.assertFalse(terminated)
        self.assertEqual(reward, config.heuristic_reward_cap)


if __name__ == "__main__":
    unittest.main()
