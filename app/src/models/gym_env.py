import gymnasium as gym
import random
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.tetris import Tetris, MoveSearcher, TetrisConfiguration, HeuristicEvaluator
from src.config import Configuration

class TetrisEnv(gym.Env):
    def __init__(self, CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
        super().__init__()

        self.CONFIG = CONFIG
        self.T_CONFIG = T_CONFIG

        assert self.CONFIG.max_board_size_h >= self.T_CONFIG.board_h, "CONFIG.max_board_size_h must be >= T_CONFIG.board_h + T_CONFIG.vanish_zone"
        assert self.CONFIG.max_board_size_w >= self.T_CONFIG.board_w, "CONFIG.max_board_size_w must be >= T_CONFIG.board_w"

        self.evaluator = HeuristicEvaluator()

        # 1. Action Space: Select an index from 0 to MAX_PLACEMENTS - 1
        self.action_space = spaces.Discrete(CONFIG.max_placements)
        
        self.observation_space = spaces.Dict({
            "boards": spaces.Box(
                low=0.0, high=1.0, 
                shape=(CONFIG.max_placements, CONFIG.max_board_size_h + T_CONFIG.vanish_zone, CONFIG.max_board_size_w), 
                dtype=np.float32
            ),
            "queues": spaces.Box(
                low=0.0, high=1.0, 
                shape=(2, T_CONFIG.max_pieces_in_view, T_CONFIG.num_piece_categories), # 7 pieces (current, hold, 5 next), 9 categories
                dtype=np.float32
            ),
            "queue_idx": spaces.Box(
                low=0, high=1, # 0 = Active, 1 = Hold
                shape=(CONFIG.max_placements,), 
                dtype=np.int64
            ),
            "game_state": spaces.Box(
                # Game state features: [combo, b2b, immediate garbage, incoming garbage]
                low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([np.inf, np.inf, CONFIG.garbage_cap, np.inf], dtype=np.float32),
                shape=(4,),
                dtype=np.float32
            ),
            "placement_mask": spaces.Box(
                low=0, high=1,
                shape=(CONFIG.max_placements,),
                dtype=bool
            )
        })
        
        self.game = None
        self.searcher = None
        self.all_placements = []


    def step(self, action):
        # =========== 1. Execute the sequence chosen by the neural network =========== 
        chosen_placement, _ = self.all_placements[action]

        reward = self.game.get_score()
        heuristic_before = (
            self.evaluator.evaluate(self.game.board).compute_total()
            if self.CONFIG.use_heuristic_rewards else 0.0
        )
        for act in chosen_placement['sequence']:
            self.game.move_active_piece(act)

        # ===========  2. Check Game Over =========== 
        terminated = self.game.is_game_over()
        truncated = False


        # ===========  3. Calculate Reward =========== 
        if terminated:
            # Penalize death — large negative reward so the model learns to survive
            reward = self.CONFIG.death_penalty
        else: 
            reward = self.game.get_score() - reward + self.CONFIG.alive_bonus # Reward is the score difference after the move
            if self.CONFIG.use_heuristic_rewards:
                # Reward board improvement, not its cumulative penalty.
                reward += self.evaluator.evaluate(self.game.board).compute_total() - heuristic_before

        reward = np.sign(reward) * np.sqrt(np.abs(reward)) / 10.0

        
        # ===========  4. Get next states =========== 
        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, {}

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize your engine
        self.game = Tetris(
            width=self.T_CONFIG.board_w,
            height=self.T_CONFIG.board_h,
            garbage_prob=self.CONFIG.garbage_prob,
            garbage_delay=self.CONFIG.garbage_delay,
            garbage_lines_probs=self.CONFIG.garbage_lines_probs,
            garbage_cap=self.CONFIG.garbage_cap,
        )
        self.searcher = MoveSearcher(self.game, self.CONFIG, self.T_CONFIG)
        
        obs = self._get_obs()
        return obs, {}

    
    def _get_obs(self):
        """Generates the observation dictionary containing board states for all placements and their corresponding queue contexts."""
        all_placements, features_dict = self.searcher.get_all_features()
        self.all_placements = all_placements # Store for step() to reference

        return features_dict
    

    def valid_action_mask(self):
        """
        CRITICAL METHOD: sb3-contrib looks for this exact function name.
        It returns a boolean array shape (MAX_PLACEMENTS,).
        True = Valid Move | False = Padded/Illegal Move
        """
        return self.searcher.valid_action_mask()

    def action_masks(self):
        return self.valid_action_mask()
    

    def get_game(self) -> Tetris:
        """Returns the underlying Tetris game instance for rendering or analysis."""
        return self.game
    


# ==========================================
# 2. Env factories
# ==========================================
def make_train_env(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    if CONFIG.n_envs > 1:
        def make_env():
            return Monitor(TetrisEnv(CONFIG, T_CONFIG))

        return SubprocVecEnv([make_env for _ in range(CONFIG.n_envs)])
    env = TetrisEnv(CONFIG, T_CONFIG)
    return Monitor(env)


def make_eval_env(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    return TetrisEnv(CONFIG, T_CONFIG)
