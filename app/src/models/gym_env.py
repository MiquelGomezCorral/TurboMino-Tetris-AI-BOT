import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

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
        self.game = Tetris(width=self.T_CONFIG.board_w, height=self.T_CONFIG.board_h)
        self.searcher = MoveSearcher(self.game, self.CONFIG, self.T_CONFIG)

        # 1. Action Space: Select an index from 0 to MAX_PLACEMENTS - 1
        self.action_space = spaces.Discrete(CONFIG.max_placements)
        
        # 2. Observation Space: Example using a flattened 10x20 binary grid
        # Shape is (50, 200). Each row is a potential future board state.
        self.board_features = T_CONFIG.board_w *  (self.T_CONFIG.board_h + self.T_CONFIG.vanish_zone)
        self.context_features = T_CONFIG.max_pieces_in_view * T_CONFIG.num_piece_categories # 56
        
        self.total_features = self.board_features + self.context_features

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
                low=np.array([0.0, 0.0]),
                high=np.array([1.0, np.inf]),
                shape=(2,),
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
        for act in chosen_placement['sequence']:
            self.game.move_active_piece(act)

        # ===========  2. Check Game Over =========== 
        terminated = self.game.is_game_over()
        truncated = False


        # ===========  3. Calculate Reward =========== 
        if terminated:
            # Penalize death — large negative reward so the model learns to survive
            reward = self.T_CONFIG.death_penalty
        else: 
            reward = self.game.get_score() - reward + self.T_CONFIG.alive_bonus # Reward is the score difference after the move
            if self.CONFIG.use_heuristic_rewards:
                reward += self.evaluator.evaluate(self.game.board).compute_total()

        reward = np.sign(reward) * np.sqrt(np.abs(reward)) / 10.0

        
        # ===========  4. Get next states =========== 
        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, {}

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize your engine
        self.game = Tetris(width=self.T_CONFIG.board_w, height=self.T_CONFIG.board_h)
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
    

    def get_game(self) -> Tetris:
        """Returns the underlying Tetris game instance for rendering or analysis."""
        return self.game
    


# ==========================================
# 2. Env factories
# ==========================================
def mask_fn(env: gym.Env):
    return env.unwrapped.valid_action_mask()


def make_train_env(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    if CONFIG.n_envs > 1:
        return DummyVecEnv([
            lambda: ActionMasker(Monitor(TetrisEnv(CONFIG, T_CONFIG)), mask_fn)
            for _ in range(CONFIG.n_envs)
        ])
    env = TetrisEnv(CONFIG, T_CONFIG)
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)
    return env


def make_eval_env(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    env = TetrisEnv(CONFIG, T_CONFIG)
    env = ActionMasker(env, mask_fn)
    return env
