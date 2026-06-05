import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.tetris import Tetris, MoveSearcher, TetrisConfiguration
from src.config import Configuration

class TetrisEnv(gym.Env):
    def __init__(self, CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
        super().__init__()

        self.CONFIG = CONFIG
        self.T_CONFIG = T_CONFIG
        
        # 1. Action Space: Select an index from 0 to MAX_PLACEMENTS - 1
        self.action_space = spaces.Discrete(CONFIG.max_placements)
        
        # 2. Observation Space: Example using a flattened 10x20 binary grid
        # Shape is (50, 200). Each row is a potential future board state.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(CONFIG.max_placements, CONFIG.board_w * CONFIG.board_h), 
            dtype=np.float32
        )
        
        self.game = None
        self.searcher = None
        self.current_placements = []

    def step(self, action):
        # 1. Execute the sequence chosen by the neural network
        chosen_placement = self.current_placements[action]
        
        reward = self.game.get_score()
        for act in chosen_placement['sequence']:
            self.game.move_active_piece(act)
            
        # 2. Calculate Reward
        reward = self.game.get_score() - reward # Reward is the score difference after the move
        
        # 3. Check Game Over
        terminated = self.game.is_game_over() # Implement this in your engine
        truncated = False
        
        # 4. Get next states
        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize your engine
        self.game = Tetris(width=self.CONFIG.board_w, height=self.CONFIG.board_h)
        self.searcher = MoveSearcher(self.game.board)
        
        obs = self._get_obs()
        return obs, {}
    
    def _get_obs(self):
        # 1. Ask the MoveSearcher for all valid placements this turn
        self.current_placements = self.searcher.get_all_placements(self.game.active_piece.type)
        
        # 2. Prepare the padded observation matrix
        obs = np.zeros((
            self.CONFIG.max_placements, 
            self.CONFIG.board_w * (self.CONFIG.board_h + self.T_CONFIG.vanish_zone)
        ), dtype=np.float32)
        
        for i, placement in enumerate(self.current_placements):
            if i >= self.CONFIG.max_placements:
                break # Hard safety bound
            
            # Convert the uint32 bitboard back to a flat float32 array of 0s and 1s
            # Alternatively, extract heuristics here (heights, holes, bumpiness)
            obs[i] = self._extract_features(placement['bitmap'])
            
        return obs

    def valid_action_mask(self):
        """
        CRITICAL METHOD: sb3-contrib looks for this exact function name.
        It returns a boolean array shape (MAX_PLACEMENTS,).
        True = Valid Move | False = Padded/Illegal Move
        """
        num_valid = len(self.current_placements)
        mask = np.zeros(self.CONFIG.max_placements, dtype=bool)
        mask[:num_valid] = True
        return mask



    def _extract_features(self, bitmap_array: np.ndarray) -> np.ndarray:
        # 1. Create a 1D array of bit-shifts for each column (0 to width-1)
        # If column 0 is the highest bit (MSB), use: self.CONFIG.board_w - 1 - np.arange(self.CONFIG.board_w)
        # If column 0 is the lowest bit (LSB), use: np.arange(self.CONFIG.board_w)
        shifts = np.arange(self.CONFIG.board_w, dtype=np.uint32)
        
        # 2. Use broadcasting to shift every row's bits and check the lowest bit (& 1)
        # bitmap_array[:, None] reshapes to (height, 1)
        # shifts[None, :] reshapes to (1, width)
        # Resulting matrix shape: (height, width)
        unpacked_2d = (bitmap_array[:, None] >> shifts) & 1
        
        # 3. Flatten and cast directly to float32 for your Neural Network
        return unpacked_2d.ravel().astype(np.float32)

    def _extract_features_2d(self, bitmap_array: np.ndarray) -> np.ndarray:
        # 1. Create bit-shifts for each column
        shifts = np.arange(self.CONFIG.board_w, dtype=np.uint32)
        
        # 2. Broadcast shift and mask to build a (board_h, board_w) matrix
        unpacked_2d = (bitmap_array[:, None] >> shifts) & 1
        
        # 3. Cast to float32 without flattening
        return unpacked_2d.astype(np.float32)
    
    def _calculate_reward(self, lines: int) -> float:
        # Reward shaping (e.g., lines cleared, surviving, penalizing holes)
        return float(lines)