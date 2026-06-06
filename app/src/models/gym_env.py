import gymnasium as gym
from gymnasium import spaces
import numpy as np
from einops import rearrange

from src.tetris import Tetris, MoveSearcher, TetrisConfiguration, PieceEnum
from src.config import Configuration

class TetrisEnv(gym.Env):
    def __init__(self, CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
        super().__init__()

        self.CONFIG = CONFIG
        self.T_CONFIG = T_CONFIG

        assert self.CONFIG.max_board_size_h >= self.T_CONFIG.board_h, "CONFIG.max_board_size_h must be >= T_CONFIG.board_h + T_CONFIG.vanish_zone"
        assert self.CONFIG.max_board_size_w >= self.T_CONFIG.board_w, "CONFIG.max_board_size_w must be >= T_CONFIG.board_w"

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
            "queue": spaces.Box(
                low=0.0, high=1.0, 
                shape=(T_CONFIG.max_pieces_in_view, T_CONFIG.num_piece_categories), # 7 pieces (current, hold, 5 next), 9 categories
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
        self.game = Tetris(width=self.T_CONFIG.board_w, height=self.T_CONFIG.board_h)
        self.searcher = MoveSearcher(self.game)
        
        obs = self._get_obs()
        return obs, {}

    
    def _get_one_hot(self, piece_val):
        arr = np.zeros(self.T_CONFIG.num_piece_categories, dtype=np.float32)
        arr[piece_val] = 1.0
        return arr
    

    def _get_obs(self):
        # 1. Ask the MoveSearcher for all valid placements this turn
        self.current_placements = self.searcher.get_all_placements()
        print(f"Found {len(self.current_placements)} valid placements for piece {self.game.active_piece.type.name}")
    
        context_list = []

        # A. Current Piece
        context_list.append(self._get_one_hot(self.game.active_piece.type.value))
        
        # B. Hold Piece (Default to 0 / N if None)
        hold_val = self.game.hold_piece.value if self.game.hold_piece else PieceEnum.N.value
        context_list.append(self._get_one_hot(hold_val))
        
        # C. Next Pieces
        queue_list = list(self.game.queue.pieces)
        for i in range(self.T_CONFIG.max_pieces_on_queue_view): 
            next_val = queue_list[i].value if i < len(queue_list) else PieceEnum.N.value
            context_list.append(self._get_one_hot(next_val))

            
        queue_matrix = np.array(context_list, dtype=np.float32) # Shape: (7, 9)
        boards_matrix = np.zeros((
            self.CONFIG.max_placements,
            self.CONFIG.max_board_size_h + self.T_CONFIG.vanish_zone,
            self.CONFIG.max_board_size_w),
        dtype=np.float32)

        pad_h = max(0, self.CONFIG.max_board_size_h - (self.T_CONFIG.board_h + self.T_CONFIG.vanish_zone))
        pad_left = max(0, (self.CONFIG.max_board_size_w - self.T_CONFIG.board_w) // 2)
        pad_right = max(0, self.CONFIG.max_board_size_w - self.T_CONFIG.board_w - pad_left)

        for i, placement in enumerate(self.current_placements):
            if i >= self.CONFIG.max_placements:
                break
            grid = self._extract_features_2d(placement['bitmap'])
            boards_matrix[i] = np.pad(grid, ((0, pad_h), (pad_left, pad_right)), constant_values=1)
            
        # 3. Return the Dict directly
        return {
            "boards": boards_matrix,
            "queue": queue_matrix,
            "placement_mask": self.valid_action_mask()
        }
    

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
        # If column 0 is the highest bit (MSB), use: self.T_CONFIG.board_w - 1 - np.arange(self.T_CONFIG.board_w)
        # If column 0 is the lowest bit (LSB), use: np.arange(self.T_CONFIG.board_w)
        shifts = np.arange(self.T_CONFIG.board_w, dtype=np.uint32)
        
        # 2. Use broadcasting to shift every row's bits and check the lowest bit (& 1)
        # bitmap_array[:, None] reshapes to (height, 1)
        # shifts[None, :] reshapes to (1, width)
        # Resulting matrix shape: (height, width)
        unpacked_2d = (bitmap_array[:, None] >> shifts) & 1
        
        # 3. Flatten and cast directly to float32 for your Neural Network
        return unpacked_2d.ravel().astype(np.float32)


    def _extract_features_2d(self, bitmap_array: np.ndarray) -> np.ndarray:
        """
        Unpacks a 1D array of bitmasks into a clean 2D float32 grid.
        Input shape: (board_h,)
        Output shape: (board_h, board_w)
        """
        shifts = np.arange(self.T_CONFIG.board_w, dtype=np.uint32)
        
        # Explicitly project 1D rows into a 2D grid setup along the horizontal axis
        expanded_rows = rearrange(bitmap_array, 'h -> h 1')
        
        # Bitwise shift and mask out individual cell bits
        unpacked_2d = (expanded_rows >> shifts) & 1
        
        return unpacked_2d.astype(np.float32)
    

    def _calculate_reward(self, lines: int) -> float:
        # Reward shaping (e.g., lines cleared, surviving, penalizing holes)
        return float(lines)