import numpy as np
from einops import rearrange
from collections import deque

from maikol_utils.print_utils import print_warn


from src.config import Configuration
from .tetris import ActionEnum, Board, ActivePiece, PieceEnum, RotationEnum, Tetris, _clear_bitmap
from .configuration import TetrisConfiguration


class MoveSearcher:
    def __init__(self, game: Tetris = None, CONFIG: Configuration = None, T_CONFIG: TetrisConfiguration = None):
        self.game = game
        self.CONFIG = CONFIG
        self.T_CONFIG = T_CONFIG
    
    def get_all_placements(self, piece_type: PieceEnum = None, prepend_hold: bool = False) -> list[dict]:
        """
        Finds all valid unique piece placement lock positions.
        Returns a list of dictionaries containing:
          - 'state': (x, y, rotation_value)
          - 'sequence': list of ActionEnum actions taken from spawn to lock
          - 'bitmap': a 1D uint32 numpy array representing the new board state
          - 'lines_cleared': int (only when clear_lines=True)
        """
        if self.game is None:
            raise ValueError("MoveSearcher must be initialized with a Tetris game instance")

        if piece_type is None:
            piece_type = self.game.active_piece.type
        board = self.game.board

        # 1. Initialize piece at spawn to find starting parameters
        spawn_piece = ActivePiece(piece_type, board.width, board.visible_height)
        start_state = (spawn_piece.x, spawn_piece.y, spawn_piece.rotation_state.value)

        # Queue holds: (x, y, rot_val, tuple_of_actions)
        queue = deque([(*start_state, ())])
        visited = {start_state}
        
        # Store unique placements mapped by their terminal (x, y, rot)
        # Keeps the shortest sequence/path to arrive at each terminal position
        placements = {}
        search_piece = ActivePiece(piece_type, board.width, board.visible_height)

        while queue:
            x, y, rot_val, path = queue.popleft()
            mask = search_piece.masks[rot_val]

            # --- LOCK CHECK & BITMAP SPECIFICATION ---
            if board.check_collision(mask, x, y - 1):
                # Unique spatial footprint: (absolute_y, absolute_shifted_mask)
                footprint = tuple(
                    (y + local_y, (row_mask << x if x >= 0 else row_mask >> abs(x)))
                    for local_y, row_mask in enumerate(mask)
                    if row_mask != 0
                )

                if footprint not in placements:
                    # Smart, fast bitwise injection into a board clone
                    new_b_rows = board.b_rows.copy()
                    for by, shifted_mask in footprint:
                        if 0 <= by < len(new_b_rows):
                            new_b_rows[by] |= shifted_mask

                    # Store using the footprint as the deduplication key
                    if self.CONFIG.clear_lines_on_placement:
                        new_b_rows, lines_cleared = _clear_bitmap(new_b_rows, board.width, board.visible_height)
                    else:
                        lines_cleared = 0

                    placements[footprint] = {
                        'state': (x, y, rot_val),
                        'sequence': list(path) + [ActionEnum.DROP] if not prepend_hold else [ActionEnum.HOLD] + list(path) + [ActionEnum.DROP],
                        'bitmap': new_b_rows,
                        'lines_cleared': lines_cleared,
                    }
            else:
                # Gravity: If it can move down, it must fall (Standard Guideline handling)
                down_state = (x, y - 1, rot_val)
                if down_state not in visited:
                    visited.add(down_state)
                    queue.append((*down_state, path + (ActionEnum.DOWN,)))

            # --- LATERAL & ROTATIONAL SEARCH BRANCHES ---
            # 1. Slide Left
            if not board.check_collision(mask, x - 1, y):
                next_state = (x - 1, y, rot_val)
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((*next_state, path + (ActionEnum.LEFT,)))

            # 2. Slide Right
            if not board.check_collision(mask, x + 1, y):
                next_state = (x + 1, y, rot_val)
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((*next_state, path + (ActionEnum.RIGHT,)))

            # 3. Rotations (CW, CCW, 180) using SRS checks
            for action in (ActionEnum.ROTATE_CW, ActionEnum.ROTATE_CCW, ActionEnum.ROTATE_180):
                search_piece.x = x
                search_piece.y = y
                search_piece.rotation_state = RotationEnum(rot_val)

                if board.attempt_rotation(search_piece, action) >= 0:
                    next_state = (search_piece.x, search_piece.y, search_piece.rotation_state.value)
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((*next_state, path + (action,)))

        return list(placements.values())
    
        
    def _get_one_hot(self, piece_val):
        """Converts a piece value into a one-hot encoded vector of length num_piece_categories."""
        arr = np.zeros(self.T_CONFIG.num_piece_categories, dtype=np.float32)
        arr[piece_val] = 1.0
        return arr
    

    def _build_single_queue_context(self, active_val: int, hold_val: int, upcoming_vals: list[int]) -> np.ndarray:
        """Helper to build the (S, C) one-hot context matrix for a specific scenario."""
        context_list = [
            self._get_one_hot(active_val),
            self._get_one_hot(hold_val)
        ]
        
        for i in range(self.T_CONFIG.max_pieces_on_queue_view):
            val = upcoming_vals[i] if i < len(upcoming_vals) else PieceEnum.N.value
            context_list.append(self._get_one_hot(val))
            
        return np.array(context_list, dtype=np.float32)

    def get_all_features(self, game_state=None) -> dict:
        """Combines placements, board states, and queue contexts into a unified observation dictionary for the model.
            Returns a dictionary with:
                - 'boards': (P, H, W) tensor of board states for each placement
                - 'queues': (2, S, C) tensor of one-hot encoded queue contexts for active and hold scenarios
                - 'queue_idx': (P,) array indicating which queue context applies to each placement
                - 'placement_mask': (P,) binary mask indicating valid placements (for padding purposes)
        
        """
        # ========== 1. Ask the MoveSearcher for all valid placements ========== 
        active_piece_type = self.game.get_active_piece_type()
        hold_piece_type, had_hold = self.game.get_hold_or_next_piece_type()

        active_placements = self.get_all_placements(piece_type=PieceEnum(active_piece_type))
        hold_placements = self.get_all_placements(piece_type=PieceEnum(hold_piece_type), prepend_hold=True)

        # ==========  2. Build the two unique queue contexts ========== 
        queue_list = self.game.get_queue()
        # Hold slot always shows the swap piece; the queue is shown post-swap (shifted when hold is empty).
        queue_list = queue_list if had_hold else queue_list[1:]
        
        # A. Active Queue Scenario
        active_q_matrix = self._build_single_queue_context(
            active_val=active_piece_type,
            hold_val=hold_piece_type,
            upcoming_vals=queue_list
        )
        
        # B. Hold Queue Scenario
        hold_q_matrix = self._build_single_queue_context(
            active_val=hold_piece_type,
            hold_val=active_piece_type,
            # Normal swap
            upcoming_vals=queue_list
        )

        # Stack into shape (2, S, C)
        queues_tensor = np.stack([active_q_matrix, hold_q_matrix])



        # ========== 3. Combine Placements and Build Boards & Mapping ========== 
        self.all_placements = []
        boards_matrix = np.zeros((
            self.CONFIG.max_placements,
            self.CONFIG.max_board_size_h + self.T_CONFIG.vanish_zone,
            self.CONFIG.max_board_size_w
        ), dtype=np.uint8)
        
        queue_idx_matrix = np.zeros(self.CONFIG.max_placements, dtype=np.int64)

        pad_h = max(0, self.CONFIG.max_board_size_h - (self.T_CONFIG.board_h + self.T_CONFIG.vanish_zone))
        pad_left = max(0, (self.CONFIG.max_board_size_w - self.T_CONFIG.board_w) // 2)
        pad_right = max(0, self.CONFIG.max_board_size_w - self.T_CONFIG.board_w - pad_left)
        # Merge active and hold placements, tagging them with their queue index
        # 0 = Active, 1 = Hold
        placements_to_process = [(p, 0) for p in active_placements] + [(p, 1) for p in hold_placements]

        for i, (placement, q_idx) in enumerate(placements_to_process):
            if i >= self.CONFIG.max_placements:
                print_warn(f"Number of placements ({len(placements_to_process)}) exceeds CONFIG.max_placements ({self.CONFIG.max_placements}). Truncating extra placements. Increase CONFIG.max_placements to capture more.")
                print_warn(f"{PieceEnum(active_piece_type) =}, {PieceEnum(hold_piece_type) =}")
                self.game.print_state()
                break
                
            grid = self._extract_features_2d(placement['bitmap'])
            target_h = boards_matrix.shape[1]
            if grid.shape[0] > target_h:
                grid = grid[:target_h]
            pad_h = max(0, target_h - grid.shape[0])
            boards_matrix[i] = np.pad(grid, ((0, pad_h), (pad_left, pad_right)), constant_values=1)
            queue_idx_matrix[i] = q_idx
            
            # Save to unified list for the step() function to execute later
            self.all_placements.append((placement, q_idx))


        if game_state is None:
            game_state = [
                self.game.get_combo(),
                self.game.get_b2b_streak(),
                self.game.get_immediate_garbage(),
                self.game.get_incoming_garbage(),
            ]

        # 4. Return the Dictionary
        return self.all_placements, {
            "boards": boards_matrix,
            "queues": queues_tensor,
            "queue_idx": queue_idx_matrix,
            "placement_mask": self.valid_action_mask(),
            "game_state": np.asarray(game_state, dtype=np.float32),
        }
    

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
        
        return unpacked_2d.astype(np.uint8)
    

    def valid_action_mask(self):
        """
        CRITICAL METHOD: sb3-contrib looks for this exact function name.
        It returns a boolean array shape (MAX_PLACEMENTS,).
        True = Valid Move | False = Padded/Illegal Move
        """
        num_valid = len(self.all_placements)
        mask = np.zeros(self.CONFIG.max_placements, dtype=bool)
        mask[:num_valid] = True
        return mask
