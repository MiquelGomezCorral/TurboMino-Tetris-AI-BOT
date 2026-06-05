from collections import deque
from .tetris import ActionEnum, Board, ActivePiece, PieceEnum, RotationEnum

class MoveSearcher:
    def __init__(self, board: Board):
        self.board = board

    def get_all_placements(self, piece_type: PieceEnum) -> list[dict]:
        """
        Finds all valid unique piece placement lock positions.
        Returns a list of dictionaries containing:
          - 'state': (x, y, rotation_value)
          - 'sequence': list of ActionEnum actions taken from spawn to lock
          - 'bitmap': a 1D uint32 numpy array representing the new board state
        """
        # 1. Initialize piece at spawn to find starting parameters
        spawn_piece = ActivePiece(piece_type, self.board.width, self.board.visible_height)
        start_state = (spawn_piece.x, spawn_piece.y, spawn_piece.rotation_state.value)

        # Queue holds: (x, y, rot_val, tuple_of_actions)
        queue = deque([(*start_state, ())])
        visited = {start_state}
        
        # Store unique placements mapped by their terminal (x, y, rot)
        # Keeps the shortest sequence/path to arrive at each terminal position
        placements = {}
        search_piece = ActivePiece(piece_type, self.board.width, self.board.visible_height)

        while queue:
            x, y, rot_val, path = queue.popleft()
            mask = search_piece.masks[rot_val]

            # --- LOCK CHECK & BITMAP SPECIFICATION ---
            if self.board.check_collision(mask, x, y - 1):
                # Unique spatial footprint: (absolute_y, absolute_shifted_mask)
                footprint = tuple(
                    (y + local_y, (row_mask << x if x >= 0 else row_mask >> abs(x)))
                    for local_y, row_mask in enumerate(mask)
                    if row_mask != 0
                )

                if footprint not in placements:
                    # Smart, fast bitwise injection into a board clone
                    new_b_rows = self.board.b_rows.copy()
                    for by, shifted_mask in footprint:
                        if 0 <= by < len(new_b_rows):
                            new_b_rows[by] |= shifted_mask

                    # Store using the footprint as the deduplication key
                    placements[footprint] = {
                        'state': (x, y, rot_val),
                        'sequence': list(path) + [ActionEnum.DROP],
                        'bitmap': new_b_rows
                    }
            else:
                # Gravity: If it can move down, it must fall (Standard Guideline handling)
                down_state = (x, y - 1, rot_val)
                if down_state not in visited:
                    visited.add(down_state)
                    queue.append((*down_state, path))
                continue # Skip lateral inputs mid-air to match true physics rules

            # --- LATERAL & ROTATIONAL SEARCH BRANCHES ---
            # 1. Slide Left
            if not self.board.check_collision(mask, x - 1, y):
                next_state = (x - 1, y, rot_val)
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((*next_state, path + (ActionEnum.LEFT,)))

            # 2. Slide Right
            if not self.board.check_collision(mask, x + 1, y):
                next_state = (x + 1, y, rot_val)
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((*next_state, path + (ActionEnum.RIGHT,)))

            # 3. Rotations (CW, CCW, 180) using SRS checks
            for action in (ActionEnum.ROTATE_CW, ActionEnum.ROTATE_CCW, ActionEnum.ROTATE_180):
                search_piece.x = x
                search_piece.y = y
                search_piece.rotation_state = RotationEnum(rot_val)

                if self.board.attempt_rotation(search_piece, action) >= 0:
                    next_state = (search_piece.x, search_piece.y, search_piece.rotation_state.value)
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((*next_state, path + (action,)))

        return list(placements.values())