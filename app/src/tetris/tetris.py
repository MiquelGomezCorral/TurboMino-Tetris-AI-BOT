import random 
from collections import deque
import numpy as np
import enum

from .scoring import SpinType, ScoringSystem


DEFAULT_GARBAGE_PROBS = (
    0.263604,
    0.155263,
    0.099349,
    0.151832,
    0.145719,
    0.087687,
    0.032314,
    0.064232,
)

# ===================================================================
#                       ENUMS
# ===================================================================

class PieceEnum(enum.Enum):
    N       = 0
    I       = 1
    O       = 2
    T       = 3
    S       = 4
    Z       = 5
    J       = 6
    L       = 7
    G       = 8

    def __str__(self):
        return str(self.value)

def char_map(char: str) -> int:
    piece = PieceEnum.__members__.get(char.upper(), PieceEnum.N)
    return piece.value

class ActionEnum(enum.Enum):
    LEFT    = 0
    RIGHT   = 1
    ROTATE_CW  = 2
    ROTATE_CCW = 3
    ROTATE_180 = 4
    DROP    = 5
    HOLD    = 6
    DOWN    = 7

ROTATION_DIR = {
    ActionEnum.ROTATE_CW:   1,
    ActionEnum.ROTATE_CCW: -1,
    ActionEnum.ROTATE_180:  2,
}

class RotationEnum(enum.IntEnum):
    SPAWN   = 0
    RIGHT   = 1
    REVERSE = 2
    LEFT    = 3


# ===================================================================
#                       CLASSES
# ===================================================================
class Queue:
    def __init__(self, initial_pieces: str=None):
        # deque is strictly O(1) for appends and pops on both ends
        self._pieces = deque(PieceEnum[c] for c in initial_pieces) if initial_pieces else deque()
        
        # Use standard list, random.shuffle is highly optimized for lists
        self.base_bag = [
            PieceEnum.I, PieceEnum.O, PieceEnum.T, 
            PieceEnum.S, PieceEnum.Z, PieceEnum.J, PieceEnum.L
        ]
        
        if len(self._pieces) == 0:
            self._add_pieces()
            self._add_pieces()
    
    def _add_pieces(self):
        bag = self.base_bag.copy()
        random.shuffle(bag)
        self._pieces.extend(bag) # O(k) extending, avoids memory reallocation

    def pop_piece(self) -> PieceEnum:
        if len(self._pieces) <= 7:
            self._add_pieces()
        
        return self._pieces.popleft() # O(1) pop

    def peek_piece(self) -> PieceEnum:
        if len(self._pieces) <= 7:
            self._add_pieces()

        return self._pieces[0] # O(1) lookup
    
    def get_queue(self) -> deque:
        if len(self._pieces) <= 7:
            self._add_pieces()
        return self._pieces
    
    def __len__(self):
        return len(self._pieces)
    
    def __str__(self):
        return f"Queue({[piece.name for piece in self._pieces]})"
    

class ActivePiece:
    '''
    Represents the currently active piece on the board.
    Precomputes all rotation states for O(1) access during gameplay.
    '''
    _BASE_SHAPES = {
        PieceEnum.I: [[0,0,0,0], [0,0,0,0], [1,1,1,1], [0,0,0,0]],
        PieceEnum.J: [[0,0,0], [1,1,1], [1,0,0]],
        PieceEnum.L: [[0,0,0], [1,1,1], [0,0,1]],
        PieceEnum.S: [[0,0,0], [1,1,0], [0,1,1]],
        PieceEnum.Z: [[0,0,0], [0,1,1], [1,1,0]],
        PieceEnum.T: [[0,0,0], [1,1,1], [0,1,0]],
        PieceEnum.O: [[1,1], [1,1]],
    }

    PRECOMPUTED_MASKS = {}
    for p_type, base in _BASE_SHAPES.items():
        base_arr = np.array(base, dtype=bool)
        rotations = []
        for i in range(4):
            rot = np.rot90(base_arr, k=i)
            row_ints = [sum((1 << j) for j, val in enumerate(row) if val) for row in rot]
            rotations.append(row_ints)
        PRECOMPUTED_MASKS[p_type] = rotations

    def __init__(self, piece_type: PieceEnum, width: int = 10, height: int = 20):
        self.width = width
        self.height = height
        self.type = piece_type
        self.reset_piece(piece_type)
        # 3. Instant O(1) lookup. No computation during the game loop.
        self.masks = self.PRECOMPUTED_MASKS[piece_type] 

    @property
    def current_mask(self):
        return self.masks[self.rotation_state.value]
    
    def reset_piece(self, piece_type: PieceEnum):
        self.type = piece_type
        self.rotation_state = RotationEnum.SPAWN
        self.masks = self.PRECOMPUTED_MASKS[piece_type]

        piece_size = len(self._BASE_SHAPES[piece_type][0])
        
        self.x = (self.width - piece_size) // 2
        self.y = self.height + 1

    def move_left(self):
        self.x -= 1
    def move_right(self):
        self.x += 1
    def move_down(self):
        self.y -= 1
    def rotate_cw(self):
        self.rotation_state = RotationEnum((self.rotation_state + 1) % 4)
    def rotate_ccw(self):
        self.rotation_state = RotationEnum((self.rotation_state - 1) % 4)
    def rotate_180(self):
        self.rotation_state = RotationEnum((self.rotation_state + 2) % 4)
    
    def set_position(self, x: int, y: int):
        self.x = x
        self.y = y
    

def _clear_bitmap(b_rows, width, visible_height, c_rows=None):
    FULL = (1 << width) - 1
    visible_slice = b_rows[:visible_height]
    full_mask = visible_slice == FULL
    cleared = int(np.sum(full_mask))
    if cleared == 0:
        return (b_rows, c_rows, 0) if c_rows is not None else (b_rows, 0)

    visible_kept = visible_slice[~full_mask]
    above_visible = b_rows[visible_height:]
    all_kept = np.concatenate((visible_kept, above_visible))

    cleared_b = b_rows.copy()
    cleared_b[:len(all_kept)] = all_kept
    cleared_b[len(all_kept):] = 0

    if c_rows is not None:
        visible_c = c_rows[:visible_height]
        kept_c = visible_c[~full_mask]
        above_c = c_rows[visible_height:]
        all_kept_c = np.vstack((kept_c, above_c)) if len(above_c) > 0 else kept_c
        cleared_c = c_rows.copy()
        cleared_c[:len(all_kept_c)] = all_kept_c
        cleared_c[len(all_kept_c):] = 0
        return cleared_b, cleared_c, cleared
    return cleared_b, cleared


class Board:
    SRS_OFFSETS_STANDARD = {
        (RotationEnum.SPAWN,   RotationEnum.RIGHT):   [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        (RotationEnum.RIGHT,   RotationEnum.SPAWN):   [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
        (RotationEnum.RIGHT,   RotationEnum.REVERSE): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
        (RotationEnum.REVERSE, RotationEnum.RIGHT):   [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        (RotationEnum.REVERSE, RotationEnum.LEFT):    [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
        (RotationEnum.LEFT,    RotationEnum.REVERSE): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        (RotationEnum.LEFT,    RotationEnum.SPAWN):   [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        (RotationEnum.SPAWN,   RotationEnum.LEFT):    [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)]
    }

    SRS_OFFSETS_I = {
        (RotationEnum.SPAWN,   RotationEnum.RIGHT):   [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
        (RotationEnum.RIGHT,   RotationEnum.SPAWN):   [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
        (RotationEnum.RIGHT,   RotationEnum.REVERSE): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
        (RotationEnum.REVERSE, RotationEnum.RIGHT):   [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
        (RotationEnum.REVERSE, RotationEnum.LEFT):    [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
        (RotationEnum.LEFT,    RotationEnum.REVERSE): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
        (RotationEnum.LEFT,    RotationEnum.SPAWN):   [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
        (RotationEnum.SPAWN,   RotationEnum.LEFT):    [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)]
    }

    SRS_OFFSETS_180 = {
        (RotationEnum.SPAWN,   RotationEnum.REVERSE): [(0, 0), (0, 1), (1, 1), (-1, 1), (1, 0), (-1, 0)],
        (RotationEnum.RIGHT,   RotationEnum.LEFT):    [(0, 0), (1, 0), (1, -2), (1, -1), (0, -2), (0, -1)],
        (RotationEnum.REVERSE, RotationEnum.SPAWN):   [(0, 0), (0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0)],
        (RotationEnum.LEFT,    RotationEnum.RIGHT):    [(0, 0), (-1, 0), (-1, -2), (-1, -1), (0, -2), (0, -1)]
    }

    FRONT_CORNERS_MAP = {
        RotationEnum.SPAWN:   [(0, 2), (2, 2)],
        RotationEnum.RIGHT:   [(0, 0), (0, 2)],
        RotationEnum.REVERSE: [(0, 0), (2, 0)],
        RotationEnum.LEFT:    [(2, 0), (2, 2)],
    }

    CORNERS = [(0, 0), (2, 0), (0, 2), (2, 2)]

    def __init__(self, width: int = 10, height: int = 20, vanish_zone: int = 4, color_map: bool = False, playfield: str = None):
        assert width > 0 and height > 0, "Width and height must be positive integers."
        assert width <= 40 and height <= 40, "Width and height must be positive integers."
        assert vanish_zone >= 0, "Vanish zone must be a non-negative integer."

        self.width = width
        self.visible_height = height
        self.vanish_zone = vanish_zone
        self.height = height + vanish_zone

        self.b_rows = np.zeros(self.height, dtype=np.uint32)
        self.color_map = color_map
        if color_map:
            self.c_rows = np.zeros((self.height, width), dtype=int)

        if playfield is not None:
            self.load_playfield(playfield)

    def load_playfield(self, playfield: str):
        flat_ints = [char_map(c) for c in playfield]

        remainder = len(flat_ints) % self.width
        if remainder != 0:
            flat_ints.extend([0] * (self.width - remainder))

        n_rows = len(flat_ints) // self.width

        if n_rows > self.height:
            extra = n_rows - self.height
            self.b_rows = np.concatenate([self.b_rows, np.zeros(extra, dtype=np.uint32)])
            if self.color_map:
                self.c_rows = np.concatenate([self.c_rows, np.zeros((extra, self.width), dtype=int)])
            self.height = n_rows

        c_matrix = np.array(flat_ints, dtype=int).reshape((n_rows, self.width))
        # c_matrix = c_matrix[::-1]

        powers_of_2 = (1 << np.arange(self.width, dtype=np.uint32))
        self.b_rows[:n_rows] = (c_matrix > 0).dot(powers_of_2).astype(np.uint32)

        if self.color_map:
            self.c_rows[:n_rows] = c_matrix

    def check_collision(self, piece_rows: list[int], x: int, y: int) -> bool:
        for local_y, row_mask in enumerate(piece_rows):
            if row_mask == 0:
                continue

            by = y + local_y
            if by < 0 or by >= self.height:
                return True

            # Handle X bounds via bit shifts
            if x < 0:
                if (row_mask >> abs(x)) << abs(x) != row_mask:
                    return True
                shifted_mask = row_mask >> abs(x)
            else:
                shifted_mask = row_mask << x
                if shifted_mask >= (1 << self.width):
                    return True

            if self.b_rows[by] & shifted_mask:
                return True

        return False

    def _burn_piece(self, piece: ActivePiece):
        for local_y, row_mask in enumerate(piece.current_mask):
            if row_mask == 0:
                continue

            by = piece.y + local_y
            if by < 0 or by >= self.height:
                continue

            shifted_mask = row_mask << piece.x if piece.x >= 0 else row_mask >> abs(piece.x)
            self.b_rows[by] |= shifted_mask

            if self.color_map:
                for local_x in range(4):
                    if row_mask & (1 << local_x):
                        bx = piece.x + local_x
                        self.c_rows[by, bx] = piece.type.value

    def _clear_lines(self) -> int:
        if self.color_map:
            cleared_b, cleared_c, cleared = _clear_bitmap(
                self.b_rows, self.width, self.visible_height, self.c_rows)
            self.b_rows[:self.visible_height] = cleared_b[:self.visible_height]
            self.c_rows[:self.visible_height] = cleared_c[:self.visible_height]
        else:
            cleared_b, cleared = _clear_bitmap(
                self.b_rows, self.width, self.visible_height)
            self.b_rows[:self.visible_height] = cleared_b[:self.visible_height]
        return cleared

    def lock_piece(self, piece: ActivePiece) -> int:
        self._burn_piece(piece)
        return self._clear_lines()

    def attempt_rotation(self, piece: ActivePiece, action: ActionEnum) -> int:
        direction = ROTATION_DIR[action]
        next_rot = (piece.rotation_state + direction) % 4
        next_mask = piece.masks[next_rot]

        if piece.type == PieceEnum.O:
            if not self.check_collision(next_mask, piece.x, piece.y):
                piece.rotation_state = RotationEnum(next_rot)
                return 0
            return -1

        if action == ActionEnum.ROTATE_180:
            offsets = self.SRS_OFFSETS_180.get((piece.rotation_state, next_rot), [(0, 0)])
        else:
            table = self.SRS_OFFSETS_I if piece.type == PieceEnum.I else self.SRS_OFFSETS_STANDARD
            offsets = table.get((piece.rotation_state, next_rot), [(0, 0)])

        for i, (dx, dy) in enumerate(offsets):
            if not self.check_collision(next_mask, piece.x + dx, piece.y + dy):
                piece.x += dx
                piece.y += dy
                piece.rotation_state = RotationEnum(next_rot)
                return i
        return -1

    def check_t_spin(self, piece: ActivePiece, last_action_was_rotation: bool, last_kick_index: int) -> SpinType:
        if piece.type != PieceEnum.T:
            return SpinType.NONE

        can_move_left = not self.check_collision(piece.current_mask, piece.x - 1, piece.y)
        can_move_right = not self.check_collision(piece.current_mask, piece.x + 1, piece.y)
        can_move_down = not self.check_collision(piece.current_mask, piece.x, piece.y - 1)
        immobile = not (can_move_left or can_move_right or can_move_down)

        if not immobile and not last_action_was_rotation:
            return SpinType.NONE

        front_corners = self.FRONT_CORNERS_MAP[piece.rotation_state]
        filled_corners = 0
        front_filled = 0

        for cx, cy in self.CORNERS:
            bx, by = piece.x + cx, piece.y + cy
            if bx < 0 or bx >= self.width or by < 0 or by >= self.height:
                is_filled = True
            else:
                is_filled = bool(self.b_rows[by] & (1 << bx))
            if is_filled:
                filled_corners += 1
                if (cx, cy) in front_corners:
                    front_filled += 1

        if filled_corners >= 3:
            if front_filled == 2 or last_kick_index == 4:
                return SpinType.REGULAR
            return SpinType.MINI
        return SpinType.NONE

    def move_piece(self, piece: ActivePiece, dx: int, dy: int) -> bool:
        can_move = not self.check_collision(piece.current_mask, piece.x + dx, piece.y + dy)
        if can_move:
            piece.x += dx
            piece.y += dy

        return can_move

    def move_piece_left(self, piece: ActivePiece) -> bool:
        return self.move_piece(piece, dx=-1, dy=0)

    def move_piece_right(self, piece: ActivePiece) -> bool:
        return self.move_piece(piece, dx=1, dy=0)

    def move_piece_down(self, piece: ActivePiece) -> bool:
        return self.move_piece(piece, dx=0, dy=-1)

    def get_ghost_y(self, piece: ActivePiece) -> int:
        ghost_y = piece.y
        while not self.check_collision(piece.current_mask, piece.x, ghost_y - 1):
            ghost_y -= 1
        return ghost_y

    def hard_drop(self, piece: ActivePiece) -> int:
        ghost_y = self.get_ghost_y(piece)
        drop_distance = piece.y - ghost_y
        piece.y = ghost_y
        return drop_distance

    def add_garbage(self, lines: int = 1, hole: int = None) -> bool:
        if lines <= 0:
            return False

        lines = min(lines, self.height)
        hole = random.randrange(self.width) if hole is None else hole
        overflow = bool(np.any(self.b_rows[-lines:]))
        garbage_row = ((1 << self.width) - 1) & ~(1 << hole)

        self.b_rows[lines:] = self.b_rows[:-lines]
        self.b_rows[:lines] = garbage_row

        if self.color_map:
            self.c_rows[lines:] = self.c_rows[:-lines]
            self.c_rows[:lines] = PieceEnum.G.value
            self.c_rows[:lines, hole] = PieceEnum.N.value

        return overflow
    
    def print_board(self, b_board=None, c_board=None, active_piece=None, include_vanish_zone=False):
        row_count = self.height if include_vanish_zone else self.visible_height
        if b_board is None:
            b_board = self.b_rows
        if c_board is None and self.color_map:
            c_board = self.c_rows

        if self.color_map:
            for y in reversed(range(row_count)):
                line = ''
                for x in range(self.width):
                    if b_board[y] & (1 << x):
                        line += PieceEnum(c_board[y, x]).name
                    else:
                        from_active = False
                        if active_piece:
                            ly = y - active_piece.y
                            lx = x - active_piece.x
                            if 0 <= ly < len(active_piece.current_mask) and 0 <= lx < 4:
                                from_active = bool(active_piece.current_mask[ly] & (1 << lx))
                        line += active_piece.type.name if from_active else '.'
                print(line)
        else:
            render_rows = b_board.copy() if include_vanish_zone else b_board[:self.visible_height].copy()

            if active_piece is not None:
                for local_y, row_mask in enumerate(active_piece.current_mask):
                    if row_mask == 0:
                        continue
                    by = active_piece.y + local_y
                    if 0 <= by < len(render_rows):
                        shifted = row_mask << active_piece.x if active_piece.x >= 0 else row_mask >> abs(active_piece.x)
                        render_rows[by] |= shifted

            for row in reversed(render_rows):
                line = ''.join('X' if row & (1 << i) else '.' for i in range(self.width))
                print(line)


# ===================================================================
#                       GAME
# ===================================================================
class Tetris:
    active_piece: ActivePiece
    hold_piece: PieceEnum | None
    queue: Queue
    board: Board
    can_hold: bool
    score_system: ScoringSystem
    last_action_was_rotation: bool
    last_kick_index: int
    incoming_garbage: deque[tuple[int, int, int]]
    garbage_prob: float
    game_over: bool

    garbage_delay: int
    garbage_probs: tuple[float, ...]
    garbage_cap: int

    def __init__(
        self, 
        width: int = 10, height: int = 20, vanish_zone: int = 5, 
        color_map: bool = False, 
        playfield: str = None, 
        next_pieces: str = None,
        active_piece: str = None,
        hold_piece: str = None,
        garbage_delay: int = 3,
        garbage_prob: float = 0.0774,
        garbage_probs: list[float] | tuple[float, ...] = DEFAULT_GARBAGE_PROBS,
        garbage_cap: int = 8,
    ):
        assert 0 <= garbage_prob <= 1, "garbage_prob must be between 0 and 1"
        assert garbage_delay >= 1, "garbage_delay must be at least 1"
        assert len(garbage_probs) == 8, "garbage_probs must contain probabilities for 1 to 8 lines"
        assert all(prob >= 0 for prob in garbage_probs) and sum(garbage_probs) > 0, (
            "garbage_probs must contain positive weight"
        )
        assert garbage_cap > 0, "garbage_cap must be positive"

        self.width = width
        self.height = height
        self.vanish_zone = vanish_zone
        self.color_map = color_map
        self.garbage_prob = garbage_prob
        self.garbage_delay = garbage_delay
        self.garbage_probs = tuple(garbage_probs)
        self.garbage_cap = garbage_cap
        self.board = Board(width, height, vanish_zone, color_map, playfield)
        self.queue = Queue(next_pieces)

        if active_piece:
            self.active_piece = ActivePiece(PieceEnum[active_piece], self.width, self.height)
        else:
            self.active_piece = ActivePiece(self.queue.pop_piece(), self.width, self.height)
        
        self.hold_piece = PieceEnum[hold_piece] if hold_piece else None
        self.can_hold = True
        self.score_system = ScoringSystem()
        self.last_action_was_rotation = False
        self.last_kick_index = -1
        self.game_over = False
        self.incoming_garbage = deque()

    def spawn_piece(self):
        self.active_piece.reset_piece(self.queue.pop_piece())
        self.game_over = self.board.check_collision(self.active_piece.current_mask, self.active_piece.x, self.active_piece.y)

    def move_active_piece(self, action: ActionEnum):
        cleared_lines = 0
        if action == ActionEnum.LEFT:
            self.board.move_piece_left(self.active_piece)
            self.last_action_was_rotation = False
        elif action == ActionEnum.RIGHT:
            self.board.move_piece_right(self.active_piece)
            self.last_action_was_rotation = False
        elif action in (ActionEnum.ROTATE_CW, ActionEnum.ROTATE_CCW, ActionEnum.ROTATE_180):
            kick_idx = self.board.attempt_rotation(self.active_piece, action)
            self.last_action_was_rotation = kick_idx >= 0
            self.last_kick_index = kick_idx
        elif action == ActionEnum.DROP:
            drop_distance = self.board.hard_drop(self.active_piece)
            spin = self.board.check_t_spin(self.active_piece, self.last_action_was_rotation, self.last_kick_index)
            cleared_lines = self.board.lock_piece(self.active_piece)
            perfect_clear = cleared_lines > 0 and all(self.board.b_rows[i] == 0 for i in range(self.board.visible_height))
            
            attack = self.score_system.evaluate_drop(cleared_lines, spin, perfect_clear, drop_distance, hard_drop=True)

            overflow = False
            # ONLY ADD GARBAGE IF THE 'MOVEMENTS ARE DONE'
            if self.garbage_prob > 0:
                overflow = self.manage_garbage(cleared_lines, attack)

            self.spawn_piece()
            self.game_over = self.game_over or overflow
            self.can_hold = True
            self.last_action_was_rotation = False
            self.last_kick_index = -1

        elif action == ActionEnum.HOLD and self.can_hold:
            if self.hold_piece is None:
                self.hold_piece = self.active_piece.type
                self.spawn_piece()
            else:
                self.hold_piece, self.active_piece.type = self.active_piece.type, self.hold_piece
                self.active_piece.reset_piece(self.active_piece.type)

            self.can_hold = False
            self.last_action_was_rotation = False
        elif action == ActionEnum.DOWN:
            self.board.move_piece_down(self.active_piece)
            self.last_action_was_rotation = False


        return cleared_lines

    def manage_garbage(self, cleared_lines: int, attack: int) -> bool:
        while attack > 0 and self.incoming_garbage:
            lines, turns, hole = self.incoming_garbage.popleft()
            cancelled = min(lines, attack)
            attack -= cancelled
            if lines > cancelled:
                self.incoming_garbage.appendleft((lines - cancelled, turns, hole))

        self.incoming_garbage = deque(
            (lines, max(0, turns - 1), hole)
            for lines, turns, hole in self.incoming_garbage
        )

        overflow = False
        if cleared_lines == 0:
            remaining_cap = self.garbage_cap
            while self.incoming_garbage and self.incoming_garbage[0][1] == 0 and remaining_cap > 0:
                lines, turns, hole = self.incoming_garbage.popleft()
                added = min(lines, remaining_cap)
                overflow = self.board.add_garbage(added, hole) or overflow
                remaining_cap -= added
                if lines > added:
                    self.incoming_garbage.appendleft((lines - added, turns, hole))

        # Garbage is queued after the turn, so it cannot be cancelled immediately.
        self.randomly_add_garbage()
        return overflow

    def randomly_add_garbage(self):
        if self.garbage_prob <= 0:
            return

        if random.random() < self.garbage_prob:
            lines = random.choices(range(1, 9), weights=self.garbage_probs, k=1)[0]
            hole = random.randrange(self.width)
            self.incoming_garbage.append((lines, self.garbage_delay, hole))

    def get_board_state(self, include_vanish_zone=False):
        if include_vanish_zone:
            return self.board.b_rows.copy()
        else:
            return self.board.b_rows[:self.board.visible_height].copy()

    def get_active_piece_info(self):
        return {
            'type': self.active_piece.type,
            'x': self.active_piece.x,
            'y': self.active_piece.y,
            'rotation': self.active_piece.rotation_state
        }

    def get_swappable_hold(self):
        return self.hold_piece

    def get_swap_piece(self):
        return self.hold_piece

    def get_next_pieces(self):
        return [piece.name for piece in self.queue.get_queue()]

    def get_score(self):
        return self.score_system.score

    def get_queue(self):
        return [p.value for p in self.queue.get_queue()]

    def is_game_over(self):
        return self.game_over
    
    def get_level(self):
        return self.score_system.level
    
    def get_active_piece_type(self):
        return self.active_piece.type.value
    
    def get_hold_piece_type(self):
        return self.hold_piece.value if self.hold_piece else PieceEnum.N.value
    
    def get_next_pieces_types(self, n=5):
        return [p.value for p in list(self.queue.get_queue())[:n]]
    
    def get_next_piece_type(self):
        return self.queue.peek_piece().value
    
    def get_combo(self):
        return self.score_system.get_combo()
    
    def get_b2b_active(self):
        return self.score_system.get_b2b_active()

    def get_b2b_streak(self):
        return self.score_system.get_b2b_streak()
    
    def get_immediate_garbage(self):
        return min(
            self.garbage_cap,
            sum(lines for lines, turns, _ in self.incoming_garbage if turns <= 1),
        )

    def get_incoming_garbage(self):
        return sum(lines for lines, _, _ in self.incoming_garbage)

    def get_lines_cleared(self):
        return self.score_system.lines_cleared_total
    def get_total_all_clears(self):
        return self.score_system.total_all_clears
    def get_total_tetrises(self):
        return self.score_system.total_tetrises
    
    def get_hold_or_next_piece_type(self):
        """Returns the piece type that would be active if the player performs a hold action right now.
        If hold is available and there is a piece in hold, returns the hold piece type.
        Otherwise, returns the next piece type from the queue.
        """
        if self.hold_piece and self.hold_piece != PieceEnum.N:
            return self.get_hold_piece_type(), True   
        else:
            return self.get_next_piece_type(), False

    def print_state(self, include_vanish_zone=False):
        # print("Current Board:")
        self.board.print_board(active_piece=self.active_piece, include_vanish_zone=include_vanish_zone)
        # print(f"Active Piece: {self.active_piece.type.name} at ({self.active_piece.x}, {self.active_piece.y}) with rotation {self.active_piece.rotation_state.name}")
        print(f"Act.: {self.active_piece.type.name} | Hold: {self.hold_piece.name if self.hold_piece else 'N'} | Next: {[p.name for p in self.queue.get_queue()]}")
        # print(f"Can Hold: {self.can_hold}")
        print(f"Combo: {self.score_system.combo if self.score_system.combo >= 0 else '---':<5}  |  B2B: {self.score_system.b2b_streak:<3} |  Last Move: {self.score_system.last_move_name if self.score_system.last_move_name else '---':<15} | Total All Clears: {self.score_system.total_all_clears: <5}")
