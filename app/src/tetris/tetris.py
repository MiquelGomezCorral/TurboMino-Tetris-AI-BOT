import random 
from collections import deque
import numpy as np
import enum

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

class ActionEnum(enum.Enum):
    LEFT    = 0
    RIGHT   = 1
    ROTATE   = 2
    DROP    = 3
    HOLD    = 4

# ===================================================================
#                       CLASSES
# ===================================================================
    
class ActivePiece:
    kind: int
    x: int
    y: int
    rotation: int

class Board:
    def __init__(self, width: int = 10, height: int = 20, color_map: bool = False, playfield: str = None):
        assert width > 0 and height > 0, "Width and height must be positive integers."

        self.width = width
        self.height = height
        
        self.b_rows = np.zeros((height, width), dtype=bool)  
        self.color_map = color_map
        if color_map:
            self.c_rows = np.zeros((height, width), dtype=int)  
        
        if playfield is not None:
            self.load_playfield(playfield)

    def load_playfield(self, playfield: str):
        for i in range(self.height):  
            for j in range(self.width):  
                index = i * self.width + j
                if index >= len(playfield):
                    break  

                # FIX: Look up by the character string name (e.g., 'G', 'N') instead of value
                cell_value = PieceEnum[playfield[index]] 

                if cell_value != PieceEnum.N:
                    self.b_rows[i, j] = True
                    if self.color_map: # Safer than hasattr check
                        self.c_rows[i, j] = cell_value.value  # Store the underlying integer enum value
            
            if index >= len(playfield):
                break

class Queue:
    def __init__(self, initial_pieces: str=None):
        # deque is strictly O(1) for appends and pops on both ends
        self.pieces = deque(PieceEnum[c] for c in initial_pieces) if initial_pieces else deque()
        
        # Use standard list, random.shuffle is highly optimized for lists
        self.base_bag = [
            PieceEnum.I, PieceEnum.O, PieceEnum.T, 
            PieceEnum.S, PieceEnum.Z, PieceEnum.J, PieceEnum.L
        ]
        
        if len(self.pieces) == 0:
            self._add_pieces()
            self._add_pieces()
    
    def _add_pieces(self):
        bag = self.base_bag.copy()
        random.shuffle(bag)
        self.pieces.extend(bag) # O(k) extending, avoids memory reallocation

    def pop_piece(self) -> PieceEnum:
        if len(self.pieces) <= 7:
            self._add_pieces()
        
        return self.pieces.popleft() # O(1) pop

    def peek_piece(self) -> PieceEnum:
        if len(self.pieces) <= 7:
            self._add_pieces()

        return self.pieces[0] # O(1) lookup
    
    def __len__(self):
        return len(self.pieces)
    
    def __str__(self):
        return f"Queue({[piece.name for piece in self.pieces]})"

# ===================================================================
#                       GAME
# ===================================================================
class Tetris:
    active_piece: ActivePiece
    board: Board

    def __init__(self, width: int = 10, height: int = 20, color_map: bool = False, playfield: str = None):
        self.board = Board(width, height, color_map, playfield)
        self.active_piece = None



