import pygame
from .tetris import PieceEnum
from dataclasses import dataclass

@dataclass 
class TetrisConfiguration:
    # --- Pygame & UI Constants ---
    CELL_SIZE = 30
    BOARD_W = 10
    BOARD_H = 20
    VANISH_ZONE = 4

    # UI Layout (Grid Coordinates)
    HOLD_OFFSET_X = 1
    BOARD_OFFSET_X = 6
    NEXT_OFFSET_X = 17

    SCREEN_WIDTH = (NEXT_OFFSET_X + 5) * CELL_SIZE
    SCREEN_HEIGHT = (BOARD_H + VANISH_ZONE) * CELL_SIZE + 4 * CELL_SIZE

    # Official Tetris Guideline Colors
    COLORS = {
        PieceEnum.N: (0, 0, 0),         # Empty
        PieceEnum.I: (84, 255, 201),     # Cyan
        PieceEnum.O: (216, 190, 75),     # Yellow
        PieceEnum.T: (190, 84, 180),     # Purple
        PieceEnum.S: (157, 206, 68),       # Green
        PieceEnum.Z: (212, 72, 80),       # Red
        PieceEnum.J: (104, 87, 192),       # Blue
        PieceEnum.L: (203, 118, 75),     # Orange
        PieceEnum.G: (44, 43, 43),   # Garbage
    }
    DISABLED_COLOR = (80, 80, 80)       # Grayed out hold piece
    GRID_COLOR = (40, 40, 40)
    BG_COLOR = (22, 23, 41)            # Vanish zone background (darker than BG)
    VZ_COLOR = (42, 43, 61)

    FONT_SIZE = 16
    TEXT_COLOR = (200, 200, 200)

    DAS_DELAY = 167  # ms before auto-repeat starts
    ARR_RATE = 33    # ms between auto-repeat moves

    # Keyboard bindings (pygame key constants → ActionEnum)
    KEYS = {
        pygame.K_a:    "LEFT",
        pygame.K_s:   "RIGHT",
        pygame.K_r: "SOFT_DROP",
        pygame.K_LEFT:      "ROTATE_CCW",
        pygame.K_RIGHT:     "ROTATE_CW",
        pygame.K_UP:        "ROTATE_180",
        pygame.K_SPACE:   "DROP",
        pygame.K_w:       "HOLD",
    }

    # Mini-shapes for UI drawing (Hold / Next Queue)
    UI_SHAPES = {
        PieceEnum.I: [(0,1), (1,1), (2,1), (3,1)],
        PieceEnum.J: [(0,0), (0,1), (1,1), (2,1)],
        PieceEnum.L: [(2,0), (0,1), (1,1), (2,1)],
        PieceEnum.S: [(1,0), (2,0), (0,1), (1,1)],
        PieceEnum.Z: [(0,0), (1,0), (1,1), (2,1)],
        PieceEnum.T: [(1,0), (0,1), (1,1), (2,1)],
        PieceEnum.O: [(1,0), (2,0), (1,1), (2,1)],
    }

def draw_cell(CONFIG: TetrisConfiguration, surface, x, y, color):
    """Draws a single Tetris block with a slight border for grid visibility."""
    rect = pygame.Rect(x * CONFIG.CELL_SIZE, y * CONFIG.CELL_SIZE, CONFIG.CELL_SIZE, CONFIG.CELL_SIZE)
    pygame.draw.rect(surface, color, rect)
    pygame.draw.rect(surface, CONFIG.GRID_COLOR, rect, 1)

def draw_ui_piece(CONFIG: TetrisConfiguration, surface, piece_type, offset_x, offset_y, disabled=False):
    """Draws a piece in the Hold or Next queues."""
    if piece_type is None or piece_type == PieceEnum.N:
        return
        
    color = CONFIG.COLORS[PieceEnum.G] if disabled else CONFIG.COLORS[piece_type]
    shape = CONFIG.UI_SHAPES.get(piece_type, [])
    
    for dx, dy in shape:
        draw_cell(CONFIG, surface, offset_x + dx, offset_y + dy, color)

def draw_text(CONFIG: TetrisConfiguration, surface, text: str, cell_x: int, cell_y: int, font_size: int = None):
    font_size = font_size or CONFIG.FONT_SIZE
    font = pygame.font.Font(None, font_size)
    label = font.render(text, True, CONFIG.TEXT_COLOR)
    px = cell_x * CONFIG.CELL_SIZE
    py = cell_y * CONFIG.CELL_SIZE
    surface.blit(label, (px, py))