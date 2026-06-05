import pygame
from .tetris import PieceEnum
from dataclasses import dataclass, field


@dataclass
class TetrisConfiguration:
    # --- Pygame & UI Constants ---
    cell_size: int = 30
    board_w: int = 10
    board_h: int = 20
    vanish_zone: int = 4

    # UI Layout (Grid Coordinates)
    hold_offset_x: int = 1
    board_offset_x: int = 6
    next_offset_x: int = 17

    screen_width: int = (next_offset_x + 5) * cell_size
    screen_height: int = (board_h + vanish_zone) * cell_size + 4 * cell_size

    # Official Tetris Guideline Colors
    colors: dict = field(default_factory=lambda: {
        PieceEnum.N: (0, 0, 0),
        PieceEnum.I: (84, 255, 201),
        PieceEnum.O: (216, 190, 75),
        PieceEnum.T: (190, 84, 180),
        PieceEnum.S: (157, 206, 68),
        PieceEnum.Z: (212, 72, 80),
        PieceEnum.J: (104, 87, 192),
        PieceEnum.L: (203, 118, 75),
        PieceEnum.G: (44, 43, 43),
    })
    disabled_color: tuple = (80, 80, 80)
    grid_color: tuple = (40, 40, 40)
    bg_color: tuple = (22, 23, 41)
    vz_color: tuple = (42, 43, 61)

    font_size: int = 16
    text_color: tuple = (200, 200, 200)
    game_over_color: tuple = (255, 60, 60)

    das_delay: int = 167
    arr_rate: int = 33

    # Keyboard bindings (pygame key constants -> ActionEnum names)
    keys: dict = field(default_factory=lambda: {
        "LEFT": pygame.K_a,
        "RIGHT": pygame.K_s,
        "SOFT_DROP": pygame.K_r,
        "ROTATE_CCW": pygame.K_LEFT,
        "ROTATE_CW": pygame.K_RIGHT,
        "ROTATE_180": pygame.K_UP,
        "DROP": pygame.K_SPACE,
        "HOLD": pygame.K_w,
        "RESET": pygame.K_p,
        "QUIT": pygame.K_q,
    })

    # Mini-shapes for UI drawing (Hold / Next Queue)
    ui_shapes: dict = field(default_factory=lambda: {
        PieceEnum.I: [(0, 1), (1, 1), (2, 1), (3, 1)],
        PieceEnum.J: [(0, 0), (0, 1), (1, 1), (2, 1)],
        PieceEnum.L: [(2, 0), (0, 1), (1, 1), (2, 1)],
        PieceEnum.S: [(1, 0), (2, 0), (0, 1), (1, 1)],
        PieceEnum.Z: [(0, 0), (1, 0), (1, 1), (2, 1)],
        PieceEnum.T: [(1, 0), (0, 1), (1, 1), (2, 1)],
        PieceEnum.O: [(1, 0), (2, 0), (1, 1), (2, 1)],
    })


def draw_cell(CONFIG: TetrisConfiguration, surface, x, y, color):
    rect = pygame.Rect(x * CONFIG.cell_size, y * CONFIG.cell_size, CONFIG.cell_size, CONFIG.cell_size)
    pygame.draw.rect(surface, color, rect)
    pygame.draw.rect(surface, CONFIG.grid_color, rect, 1)


def draw_ui_piece(CONFIG: TetrisConfiguration, surface, piece_type, offset_x, offset_y, disabled=False):
    if piece_type is None or piece_type == PieceEnum.N:
        return

    color = CONFIG.colors[PieceEnum.G] if disabled else CONFIG.colors[piece_type]
    shape = CONFIG.ui_shapes.get(piece_type, [])

    for dx, dy in shape:
        draw_cell(CONFIG, surface, offset_x + dx, offset_y + dy, color)


def draw_text(CONFIG: TetrisConfiguration, surface, text: str, cell_x: int, cell_y: int, font_size: int = None, color: tuple = None):
    font_size = font_size or CONFIG.font_size
    font = pygame.font.Font(None, font_size)
    c = color or CONFIG.text_color
    label = font.render(text, True, c)
    px = cell_x * CONFIG.cell_size
    py = cell_y * CONFIG.cell_size
    surface.blit(label, (px, py))
