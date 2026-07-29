import pygame
from dataclasses import dataclass, field
from maikol_utils.print_utils import print_separator

from .tetris import PieceEnum


@dataclass
class TetrisConfiguration:
    # --- Settable from CLI ---
    cell_size: int = 30
    board_w: int = 10
    board_h: int = 20
    vanish_zone: int = 5
    death_penalty: int = -250
    alive_bonus: int = 10
    max_pieces_on_queue_view: int = 5
    max_pieces_in_view: int = max_pieces_on_queue_view + 2 # + Hold + Active
    num_piece_categories: int = len(PieceEnum) - 1
    
    # --- Layout constants ---
    sidebar_cols: int = 5
    gap_cols: int = 1
    min_content_rows: int = 14
    bottom_padding_rows: int = 4

    # --- Computed layout (set in __post_init__) ---
    hold_offset_x: int = 1
    board_offset_x: int = 6
    garbage_bar_offset_x: int = 10
    next_offset_x: int = 17
    total_height: int = 24
    screen_width: int = 660
    screen_height: int = 840

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
        PieceEnum.G: (70, 70, 70),
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

    def __post_init__(self):
        self.hold_offset_x = 1
        self.board_offset_x = self.sidebar_cols + self.gap_cols
        self.garbage_bar_offset_x = self.board_offset_x - self.gap_cols
        self.next_offset_x = self.board_offset_x + self.board_w + self.gap_cols
        self.screen_width = (self.next_offset_x + self.sidebar_cols) * self.cell_size

        content_rows = max(self.board_h + self.vanish_zone, self.min_content_rows)
        self.total_height = content_rows
        self.screen_height = content_rows * self.cell_size + self.bottom_padding_rows * self.cell_size

    def print_config(self):
        print_separator("TETRIS CONFIG", sep_type="SHORT")
        for field_name, value in self.__dict__.items():
            print(f"- {field_name}: {value}")


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


def draw_garbage_bar(CONFIG: TetrisConfiguration, surface, incoming_garbage, board_height, visible_height):
    groups = {}
    for lines, turns, _ in incoming_garbage:
        groups[turns] = groups.get(turns, 0) + lines

    garbage_color = CONFIG.colors[PieceEnum.G]
    screen_y = board_height - 1
    max_screen_y = board_height - visible_height

    for turns, lines in sorted(groups.items()):
        color = CONFIG.game_over_color if turns <= 1 else garbage_color
        for _ in range(lines):
            if screen_y < max_screen_y:
                return
            draw_cell(CONFIG, surface, CONFIG.garbage_bar_offset_x, screen_y, color)
            screen_y -= 1


def draw_text(CONFIG: TetrisConfiguration, surface, text: str, cell_x: int, cell_y: int, font_size: int = None, color: tuple = None):
    font_size = font_size or CONFIG.font_size
    font = pygame.font.Font(None, font_size)
    c = color or CONFIG.text_color
    label = font.render(text, True, c)
    px = cell_x * CONFIG.cell_size
    py = cell_y * CONFIG.cell_size
    surface.blit(label, (px, py))
