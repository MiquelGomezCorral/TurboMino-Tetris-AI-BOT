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
    max_pieces_on_queue_view: int = 5
    max_pieces_in_view: int = max_pieces_on_queue_view + 2  # + Hold + Active
    num_piece_categories: int = len(PieceEnum) - 1
    showcase_delay: float = 0.01

    # --- Layout constants ---
    sidebar_cols: int = 5
    gap_cols: int = 1
    min_content_rows: int = 14
    bottom_padding_rows: int = 4

    # --- Computed layout (set in __post_init__) ---
    hold_offset_x: int = 1
    board_offset_x: int = 6
    garbage_bar_offset_x: int = -1
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
        self.garbage_bar_offset_x += self.board_offset_x - self.gap_cols 
        self.next_offset_x = self.board_offset_x + self.board_w + self.gap_cols
        self.screen_width = (self.next_offset_x + self.sidebar_cols) * self.cell_size

        content_rows = max(self.board_h + self.vanish_zone, self.min_content_rows)
        self.total_height = content_rows
        self.screen_height = content_rows * self.cell_size + self.bottom_padding_rows * self.cell_size

    def print_config(self):
        print_separator("TETRIS CONFIG", sep_type="SHORT")
        for field_name, value in self.__dict__.items():
            print(f"- {field_name}: {value}")
