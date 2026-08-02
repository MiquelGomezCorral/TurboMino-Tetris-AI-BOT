import pygame

from .tetris import PieceEnum
from .configuration import TetrisConfiguration


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
