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


def render_game(CONFIG: TetrisConfiguration, surface, game, show_ghost: bool = True):
    """Draws the full game state (board, active piece, hold, next, stats) onto the surface."""
    total_h = game.board.height      # 24 (visible 20 + vanish 4)
    vis_h = game.board.visible_height

    # 1. Draw Vanish Zone Background
    vz_rect = pygame.Rect(
        CONFIG.board_offset_x * CONFIG.cell_size, 0,
        CONFIG.board_w * CONFIG.cell_size,
        (total_h - vis_h) * CONFIG.cell_size
    )
    pygame.draw.rect(surface, CONFIG.vz_color, vz_rect)

    # 2. Draw Board Stack & Grid (y=0..total_h-1)
    for y in range(total_h):
        for x in range(game.board.width):
            screen_y = (total_h - 1 - y)
            color_val = game.board.c_rows[y, x]
            if color_val != 0:
                piece_enum = PieceEnum(color_val)
                draw_cell(CONFIG, surface, CONFIG.board_offset_x + x, screen_y, CONFIG.colors[piece_enum])
            elif y < vis_h:
                # Empty cell in visible area → draw grid border
                rect = pygame.Rect(
                    (CONFIG.board_offset_x + x) * CONFIG.cell_size,
                    screen_y * CONFIG.cell_size,
                    CONFIG.cell_size, CONFIG.cell_size
                )
                pygame.draw.rect(surface, CONFIG.grid_color, rect, 1)

    # 3. Draw incoming garbage
    draw_garbage_bar(CONFIG, surface, tuple(game.incoming_garbage), total_h, vis_h)

    # 4. Draw Active Piece (all rows, including vanish zone)
    active = game.active_piece
    color = CONFIG.colors[active.type]
    for local_y, row_mask in enumerate(active.current_mask):
        if row_mask == 0: continue
        by = active.y + local_y
        if 0 <= by < total_h:
            screen_y = (total_h - 1 - by)
            for local_x in range(4):
                if row_mask & (1 << local_x):
                    bx = active.x + local_x
                    draw_cell(CONFIG, surface, CONFIG.board_offset_x + bx, screen_y, color)

    # 5. Draw Ghost Piece (all rows, including vanish zone)
    if show_ghost:
        ghost_y = game.board.get_ghost_y(active)
        ghost_color = [int(c * 0.2 + 255 * 0.8) for c in color]
        for local_y, row_mask in enumerate(active.current_mask):
            if row_mask == 0: continue
            by = ghost_y + local_y
            if 0 <= by < total_h:
                screen_y = (total_h - 1 - by)
                for local_x in range(4):
                    if row_mask & (1 << local_x):
                        bx = active.x + local_x
                        rect = pygame.Rect((CONFIG.board_offset_x + bx) * CONFIG.cell_size, screen_y * CONFIG.cell_size, CONFIG.cell_size, CONFIG.cell_size)
                        pygame.draw.rect(surface, ghost_color, rect, 2)

    # 6. Draw Hold Piece
    hold_piece = game.get_swap_piece()
    draw_ui_piece(CONFIG, surface, hold_piece, CONFIG.hold_offset_x, 1, disabled=not game.can_hold)

    # 7. Draw Stats (Level, Lines, Combo) under hold piece
    ss = game.score_system
    if ss.get_combo() <= 0:
        combo_str = "---"
    else:
        combo_str = f"x{ss.get_combo()}"
    draw_text(CONFIG, surface, f"Level {ss.level}", CONFIG.hold_offset_x, 5)
    draw_text(CONFIG, surface, f"Lines {ss.lines_cleared_total}", CONFIG.hold_offset_x, 6)
    draw_text(CONFIG, surface, f"Combo {combo_str}", CONFIG.hold_offset_x, 7)
    draw_text(CONFIG, surface, f"B2B active {'ON' if ss.get_b2b_active() else 'OFF'}", CONFIG.hold_offset_x, 8)
    draw_text(CONFIG, surface, f'Move: {ss.last_move_name if ss.last_move_name else "---"}', CONFIG.hold_offset_x, 9)

    live_totals = (
        ("Moves", ss.total_placements),
        ("Tetrises", ss.total_tetrises),
        ("All Clears", ss.total_all_clears),
        ("T-Spin", ss.total_t_spins["T-Spin"]),
        ("TS Single", ss.total_t_spins["T-Spin Single"]),
        ("TS Double", ss.total_t_spins["T-Spin Double"]),
        ("TS Triple", ss.total_t_spins["T-Spin Triple"]),
        ("Mini", ss.total_t_spins["T-Spin Mini"]),
        ("Mini Single", ss.total_t_spins["T-Spin Mini Single"]),
        ("Mini Double", ss.total_t_spins["T-Spin Mini Double"]),
    )
    for row, (label, count) in enumerate(live_totals, start=11):
        draw_text(CONFIG, surface, f"{label} {count}", CONFIG.hold_offset_x, row)

    # 8. Draw Next Queue (Next 5 pieces)
    next_pieces = game.get_next_pieces()[:5] # Returns list of string names
    for i, piece_name in enumerate(next_pieces):
        piece_type = PieceEnum[piece_name]
        # Space them out vertically by 3 cells
        draw_ui_piece(CONFIG, surface, piece_type, CONFIG.next_offset_x, 1 + (i * 3))

    # 9. Draw Score under next pieces
    draw_text(CONFIG, surface, f"Score {ss.score}", CONFIG.next_offset_x, 16)

    # 10. Game Over overlay
    if game.game_over:
        draw_text(CONFIG, surface, "GAME OVER", CONFIG.board_offset_x + 2, total_h + 1, font_size=50, color=CONFIG.game_over_color)
        draw_text(CONFIG, surface, "Press R to restart", CONFIG.board_offset_x + 2.5, total_h + 2.5, font_size=30, color=CONFIG.game_over_color)
