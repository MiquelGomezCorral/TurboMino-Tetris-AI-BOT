import pygame
import sys

from src.tetris import Tetris, PieceEnum, ActionEnum, draw_cell, draw_ui_piece, draw_text, TetrisConfiguration

def play_tetris_game(CONFIG: TetrisConfiguration):
    pygame.init()
    screen = pygame.display.set_mode((CONFIG.screen_width, CONFIG.screen_height))
    pygame.display.set_caption("Tetris Engine")
    clock = pygame.time.Clock()

    # Initialize Engine (Must use color_map=True to render official colors)
    game = Tetris(width=CONFIG.board_w, height=CONFIG.board_h, color_map=True)

    # Gravity Timer
    GRAVITY_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(GRAVITY_EVENT, 1000) # 1 block per second at Level 1

    # DAS/ARR state: key_code -> (last_action_time, das_started)
    das_state = {}
    das_actions = {}
    action_by_key = {v: k for k, v in CONFIG.keys.items()}
    for action_name, key_code in CONFIG.keys.items():
        if action_name == "LEFT":
            das_actions[key_code] = lambda piece: game.board.move_piece_left(piece)
        elif action_name == "RIGHT":
            das_actions[key_code] = lambda piece: game.board.move_piece_right(piece)
        elif action_name == "SOFT_DROP":
            das_actions[key_code] = lambda piece: game.board.move_piece_down(piece)

    running = True
    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # --- Input Handling ---
            elif event.type == pygame.KEYDOWN:
                if event.key == CONFIG.keys.get("RESET"):
                    game = Tetris(width=CONFIG.board_w, height=CONFIG.board_h, color_map=True)
                    das_state.clear()
                    continue

                if event.key == CONFIG.keys.get("QUIT"):
                    running = False
                    continue

                if game.game_over:
                    continue

                action_name = action_by_key.get(event.key)
                if action_name is None:
                    pass
                elif action_name == "SOFT_DROP":
                    game.board.move_piece_down(game.active_piece)
                else:
                    game.move_active_piece(ActionEnum[action_name])

                if event.key in das_actions:
                    das_state[event.key] = (now, False)

            elif event.type == pygame.KEYUP:
                das_state.pop(event.key, None)

            # --- Gravity ---
            elif event.type == GRAVITY_EVENT:
                if not game.game_over:
                    game.board.move_piece_down(game.active_piece)

        # --- DAS/ARR Auto-Repeat ---
        if not game.game_over:
            for key in list(das_state.keys()):
                last_time, das_started = das_state[key]
                elapsed = now - last_time

                if not das_started:
                    if elapsed < CONFIG.das_delay:
                        continue
                    das_state[key] = (now, True)
                    das_actions[key](game.active_piece)
                    continue

                if elapsed < CONFIG.arr_rate:
                    continue

                das_state[key] = (now, True)
                das_actions[key](game.active_piece)

        # --- Rendering ---
        screen.fill(CONFIG.bg_color)

        total_h = game.board.height      # 24 (visible 20 + vanish 4)
        vis_h = game.board.visible_height

        # 1. Draw Vanish Zone Background
        vz_rect = pygame.Rect(
            CONFIG.board_offset_x * CONFIG.cell_size, 0,
            CONFIG.board_w * CONFIG.cell_size,
            (total_h - vis_h) * CONFIG.cell_size
        )
        pygame.draw.rect(screen, CONFIG.vz_color, vz_rect)

        # 2. Draw Board Stack & Grid (y=0..total_h-1)
        for y in range(total_h):
            for x in range(game.board.width):
                screen_y = (total_h - 1 - y)
                color_val = game.board.c_rows[y, x]
                if color_val != 0:
                    piece_enum = PieceEnum(color_val)
                    draw_cell(CONFIG, screen, CONFIG.board_offset_x + x, screen_y, CONFIG.colors[piece_enum])
                elif y < vis_h:
                    # Empty cell in visible area → draw grid border
                    rect = pygame.Rect(
                        (CONFIG.board_offset_x + x) * CONFIG.cell_size,
                        screen_y * CONFIG.cell_size,
                        CONFIG.cell_size, CONFIG.cell_size
                    )
                    pygame.draw.rect(screen, CONFIG.grid_color, rect, 1)

        # 3. Draw Active Piece (all rows, including vanish zone)
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
                        draw_cell(CONFIG, screen, CONFIG.board_offset_x + bx, screen_y, color)

        # 4. Draw Ghost Piece (all rows, including vanish zone)
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
                        pygame.draw.rect(screen, ghost_color, rect, 2)

        # 4. Draw Hold Piece
        hold_piece = game.get_swap_piece()
        draw_ui_piece(CONFIG, screen, hold_piece, CONFIG.hold_offset_x, 1, disabled=not game.can_hold)

        # 5. Draw Stats (Level, Lines, Combo) under hold piece
        ss = game.score_system
        if ss.combo < 0:
            combo_str = "---"
        elif ss.combo == 0:
            combo_str = "x1"
        else:
            combo_str = f"x{ss.combo + 1}"
        draw_text(CONFIG, screen, f"Level {ss.level}", CONFIG.hold_offset_x, 5)
        draw_text(CONFIG, screen, f"Lines {ss.lines_cleared_total}", CONFIG.hold_offset_x, 6)
        draw_text(CONFIG, screen, f"Combo {combo_str}", CONFIG.hold_offset_x, 7)
        draw_text(CONFIG, screen, f"B2B active {'ON' if ss.b2b_active else 'OFF'}", CONFIG.hold_offset_x, 8)
        draw_text(CONFIG, screen, f'Move: {ss.last_move_name if ss.last_move_name else "---"}', CONFIG.hold_offset_x, 9)

        # 6. Draw Next Queue (Next 5 pieces)
        next_pieces = game.get_next_pieces()[:5] # Returns list of string names
        for i, piece_name in enumerate(next_pieces):
            piece_type = PieceEnum[piece_name]
            # Space them out vertically by 3 cells
            draw_ui_piece(CONFIG, screen, piece_type, CONFIG.next_offset_x, 1 + (i * 3))

        # 7. Draw Score under next pieces
        draw_text(CONFIG, screen, f"Score {ss.score}", CONFIG.next_offset_x, 16)

        # 8. Game Over overlay
        if game.game_over:
            draw_text(CONFIG, screen, "GAME OVER", CONFIG.board_offset_x + 2, total_h + 1, font_size=50, color=CONFIG.game_over_color)
            draw_text(CONFIG, screen, "Press R to restart", CONFIG.board_offset_x + 2.5, total_h + 2.5, font_size=30, color=CONFIG.game_over_color)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()