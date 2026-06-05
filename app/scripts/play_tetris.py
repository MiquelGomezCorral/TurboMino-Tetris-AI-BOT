import pygame
import sys

from src.tetris import Tetris, PieceEnum, ActionEnum, draw_cell, draw_ui_piece, draw_text, TetrisConfiguration

def play_tetris_game():
    CONFIG = TetrisConfiguration()  # Use dataclass for constants and config
    pygame.init()
    screen = pygame.display.set_mode((CONFIG.SCREEN_WIDTH, CONFIG.SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris Engine")
    clock = pygame.time.Clock()

    # Initialize Engine (Must use color_map=True to render official colors)
    game = Tetris(width=CONFIG.BOARD_W, height=CONFIG.BOARD_H, color_map=True)

    # Gravity Timer
    GRAVITY_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(GRAVITY_EVENT, 1000) # 1 block per second at Level 1

    # DAS/ARR state: key_code -> (last_action_time, das_started)
    das_state = {}
    das_actions = {}
    for k, v in CONFIG.KEYS.items():
        if v == "LEFT":
            das_actions[k] = lambda piece: game.board.move_piece_left(piece)
        elif v == "RIGHT":
            das_actions[k] = lambda piece: game.board.move_piece_right(piece)
        elif v == "SOFT_DROP":
            das_actions[k] = lambda piece: game.board.move_piece_down(piece)

    running = True
    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # --- Input Handling ---
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    game = Tetris(width=CONFIG.BOARD_W, height=CONFIG.BOARD_H, color_map=True)
                    das_state.clear()
                    continue

                if event.key == pygame.K_q:
                    running = False
                    continue

                action_name = CONFIG.KEYS.get(event.key)
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
                # If it hits the floor/stack during gravity, let it sit (lock delay mechanics normally apply here)
                game.board.move_piece_down(game.active_piece)

        # --- DAS/ARR Auto-Repeat ---
        for key in list(das_state.keys()):
            last_time, das_started = das_state[key]
            elapsed = now - last_time

            if not das_started:
                if elapsed < CONFIG.DAS_DELAY:
                    continue
                das_state[key] = (now, True)
                das_actions[key](game.active_piece)
                continue

            if elapsed < CONFIG.ARR_RATE:
                continue

            das_state[key] = (now, True)
            das_actions[key](game.active_piece)

        # --- Rendering ---
        screen.fill(CONFIG.BG_COLOR)

        total_h = game.board.height      # 24 (visible 20 + vanish 4)
        vis_h = game.board.visible_height

        # 1. Draw Vanish Zone Background
        vz_rect = pygame.Rect(
            CONFIG.BOARD_OFFSET_X * CONFIG.CELL_SIZE, 0,
            CONFIG.BOARD_W * CONFIG.CELL_SIZE,
            (total_h - vis_h) * CONFIG.CELL_SIZE
        )
        pygame.draw.rect(screen, CONFIG.VZ_COLOR, vz_rect)

        # 2. Draw Board Stack & Grid (y=0..total_h-1)
        for y in range(total_h):
            for x in range(game.board.width):
                screen_y = (total_h - 1 - y)
                color_val = game.board.c_rows[y, x]
                if color_val != 0:
                    piece_enum = PieceEnum(color_val)
                    draw_cell(CONFIG, screen, CONFIG.BOARD_OFFSET_X + x, screen_y, CONFIG.COLORS[piece_enum])
                elif y < vis_h:
                    # Empty cell in visible area → draw grid border
                    rect = pygame.Rect(
                        (CONFIG.BOARD_OFFSET_X + x) * CONFIG.CELL_SIZE,
                        screen_y * CONFIG.CELL_SIZE,
                        CONFIG.CELL_SIZE, CONFIG.CELL_SIZE
                    )
                    pygame.draw.rect(screen, CONFIG.GRID_COLOR, rect, 1)

        # 3. Draw Active Piece (all rows, including vanish zone)
        active = game.active_piece
        color = CONFIG.COLORS[active.type]
        for local_y, row_mask in enumerate(active.current_mask):
            if row_mask == 0: continue
            by = active.y + local_y
            if 0 <= by < total_h:
                screen_y = (total_h - 1 - by)
                for local_x in range(4):
                    if row_mask & (1 << local_x):
                        bx = active.x + local_x
                        draw_cell(CONFIG, screen, CONFIG.BOARD_OFFSET_X + bx, screen_y, color)

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
                        rect = pygame.Rect((CONFIG.BOARD_OFFSET_X + bx) * CONFIG.CELL_SIZE, screen_y * CONFIG.CELL_SIZE, CONFIG.CELL_SIZE, CONFIG.CELL_SIZE)
                        pygame.draw.rect(screen, ghost_color, rect, 2)

        # 4. Draw Hold Piece
        hold_piece = game.get_swap_piece()
        draw_ui_piece(CONFIG, screen, hold_piece, CONFIG.HOLD_OFFSET_X, 1, disabled=not game.can_hold)

        # 5. Draw Stats (Level, Lines, Combo) under hold piece
        ss = game.score_system
        if ss.combo < 0:
            combo_str = "---"
        elif ss.combo == 0:
            combo_str = "x1"
        else:
            combo_str = f"x{ss.combo + 1}"
        draw_text(CONFIG, screen, f"Level {ss.level}", CONFIG.HOLD_OFFSET_X, 5)
        draw_text(CONFIG, screen, f"Lines {ss.lines_cleared_total}", CONFIG.HOLD_OFFSET_X, 6)
        draw_text(CONFIG, screen, f"Combo {combo_str}", CONFIG.HOLD_OFFSET_X, 7)
        draw_text(CONFIG, screen, f"B2B active {'ON' if ss.b2b_active else 'OFF'}", CONFIG.HOLD_OFFSET_X, 8)

        # 6. Draw Next Queue (Next 5 pieces)
        next_pieces = game.get_next_pieces()[:5] # Returns list of string names
        for i, piece_name in enumerate(next_pieces):
            piece_type = PieceEnum[piece_name]
            # Space them out vertically by 3 cells
            draw_ui_piece(CONFIG, screen, piece_type, CONFIG.NEXT_OFFSET_X, 1 + (i * 3))

        # 7. Draw Score under next pieces
        draw_text(CONFIG, screen, f"Score {ss.score}", CONFIG.NEXT_OFFSET_X, 16)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()