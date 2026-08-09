import pygame
import sys

from src.tetris import Tetris, ActionEnum, TetrisConfiguration
from src.tetris.visualization import render_game

def play_tetris_game(CONFIG: TetrisConfiguration):
    pygame.init()
    screen = pygame.display.set_mode((CONFIG.screen_width, CONFIG.screen_height))
    pygame.display.set_caption("Tetris Engine")
    clock = pygame.time.Clock()

    # Initialize Engine (Must use color_map=True to render official colors)
    game = Tetris(width=CONFIG.board_w, height=CONFIG.board_h, vanish_zone=CONFIG.vanish_zone, color_map=True)

    # Gravity Timer
    GRAVITY_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(GRAVITY_EVENT, 1000) # 1 block per second at Level 1

    # DAS/ARR state: key_code -> (last_action_time, das_started)
    das_state = {}
    das_actions = {}
    keys = {
        "LEFT": pygame.K_a, "RIGHT": pygame.K_s, "SOFT_DROP": pygame.K_r,
        "ROTATE_CCW": pygame.K_LEFT, "ROTATE_CW": pygame.K_RIGHT,
        "ROTATE_180": pygame.K_UP, "DROP": pygame.K_SPACE, "HOLD": pygame.K_w,
        "RESET": pygame.K_p, "QUIT": pygame.K_q,
    }
    action_by_key = {v: k for k, v in keys.items()}
    for action_name, key_code in keys.items():
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
                if event.key == keys["RESET"]:
                    game = Tetris(width=CONFIG.board_w, height=CONFIG.board_h, vanish_zone=CONFIG.vanish_zone, color_map=True)
                    das_state.clear()
                    continue

                if event.key == keys["QUIT"]:
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
        render_game(CONFIG, screen, game)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()
