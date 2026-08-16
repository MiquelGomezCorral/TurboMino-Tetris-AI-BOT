import pygame
import sys

from src.tetris import Tetris, ActionEnum, TetrisConfiguration
from src.tetris.visualization import render_game


class HumanController:
    """Keyboard controls shared by the normal game and PvE showcase."""

    keys = {
        "LEFT": pygame.K_a, "RIGHT": pygame.K_s, "SOFT_DROP": pygame.K_r,
        "ROTATE_CCW": pygame.K_LEFT, "ROTATE_CW": pygame.K_RIGHT,
        "ROTATE_180": pygame.K_UP, "DROP": pygame.K_SPACE, "HOLD": pygame.K_w,
        "RESET": pygame.K_p, "QUIT": pygame.K_q,
    }

    def __init__(self, game):
        self.action_by_key = {v: k for k, v in self.keys.items()}
        self.repeat_keys = {
            self.keys["LEFT"], self.keys["RIGHT"], self.keys["SOFT_DROP"]
        }
        self.das_state = {}
        self.game = game

    def set_game(self, game):
        self.game = game
        self.das_state.clear()

    def handle_event(self, event, now, enabled=True):
        if event.type == pygame.KEYUP:
            self.das_state.pop(event.key, None)
            return None

        if event.type != pygame.KEYDOWN:
            return None
        if event.key == self.keys["RESET"]:
            self.das_state.clear()
            return "reset"
        if event.key == self.keys["QUIT"]:
            return "quit"
        if not enabled:
            return None
        if self.game.game_over:
            return None

        action_name = self.action_by_key.get(event.key)
        if action_name == "SOFT_DROP":
            self.game.board.move_piece_down(self.game.active_piece)
        elif action_name is not None:
            self.game.move_active_piece(ActionEnum[action_name])

        if event.key in self.repeat_keys:
            self.das_state[event.key] = (now, False)
        return "locked" if action_name == "DROP" else None

    def update(self, now, CONFIG):
        if self.game.game_over:
            return

        for key in list(self.das_state):
            last_time, das_started = self.das_state[key]
            elapsed = now - last_time
            if not das_started:
                if elapsed < CONFIG.das_delay:
                    continue
                self.das_state[key] = (now, True)
                self._repeat_action(key)
                continue
            if elapsed < CONFIG.arr_rate:
                continue
            self.das_state[key] = (now, True)
            self._repeat_action(key)

    def _repeat_action(self, key):
        if key == self.keys["LEFT"]:
            self.game.board.move_piece_left(self.game.active_piece)
        elif key == self.keys["RIGHT"]:
            self.game.board.move_piece_right(self.game.active_piece)
        elif key == self.keys["SOFT_DROP"]:
            self.game.board.move_piece_down(self.game.active_piece)


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

    controller = HumanController(game)

    running = True
    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # --- Input Handling ---
            elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
                result = controller.handle_event(event, now)
                if result == "reset":
                    game = Tetris(width=CONFIG.board_w, height=CONFIG.board_h, vanish_zone=CONFIG.vanish_zone, color_map=True)
                    controller.set_game(game)
                    continue
                if result == "quit":
                    running = False
                    continue


            # --- Gravity ---
            elif event.type == GRAVITY_EVENT:
                if not game.game_over:
                    game.board.move_piece_down(game.active_piece)

        # --- DAS/ARR Auto-Repeat ---
        controller.update(now, CONFIG)

        # --- Rendering ---
        screen.fill(CONFIG.bg_color)
        render_game(CONFIG, screen, game)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()
