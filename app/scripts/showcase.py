
"""Run a trained agent against the engine, rendered in the terminal or the Pygame UI."""

import os
import sys
import time

import pygame

from maikol_utils.print_utils import print_separator

from src.models import TetrisEnv, load_model
from src.config import Configuration
from src.tetris import TetrisConfiguration
from src.tetris.visualization import render_game


from src.config import Configuration
from src.tetris import TetrisConfiguration

def showcase_model(CONFIG: Configuration, T_CONFIG: TetrisConfiguration, use_ui: bool = False):
    """Run the trained agent. use_ui=True renders in the Pygame window, False renders in the terminal."""
    if use_ui:
        play_agent_ui(CONFIG, T_CONFIG)
    else:
        play_agent_terminal(CONFIG, T_CONFIG)


def clear_terminal():
    """Clears the console for smooth animation."""
    os.system('cls' if os.name == 'nt' else 'clear')


def _run_agent(CONFIG: Configuration, T_CONFIG: TetrisConfiguration, on_frame, delay: float):
    """Shared agent driver: loads the model and steps the env, calling on_frame(game, pieces_placed) before each move."""
    if not os.path.exists(CONFIG.model_path):
        print(f"[!] Model not found at {CONFIG.model_path}. Please train the model first.")
        return None

    print(f"[*] Loading model from {CONFIG.model_path}...")
    env = TetrisEnv(CONFIG, T_CONFIG, color_map=True)
    model = load_model(CONFIG, T_CONFIG, env=env, model_path=CONFIG.model_path)

    obs, _ = env.reset()
    done = False
    pieces_placed = 0
    max_combo = 0

    while not done:
        game = env.get_game()
        on_frame(game, pieces_placed)

        action_masks = env.unwrapped.valid_action_mask()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        max_combo = max(max_combo, game.score_system.combo)

        pieces_placed += 1
        done = terminated or truncated
        time.sleep(delay)

    return game, pieces_placed, max_combo


def _print_final_stats(game, pieces_placed, max_combo):
    print("\nFinal Stats:")
    print(f"- Level: {game.get_level()}")
    print(f"- Total Score: {game.score_system.score}")
    print(f"- Lines Cleared: {game.score_system.lines_cleared_total}")
    print(f"- Pieces Placed: {pieces_placed}")
    print(f"- Max Combo: {max_combo if max_combo > 0 else '---':<5}")
    print(f"- Total All Clears: {game.score_system.total_all_clears: <5}")


def play_agent_terminal(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    """Runs the agent with console output (default showcase behavior)."""
    print("[*] Starting showcase in 2 seconds...")
    time.sleep(2)

    def on_frame(game, pieces_placed):
        clear_terminal()
        print_separator(f"AI TETRIS SHOWCASE  |  Pieces: {pieces_placed}", sep_type="SHORT")
        print(f"Score: {game.score_system.score:<10}  |  Lines: {game.score_system.lines_cleared_total:<10}  |  Level: {game.score_system.level}")
        print_separator("", sep_type="SHORT")
        game.print_state(include_vanish_zone=True)

    result = _run_agent(CONFIG, T_CONFIG, on_frame, delay=0.01)
    if result is None:
        return
    game, pieces_placed, max_combo = result

    clear_terminal()
    print_separator("GAME OVER", sep_type="START")
    game.print_state(include_vanish_zone=False)
    _print_final_stats(game, pieces_placed, max_combo)


def play_agent_ui(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    """Runs the agent rendered in the Pygame window."""
    pygame.init()
    screen = pygame.display.set_mode((T_CONFIG.screen_width, T_CONFIG.screen_height))
    pygame.display.set_caption("AI Tetris Showcase")
    clock = pygame.time.Clock()

    def on_frame(game, pieces_placed):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        screen.fill(T_CONFIG.bg_color)
        render_game(T_CONFIG, screen, game)
        pygame.display.flip()
        clock.tick(60)

    result = _run_agent(CONFIG, T_CONFIG, on_frame, delay=0.05)
    if result is None:
        pygame.quit()
        return
    game, pieces_placed, max_combo = result

    time.sleep(2)
    pygame.quit()
    _print_final_stats(game, pieces_placed, max_combo)
