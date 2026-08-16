
"""Run a trained agent against the engine, rendered in the terminal or the Pygame UI."""

import os
import random
import sys
import time
from copy import deepcopy
from math import ceil

import pygame

from maikol_utils.print_utils import print_separator

from src.models import TetrisEnv, load_model
from src.config import Configuration
from src.tetris import Tetris, TetrisConfiguration
from scripts.play_tetris import HumanController
from src.tetris.visualization import render_game

def showcase_model(
    CONFIG: Configuration,
    T_CONFIG: TetrisConfiguration,
    use_ui: bool = False,
    pve: bool = False,
):
    """Run the trained agent. use_ui=True renders in the Pygame window, False renders in the terminal."""
    if pve:
        play_agent_pve(CONFIG, T_CONFIG)
    elif use_ui:
        play_agent_ui(CONFIG, T_CONFIG)
    else:
        play_agent_terminal(CONFIG, T_CONFIG)


def clear_terminal():
    """Clears the console for smooth animation."""
    os.system('cls' if os.name == 'nt' else 'clear')


def _run_agent(CONFIG: Configuration, T_CONFIG: TetrisConfiguration, on_frame):
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
        if T_CONFIG.showcase_delay > 0:
            time.sleep(T_CONFIG.showcase_delay)

    return game, pieces_placed, max_combo


def _print_final_stats(game, pieces_placed, max_combo):
    print("\nFinal Stats:")
    print(f"- Level: {game.get_level()}")
    print(f"- Total Score: {game.score_system.score}")
    print(f"- Lines Cleared: {game.score_system.lines_cleared_total}")
    print(f"- Pieces Placed: {pieces_placed}")
    print(f"- Max Combo: {max_combo if max_combo > 0 else '---':<5}")
    print(f"- Total All Clears: {game.score_system.total_all_clears: <5}")
    for spin_name, count in game.score_system.total_t_spins.items():
        print(f"- {spin_name}: {count:<5}")


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

    result = _run_agent(CONFIG, T_CONFIG, on_frame)
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

    result = _run_agent(CONFIG, T_CONFIG, on_frame)
    if result is None:
        pygame.quit()
        return
    game, pieces_placed, max_combo = result

    time.sleep(2)
    pygame.quit()
    _print_final_stats(game, pieces_placed, max_combo)


def play_agent_pve(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    """Run a human-versus-agent round with player-sent garbage only."""
    if not os.path.exists(CONFIG.model_path):
        print(f"[!] Model not found at {CONFIG.model_path}. Please train the model first.")
        return

    pve_config = deepcopy(CONFIG)
    pve_config.garbage_prob = 0.0
    pve_config.garbage_delay = 7

    ai_garbage_rng = random.Random()
    human_garbage_rng = random.Random()
    env = TetrisEnv(pve_config, T_CONFIG, color_map=True, garbage_rng=ai_garbage_rng)
    print(f"[*] Loading model from {pve_config.model_path}...")
    model = load_model(pve_config, T_CONFIG, env=env, model_path=pve_config.model_path)

    pygame.init()
    gap = T_CONFIG.cell_size * 2
    screen_width = T_CONFIG.screen_width * 2 + gap
    screen = pygame.display.set_mode((screen_width, T_CONFIG.screen_height))
    pygame.display.set_caption("AI Tetris PvE Showcase")
    clock = pygame.time.Clock()
    gravity_event = pygame.USEREVENT + 2
    pygame.time.set_timer(gravity_event, 1000)

    board_surfaces = [
        pygame.Surface((T_CONFIG.screen_width, T_CONFIG.screen_height)),
        pygame.Surface((T_CONFIG.screen_width, T_CONFIG.screen_height)),
    ]
    round_number = 0
    countdown_ms = 3000
    human = None
    controller = None
    ai_game = None
    obs = None
    countdown_end = 0
    ai_next_step = 0
    match_over = False
    winner = ""

    def reset_round(now):
        nonlocal human, controller, ai_game, obs, countdown_end, ai_next_step
        nonlocal match_over, winner, round_number

        round_seed = pve_config.seed + round_number
        obs, _ = env.reset(seed=round_seed)
        ai_game = env.get_game()
        human = Tetris(
            width=T_CONFIG.board_w,
            height=T_CONFIG.board_h,
            vanish_zone=T_CONFIG.vanish_zone,
            color_map=True,
            garbage_prob=0,
            garbage_delay=pve_config.garbage_delay,
            garbage_lines_probs=pve_config.garbage_lines_probs,
            garbage_cap=pve_config.garbage_cap,
            piece_seed=round_seed,
            garbage_rng=human_garbage_rng,
        )
        controller = HumanController(human)
        countdown_end = now + countdown_ms
        ai_next_step = countdown_end
        match_over = False
        winner = ""
        round_number += 1

    def route_attack(source, target):
        attack = source.last_outgoing_attack
        if attack > 0 and not target.game_over:
            target.queue_garbage(attack)
        return attack

    def finish_round():
        nonlocal match_over, winner
        if not human.game_over and not ai_game.game_over:
            return
        match_over = True
        if human.game_over and ai_game.game_over:
            winner = "ROUND OVER"
        elif human.game_over:
            winner = "AI WINS"
        else:
            winner = "PLAYER WINS"

    def draw_label(x, text):
        font = pygame.font.Font(None, 30)
        label = font.render(text, True, T_CONFIG.text_color)
        screen.blit(label, (x + T_CONFIG.cell_size, 4))

    def draw_overlay(now):
        if match_over:
            text = f"{winner}  |  Press P to restart"
        elif now < countdown_end:
            seconds = max(1, ceil((countdown_end - now) / 1000))
            text = str(seconds)
        else:
            text = ""
        if not text:
            return
        font = pygame.font.Font(None, 52 if match_over else 90)
        label = font.render(text, True, T_CONFIG.game_over_color)
        rect = label.get_rect(center=(screen_width // 2, T_CONFIG.screen_height // 2))
        screen.blit(label, rect)

    reset_round(pygame.time.get_ticks())
    running = True
    while running:
        now = pygame.time.get_ticks()
        playing = now >= countdown_end and not match_over

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            result = controller.handle_event(event, now, enabled=playing)
            if result == "quit":
                running = False
            elif result == "reset":
                reset_round(now)
                playing = False
            elif result == "locked" and playing:
                route_attack(human, ai_game)
                obs = env.refresh_observation()
                finish_round()

            if event.type == gravity_event and playing and not human.game_over:
                human.board.move_piece_down(human.active_piece)

        if playing and not match_over:
            controller.update(now, T_CONFIG)
            if now >= ai_next_step and not ai_game.game_over:
                action_masks = env.valid_action_mask()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                route_attack(ai_game, human)
                finish_round()
                if not (terminated or truncated) and not match_over:
                    ai_next_step = now + max(0, int(T_CONFIG.showcase_delay * 1000))

        for surface, game in zip(board_surfaces, (human, ai_game)):
            surface.fill(T_CONFIG.bg_color)
            render_game(T_CONFIG, surface, game, show_game_over=False)

        screen.fill(T_CONFIG.bg_color)
        screen.blit(board_surfaces[0], (0, 0))
        screen.blit(board_surfaces[1], (T_CONFIG.screen_width + gap, 0))
        draw_label(0, "PLAYER")
        draw_label(T_CONFIG.screen_width + gap, "AI")
        draw_overlay(now)
        pygame.display.flip()
        clock.tick(60)

    pygame.time.set_timer(gravity_event, 0)
    pygame.quit()
