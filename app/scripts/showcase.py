import os
import time
from sb3_contrib import MaskablePPO

# Import your environment and configs
from src.models import TetrisEnv 
from src.config import Configuration
from src.tetris import TetrisConfiguration

def clear_terminal():
    """Clears the console for smooth animation."""
    os.system('cls' if os.name == 'nt' else 'clear')

def showcase_model(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    # --- 1. Configurations ---
    if not os.path.exists(CONFIG.model_path):
        print(f"[!] Model not found at {CONFIG.model_path}. Please train the model first.")
        return

    # --- 2. Environment Initialization ---
    # We do not strictly need the ActionMasker wrapper here because we will 
    # manually pass the mask to the predict() function.
    env = TetrisEnv(CONFIG, T_CONFIG)

    # --- 3. Load the Model ---
    print(f"[*] Loading model from {CONFIG.model_path}...")
    model = MaskablePPO.load(CONFIG.model_path)
    print("[*] Model loaded successfully. Starting showcase in 2 seconds...")
    time.sleep(2)

    # --- 4. Game Loop ---
    obs, _ = env.reset()
    done = False
    pieces_placed = 0

    while not done:
        clear_terminal()
        
        # Access the underlying engine to print the board
        game = env.unwrapped.game
        
        # Header Stats
        print("="*30)
        print(f"  AI TETRIS SHOWCASE  |  Pieces: {pieces_placed}")
        print(f"  Score: {game.score_system.score}  |  Lines: {game.score_system.lines_cleared_total}  |  Level: {game.score_system.level}")
        print("="*30)
        
        # Print the board state
        game.print_state(include_vanish_zone=False)
        
        # 1. Get the valid action mask for the current state
        action_masks = env.unwrapped.valid_action_mask()
        
        # 2. Predict the best move deterministically
        # deterministic=True ensures the AI plays its absolute best rather than exploring
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        
        # 3. Execute the chosen placement
        obs, reward, terminated, truncated, info = env.step(action)
        
        pieces_placed += 1
        done = terminated or truncated
        
        # Sleep to make the console animation viewable (adjust for speed)
        time.sleep(0.15) 

    # --- 5. Game Over ---
    clear_terminal()
    game = env.unwrapped.game
    print("="*30)
    print("          GAME OVER")
    print("="*30)
    game.print_state(include_vanish_zone=False)
    print("\nFinal Stats:")
    print(f"- Total Score: {game.score_system.score}")
    print(f"- Lines Cleared: {game.score_system.lines_cleared_total}")
    print(f"- Pieces Placed: {pieces_placed}")
    print(f"- Level: {game.get_level()}")