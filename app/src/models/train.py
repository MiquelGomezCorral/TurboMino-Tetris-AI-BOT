from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker

from src.config import Configuration
from src.models.gym_env import TetrisEnv
from src.models.TurboMino import TetrisFeatureExtractor

import gymnasium as gym
import numpy as np

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()


def train_ppo(CONFIG: Configuration):
    # 1. Create Env and Wrap it with ActionMasker
    env = TetrisEnv()
    env = ActionMasker(env, mask_fn)

    # 2. Initialize Maskable PPO
    model = MaskablePPO("MlpPolicy", env, verbose=1)

    # 3. Train
    model.learn(total_timesteps=CONFIG.total_timesteps)

    model.save("tetris_ppo_bot")



    policy_kwargs = dict(
        features_extractor_class=TetrisFeatureExtractor,
        features_extractor_kwargs=dict(
            max_placements=CONFIG.max_placements, 
            features_dim=CONFIG.max_placements * 16
        ),
        net_arch=[128, 128] 
    )

    model = MaskablePPO(
        "MultiInputPolicy", # CRITICAL: Change this from MlpPolicy
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=1
    )