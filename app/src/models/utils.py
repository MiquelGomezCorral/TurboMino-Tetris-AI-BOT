
import os

from sb3_contrib import MaskablePPO
from src.config import Configuration
from src.tetris import TetrisConfiguration
from .gym_env import TetrisEnv
from .TurboMino import TurboMinoEncoder

def load_model(CONFIG: Configuration, T_CONFIG: TetrisConfiguration, env=None, lr_schedule=None, model_path=None):
    model_path = model_path or CONFIG.final_model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f" - Loading model: {model_path}")

    env = env if env is not None else TetrisEnv(CONFIG, T_CONFIG)

    if model_path.endswith('.ckpt'):
        from .TurboMino import TurboMinoModule
        model = TurboMinoModule.load_from_checkpoint(
            model_path,
            CONFIG=CONFIG, 
            T_CONFIG=T_CONFIG, 
            observation_space=env.observation_space
        )

    else:
        learning_rate = lr_schedule if lr_schedule is not None else CONFIG.learning_rate
        model = MaskablePPO.load(
            model_path,
            env=env, 
            tensorboard_log=CONFIG.log_dir,
            custom_objects={
                "learning_rate": learning_rate,
                "n_steps": CONFIG.rollout_steps(),
                "batch_size": CONFIG.batch_size,
                "ent_coef": CONFIG.ent_coef,
                "clip_range": CONFIG.clip_range,
                "gamma": CONFIG.gamma,
                'gae_lambda': CONFIG.gae_lambda,
                "target_kl": CONFIG.target_kl,
            },
        )

    return model


def create_fresh_model(CONFIG: Configuration, T_CONFIG: TetrisConfiguration, env=None, lr_schedule=None):
    env = env if env is not None else TetrisEnv(CONFIG, T_CONFIG)
    policy_kwargs = dict(
        features_extractor_class=TurboMinoEncoder,
        features_extractor_kwargs=dict(T_CONFIG=T_CONFIG, CONFIG=CONFIG),
        net_arch=dict(pi=CONFIG.net_arch, vf=CONFIG.net_arch),
    )
    return MaskablePPO(
        "MultiInputPolicy", env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,
        n_steps=CONFIG.rollout_steps(),
        batch_size=CONFIG.batch_size,
        ent_coef=CONFIG.ent_coef,
        clip_range=CONFIG.clip_range,
        gamma=CONFIG.gamma,
        gae_lambda=CONFIG.gae_lambda,
        tensorboard_log=CONFIG.log_dir,
        verbose=0,
        target_kl=CONFIG.target_kl,
        seed=CONFIG.seed,
    )
