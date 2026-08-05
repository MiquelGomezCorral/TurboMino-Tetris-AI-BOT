
from functools import partial
import os

from gymnasium import spaces
import numpy as np
from torch import nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.torch_layers import MlpExtractor
from src.config import Configuration
from src.tetris import TetrisConfiguration
from .gym_env import TetrisEnv
from .TurboMino import TurboMinoEncoder


class PlacementLogits(nn.Module):
    """Applies one shared logit head to every placement feature."""

    def __init__(self, features_dim: int, n_actions: int):
        super().__init__()
        if features_dim % n_actions:
            raise ValueError("features_dim must be divisible by the action count")
        self.n_actions = n_actions
        per_placement = features_dim // n_actions
        self.head = nn.Linear(per_placement, 1)

    def forward(self, features):
        placements = features.reshape(features.shape[0], self.n_actions, -1)
        return self.head(placements).squeeze(-1)


class AfterstateActorCriticPolicy(MaskableActorCriticPolicy):
    """Uses shared afterstate scores directly as discrete-action logits."""

    def _build_mlp_extractor(self) -> None:
        critic_arch = self.net_arch.get("vf", []) if isinstance(self.net_arch, dict) else self.net_arch
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=dict(pi=[], vf=critic_arch),
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()
        self.action_net = PlacementLogits(self.features_dim, self.action_space.n)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )


def _only_board_dtype_differs(saved_space, current_space) -> bool:
    if not isinstance(saved_space, spaces.Dict) or not isinstance(current_space, spaces.Dict):
        return False
    if saved_space.spaces.keys() != current_space.spaces.keys():
        return False

    for key in saved_space.spaces:
        saved_subspace = saved_space[key]
        current_subspace = current_space[key]
        if key == "boards":
            if not isinstance(saved_subspace, spaces.Box) or not isinstance(current_subspace, spaces.Box):
                return False
            if saved_subspace.shape != current_subspace.shape:
                return False
            if not np.array_equal(saved_subspace.low, current_subspace.low):
                return False
            if not np.array_equal(saved_subspace.high, current_subspace.high):
                return False
            if saved_subspace.dtype == current_subspace.dtype:
                return False
        elif saved_subspace != current_subspace:
            return False

    return True


def _attach_compact_env(model, env):
    wrapped_env = model._wrap_env(env, model.verbose)
    model.observation_space = wrapped_env.observation_space
    model.policy.observation_space = wrapped_env.observation_space
    model.n_envs = wrapped_env.num_envs
    model.set_env(wrapped_env)
    model.rollout_buffer = model.rollout_buffer_class(
        model.n_steps,
        model.observation_space,
        model.action_space,
        model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
        **model.rollout_buffer_kwargs,
    )


def load_model(CONFIG: Configuration, T_CONFIG: TetrisConfiguration, env=None, lr_schedule=None, model_path=None):
    model_path = model_path or CONFIG.final_model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
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
        custom_objects = {
            "learning_rate": learning_rate,
            "n_steps": CONFIG.rollout_steps(),
            "batch_size": CONFIG.batch_size,
            "n_epochs": CONFIG.n_epochs,
            "ent_coef": CONFIG.ent_coef,
            "clip_range": CONFIG.clip_range,
            "gamma": CONFIG.gamma,
            'gae_lambda': CONFIG.gae_lambda,
            "target_kl": CONFIG.target_kl,
        }
        try:
            model = MaskablePPO.load(
                model_path,
                env=env,
                tensorboard_log=CONFIG.log_dir,
                custom_objects=custom_objects,
            )
        except ValueError:
            saved_data, _, _ = load_from_zip_file(model_path, device="cpu")
            if not _only_board_dtype_differs(saved_data["observation_space"], env.observation_space):
                raise
            model = MaskablePPO.load(
                model_path,
                env=None,
                tensorboard_log=CONFIG.log_dir,
                custom_objects=custom_objects,
            )
            _attach_compact_env(model, env)

    return model


def create_fresh_model(CONFIG: Configuration, T_CONFIG: TetrisConfiguration, env=None, lr_schedule=None):
    env = env if env is not None else TetrisEnv(CONFIG, T_CONFIG)
    policy_kwargs = dict(
        features_extractor_class=TurboMinoEncoder,
        features_extractor_kwargs=dict(T_CONFIG=T_CONFIG, CONFIG=CONFIG),
        net_arch=dict(vf=CONFIG.net_arch),
    )
    return MaskablePPO(
        AfterstateActorCriticPolicy, env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,
        n_steps=CONFIG.rollout_steps(),
        batch_size=CONFIG.batch_size,
        n_epochs=CONFIG.n_epochs,
        ent_coef=CONFIG.ent_coef,
        clip_range=CONFIG.clip_range,
        gamma=CONFIG.gamma,
        gae_lambda=CONFIG.gae_lambda,
        tensorboard_log=CONFIG.log_dir,
        verbose=0,
        target_kl=CONFIG.target_kl,
        seed=CONFIG.seed,
    )
