# Reinforcement Learning with TorchRL Reference

Comprehensive guide to Deep Reinforcement Learning using TorchRL 0.10.x, Gymnasium 1.0+, DQN, and PPO.

## Critical: Use Gymnasium, NOT OpenAI Gym

**OpenAI Gym is DEPRECATED.** Always use Gymnasium:

```python
# ❌ WRONG - Deprecated
import gym
env = gym.make("CartPole-v1")

# ✅ CORRECT - Use Gymnasium
import gymnasium as gym
env = gym.make("CartPole-v1")
```

### Gymnasium API (5 Return Values)

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset()

while True:
    action = env.action_space.sample()
    # 5 return values - NOT 4!
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        break

env.close()
```

## TorchRL Overview

TorchRL is PyTorch's official RL library providing:
- Environment wrappers (GymEnv, PettingZooEnv)
- Data collectors (SyncDataCollector, MultiSyncDataCollector)
- Replay buffers with tensor storage
- Loss modules (DQNLoss, ClipPPOLoss)
- Advantage estimation (GAE)

### Installation

```bash
pip install torchrl gymnasium
```

## TorchRL Environment Wrappers

### GymEnv - Single Environment

```python
from torchrl.envs import GymEnv
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create environment
env = GymEnv("CartPole-v1", device=device)

# Get specs
print(f"Observation spec: {env.observation_spec}")
print(f"Action spec: {env.action_spec}")

# Reset and step
td = env.reset()
print(f"Initial state: {td}")

action = env.action_spec.rand()
td = env.step(td.set("action", action))
print(f"Next state: {td}")
```

### Parallel Environments

```python
from torchrl.envs import ParallelEnv, GymEnv

# Run 8 environments in parallel
def make_env():
    return GymEnv("CartPole-v1")

env = ParallelEnv(8, make_env)
td = env.reset()  # Shape: [8, ...]
```

### TransformedEnv - Preprocessing

```python
from torchrl.envs import TransformedEnv, GymEnv
from torchrl.envs.transforms import (
    RewardSum, StepCounter, TransformedEnv,
    ObservationNorm, RewardClipping
)

base_env = GymEnv("CartPole-v1")
env = TransformedEnv(
    base_env,
    transform=Compose(
        ObservationNorm(in_keys=["observation"]),
        RewardClipping(-1, 1),
        StepCounter(),
        RewardSum(),
    )
)
```

## Deep Q-Networks (DQN)

DQN uses a neural network to approximate Q-values with experience replay and target networks for stability.

### Q-Network Architecture

```python
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# Wrap for TorchRL
qnet = QNetwork(state_dim=4, action_dim=2)
qvalue_module = TensorDictModule(
    qnet,
    in_keys=["observation"],
    out_keys=["action_value"]
)
```

### DQN with TorchRL

```python
import torch
from torchrl.envs import GymEnv
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.modules import QValueActor, EGreedyModule
from torchrl.objectives import DQNLoss, HardUpdate
from tensordict.nn import TensorDictModule, TensorDictSequential

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment
env = GymEnv("CartPole-v1", device=device)
state_dim = env.observation_spec["observation"].shape[-1]
action_dim = env.action_spec.space.n

# Q-Network
qnet = nn.Sequential(
    nn.Linear(state_dim, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, action_dim)
)
qvalue_module = TensorDictModule(
    qnet, in_keys=["observation"], out_keys=["action_value"]
)

# Actor with epsilon-greedy exploration
actor = QValueActor(qvalue_module, spec=env.action_spec)
exploration_module = EGreedyModule(
    spec=env.action_spec,
    eps_init=1.0,
    eps_end=0.05,
    annealing_num_steps=10000
)
policy = TensorDictSequential(actor, exploration_module)

# Replay buffer
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=100_000, device=device),
    batch_size=64
)

# Data collector
collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=128,
    total_frames=100_000,
    device=device
)

# Loss module
loss_module = DQNLoss(
    value_network=actor,
    loss_function="smooth_l1",
    delay_value=True
)
loss_module.make_value_estimator(gamma=0.99)

# Target network updater
target_updater = HardUpdate(
    loss_module,
    value_network_update_interval=1000
)

# Optimizer
optimizer = torch.optim.Adam(loss_module.parameters(), lr=1e-3)
```

### DQN Training Loop

```python
for i, data in enumerate(collector):
    # Add to replay buffer
    replay_buffer.extend(data)

    if len(replay_buffer) < 1000:
        continue

    # Sample batch
    batch = replay_buffer.sample()

    # Compute loss
    loss_dict = loss_module(batch)
    loss = loss_dict["loss"]

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
    optimizer.step()

    # Update target network
    target_updater.step()

    # Update exploration epsilon
    exploration_module.step(data.numel())

    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss.item():.4f}")
```

## Proximal Policy Optimization (PPO)

PPO is an on-policy algorithm using clipped surrogate objective for stable training.

### Actor-Critic Architecture

```python
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator
from torch.distributions import Categorical

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.policy_head(features), self.value_head(features)
```

### PPO with TorchRL

```python
import torch
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import RewardSum, StepCounter
from torchrl.collectors import SyncDataCollector
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment with transforms
base_env = GymEnv("CartPole-v1", device=device)
env = TransformedEnv(base_env, StepCounter())

state_dim = env.observation_spec["observation"].shape[-1]
action_dim = env.action_spec.space.n

# Actor network
actor_net = nn.Sequential(
    nn.Linear(state_dim, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, action_dim)
)
actor_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["logits"]
)
actor = ProbabilisticActor(
    actor_module,
    in_keys=["logits"],
    out_keys=["action"],
    distribution_class=torch.distributions.Categorical,
    return_log_prob=True
)

# Critic network
critic_net = nn.Sequential(
    nn.Linear(state_dim, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 1)
)
critic = ValueOperator(
    TensorDictModule(critic_net, in_keys=["observation"], out_keys=["state_value"])
)

# Move to device
actor = actor.to(device)
critic = critic.to(device)

# Advantage estimation (GAE)
advantage_module = GAE(
    gamma=0.99,
    lmbda=0.95,
    value_network=critic,
    average_gae=True
)

# PPO Loss
loss_module = ClipPPOLoss(
    actor_network=actor,
    critic_network=critic,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    critic_coef=0.5,
    normalize_advantage=True
)

# Data collector
collector = SyncDataCollector(
    env,
    actor,
    frames_per_batch=2048,
    total_frames=100_000,
    device=device
)

# Optimizer
optimizer = torch.optim.Adam(loss_module.parameters(), lr=3e-4)
```

### PPO Training Loop

```python
num_epochs = 10  # PPO epochs per batch

for batch_idx, data in enumerate(collector):
    # Compute advantages
    with torch.no_grad():
        advantage_module(data)

    # PPO update epochs
    for epoch in range(num_epochs):
        # Mini-batch updates
        for minibatch in data.split(256):
            loss_dict = loss_module(minibatch)

            loss = (
                loss_dict["loss_objective"] +
                loss_dict["loss_critic"] +
                loss_dict["loss_entropy"]
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 0.5)
            optimizer.step()

    # Logging
    episode_reward = data["next", "reward"].sum().item()
    print(f"Batch {batch_idx}, Reward: {episode_reward:.2f}")
```

## Replay Buffers

### Basic Replay Buffer

```python
from torchrl.data import ReplayBuffer, LazyTensorStorage

buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=100_000),
    batch_size=64
)

# Add data
buffer.extend(data)

# Sample
batch = buffer.sample()
```

### Prioritized Experience Replay

```python
from torchrl.data import PrioritizedReplayBuffer, LazyTensorStorage

buffer = PrioritizedReplayBuffer(
    storage=LazyTensorStorage(max_size=100_000),
    alpha=0.6,
    beta=0.4,
    batch_size=64
)
```

## Multi-Agent RL with PettingZoo

```python
from torchrl.envs import PettingZooEnv

# Cooperative/competitive environments
env = PettingZooEnv(
    env="simple_spread_v3",
    parallel=True
)

td = env.reset()
print(f"Agents: {env.agents}")
```

## Distributed Training

### Parallel Data Collection

```python
from torchrl.collectors import MultiSyncDataCollector

collector = MultiSyncDataCollector(
    [make_env] * 4,  # 4 parallel collectors
    policy,
    frames_per_batch=512,
    total_frames=100_000,
    device=device
)
```

### Multi-GPU with torchrun

```bash
# Launch distributed PPO
torchrun --nproc_per_node=4 train_ppo.py
```

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend='nccl')
local_rank = int(os.environ["LOCAL_RANK"])
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

## Key Hyperparameters

### DQN Hyperparameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| learning_rate | 1e-4 to 1e-3 | Adam learning rate |
| gamma | 0.99 | Discount factor |
| epsilon_start | 1.0 | Initial exploration rate |
| epsilon_end | 0.01-0.05 | Final exploration rate |
| epsilon_decay | 10000-50000 steps | Annealing schedule |
| buffer_size | 100_000 - 1_000_000 | Replay buffer capacity |
| batch_size | 32-128 | Training batch size |
| target_update | 1000-10000 steps | Target network update frequency |

### PPO Hyperparameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| learning_rate | 3e-4 | Adam learning rate |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.95 | GAE lambda |
| clip_epsilon | 0.2 | PPO clip range |
| entropy_coef | 0.01 | Entropy bonus coefficient |
| value_coef | 0.5 | Value loss coefficient |
| max_grad_norm | 0.5 | Gradient clipping |
| num_epochs | 4-10 | PPO epochs per batch |
| mini_batch_size | 64-256 | Mini-batch size |
| frames_per_batch | 2048 | Rollout length |

## Best Practices

1. **Always use Gymnasium** - OpenAI Gym is deprecated
2. **Start simple** - CartPole before Atari
3. **Monitor training** - Use TensorBoard or W&B
4. **Tune one thing at a time** - Systematic hyperparameter search
5. **Use vectorized environments** - Faster data collection
6. **Clip gradients** - Prevents training instability
7. **Normalize observations** - Helps neural network learning
8. **Use torch.compile** - Faster training (CUDA only)

## Common Issues

### NaN Loss
- Check for division by zero in advantage normalization
- Verify reward scaling is reasonable
- Ensure learning rate isn't too high

### No Learning
- Increase exploration (epsilon/entropy)
- Check environment reward structure
- Verify network is receiving gradients

### Unstable Training
- Reduce learning rate
- Increase batch size
- Adjust clip range (PPO)
- Use target network (DQN)
