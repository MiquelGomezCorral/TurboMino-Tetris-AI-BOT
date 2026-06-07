---
name: mastering-pytorch-rl-nlp
description: |
  Expert guidance for PyTorch development covering Deep Reinforcement Learning and NLP Transformers.
  This skill provides comprehensive knowledge for building RL agents with TorchRL (DQN, PPO) and
  NLP systems with HuggingFace Transformers. Use this skill when working with PyTorch 2.7+,
  implementing reinforcement learning algorithms, fine-tuning transformer models, or deploying
  ML systems to production. Includes current best practices, verified library versions (Dec 2025),
  and warnings about deprecated APIs.
version: 1.0.0
category: machine-learning
triggers:
  - pytorch
  - torch
  - torchrl
  - reinforcement learning
  - RL with pytorch
  - DQN
  - Deep Q-Network
  - PPO
  - Proximal Policy Optimization
  - policy gradient
  - huggingface
  - transformers
  - BERT
  - GPT
  - fine-tuning
  - torch.compile
  - quantization
  - TorchServe
  - gymnasium
  - gym environment
  - NLP with PyTorch
  - transformer models
  - PEFT
  - LoRA
author: Rick Hightower
license: MIT
tags:
  - pytorch
  - deep-learning
  - reinforcement-learning
  - nlp
  - transformers
  - torchrl
  - huggingface
---

# Mastering PyTorch: Deep RL and NLP

Expert guidance for PyTorch 2.7+ development covering Deep Reinforcement Learning with TorchRL and NLP with HuggingFace Transformers.

## Verified Library Versions (December 2025)

| Library | Version | Notes |
|---------|---------|-------|
| PyTorch | 2.9.1+ | Use 2.7+ minimum, CUDA 12.4 recommended |
| TorchRL | 0.10.x | GymEnv, SyncDataCollector, ClipPPOLoss |
| HuggingFace Transformers | 4.56.2+ | AutoTokenizer, Trainer, pipeline |
| Gymnasium | 1.0.0+ | **OpenAI Gym is DEPRECATED** |
| PEFT | Current | LoRA fine-tuning |
| PettingZoo | Current | Multi-agent RL |

## Deprecation Warnings

**ALWAYS avoid these deprecated patterns:**

| Deprecated | Use Instead |
|------------|-------------|
| `import gym` | `import gymnasium as gym` |
| `evaluation_strategy` in Trainer | `eval_strategy` |
| CUDA < 12.1 | CUDA 12.4 |
| `env.step()` returning 4 values | Use 5 values: `obs, reward, terminated, truncated, info` |

## Quick Reference

### Device Setup (All Platforms)

```python
import torch

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("xpu") if hasattr(torch.backends, 'xpu') and torch.backends.xpu.is_available() else
    torch.device("cpu")
)
model = model.to(device)
model = torch.compile(model)  # Optimize for speed
```

### TorchRL Quick Start (DQN)

```python
from torchrl.envs import GymEnv
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.objectives import DQNLoss, HardUpdate

env = GymEnv("CartPole-v1", device=device)
collector = SyncDataCollector(env, policy, frames_per_batch=128, total_frames=100_000)
replay_buffer = ReplayBuffer(storage=LazyTensorStorage(10_000))
loss_module = DQNLoss(value_network=qnet, loss_function="smooth_l1", delay_value=True)
```

### TorchRL Quick Start (PPO)

```python
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

loss_fn = ClipPPOLoss(actor_network=actor, critic_network=critic, clip_epsilon=0.2, entropy_coef=0.01)
advantage_fn = GAE(value_network=critic, gamma=0.99, lmbda=0.95)
```

### HuggingFace Quick Start

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Simple inference
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love PyTorch!")

# Fine-tuning
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./results", eval_strategy="epoch",  # NOT evaluation_strategy
    learning_rate=2e-5, per_device_train_batch_size=8, num_train_epochs=2, fp16=True
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
```

## Detailed Guides

For comprehensive coverage, load the appropriate guide:

| Topic | Guide | When to Load |
|-------|-------|--------------|
| Tensors, Autograd, nn.Module | `references/pytorch-fundamentals.md` | PyTorch basics, device management |
| TorchRL, DQN, PPO | `references/reinforcement-learning.md` | RL algorithms, environments |
| HuggingFace, BERT, Fine-tuning | `references/nlp-transformers.md` | NLP tasks, transformer models |
| torch.compile, Quantization, DDP | `references/optimization-deployment.md` | Production, performance |
| CLIP, RLHF, Ethics | `references/advanced-topics.md` | Multi-modal, responsible AI |

## Common Patterns

### Gymnasium Environment (Modern API)

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset()

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)  # 5 values!
    done = terminated or truncated
    if done:
        break
```

### Training Loop with AMP

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in dataloader:
    optimizer.zero_grad()
    with autocast():
        loss = compute_loss(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### PEFT/LoRA Fine-tuning

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
# Now fine-tune with much fewer parameters
```

## Installation

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# RL libraries
pip install torchrl gymnasium pettingzoo

# NLP libraries
pip install transformers datasets peft accelerate

# Experiment tracking
pip install tensorboard wandb
```

## Apple Silicon (M2/M3/M4) Support

PyTorch MPS backend enables GPU acceleration on Apple Silicon Macs.

### MPS Setup

```python
import torch
import os

# Enable MPS fallback for unsupported operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = model.to(device)
```

### MPS Limitations (Dec 2025)
- **Not all operations supported** - Some fall back to CPU
- **No distributed training** - Single GPU only
- **No float64 support** - Use float32
- **SDPA can be unstable** - Use eager attention if issues occur
- **torch.compile limited** - Test thoroughly, may need to disable
- **~3x slower than RTX 4090** - But 80% lower energy consumption

### Best for Apple Silicon
- Prototyping and development
- Light to medium training workloads
- Local inference
- Learning and experimentation

## Key Concepts

### Reinforcement Learning
- **Agent**: Learns from environment interactions
- **Environment**: Provides states, rewards (use Gymnasium, not gym)
- **Policy**: Maps states to actions (actor in PPO)
- **Value Function**: Estimates future rewards (critic in PPO)
- **Experience Replay**: Stores transitions for stable learning (DQN)
- **Target Network**: Slowly-updated copy for stable Q-targets (DQN)

### NLP/Transformers
- **Tokenizer**: Converts text to token IDs
- **Encoder**: Processes input (BERT-style)
- **Decoder**: Generates output (GPT-style)
- **Fine-tuning**: Adapts pre-trained model to specific task
- **PEFT/LoRA**: Parameter-efficient fine-tuning (fewer trainable params)

### Optimization
- **torch.compile**: JIT compilation for faster execution
- **Mixed Precision (AMP)**: fp16/bf16 for speed and memory
- **Quantization**: Reduce model size (int8)
- **DDP**: Distributed training across GPUs
- **torchrun**: Launch distributed training
