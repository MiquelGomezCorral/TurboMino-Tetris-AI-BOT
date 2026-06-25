# TurboMino: Tetris AI Bot

[English](#english) | [Español](#español)

---

<a name="english"></a>
## English

### About
Tetris AI bot that plays autonomously using deep reinforcement learning. Trained with PPO (Stable-Baselines3) on a custom Gymnasium environment. Uses a CNN with Rotary Position Embeddings (RoPE), bit-packed uint32 board representation, and exhaustive piece placement search via BFS. Includes an interactive Pygame mode for human play or watching the agent.

### Setup & Installation

```bash
conda create --name TETRIO_env python=3.13 -y
conda activate TETRIO_env

uv pip install -r requirements.txt
pip install -e app/
```

### Usage

```bash
python app/main.py play-tetris -W 10 -H 20     # Interactive Pygame window
python app/main.py test                          # Ad-hoc test slot
```

### Training
Launch PPO training in background with a config file from `configs/`:
```bash
bash scripts/train_ppo.sh config_curr_ppo.yaml
```
Stop training:
```bash
bash scripts/stop_training.sh config_curr_ppo.yaml
```

### Architecture
- **Tetris engine** (`src/tetris/`): Core game logic, bit-packed uint32 board rows, MoveSearcher (BFS over all piece rotations/translations)
- **Gymnasium env** (`src/models/`): `TetrisEnv` with configurable board size, padded to max dimensions
- **RL model**: CNN + Rotary Position Embeddings (RoPE) trained with PPO (Stable-Baselines3 + sb3-contrib)
- **Visualization**: Pygame interactive mode with real-time rendering

### Dataset
[Tetr.io Top Players Replays](https://www.kaggle.com/datasets/n3koasakura/tetr-io-top-players-replays)

*Maintained by [MiquelGomezCorral](https://miquelgc.net)*

---

<a name="español"></a>
## Español

### Sobre el proyecto
Bot de Tetris que juega automáticamente usando aprendizaje por refuerzo profundo. Entrenado con PPO (Stable-Baselines3) sobre un entorno Gymnasium personalizado. Usa una CNN con Rotary Position Embeddings (RoPE), representación del tablero con filas empaquetadas en uint32, y búsqueda exhaustiva de colocaciones de piezas mediante BFS. Incluye modo interactivo con Pygame para jugar o ver al agente.

### Configuración e Instalación

```bash
conda create --name TETRIO_env python=3.13 -y
conda activate TETRIO_env

uv pip install -r requirements.txt
pip install -e app/
```

### Uso

```bash
python app/main.py play-tetris -W 10 -H 20     # Ventana interactiva Pygame
python app/main.py test                          # Test ad-hoc
```

### Entrenamiento
Lanzar entrenamiento PPO en segundo plano con un config de `configs/`:
```bash
bash scripts/train_ppo.sh config_curr_ppo.yaml
```
Detener entrenamiento:
```bash
bash scripts/stop_training.sh config_curr_ppo.yaml
```

### Arquitectura
- **Motor Tetris** (`src/tetris/`): Lógica del juego, filas uint32 bit-packed, MoveSearcher (BFS sobre rotaciones/traslaciones)
- **Entorno Gymnasium** (`src/models/`): `TetrisEnv` con tamaño configurable, padding a dimensiones máximas
- **Modelo RL**: CNN + Rotary Position Embeddings (RoPE) entrenado con PPO (Stable-Baselines3 + sb3-contrib)
- **Visualización**: Modo interactivo Pygame con renderizado en tiempo real

### Dataset
[Tetr.io Top Players Replays](https://www.kaggle.com/datasets/n3koasakura/tetr-io-top-players-replays)

*Mantenido por [MiquelGomezCorral](https://miquelgc.net)*
