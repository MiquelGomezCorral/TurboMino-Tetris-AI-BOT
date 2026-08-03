# AGENTS.md — TurboMino Tetris AI BOT

## Setup

```bash
conda activate TETRIO_env                            # Python 3.13
uv pip install -r requirements.txt
pip install -e app/                                   # editable install, makes src/ + scripts/ top-level packages
```

`requirements.txt` is incomplete — missing `torch`, `gymnasium`, `stable-baselines3`, `sb3-contrib`, `numpy`, `einops`. These are pre-installed in the conda env.

## Run / Dev

```bash
python app/main.py play-tetris -W 10 -H 20            # interactive Pygame window
python app/main.py --config <config.yaml> train-ppo   # CLI: train PPO model
```

## Training Launcher

```bash
bash scripts/train_ppo.sh config_curr_ppo.yaml        # launch PPO in background (nohup)
bash scripts/stop_training.sh config_curr_ppo.yaml    # kill that background process
```

Logs and PID files go into `logs/<config_name>.log` / `logs/<config_name>.pid`.

Tests use the standard library `unittest`; there is no linter/formatter,
pre-commit, or CI configuration.

```bash
python -m unittest discover -s app/tests -v
```

## Architecture

```
app/main.py         → CLI (argparse: play-tetris, test)
app/scripts/        → Pygame game loop
app/src/tetris/     → core engine, MoveSearcher, scoring, viz
app/src/models/     → Gymnasium env (TetrisEnv), PPO training, CNN+RoPE model
app/src/config/     → Configuration dataclass (paths, RL params, max_board_size)
```

Two config classes, don't mix them:
- `Configuration` (`src/config/config.py`) — model/RL params, max board size for padding
- `TetrisConfiguration` (`src/tetris/visualization.py`) — board dimensions, rendering, keybinds, color map

## Key details

- Board rows are **bit-packed uint32** arrays (`b_rows`, `c_rows`). Unpacked by `_extract_features_2d` with bit shifts.
- `_clear_bitmap()` is a shared utility in `tetris.py` (not `algorithms.py`). Returns `(cleared_b, cleared)` without c_rows, or `(cleared_b, cleared_c, cleared)` with c_rows.
- `gym_env.py` pads rendered boards to `max_board_size_h × max_board_size_w` with `constant_values=1` (filled/wall cells). Board is centered horizontally, pinned to top.
- `app/app.egg-info/` is stale — safe to ignore or regenerate.
- Playfield test strings: `char_map` maps piece enum names (I, O, T, S, Z, J, L, G). Use `'G'` for garbage/filled cells, not `'X'`.
- `MoveSearcher.get_all_placements()` does BFS over all rotation + translation placements. Used by both the RL env and any external placement queries.
- PPO reward defaults are survival-only: `alive_reward=0.1`, `death_penalty=-5.0`, with game and heuristic rewards disabled. See `docs/scoring-and-garbage.md`.
