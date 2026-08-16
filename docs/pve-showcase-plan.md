# PvE Showcase Implementation Contract

This document records the approved scope for the `showcase --pve` mode.

## Rules

- PvE renders a human board and a trained-agent board in one Pygame window.
- A three-second countdown runs before each round and after each reset.
- The round ends as soon as either board tops out.
- `P` resets both boards and starts the countdown again.
- Human input is processed every frame. The AI uses `showcase_delay` between placements.
- PvE games use `garbage_prob=0` and `garbage_delay=7`.
- `garbage_prob=0` disables only random garbage generation. Externally queued garbage still ages, cancels, and rises normally.
- A placement's attack first cancels the player's pending incoming garbage. Any remainder is sent to the opponent.
- Both players receive the same seeded 7-bag sequence, while each game owns its own RNG so different play speeds do not desynchronize the bags.
- Garbage holes use separate persistent RNGs from piece bags, so resetting the shared piece seed does not repeat the same incoming-hole sequence.

## Smallest code changes

- `Tetris` exposes `queue_garbage()` and `last_outgoing_attack`.
- Garbage management runs after every locked piece, regardless of `garbage_prob`.
- `Queue` and `Tetris` accept an optional piece seed.
- `TetrisEnv` refreshes observations after opponent garbage is queued.
- `showcase.py` owns the PvE round loop and routes attacks in both directions.
- `visualization.py` accepts a board-position offset and can omit its local game-over prompt.
- `main.py` adds the `--pve` showcase flag.

## Verification

Focused tests cover cancellation, external garbage with random generation disabled, seven-turn delay, and seeded queue equality. The full command is:

```bash
python -m unittest discover -s app/tests -v
```
