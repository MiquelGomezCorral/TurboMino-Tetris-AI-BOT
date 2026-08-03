# Scoring, Attack, Combo, B2B, and Garbage Rules

This document describes the rules currently implemented by TurboMino. It is a code-level reference for the engine and training environment, not a claim that every value exactly matches the live TETR.IO multiplayer rules.

The main implementation is in:

- `app/src/tetris/scoring.py`: score, attack, combo, B2B, and move names.
- `app/src/tetris/tetris.py`: spin detection, perfect-clear detection, garbage cancellation, delay, insertion, and game over.
- `app/src/models/gym_env.py`: reinforcement-learning observation and reward handling.

## State

A new `ScoringSystem` starts with:

| State | Initial value |
|---|---:|
| Score | 0 |
| Level | 1 |
| Total cleared lines | 0 |
| Combo | 0 |
| B2B streak | 0 |
| Total perfect clears | 0 |
| Total Tetrises | 0 |

Combo uses the same non-negative convention throughout the engine, dataset, and model observation:

- `0`: no active combo.
- `1`: the first consecutive placement that clears at least one line.
- `2`: the second consecutive line-clearing placement.
- `n`: the `n`th consecutive line-clearing placement.

The model receives this state vector as `float32`:

```text
[combo, b2b_streak, immediate_garbage, total_incoming_garbage]
```

Precomputed observations created before the non-negative combo convention may still contain `-1` for no combo and `0` for the first clear. Those files must be regenerated rather than mixed with current observations.

## Drop Processing

Scoring is evaluated only when a piece is hard-dropped and locked. The engine processes a lock in this order:

1. Hard-drop the active piece and obtain the drop distance.
2. Detect whether the placement is a regular T-spin, mini T-spin, or not a spin.
3. Lock the piece and clear complete lines.
4. Detect a perfect clear.
5. Update score, combo, B2B, and attack.
6. Cancel or insert incoming garbage when garbage simulation is enabled.
7. Spawn the next piece.
8. End the game if spawning collides or garbage overflow occurred.

Horizontal movement, rotation, hold, and soft downward movement do not directly call the scoring system.

## Drop Score

A hard drop adds:

```text
2 * drop_distance
```

`ScoringSystem.evaluate_drop()` also supports `hard_drop=False`, which would add `drop_distance`, but the current engine calls it with `hard_drop=True`.

Soft downward movement currently awards no score.

## Line-Clear Score

### Base values

| Clear | Base score |
|---|---:|
| Single | 100 |
| Double | 300 |
| Triple | 500 |
| Tetris | 800 |
| Mini T-spin Single | 200 |
| Mini T-spin Double | 400 |
| T-spin Single | 800 |
| T-spin Double | 1200 |
| T-spin Triple | 1600 |

A zero-line T-spin receives no base score in the current implementation.

### Difficult clears

The following clears are difficult:

- Any regular T-spin line clear.
- Any mini T-spin line clear.
- A Tetris.
- Any perfect clear.

If the previous B2B streak is active, the base score of a difficult clear is multiplied by `1.5` and truncated to an integer.

### Perfect-clear bonus

The following bonus is added after the B2B multiplier:

| Lines cleared | Perfect-clear score bonus |
|---|---:|
| 1 | 800 |
| 2 | 1200 |
| 3 | 1800 |
| 4 | 2000 |

The perfect-clear bonus itself is therefore not multiplied by B2B.

### Combo score bonus

For a line-clearing placement:

```text
additional_combo_clears = combo - 1
combo_score_bonus = 50 * additional_combo_clears
```

The first clear has combo `1`, so its combo score bonus is `0`. The second consecutive clear receives `50`, the third receives `100`, and so on.

### Final score formula

```text
clear_score = (b2b_adjusted_base + perfect_clear_bonus + combo_score_bonus) * current_level
```

The current level is used before adding the newly cleared lines. After scoring:

```text
total_lines += lines_cleared
level = floor(total_lines / 10) + 1
```

Crossing a ten-line boundary therefore changes the multiplier for the next clear.

## Combo Rules

- Any placement that clears at least one line increments combo by `1`.
- Any placement that clears no lines resets combo to `0`.
- Combo affects both score and attack.
- Combo never becomes negative.

For attacks, `additional_combo_clears = combo - 1` preserves the intended behavior while exposing the natural `0, 1, 2, ...` state convention.

## B2B Rules

The B2B streak counts consecutive difficult clears.

- A difficult clear increments `b2b_streak`.
- A zero-line placement leaves `b2b_streak` unchanged.
- A non-difficult line clear resets `b2b_streak` to `0`.
- B2B is active whenever `b2b_streak > 0`.
- A difficult clear receives the B2B bonus when B2B was already active before that clear.

Consequently, the first difficult clear starts the streak but receives no B2B bonus. The second consecutive difficult clear receives it.

### B2B surge

Breaking a B2B streak of at least four difficult clears produces a surge attack equal to the old streak length:

```text
surge = previous_b2b_streak if previous_b2b_streak >= 4 else 0
```

The surge is added to the attack of the non-difficult clear that broke the streak.

## Attack Calculation

Attack is measured in garbage lines. The engine uses it to cancel queued incoming garbage; excess attack is not sent to an opponent.

### Base attack

| Clear | Base attack |
|---|---:|
| Single | 0 |
| Double | 1 |
| Triple | 2 |
| Tetris | 4 |
| Mini T-spin Single | 0 |
| Mini T-spin Double | 1 |
| T-spin Single | 2 |
| T-spin Double | 4 |
| T-spin Triple | 6 |

### B2B attack bonus

A difficult clear made while B2B was already active adds `1` to base attack before combo scaling.

### Combo attack scaling

Let:

```text
c = additional_combo_clears = combo - 1
```

When attack is already positive after the B2B bonus:

```text
attack = floor(attack * (1 + 0.25 * c))
```

When attack is still zero and `c >= 2`, combo alone creates attack:

```text
attack = floor(ln(1 + 1.25 * c))
```

`ln` is the natural logarithm. For example, consecutive Singles produce attacks `0`, `0`, and `1` on combo values `1`, `2`, and `3`.

### Perfect-clear attack bonus

A perfect clear adds `5` attack after B2B and combo calculations.

### Final attack

```text
final_attack = scaled_attack + perfect_clear_attack_bonus + b2b_surge
```

## Worked Examples

All examples assume level `1` and omit hard-drop distance score.

### Consecutive Singles

| Placement | Combo | Score gained | Attack |
|---|---:|---:|---:|
| First Single | 1 | 100 | 0 |
| Second Single | 2 | 150 | 0 |
| Third Single | 3 | 200 | 1 |

### Consecutive Tetrises

| Placement | Combo | B2B streak after clear | Score gained | Attack |
|---|---:|---:|---:|---:|
| First Tetris | 1 | 1 | 800 | 4 |
| Second Tetris | 2 | 2 | 1250 | 6 |

For the second Tetris:

```text
score = (floor(800 * 1.5) + 50) = 1250
attack = floor((4 + 1) * 1.25) = 6
```

### Perfect-clear Single

A first-clear Single that empties the board:

- Starts combo at `1`.
- Starts B2B at `1` because every perfect clear is difficult.
- Gains `100 + 800 = 900` score.
- Produces `5` attack.

## T-spin Detection

Only the T piece can produce a T-spin classification.

The detector first checks whether the T piece is immobile: it cannot move left, right, or down. Detection continues when either the piece is immobile or the last successful action was a rotation. Otherwise, the placement is not a T-spin.

The detector then examines the four corners around the T piece:

- Board boundaries count as occupied corners.
- At least three occupied corners are required.
- Two occupied front corners produce a regular T-spin.
- Rotation kick index `4` also upgrades the result to a regular T-spin.
- Otherwise, three occupied corners produce a mini T-spin.

Lateral or downward movement after a rotation clears the last-rotation flag. A zero-line spin can still be named as a spin, but it currently awards no spin score or attack and resets combo to `0`.

## Perfect-clear Detection

A placement is a perfect clear when:

- It clears at least one line.
- Every row in the visible playfield is empty after line clearing.

The vanish zone is not included in this emptiness check.

## Incoming Garbage

Each queued garbage packet is represented as:

```text
(lines, turns_remaining, hole_column)
```

Packets are processed in FIFO order.

### Turn order

After every locked piece, when garbage simulation is enabled:

1. Current attack cancels queued garbage from the front.
2. Every surviving packet's delay decreases by `1`, with a minimum of `0`.
3. If the placement cleared no lines, ready packets are inserted up to the per-turn cap.
4. A new random packet may be queued.

Because random garbage is queued last, attack from the same placement cannot cancel that new packet. It can be cancelled on a later placement.

### Cancellation

Attack cancels incoming lines one-for-one:

```text
cancelled = min(packet_lines, remaining_attack)
```

Partially cancelled packets remain at the front with the same delay and hole. Attack remaining after the queue becomes empty is discarded because this engine does not simulate an opponent receiving outgoing garbage.

### Delay and insertion

The default delay is `3`. A packet queued with that delay becomes ready after three later locked pieces, unless it is cancelled first.

Ready garbage is inserted only after a placement that clears no lines. A ready packet remains queued during a line-clearing placement.

At most `garbage_cap` lines are inserted per eligible placement. The default cap is `8`. Any remainder stays at the front of the queue with delay `0`.

Each packet uses one hole column for all of its rows. Adding garbage shifts the board upward and fills the new bottom rows except for that hole. If occupied cells are pushed beyond the board height, garbage overflow causes game over.

### Random generation defaults

After processing existing garbage, a new packet is generated with probability:

```text
garbage_prob = 0.0774
```

The default weighted line-count distribution is:

| Lines | Weight | Approximate probability |
|---|---:|---:|
| 1 | 0.263604 | 26.3604% |
| 2 | 0.155263 | 15.5263% |
| 3 | 0.099349 | 9.9349% |
| 4 | 0.151832 | 15.1832% |
| 5 | 0.145719 | 14.5719% |
| 6 | 0.087687 | 8.7687% |
| 7 | 0.032314 | 3.2314% |
| 8 | 0.064232 | 6.4232% |

The hole column is selected uniformly from the board width.

When `garbage_prob <= 0`, the game does not call garbage management. This disables random generation as well as delay, cancellation, and insertion processing.

### Garbage observation values

`get_incoming_garbage()` returns the total number of lines in all queued packets.

`get_immediate_garbage()` returns the sum of packets with `turns_remaining <= 1`, capped at `garbage_cap`:

```text
immediate_garbage = min(garbage_cap, ready_or_near_ready_lines)
```

## Reinforcement-learning Reward

Engine score and attack are separate from the Gymnasium reward.

By default, the environment uses survival-only rewards. For a non-terminal step,
the reward is:

```text
alive_reward
```

The default `alive_reward` is `0.1`. When enabled, game-event rewards are added
for cleared lines, all clears, and regular T-spins. Heuristic rewards can also be
enabled as a scaled and capped change in the board heuristic.

For a terminal step, the reward is only `death_penalty`, whose default is `-5.0`.
Game-event and heuristic rewards are not added to terminal transitions.

The default reward configuration is:

```yaml
use_survival_rewards: true
use_game_rewards: false
use_heuristic_rewards: false
alive_reward: 0.1
death_penalty: -5.0
```

Optional reward values are configured with `line_clear_reward`,
`all_clear_reward`, `t_spin_reward`, `heuristic_reward_scale`, and
`heuristic_reward_cap`.

## Tested Behavior

`app/tests/test_garbage.py` currently covers:

- Garbage insertion and overflow.
- Configured garbage delay.
- Incoming garbage cancellation and per-turn insertion cap.
- Tetris, T-spin, B2B, surge, and perfect-clear attacks.
- Combo values `0`, `1`, `2`, and `3`.
- Representative Combo score and attack progression for Singles.
- Garbage generation only after a locked piece.
- Model game-state overrides.

`app/tests/test_rewards.py` covers:

- Placement-event recording without changing player-facing score behavior.
- Game rewards reading placement events instead of cumulative engine score.
- Terminal rewards excluding game-event and heuristic rewards.
- Heuristic reward scaling and capping.

The tests can be run with:

```bash
conda run -n TETRIO_env python -m unittest discover -s app/tests -v
```

## PPO Training Findings

The previous curriculum run in `logs/config_curr_ppo.yaml.log` showed that the
training pipeline was active and learning short-horizon behavior, but it did not
meet the long-horizon survival objective:

- Mean pieces improved from about `29` early in training to about `39` at its
  best observed plateau.
- The final recorded validation was about `35.7` mean pieces at roughly
  `10M` steps.
- The `survival` metric remained `0%` because it counts episodes reaching the
  configured `max_eval_pieces` threshold, not partial survival progress.
- Score-pass percentages saturated and were therefore not a reliable proxy for
  survival quality.

These results support a reward-objective mismatch as the primary hypothesis,
not a conclusion that PPO or the environment failed completely. The next
curriculum run should be evaluated against survival milestones and mean pieces,
not score alone. The current curriculum config uses `max_eval_pieces: 500` as
its strict long-horizon gate.

The survival strategy is intentionally staged: establish a survival-only
baseline first, then test one secondary signal at a time. Hard-drop distance,
combo score, B2B score, and other player-facing score details are not default
survival objectives because they can reward point accumulation without proving
that the board remains playable.

The survival-only reward redesign has been implemented, but its training result
must be measured in a fresh run. No claim of improved PPO performance should be
made until that comparison is complete.
