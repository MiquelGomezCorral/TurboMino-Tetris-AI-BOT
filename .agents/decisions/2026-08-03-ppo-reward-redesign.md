# PPO Reward Redesign

Date: 2026-08-03
Status: accepted

## Context

The previous PPO curriculum run improved short-horizon behavior but plateaued
around 35-39 average pieces. The final recorded validation was about 35.7 mean
pieces at roughly 10M steps, while the 500-piece survival metric remained 0%.
The old reward mixed player-facing score deltas, a survival bonus, heuristic
values, and signed square-root compression. That objective did not directly
match the desired behavior: keeping the stack playable for a long time.

## Decisions

- Optimize for long-horizon survival, not for accumulating player-facing score
  points. A high score is useful as a diagnostic, but it is not the training
  objective or the curriculum success criterion.
- Use survival as the default PPO objective.
- Give `+0.1` for every non-terminal placement.
- Give `-5.0` for a terminal transition.
- Make terminal reward exclusive; do not add game-event or heuristic rewards
  after termination.
- Stop using player-facing score deltas as the default PPO reward.
- Remove signed square-root reward compression. PPO already normalizes
  advantages, and arbitrary nonlinear reward transforms can change policy
  preferences.
- Keep game-event rewards opt-in and separate from engine score. Supported
  events are cleared lines, all clears, and regular T-spins. Mini T-spins are
  not separately rewarded.
- Keep heuristic shaping opt-in, scaled by `heuristic_reward_scale` and capped
  by `heuristic_reward_cap`.
- Keep player-facing score and attack calculations unchanged.
- Keep the 500-piece curriculum gate strict. Do not lower it to mask failure;
  also inspect mean pieces and survival milestones because the current survival
  percentage is threshold-based.

## Survival Enhancement Strategy

The policy should learn to preserve a playable board over many placements. The
chosen approach is therefore:

1. Give a small positive reward for each placement that keeps the game alive.
2. Apply a clear terminal penalty when the board becomes unplayable.
3. Avoid rewarding hard-drop distance, combo score, B2B score, or other engine
   score details unless they are deliberately enabled as secondary experiments.
4. Keep optional line-clear, all-clear, T-spin, and heuristic signals separate
   so they can be tested without changing the survival baseline.
5. Evaluate survival directly at multiple horizons instead of relying on score
   pass rates or a single threshold.

This is a staged experiment: first establish a survival-only baseline, then add
one secondary signal at a time only if it improves survival without creating a
new scoring shortcut.

## Evidence

- `app/src/models/gym_env.py` now calculates reward from survival and optional
  event/heuristic components instead of score deltas.
- `app/src/tetris/scoring.py` records a `PlacementEvent` for the latest locked
  piece without changing score calculation.
- `app/tests/test_rewards.py` covers event recording, reward source selection,
  terminal exclusivity, and heuristic caps.
- The old run's score-pass metric saturated while long-horizon survival stayed
  at zero, so score alone is not an adequate training success metric.

## Validation Status

Focused reward and garbage tests passed previously with `unittest`. A fresh
survival-only PPO run has not yet been evaluated, so this decision changes the
training objective but does not yet prove better policy performance.

## Follow-up

- Run a fresh 4-wide survival-only pilot.
- Compare mean pieces and survival at 50, 100, 250, and 500 pieces with the
  previous baseline.
- Measure random and heuristic baselines if the result remains unclear.
