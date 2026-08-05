# TurboMino Afterstate Optimization

Date: 2026-08-05
Status: accepted for experimentation

## Goal

Make TurboMino learn a reliable quality score for every legal Tetris placement,
then choose the best valid afterstate for long-horizon survival.

## Current Decision Path

1. `MoveSearcher` BFS enumerates active and hold placements with rotations,
   SRS kicks, translations, and hard drop.
2. Each candidate is represented by its post-lock, post-line-clear binary board.
3. `BoardEncoder` maps each candidate board to one token.
4. `PieceEncoder` encodes current, hold, and upcoming pieces for two queue
   scenarios; `queue_idx` selects the scenario for each candidate.
5. Each board token attends only to its own queue context and game-state token.
6. `placement_head` emits one scalar per candidate.
7. Invalid candidates are masked.

This is an afterstate evaluator: the main input to the placement score is the
resulting board after the action, not a generic state-action pair.

## Main Architectural Finding

`TurboMinoEncoder` produces one score per placement, but SB3 currently feeds
the flattened placement scores through a dense `156 -> 156` actor MLP and an
additional action layer. This makes each action depend on candidate index and
on the other candidates. It is not permutation-equivariant and weakens the
meaning of the shared placement scorer.

Preferred target:

```text
candidate afterstate -> shared placement scorer -> one action logit
valid mask -> masked softmax
```

Use the shared scalar directly as the actor logit. Keep cross-candidate set
aggregation, if needed, in the critic rather than the actor.

## PPO Stability Decisions

- Remove CNN `Dropout(0.5)` for PPO. Rollout collection runs in eval mode while
  PPO updates run in train mode, so dropout corrupts old/new policy ratios.
- Replace CNN `BatchNorm2d` with stateless `GroupNorm`, or freeze BatchNorm
  permanently. Rollout and PPO update modes otherwise use different statistics.
- Keep `LayerNorm` in transformer and placement heads.
- Remove the learned `feature_scale=10`, or initialize it to `1`. LayerNorm
  already handles scale; an unconstrained gain is unnecessary.
- Keep PPO Adam, the current linear learning-rate schedule, clip range `0.1`,
  target KL `0.015`, and gradient clipping until the structural fixes are
  measured.
- Do not introduce AdamW or cosine warm restarts into PPO yet. Restarts can
  produce KL spikes and have no evidence of benefit here.

## Capacity Defaults

The current policy is about 2.05M parameters, so capacity is not the first
constraint.

- Keep `channels=32` initially.
- Test `wide_k=2` against `wide_k=4`; `wide_k=4` makes the board encoder about
  1.02M parameters, while `wide_k=2` is about 326K.
- Keep `d_model=156`, `n_heads=4`, and `n_piece_layers=2` for the first
  structural experiment.
- Consider `d_model=128`, `n_heads=4`, and one piece layer only after the
  actor and PPO mode issues are resolved.
- Keep board-to-piece attention. Do not enable global piece-to-board attention
  in the actor before adding explicit candidate metadata.

## Observation Gaps

The post-placement board is sufficient for geometric survival evaluation, but
not for exact event prediction.

`MoveSearcher` computes or has access to placement metadata that is currently
discarded, including `lines_cleared`. T-spin status depends on rotation and
kick history, and can be lost when candidates are deduplicated by final
footprint. If score, attack, or T-spin behavior becomes a target, expose per-
placement:

- `lines_cleared`
- `all_clear`
- `spin_type`

Generate spin metadata during candidate simulation and preserve scoring-
distinct routes. Do not add this complexity for a survival-only experiment.

Normalize `game_state` before projection because combo, B2B, immediate garbage,
and incoming garbage have different scales.

## Pretraining Separation

AdamW and `CosineAnnealingWarmRestarts` currently apply only to Lightning
supervised pretraining. PPO uses SB3 Adam. `transfer_encoder_weights()` has no
current call site, and PPO training rejects `.ckpt` resumes, so pretraining
optimizer tuning cannot affect the active PPO curriculum until transfer is
connected.

The pretraining callback configuration is also stale: the model logs
`val/acc_top1`, while `train_tetrio.py` monitors `val_acc`. Fix that before
trusting pretraining checkpoints. With one 100-epoch run, ordinary cosine
decay is more accurate than a warm-restart scheduler with `T_0=epochs`.

## Experiment Order

Run controlled experiments, changing one group at a time:

1. Current architecture with PPO dropout disabled and CNN normalization made
   train/eval invariant.
2. Same baseline with direct shared placement logits instead of the dense actor
   MLP; retain the current critic initially.
3. Compare PPO `n_epochs=10` versus `n_epochs=5`; current settings perform
   7,680 optimizer steps per 73,728-sample rollout.
4. Compare `wide_k=4` versus `wide_k=2`.
5. Normalize `game_state`.
6. Add masked set pooling to the critic only if explained variance remains poor.
7. Add placement event metadata only for score/event-focused training.

Track approximate KL, clip fraction, explained variance, validation placement
ranking, mean pieces, and survival at 50/100/250/500 pieces.

## External Reference

The afterstate actor design is also supported by:

https://arxiv.org/html/2603.26765v1#S4.SS1

Its conclusion is directionally consistent with this project: evaluating
deterministic afterstates directly can outperform a generic action-value actor
with fewer parameters.
