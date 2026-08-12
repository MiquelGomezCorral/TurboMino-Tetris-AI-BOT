# Tetrio Training Performance

## Current Diagnosis

- Training is GPU-bound: the live run sustained 4.23 batches/s at 99% GPU use.
- The precomputed loader used about 13% of one CPU core and 1.3 MB/s disk, so it was not starving the GPU.
- A state contains many candidate boards. The CNN runs once per valid candidate, not once per state.
- The train split contains 6,496,451 states. At batch size 96 and 4.23 batches/s, one full epoch takes about 4.4 hours.

## Applied

- [x] Train Tetrio with BF16 mixed precision.
- [x] Enable high-precision float32 matrix multiplication for remaining float32 operations.
- [x] Show validation loss and top-1/3/5/10 accuracy; select checkpoints by `val/acc_top10`.
- [x] Downsample after the CNN stem so wide residual convolutions run at lower resolution.

## Next Experiments

- [ ] Benchmark 500 warm-up-free batches with the current smaller CNN and BF16; record states/s, peak VRAM, and validation accuracy.
- [ ] Test the largest BF16 batch that fits in VRAM, then keep the best states/s result.
- [ ] Decide whether training should use all frame states or a fixed random sample budget per epoch.

## Epoch Resume

- Every completed validation epoch saves `models/pretrain_model/<exp_name>/last.ckpt` with model, optimizer, scheduler, and callback state.
- The best `val/acc_top10` checkpoint is stored beside it.
- Resume from the last completed epoch with `bash scripts/train_tetrio.sh <config>.yaml models/pretrain_model/<exp_name>/last.ckpt`.
- `tetrio_epochs` remains the total target epoch count after resuming.

## Deferred Work

- [x] Remove the empty precomputed sample at `data/precomputed/train/0066/066127.npz`.
- [ ] Make preprocessing writes atomic.
- [ ] Replace millions of individual NPZ files with compressed dataset shards.
- [ ] Store binary boards as uint8 and pad only to the batch placement count.
- [ ] Encode each base board once and score candidate placements from compact placement features instead of running the CNN per candidate.
