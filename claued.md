I'll dig into the diagram and your current code before writing anything, since the architecture has a few subtle pieces (the per-placement broadcast, the cache, the bidirectional cross-attention).

Let me set up and verify my implementation actually runs with the right shapes before handing it over.Libraries are ready. Now I'll write the cleaned-up extractor.Now let me validate shapes and gradient flow with mocked configs (so the real `src`/SB3 imports don't block testing).Everything checks out: correct `(B, max_placements)` output, masking works, all 77 parameter groups receive gradients, ~354K params, ~97 ms on CPU for B=8/M=200 (far faster on GPU since attention uses the flash-SDPA kernel).Here's what the rewrite does and the decisions behind it.

**The data flow** matches your diagram. Boards `(B, M, H, W)` run through one shared CNN (the `(b m) 1 h w` trick is preserved — "each a different forward" with weight sharing), producing one token per placement. The queue runs through embedding → RoPE → self-attention **once** and is reused across all M placements — that's the "CACHE / same between most" note made concrete, and it's why this stays cheap even at M=200.

**The two cross-attention heads** you wanted between the CNN and the embeddings:
- `board_to_piece` — each placement queries the piece context ("given this queue, does this placement make sense?").
- `piece_to_board` — the pieces query the whole placement set, then pool into one board-aware summary that gets broadcast back. This is the second green block feeding the concat.

The concat is `[board, board←piece, piece←board]`, which is the three streams arriving at your `Concat` node (raw CNN + the two attentions).

**The hidden part / output.** Rather than a giant flattened MLP, the `placement_head` is a small MLP applied per-placement with shared weights over the M axis. That gives you permutation-equivariance (placement 5 is scored the same way as placement 50) and outputs exactly `CONFIG.max_placements` values — one scalar value per candidate. `features_dim` is set to `max_placements` directly, so I dropped the old `* 16` flatten hack and the three-tuple return (SB3 needs a single tensor).

**A few things worth flagging:**
- I added a learned **role embedding** (current / hold / next) on top of RoPE — that directly serves your "must learn difference between the swap and the incoming ones" note. The default assumes queue order `[current, hold, next...]`; adjust `roles` in `PieceEncoder` if yours differs.
- **Masking** is handled if you pass `observations["placement_mask"]` of shape `(B, M)`: invalid placements are excluded from the `piece→board` attention and set to `-1e9` at the output. That value treats the output as action logits. If you're using these as Q-value/regression targets instead, mask in the loss rather than relying on `-1e9`, since that magnitude will wreck a regression objective.
- For SB3, since the extractor already emits per-placement values, you'll likely want `net_arch=[]` (or a custom head) so the policy doesn't stack another MLP on top and dilute them.

One assumption I should surface: I treated `M` (the boards axis) as equal to `max_placements` with padding, since your original reshape implied that. If `M` is instead a fixed small number and placements are encoded differently, let me know and I'll rework the placement axis.