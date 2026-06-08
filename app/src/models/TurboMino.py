import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import nn
import pytorch_lightning as pl

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.config import Configuration
from src.tetris import TetrisConfiguration



# ----------------------------------------------------------------------------- #
#  Attention primitives (self + cross share one implementation)                 #
# ----------------------------------------------------------------------------- #
class Attention(nn.Module):
    """Multi-head attention. Self-attention when `x_kv is None`, else cross."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv=None, key_mask=None):
        x_kv = x_q if x_kv is None else x_kv

        q = rearrange(self.to_q(x_q), "b n (h d) -> b h n d", h=self.n_heads)
        k = rearrange(self.to_k(x_kv), "b m (h d) -> b h m d", h=self.n_heads)
        v = rearrange(self.to_v(x_kv), "b m (h d) -> b h m d", h=self.n_heads)

        # key_mask: (B, M) bool, True = keep. Broadcast to (B, 1, 1, M) for SDPA.
        attn_mask = rearrange(key_mask, "b m -> b 1 1 m") if key_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # flash kernel
        return self.proj(rearrange(out, "b h n d -> b n (h d)"))


class TransformerBlock(nn.Module):
    """Pre-norm residual block usable for both self- and cross-attention."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x_q, x_kv=None, key_mask=None):
        kv = self.norm_kv(x_kv) if x_kv is not None else None
        x = x_q + self.attn(self.norm_q(x_q), kv, key_mask=key_mask)
        x = x + self.ff(self.norm_ff(x))
        return x


# ----------------------------------------------------------------------------- #
#  Board encoder (CNN with residual skip — shared across every placement)       #
# ----------------------------------------------------------------------------- #
class ConvResBlock(nn.Module):
    """The 'Skip' connection from the diagram: conv -> conv with a residual add."""

    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding='same'),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding='same'),
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


class BoardEncoder(nn.Module):
    """Turns each candidate board (H, W) into a single d_model token.

    The same weights run over all M placements, so the network treats each
    placement identically (permutation-equivariant over the M axis).
    """

    def __init__(self, height: int, width: int, d_model: int, ch: int = 16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, ch, 3, padding='same'), nn.GELU())
        self.res = ConvResBlock(ch)
        self.expand = nn.Sequential(nn.Conv2d(ch, 2 * ch, 3, padding='same'), nn.GELU())
        self.pool = nn.MaxPool2d(2)

        flat_dim = 2 * ch * (height // 2) * (width // 2)
        self.proj = nn.Linear(flat_dim, d_model)

    def forward(self, boards):  # (B, M, H, W) -> (B, M, d_model)
        b, m = boards.shape[:2]
        x = rearrange(boards, "b m h w -> (b m) 1 h w")
        x = self.pool(self.expand((self.res(self.stem(x)))))
        x = rearrange(x, "bm c h w -> bm (c h w)")
        x = self.proj(x)
        return rearrange(x, "(b m) d -> b m d", b=b, m=m)


# ----------------------------------------------------------------------------- #
#  Positional encoding                                                          #
# ----------------------------------------------------------------------------- #
class RoPE1D(nn.Module):
    """Rotary positional encoding for the piece queue (current / hold / next-k)."""

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE."

        position = rearrange(torch.arange(max_seq_len), "s -> s 1")
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        freqs = position * div_term  # (S, d_model // 2)

        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, x):  # x: (B, S, d_model)
        seq_len = x.shape[1]

        x_1, x_2 = rearrange(x, "b s (d c) -> c b s d", c=2)
        cos = rearrange(self.cos[:seq_len], "s d -> 1 s d")
        sin = rearrange(self.sin[:seq_len], "s d -> 1 s d")

        out_1 = x_1 * cos - x_2 * sin
        out_2 = x_1 * sin + x_2 * cos
        return rearrange([out_1, out_2], "c b s d -> b s (d c)")

# ----------------------------------------------------------------------------- #
#  Piece-queue encoder (embedding + role + RoPE + self-attention; cached)        #
# ----------------------------------------------------------------------------- #

class PieceEncoder(nn.Module):
    """Encodes [current, hold/swap, next_1..next_k] into context tokens.

    Computed once per board state and reused for every placement (the 'CACHE
    / same between most' note). A learned role embedding lets the net tell the
    swap piece apart from the incoming queue ('must learn difference between
    the swap and the incoming ones').
    """

    NUM_ROLES = 3  # 0 = current, 1 = hold/swap, 2 = upcoming

    def __init__(self, num_categories: int, d_model: int, n_heads: int,
                 n_layers: int, max_pieces: int):
        super().__init__()
        self.embed = nn.Linear(num_categories, d_model)
        self.role_embed = nn.Embedding(self.NUM_ROLES, d_model)
        self.rope = RoPE1D(d_model, max_pieces)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )

        roles = torch.full((max_pieces,), 2, dtype=torch.long)
        roles[0] = 0
        if max_pieces > 1:
            roles[1] = 1
        self.register_buffer("role_ids", roles, persistent=False)

    def forward(self, queue):  # (B, S, num_categories) -> (B, S, d_model)
        x = self.embed(queue) + self.role_embed(self.role_ids)  # (S, d) broadcasts
        x = self.rope(x)
        for block in self.layers:
            x = block(x)
        return x


# ----------------------------------------------------------------------------- #
#  Feature extractor                                                            #
# ----------------------------------------------------------------------------- #
class TurboMinoEncoder(BaseFeaturesExtractor):
    """Scores every candidate placement.

    Pipeline
    --------
    boards (B, M, H, W) --CNN-------> board tokens (B, M, d)
    queue  (B, S, C)    --emb+rope+SA-> piece tokens (B, S, d)   [computed once]

    board -> piece  cross-attn : each placement pulls in the piece context
    piece -> board  cross-attn : the pieces survey the whole placement set,
                                 pooled into one board-aware summary

    concat[ board, board<-piece, piece<-board ] -> shared MLP -> 1 value / placement

    Output: (B, max_placements). Invalid placements are masked when an optional
    `placement_mask` (B, M) is provided in the observation.
    """

    MASK_VALUE = -1e9  # masked-out placements (treat as action logits downstream)

    def __init__(
        self, 
        observation_space,
        T_CONFIG: TetrisConfiguration, 
        CONFIG: Configuration,
    ):
        super().__init__(observation_space, features_dim=CONFIG.max_placements * CONFIG.features_per_placement)
        self.CONFIG = CONFIG
        self.T_CONFIG = T_CONFIG

        d_model = getattr(CONFIG, "d_model", 64)
        n_heads = getattr(CONFIG, "n_heads", 4)
        n_self_layers = getattr(CONFIG, "n_piece_layers", 2)
        head_hidden = getattr(CONFIG, "head_hidden", 128)

        _, height, width = observation_space["boards"].shape


        self.board_encoder = BoardEncoder(height, width, d_model)
        self.piece_encoder = PieceEncoder(
            num_categories=T_CONFIG.num_piece_categories,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_self_layers,
            max_pieces=T_CONFIG.max_pieces_in_view,
        )

        # The two cross-attention heads bridging the CNN and the piece embeddings.
        self.board_to_piece = TransformerBlock(d_model, n_heads)  # Q=board, KV=piece
        self.piece_to_board = TransformerBlock(d_model, n_heads)  # Q=piece, KV=board
        
        # To make markovian the combos done by the game
        self.game_state_proj = nn.Linear(2, d_model)
        
        # Per-placement value head ('Oculto' MLP), shared across the M placements.
        self.feature_scale = nn.Parameter(torch.tensor(10.0))
        self.placement_head = nn.Sequential(
            nn.Linear(2 * d_model, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, CONFIG.features_per_placement),
        )


    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        boards = observations["boards"]              # (B, M, H, W)
        queues = observations["queues"]              # (B, 2, S, C)
        queue_idx = observations["queue_idx"].long() # (B, M)
        mask = observations.get("placement_mask")    # (B, M) or None
        key_mask = mask.bool() if mask is not None else None

        B, M = boards.shape[:2]
        _, num_queues, S, C = queues.shape

        # 1. Encode the Board
        board_tok = self.board_encoder(boards)       # (B, M, d)

        # 2. Encode the Unique Queues Efficiently (Runs only 2 times per batch item, not M times)
        flat_queues = rearrange(queues, "b k s c -> (b k) s c")
        flat_piece_toks = self.piece_encoder(flat_queues)
        # Reshape back to (Batch, 2, Seq, d_model)
        piece_toks = rearrange(flat_piece_toks, "(b k) s d -> b k s d", b=B, k=num_queues)

        # 3. Gather Mapping: Assign the correct piece context to each of the M placements
        batch_indices = torch.arange(B, device=boards.device).view(B, 1)
        # Advanced indexing extracts the specific queue for each placement
        # piece_tok_per_placement shape: (B, M, S, d)
        piece_tok_per_placement = piece_toks[batch_indices, queue_idx] 

        # Game observation as a token so it adds to the board more info
        gs = observations["game_state"]                          # (B, 2)
        gs_tok = self.game_state_proj(gs)                        # (B, d)
        gs_tok = gs_tok.unsqueeze(1)                             # (B, 1, d)
        gs_tok_expanded = gs_tok.unsqueeze(1).expand(-1, M, -1, -1)  # (B, M, 1, d)

        # 4. Board -> Piece Cross-Attention (Isolated Tunnels)
        # We temporarily collapse (B, M) into a single batch dimension. 
        # This mathematically guarantees Placement X only attends to Queue X, with zero crosstalk.
        board_tok_flat = rearrange(board_tok, "b m d -> (b m) 1 d")
        piece_tok_flat = rearrange(piece_tok_per_placement, "b m s d -> (b m) s d")
        piece_tok_flat = torch.cat([
            piece_tok_flat,                                       # (B*M, S, d)
            rearrange(gs_tok_expanded, "b m 1 d -> (b m) 1 d"),  # (B*M, 1, d)
        ], dim=1)  
        
        b_from_p_flat = self.board_to_piece(board_tok_flat, piece_tok_flat)
        b_from_p = rearrange(b_from_p_flat, "(b m) 1 d -> b m d", b=B, m=M)

        # 5. Piece -> Board (The Global Summary) — currently not used in fused
        # active_piece_tok = piece_toks[:, 0, :, :] # (B, S, d)
        # p_from_b = self.piece_to_board(active_piece_tok, board_tok, key_mask=key_mask)
        # p_summary = reduce(p_from_b, "b s d -> b d", "mean")          # (B, d)

        # 6. Fuse and Score
        fused = torch.cat([board_tok, b_from_p], dim=-1)              # (B, M, 2d)
        fused = fused * self.feature_scale
        values = self.placement_head(fused)                           # (B, M, f)
        final_features = rearrange(values, "b m f -> b (m f)")        # (B, M * f)
        return final_features
    


# ----------------------------------------------------------------------------- #
#  Pretrain module                                                            #
# ----------------------------------------------------------------------------- #
class TurboMinoModule(pl.LightningModule):
    """
    Pretraining wrapper for TurboMinoEncoder.
    
    Task: given a board state + piece queue, predict which placement index
    the expert (e.g. heuristic agent) would choose.
    
    Input batch: (observations_dict, target_placement_idx)
      observations_dict keys: boards, queues, queue_idx, placement_mask
      target: (B,) long — index of the correct placement in [0, M)
    """
    def __init__(
        self, 
        CONFIG: Configuration, 
        T_CONFIG: TetrisConfiguration,
        observation_space,
        weights=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["weights"])
        self.CONFIG = CONFIG
        self.T_CONFIG = T_CONFIG

        self.encoder = TurboMinoEncoder(observation_space, T_CONFIG, CONFIG)

        # Project (M * f) back to M logits for placement classification
        self.classifier_head = nn.Linear(
            CONFIG.features_per_placement, 1  # applied per-placement, shared weights
        )
        

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=getattr(CONFIG, "label_smoothing", 0.0)
        )

        if weights is not None:
            self.load_state_dict(weights)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns placement logits (B, M)."""
        features = self.encoder(observations)                        # (B, M * f)
        B = features.shape[0]
        M = self.CONFIG.max_placements
        f = self.CONFIG.features_per_placement

        per_placement = features.view(B, M, f)                       # (B, M, f)
        logits = self.classifier_head(per_placement).squeeze(-1)     # (B, M)

        # Mask invalid placements before loss / argmax
        mask = observations.get("placement_mask")
        if mask is not None:
            logits = logits.masked_fill(mask.bool(), TurboMinoEncoder.MASK_VALUE)

        return logits                                                 # (B, M)

    def _shared_step(self, batch, stage: str):
        observations, targets = batch   # targets: (B,) long, index in [0, M)

        logits = self(observations)     # (B, M)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()

        # Top-3 acc is useful when M=128 — exact match is hard
        top3 = (
            logits.topk(3, dim=1).indices == targets.unsqueeze(1)
        ).any(dim=1).float().mean()

        self.log(f"{stage}/loss",     loss,  prog_bar=True,  on_epoch=True, on_step=False)
        self.log(f"{stage}/acc_top1", acc,   prog_bar=True,  on_epoch=True, on_step=False)
        self.log(f"{stage}/acc_top3", top3,  prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.CONFIG.learning_rate,
            weight_decay=self.CONFIG.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.CONFIG.epochs,
            eta_min=self.CONFIG.eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss",
            },
        }

    def transfer_encoder_weights(self, rl_policy):
        """
        Copy pretrained encoder weights into an SB3 policy's features extractor.
        Call after pretraining, before RL training.
        """
        rl_policy.features_extractor.load_state_dict(
            self.encoder.state_dict(), strict=True
        )

