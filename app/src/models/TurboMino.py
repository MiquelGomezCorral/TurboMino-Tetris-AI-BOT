import torch
from einops import rearrange, repeat
import numpy as np
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.config import Configuration
from src.tetris import TetrisConfiguration

class TetrisFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, T_CONFIG: TetrisConfiguration, CONFIG: Configuration, observation_space):
        super().__init__(observation_space, CONFIG.features_dim)
        
        self.CONFIG = CONFIG
        self.T_CONFIG = T_CONFIG
        
        # --- 1. Board CNN ---
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        _, H, W = observation_space["boards"].shape
        self.cnn_output_dim = 32 * (H // 2) * (W // 2)
        
        # --- 2. Sequence Queue Pipeline ---
        self.d_model = 16 
        self.piece_embedding = nn.Linear(self.T_CONFIG.num_piece_categories, self.d_model)
        self.rope = RoPE1D(self.d_model, max_seq_len=self.T_CONFIG.max_pieces_in_view)
        self.seq_flatten_dim = self.T_CONFIG.max_pieces_in_view * self.d_model
        
        # --- 3. Combiner ---
        self.placement_evaluator = nn.Sequential(
            nn.Linear(self.cnn_output_dim + self.seq_flatten_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        
        self._features_dim = self.CONFIG.max_placements * 16

    def forward(self, observations: dict[str, torch.Tensor]):
        # SB3 passes a dict of tensors
        boards = observations["boards"] # Shape: (Batch, M, 20, 10)
        queue = observations["queue"]   # Shape: (Batch, T_CONFIG.max_pieces_in_view, T_CONFIG.num_piece_categories)
        
        B, M, H, W = boards.shape
        
        # 1. Process Boards
        board_img = rearrange(boards, 'b m h w -> (b m) 1 h w')
        board_features = self.cnn(board_img) # Shape: (B*M, cnn_out)
        
        # 2. Process Queue
        piece_embeds = self.piece_embedding(queue) # Shape: (B, T_CONFIG.max_pieces_in_view, 16)
        rotated_embeds = self.rope(piece_embeds)   # Shape: (B, T_CONFIG.max_pieces_in_view, 16)
        queue_features = rearrange(rotated_embeds, 'b seq d -> b (seq d)') # Shape: (B, T_CONFIG.max_pieces_in_view * 16)
        
        # 3. Broadcast Queue to match Boards
        # This duplicates the queue context for each of the M placements instantly
        queue_expanded = repeat(queue_features, 'b f -> (b m) f', m=M)
        
        # 4. Combine and Evaluate
        combined = torch.cat([board_features, queue_expanded], dim=1)
        placement_values = self.placement_evaluator(combined) # Shape: (B*M, 16)
        
        final_features = rearrange(placement_values, '(b m) out -> b (m out)', b=B, m=M)
        return board_features, queue_expanded, final_features
    

    
class RoPE1D(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 7):
        super().__init__()
        self.d_model = d_model
        
        assert d_model % 2 == 0, "Feature dimension must be even for RoPE."
        
        # 's -> s 1' replaces .unsqueeze(1)
        position = rearrange(torch.arange(max_seq_len), 's -> s 1')
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        freqs = position * div_term 
        
        self.register_buffer('cos', torch.cos(freqs))
        self.register_buffer('sin', torch.sin(freqs))


    def forward(self, x):
        seq_len = x.shape[1]
        
        # 'b s (d c) -> c b s d' handles the split and unpacking in one step
        x_1, x_2 = rearrange(x, 'b s (d c) -> c b s d', c=2)
        
        # 's d -> 1 s d' replaces .unsqueeze(0)
        cos = rearrange(self.cos[:seq_len], 's d -> 1 s d')
        sin = rearrange(self.sin[:seq_len], 's d -> 1 s d')
        
        out_1 = x_1 * cos - x_2 * sin
        out_2 = x_1 * sin + x_2 * cos
        
        # einops automatically stacks the list and interleaves back to d_model
        return rearrange([out_1, out_2], 'c b s d -> b s (d c)')