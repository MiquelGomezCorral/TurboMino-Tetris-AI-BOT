import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import Configuration
from src.tetris import MoveSearcher, Board, TetrisConfiguration, Tetris



def load_tetrio_data(CONFIG: Configuration, T_CONFIG: TetrisConfiguration, use_transforms=True):
    """
    Load and preprocess the tetrio classification dataset based on the provided configuration.
    Args:
        CONFIG (Configuration): The configuration object containing data loading parameters.
        T_CONFIG (TetrisConfiguration): The configuration object containing tetris-specific parameters.
        use_transforms (bool): Whether to apply data augmentations (default: True).
    Returns:
        Tuple: A tuple containing the training and testing data (x_train, x_test, y_train, y_test).
    """
    # ============== Load data ============== 
    df_train = pd.read_csv(CONFIG.tetrio_train)
    df_test = pd.read_csv(CONFIG.tetrio_test)
    df_val = pd.read_csv(CONFIG.tetrio_val)

    # ============== Create datasets dataloaders ==============
    train_dataset = TetrioDataset(df_train, CONFIG, T_CONFIG)
    test_dataset  = TetrioDataset(df_test, CONFIG, T_CONFIG)
    val_dataset   = TetrioDataset(df_val, CONFIG, T_CONFIG)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=CONFIG.batch_size, shuffle=False)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG.batch_size, shuffle=False)

    return train_loader, test_loader, val_loader


# ========================================================
#                     Dataset
# ========================================================

class TetrioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
        self.df = df
        self.CONFIG = CONFIG
        self.T_CONFIG = T_CONFIG

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        playfield_next = self.df.iloc[idx]['playfield_next']

        game = Tetris(
            playfield=self.df.iloc[idx]['playfield'],
            next_pieces=self.df.iloc[idx]['next'],
            active_piece=self.df.iloc[idx]['real_current'],
            hold_piece=self.df.iloc[idx]['real_hold']
        )

        searcher = MoveSearcher(game, self.CONFIG, self.T_CONFIG)
        _, features = searcher.get_all_features()

        board = Board(game.width, game.height, game.vanish_zone, game.color_map, playfield_next)
        idx = find_board_index(board, features['boards'])

        return features, idx
    
    def __getitem__(self, row_idx):
        row = self.df.iloc[row_idx]

        game = Tetris(
            playfield=row['playfield'],
            next_pieces=row['next'],
            active_piece=row['real_current'],
            hold_piece=row['real_hold'],
        )

        searcher = MoveSearcher(game, self.CONFIG, self.T_CONFIG)
        _, features = searcher.get_all_features()

        board = Board(game.width, game.height, game.vanish_zone, game.color_map, row['playfield_next'])
        target = find_board_index(board, features['boards'])

        assert target != -1, f"Board not found in placements for row {row_idx}"

        # Pad all variable-length tensors to max_placements
        M_actual = features['boards'].shape[0]
        M_max    = self.CONFIG.max_placements

        def pad_first_dim(arr, max_len, pad_value=0.0):
            pad = np.zeros((max_len - arr.shape[0], *arr.shape[1:]), dtype=arr.dtype)
            return np.concatenate([arr, pad], axis=0)

        boards    = pad_first_dim(features['boards'],    M_max)        # (M, H, W)
        queues    = features['queues']                                 # (2, S, C) — not placement-dim
        queue_idx = pad_first_dim(features['queue_idx'], M_max)       # (M,)
        heuristics = pad_first_dim(features['heuristics'], M_max)     # (M, h) if present

        # Mask: True = invalid (padded) slot
        placement_mask = np.zeros(M_max, dtype=np.float32)
        placement_mask[M_actual:] = 1.0

        obs = {
            "boards":          torch.from_numpy(boards).float(),
            "queues":          torch.from_numpy(queues).float(),
            "queue_idx":       torch.from_numpy(queue_idx).long(),
            "heuristics":      torch.from_numpy(heuristics).float(),
            "placement_mask":  torch.from_numpy(placement_mask).float(),
        }

        return obs, torch.tensor(target, dtype=torch.long)
    


def dense_batch_to_bitrows(boards_batch: np.ndarray, width: int) -> np.ndarray:
    """
    Convert dense float32 batch (N, H, W) → bitrows (N, H) uint32.
    boards_batch: shape (N, H, W), values 0.0/1.0
    """
    powers = (1 << np.arange(width, dtype=np.uint32))          # (W,)
    binary = (boards_batch > 0.5).astype(np.uint32)             # (N, H, W)
    return binary.dot(powers)                                    # (N, H) uint32

def find_board_index(board: Board, boards_batch: np.ndarray) -> int:
    """
    Return the index in boards_batch (N, H, W) that matches board.
    Returns -1 if not found.
    
    boards_batch: shape (N, visible_height, width) float32
    """
    N, H, W = boards_batch.shape
    
    # Convert the query board to bitrows
    query = board.b_rows[:H].astype(np.uint32)                  # (H,)
    
    # Convert entire batch to bitrows
    batch_bits = dense_batch_to_bitrows(boards_batch, W)         # (N, H)
    
    # XOR each row against query, OR-reduce across rows — zero means exact match
    diff = (batch_bits ^ query[None, :]).any(axis=1)             # (N,) bool
    
    indices = np.where(~diff)[0]
    return int(indices[0]) if len(indices) > 0 else -1