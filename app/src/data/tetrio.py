import os, glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import Configuration
from src.tetris import MoveSearcher, Board, PieceEnum, TetrisConfiguration, Tetris, PIECE_MAPPING


MIRRORED_PIECE_INDICES = np.array([
    PIECE_MAPPING.get(PieceEnum(i), PieceEnum(i)).value
    for i in range(PieceEnum.L.value + 1)
])
J_L_SWAP = str.maketrans("JL", "LJ")

def load_tetrio_loader(CONFIG: Configuration, T_CONFIG: TetrisConfiguration, path: str, augment: bool = False):
    print(f" - Loading Tetrio data from {path}...")
    df = pd.read_csv(path)
    dataset = TetrioDataset(df, CONFIG, T_CONFIG, augment=augment)
    return DataLoader(dataset, batch_size=CONFIG.batch_size, shuffle=True)

def load_tetrio_data(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    """
    Load and preprocess the tetrio classification dataset based on the provided configuration.
    Args:
        CONFIG (Configuration): The configuration object containing data loading parameters.
        T_CONFIG (TetrisConfiguration): The configuration object containing tetris-specific parameters.
    Returns:
        Tuple: A tuple containing the training and testing data (x_train, x_test, y_train, y_test).
    """
    # ============== Load data ============== 
    train_loader = load_tetrio_loader(CONFIG, T_CONFIG, CONFIG.tetrio_train, augment=True)
    test_loader = load_tetrio_loader(CONFIG, T_CONFIG, CONFIG.tetrio_test)
    val_loader = load_tetrio_loader(CONFIG, T_CONFIG, CONFIG.tetrio_val)


    return train_loader, test_loader, val_loader


# ========================================================
#                     Dataset
# ========================================================

class TetrioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, CONFIG: Configuration, T_CONFIG: TetrisConfiguration,
                 augment: bool = False):
        self.df = df
        self.CONFIG = CONFIG
        self.T_CONFIG = T_CONFIG
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.augment and np.random.rand() < self.CONFIG.aug_prob:
            row = flip_rows(row, self.T_CONFIG.board_w)

        reference = row['playfield_next']
        if row['cleared'] == 0:
            reference = reference[int(row['immediate_garbage']) * self.T_CONFIG.board_w:]

        pf_rows = (len(row['playfield']) + self.T_CONFIG.board_w - 1) // self.T_CONFIG.board_w
        game_h = max(self.T_CONFIG.board_h, pf_rows)
        game = Tetris(
            playfield=row['playfield'],
            next_pieces=row['next'],
            active_piece=row['real_current'].translate(J_L_SWAP),
            hold_piece=row['real_hold'].translate(J_L_SWAP),
            height=game_h,
            vanish_zone=self.T_CONFIG.vanish_zone,
        )

        searcher = MoveSearcher(game, self.CONFIG, self.T_CONFIG)
        game_state = [
            float(row['combo']),
            float(row['btb']),
            float(row['immediate_garbage']),
            float(row['incoming_garbage']),
        ]
        placements, features = searcher.get_all_features(game_state)
        M_actual = len(placements)

        board = Board(
            game.width, game.height, game.vanish_zone, game.color_map, reference
        )
        target = find_board_index(board, features['boards'][:M_actual])

        assert target != -1, f"Board not found in placements for row {idx}, {target=}."

        # Pad all variable-length tensors to max_placements
        M_max    = self.CONFIG.max_placements

        def pad_first_dim(arr, max_len, pad_value=0.0):
            pad = np.zeros((max_len - arr.shape[0], *arr.shape[1:]), dtype=arr.dtype)
            return np.concatenate([arr, pad], axis=0)

        boards    = pad_first_dim(features['boards'],    M_max)        # (M, H, W)
        queues    = features['queues']                                 # (2, S, C) — not placement-dim
        queue_idx = pad_first_dim(features['queue_idx'], M_max)       # (M,)

        placement_mask = np.zeros(M_max, dtype=bool)
        placement_mask[:M_actual] = True

        obs = {
            "boards":          torch.from_numpy(boards).float(),
            "queues":          torch.from_numpy(queues).float(),
            "queue_idx":       torch.from_numpy(queue_idx).long(),
            "game_state":      torch.from_numpy(features['game_state']).float(),
            "placement_mask":  torch.from_numpy(placement_mask),
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


# ========================================================
#              Precomputed Dataset
# ========================================================

class PrecomputedTetrioDataset(Dataset):
    def __init__(self, data_dir: str, CONFIG: Configuration, augment: bool = False):
        self.data_dir = data_dir
        self.CONFIG = CONFIG
        self.augment = augment
        self.paths = sorted(glob.glob(os.path.join(data_dir, '*', '*.npz')))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])

        boards     = data['boards']       # (M_actual, H, W)
        queues     = data['queues']       # (2, S, C)
        queue_idx  = data['queue_idx']    # (M_actual,)
        game_state = data['game_state']   # (G,)
        target     = int(data['target'])

        if self.augment and np.random.rand() < self.CONFIG.aug_prob:
            boards = boards[..., ::-1].copy()
            queues = queues[..., MIRRORED_PIECE_INDICES]

        M_actual = boards.shape[0]
        M_max    = self.CONFIG.max_placements

        def pad_first_dim(arr, max_len, pad_value=0.0):
            pad = np.zeros((max_len - arr.shape[0], *arr.shape[1:]), dtype=arr.dtype)
            return np.concatenate([arr, pad], axis=0)

        boards    = pad_first_dim(boards, M_max)
        queue_idx = pad_first_dim(queue_idx, M_max)

        placement_mask = np.zeros(M_max, dtype=bool)
        placement_mask[:M_actual] = True

        obs = {
            "boards":          torch.from_numpy(boards).float(),
            "queues":          torch.from_numpy(queues).float(),
            "queue_idx":       torch.from_numpy(queue_idx).long(),
            "game_state":      torch.from_numpy(game_state).float(),
            "placement_mask":  torch.from_numpy(placement_mask),
        }

        return obs, torch.tensor(target, dtype=torch.long)


def load_precomputed_tetrio_data(CONFIG: Configuration):
    print(' - Loading precomputed Tetrio train...')
    train_dataset = PrecomputedTetrioDataset(CONFIG.precomputed_train, CONFIG, augment=True)
    print(' - Loading precomputed Tetrio test...')
    test_dataset  = PrecomputedTetrioDataset(CONFIG.precomputed_test, CONFIG)
    print(' - Loading precomputed Tetrio val...')
    val_dataset   = PrecomputedTetrioDataset(CONFIG.precomputed_val, CONFIG)

    loader_kwargs = {
        "batch_size": CONFIG.batch_size,
        "num_workers": CONFIG.num_workers,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader  = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    val_loader   = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    return train_loader, test_loader, val_loader



def flip_rows(row, width: int = 10):
    """Flips the playfield and next queue rows horizontally for data augmentation."""
    row = row.copy()

    def _flip_field(pf: str) -> str:
        return ''.join(
            pf[i:i + width].ljust(width, 'N')[::-1]
            for i in range(0, len(pf), width)
        )

    def _flip_piece(piece: str) -> str:
        piece = PieceEnum[piece]
        return PIECE_MAPPING.get(piece, piece).name

    row['playfield'] = _flip_field(row['playfield'])
    row['playfield_next'] = _flip_field(row['playfield_next'])


    for field in ('real_current', 'placed', 'real_hold', 'hold'):
        row[field] = _flip_piece(row[field])
    row['next'] = ''.join(_flip_piece(piece) for piece in row['next'])

    return row
