import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import Configuration
from src.data.tetrio import J_L_SWAP, find_board_index
from src.tetris import Board, MoveSearcher, Tetris, TetrisConfiguration

from maikol_utils.print_utils import print_separator


_df = None
_CONFIG = None
_T_CONFIG = None
_OUT_DIR = None
columns = [
    'game_id', 'subframe', 'playfield', 'playfield_next', 'placed', 'hold',
    'real_current', 'real_hold', 'next', 'won', 'rating',
    'immediate_garbage', 'incoming_garbage', 'btb', 'combo', 'cleared',
]

def _process_raw_data(CONFIG):
    print(' - Loading raw dataset...')
    raw_df = pd.read_csv(CONFIG.raw_dataset_path)

    
    print(' - Cleaning data...')
    raw_df = raw_df.sort_values(by=['game_id', 'subframe'])
    raw_df['subframe'] = raw_df.groupby('game_id').cumcount()
    raw_df['playfield'] = raw_df['playfield'].fillna('N')
    raw_df['playfield_next'] = raw_df['playfield'].shift(-1).fillna('N')
    raw_df = raw_df[raw_df.duplicated(subset=['game_id'], keep='last')].copy()

    print(' - Fixing hold and current...')
    prev_hold = raw_df.groupby('game_id')['hold'].shift(1).fillna('N')
    hold_changed = prev_hold != raw_df['hold']
    raw_df['real_current'] = np.where(hold_changed, raw_df['hold'], raw_df['placed'])
    raw_df['real_hold'] = np.where(hold_changed, raw_df['placed'], raw_df['hold'])

    empty_hold = raw_df['real_hold'] == 'N'
    raw_df['real_hold'] = np.where(empty_hold, raw_df['next'].str[0], raw_df['real_hold'])
    raw_df['next'] = np.where(empty_hold, raw_df['next'].str[1:], raw_df['next'])

    print(' - Clipping immediate garbage...')
    raw_df['immediate_garbage'] = raw_df['immediate_garbage'].clip(upper=8)

    return raw_df[columns]


def _split_data(processed_df, CONFIG):
    unique_games = pd.Series(processed_df['game_id'].unique()).sample(
        frac=1, random_state=CONFIG.seed
    )
    train_end = int(len(unique_games) * (1 - CONFIG.test_size - CONFIG.val_size))
    val_end = train_end + int(len(unique_games) * CONFIG.val_size)

    train_df = processed_df[processed_df['game_id'].isin(unique_games.iloc[:train_end])].copy()
    train_df = train_df.sample(frac=1, random_state=CONFIG.seed).reset_index(drop=True)
    val_df = processed_df[processed_df['game_id'].isin(unique_games.iloc[train_end:val_end])].copy()
    test_df = processed_df[processed_df['game_id'].isin(unique_games.iloc[val_end:])].copy()
    return train_df, val_df, test_df


def _compute_row(row, CONFIG, T_CONFIG):
    pf_rows = (len(row['playfield']) + T_CONFIG.board_w - 1) // T_CONFIG.board_w
    game_h = max(T_CONFIG.board_h, pf_rows)
    game = Tetris(
        playfield=row['playfield'],
        next_pieces=row['next'],
        active_piece=row['real_current'].translate(J_L_SWAP),
        hold_piece=row['real_hold'].translate(J_L_SWAP),
        height=game_h,
        vanish_zone=T_CONFIG.vanish_zone,
    )

    searcher = MoveSearcher(game, CONFIG, T_CONFIG)
    game_state = [
        float(row['combo']),
        float(row['btb']),
        float(row['immediate_garbage']),
        float(row['incoming_garbage']),
    ]
    placements, features = searcher.get_all_features(game_state)
    actual = min(len(placements), CONFIG.max_placements)

    reference = row['playfield_next']
    if row['cleared'] == 0:
        reference = reference[int(row['immediate_garbage']) * T_CONFIG.board_w:]
    board = Board(T_CONFIG.board_w, game_h, T_CONFIG.vanish_zone, False, reference)
    target = find_board_index(board, features['boards'][:actual])
    if target == -1:
        return None

    return features, target, actual


def _init_worker(df, CONFIG, T_CONFIG, out_dir):
    global _df, _CONFIG, _T_CONFIG, _OUT_DIR
    _df = df
    _CONFIG = CONFIG
    _T_CONFIG = T_CONFIG
    _OUT_DIR = out_dir


def _precompute_row(idx):
    path = os.path.join(_OUT_DIR, f'{idx // 1000:04d}', f'{idx:06d}.npz')
    if os.path.exists(path):
        return idx, True

    result = _compute_row(_df.iloc[idx], _CONFIG, _T_CONFIG)
    if result is None:
        return idx, False

    features, target, actual = result
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        boards=features['boards'][:actual].astype(np.float32),
        queues=features['queues'].astype(np.float32),
        queue_idx=features['queue_idx'][:actual].astype(np.int64),
        game_state=features['game_state'],
        target=np.int64(target),
    )
    return idx, True


def _precompute_split(df, name, out_dir, CONFIG, T_CONFIG):

    os.makedirs(out_dir, exist_ok=True)
    n = len(df)
    workers = max(1, cpu_count() * 3 // 4)
    started = time.time()
    skipped = 0
    with Pool(workers, initializer=_init_worker, initargs=(df, CONFIG, T_CONFIG, out_dir)) as pool:
        progress = pool.imap_unordered(_precompute_row, range(n), chunksize=50)
        with tqdm(progress, total=n, desc=name, unit='rows') as bar:
            for idx, ok in bar:
                if not ok:
                    row = df.iloc[idx]
                    skipped += 1
                    bar.set_postfix(skipped=skipped)
                    tqdm.write(f"WARNING: skipping game {row['game_id']}, frame {row['subframe']}")

    elapsed = time.time() - started
    print(f'{name}: {n} rows in {elapsed:.0f}s, skipped {skipped}')


def process_tetrio_data(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    print_separator("PROCESSING TETRIO DATA", sep_type="LONG")
    processed_df = _process_raw_data(CONFIG)
    print_separator("SPLITTING DATA", sep_type="LONG")
    train_df, val_df, test_df = _split_data(processed_df, CONFIG)


    print_separator("SAVING DATA", sep_type="LONG")

    print(f" - Saving processed dataset to {CONFIG.processed_dataset_path}...")
    processed_df.to_csv(CONFIG.processed_dataset_path, index=False)


    splits = {
        'train': (train_df, CONFIG.tetrio_train, CONFIG.precomputed_train),
        'val': (val_df, CONFIG.tetrio_val, CONFIG.precomputed_val),
        'test': (test_df, CONFIG.tetrio_test, CONFIG.precomputed_test),
    }
    for name, (df, csv_path, path) in splits.items():
        print(f" - Saving       {name} split to {csv_path}...")
        df.to_csv(csv_path, index=False)
        print(f" - Precomputing {name} split to {path}...")
        _precompute_split(df, name, path, CONFIG, T_CONFIG)
