import numpy as np
from typing import NamedTuple

from .tetris import Board

class HeuristicsResult(NamedTuple):
    blocks: int
    weighted_blocks: int
    clearable_lines: int
    roughness: int
    col_holes: int
    connected_holes: int
    blocks_above_holes: int
    pit_hole_percent: float
    deepest_well: int


    base_weights = {
        'blocks': -0.01,
        'weighted_blocks': -0.1,
        'clearable_lines': 1.0,
        'roughness': -0.25,
        'col_holes': -1.0,
        'connected_holes': -1.0,
        'blocks_above_holes': -0.5,
        'pit_hole_percent': -0.5,
        'deepest_well': 0.5,

        'total': 1.0,  # Optional: weight for the total score itself if you want to include it
    }

    def __str__(self):
        return (f"HeuristicsResult(\n"
                f"  blocks={self.blocks}, \n"
                f"  weighted_blocks={self.weighted_blocks}, \n"
                f"  clearable_lines={self.clearable_lines}, \n" 
                f"  roughness={self.roughness}, \n"
                f"  col_holes={self.col_holes}, \n"
                f"  connected_holes={self.connected_holes}, \n"
                f"  blocks_above_holes={self.blocks_above_holes}, \n"
                f"  pit_hole_percent={self.pit_hole_percent:.2f}, \n" 
                f"  deepest_well={self.deepest_well}\n"
                ")")
    
    def compute_total(self, weights: dict=None) -> float:
        """Compute a weighted sum of heuristics based on provided weights."""
        total = 0.0
        if weights is None:
            weights = self.base_weights
        for key, weight in weights.items():
            value = getattr(self, key, 0)
            total += weight * value
        return total * weights.get('total', 1.0)  # Apply total weight if specified


class HeuristicEvaluator:
    """
    Fast bitboard heuristic evaluator for Tetris.

    All heavy lifting uses NumPy bitwise ops on the b_rows uint32 array.
    Column heights are computed once and reused across all heuristics.
    """

    # Precomputed popcount table for 16-bit halves (built once at class load)
    _POPCOUNT16 = np.zeros(65536, dtype=np.int32)
    for _i in range(65536):
        _POPCOUNT16[_i] = bin(_i).count('1')

    def __init__(self):
        powers = np.arange(24, dtype=np.float32)
        self._block_weights = np.minimum(0.01 * (2 ** powers), 10.24)

    def reset(self) -> None:
        ...
    # ------------------------------------------------------------------
    # Core low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _popcount_rows(b_rows: np.ndarray) -> np.ndarray:
        """Popcount each uint32 row → int32 array of same length."""
        lo = (b_rows & 0xFFFF).astype(np.int32)
        hi = (b_rows >> 16).astype(np.int32)
        pc = HeuristicEvaluator._POPCOUNT16
        return pc[lo] + pc[hi]

    @staticmethod
    def _col_heights(b_rows: np.ndarray, width: int, visible_height: int) -> np.ndarray:
        """
        Return int array[width] of column heights (0-based from bottom).
        Uses a vectorised approach: for each column bit, find the topmost set row.
        """
        rows = b_rows[:visible_height]  # shape (H,)
        # col_mask[c] = 1 << c
        col_masks = np.uint32(1) << np.arange(width, dtype=np.uint32)  # (W,)
        # occupied[r, c] = bool
        occupied = (rows[:, None] & col_masks[None, :]) != 0  # (H, W)
        # height = index of topmost occupied row + 1; 0 if column empty
        # np.argmax on reversed axis gives first True from top
        # Trick: multiply row indices and take max
        row_indices = np.arange(visible_height, dtype=np.int32)  # (H,)
        heights = np.where(occupied, row_indices[:, None] + 1, 0).max(axis=0)  # (W,)
        return heights  # int32 array shape (W,)

    @staticmethod
    def _hole_mask(b_rows: np.ndarray, width: int, visible_height: int,
                   col_heights: np.ndarray) -> np.ndarray:
        """
        Return bool array (visible_height, width) where True = hole cell.
        A hole is an empty cell strictly below a column's top surface.
        """
        rows = b_rows[:visible_height]
        col_masks = np.uint32(1) << np.arange(width, dtype=np.uint32)
        occupied = (rows[:, None] & col_masks[None, :]) != 0  # (H, W)
        row_idx = np.arange(visible_height, dtype=np.int32)   # (H,)
        # cell (r, c) is a hole if: not occupied AND r < col_heights[c]
        below_surface = row_idx[:, None] < col_heights[None, :]  # (H, W)
        return below_surface & ~occupied  # (H, W) bool

    # ------------------------------------------------------------------
    # Individual heuristics (all accept board + precomputed helpers)
    # ------------------------------------------------------------------

    def blocks(self, b_rows: np.ndarray, visible_height: int) -> int:
        """Total filled cells in visible area."""
        return int(self._popcount_rows(b_rows[:visible_height]).sum())

    def weighted_blocks(self, b_rows: np.ndarray, visible_height: int) -> float:
        row_pops = self._popcount_rows(b_rows[:visible_height])
        return float((row_pops * self._block_weights[:visible_height]).sum())

    def clearable_lines(self, b_rows: np.ndarray, width: int, visible_height: int) -> int:
        """
        Maximum lines an I-piece (vertical) can clear in a single placement.
        We scan every column: count how many of the 4 rows starting at ghost
        position would be full after placing the I-piece there.
        Fast path: find rows that are full except exactly one column gap,
        then count consecutive such rows per column.

        Simpler correct version: for each column, find the lowest gap,
        drop a vertical I there, count how many of the 4 rows become full.
        """
        full_row = np.uint32((1 << width) - 1)
        rows = b_rows[:visible_height]
        # For each row, compute missing = popcount of ~row & full_row = width - popcount(row)
        row_pops = self._popcount_rows(rows)
        missing = width - row_pops  # (H,)

        col_masks = np.uint32(1) << np.arange(width, dtype=np.uint32)
        col_heights = self._col_heights(b_rows, width, visible_height)

        best = 0
        for c in range(width):
            mask = col_masks[c]
            # drop position: first empty row in this column from bottom
            drop_y = col_heights[c]  # first empty row
            if drop_y + 4 > visible_height:
                continue
            # rows drop_y .. drop_y+3 get one extra block in column c
            # row r becomes full if missing[r] == 1 and that missing bit is col c,
            # or missing[r] == 0 already
            count = 0
            for r in range(drop_y, drop_y + 4):
                if missing[r] == 0:
                    count += 1
                elif missing[r] == 1 and not (rows[r] & mask):
                    count += 1
            if count > best:
                best = count
        return best

    def roughness(self, col_heights: np.ndarray) -> int:
        """Sum of absolute height differences between adjacent columns."""
        diffs = np.abs(np.diff(col_heights.astype(np.int32)))
        return int(diffs.sum())

    def col_holes(self, hole_mask: np.ndarray) -> int:
        """Number of columns that contain at least one hole."""
        return int(hole_mask.any(axis=0).sum())

    def connected_holes(self, hole_mask: np.ndarray) -> int:
        """
        Number of vertically connected hole groups (contiguous vertical runs).
        Count transitions: cells where hole[r,c]=True and hole[r-1,c]=False (or r==0).
        """
        # A new connected group starts when current row is hole and row above is not
        starts = hole_mask.copy()
        starts[1:] &= ~hole_mask[:-1]  # mask out continuations
        return int(starts.sum())

    def blocks_above_holes(self, b_rows: np.ndarray, width: int, visible_height: int,
                           hole_mask: np.ndarray) -> int:
        """
        Number of filled blocks that are directly above any hole column.
        For each hole cell, count filled cells above it in the same column.
        Efficient: for each column, if it has holes, sum filled cells above lowest hole.
        """
        col_masks = np.uint32(1) << np.arange(width, dtype=np.uint32)
        rows = b_rows[:visible_height]
        occupied = (rows[:, None] & col_masks[None, :]) != 0  # (H, W)

        total = 0
        for c in range(width):
            col_hole = hole_mask[:, c]
            if not col_hole.any():
                continue
            # lowest hole row in this column
            lowest_hole = int(np.argmax(col_hole))
            # count filled cells strictly above lowest_hole
            total += int(occupied[lowest_hole + 1:, c].sum())
        return total

    def pit_hole_percent(self, b_rows: np.ndarray, width: int, visible_height: int,
                         col_heights: np.ndarray, hole_mask: np.ndarray) -> float:
        """
        Ratio: pits / (pits + holes).
        A pit is an empty non-hole cell with filled cells on both sides (or board edge).
        """
        rows = b_rows[:visible_height]
        col_masks = np.uint32(1) << np.arange(width, dtype=np.uint32)
        occupied = (rows[:, None] & col_masks[None, :]) != 0  # (H, W)

        row_idx = np.arange(visible_height, dtype=np.int32)
        # empty cells in visible area
        empty = ~occupied
        # above surface: cells at or above col height are not candidates
        below_surface = row_idx[:, None] < col_heights[None, :]
        # candidate empty cells: empty AND below surface AND not a hole
        candidate = empty & below_surface & ~hole_mask

        # Left filled: col 0 → board edge counts as filled
        left_filled = np.empty_like(candidate)
        left_filled[:, 0] = True  # board edge
        left_filled[:, 1:] = occupied[:, :-1]

        # Right filled: last col → board edge counts as filled
        right_filled = np.empty_like(candidate)
        right_filled[:, -1] = True
        right_filled[:, :-1] = occupied[:, 1:]

        pit_mask = candidate & left_filled & right_filled
        n_pits = int(pit_mask.sum())
        n_holes = int(hole_mask.sum())
        denom = n_pits + n_holes
        return n_pits / denom if denom > 0 else 0.0

    def deepest_well(self, b_rows: np.ndarray, width: int, visible_height: int,
                     hole_mask: np.ndarray) -> int:
        """
        Row index (0 = bottom) of the lowest empty non-hole cell.
        Returns -1 if no such cell exists (board is full or all empties are holes).
        """
        rows = b_rows[:visible_height]
        col_masks = np.uint32(1) << np.arange(width, dtype=np.uint32)
        occupied = (rows[:, None] & col_masks[None, :]) != 0  # (H, W)
        empty_non_hole = ~occupied & ~hole_mask
        if not empty_non_hole.any():
            return -1
        return int(np.argmax(empty_non_hole.any(axis=1)))

    # ------------------------------------------------------------------
    # Main entry point — compute ALL heuristics in one pass
    # ------------------------------------------------------------------

    def evaluate(self, board: Board) -> HeuristicsResult:
        """
        Compute all heuristics for the given Board instance.
        Returns a HeuristicsResult namedtuple.

        Shared intermediates (col_heights, hole_mask) are computed once.
        """
        b_rows = board.b_rows
        width = board.width
        visible_height = board.visible_height

        # --- Shared intermediates ---
        col_heights = self._col_heights(b_rows, width, visible_height)
        hole_mask = self._hole_mask(b_rows, width, visible_height, col_heights)

        return HeuristicsResult(
            blocks=self.blocks(b_rows, visible_height),
            weighted_blocks=self.weighted_blocks(b_rows, visible_height),
            clearable_lines=self.clearable_lines(b_rows, width, visible_height),
            roughness=self.roughness(col_heights),
            col_holes=self.col_holes(hole_mask),
            connected_holes=self.connected_holes(hole_mask),
            blocks_above_holes=self.blocks_above_holes(b_rows, width, visible_height, hole_mask),
            pit_hole_percent=self.pit_hole_percent(b_rows, width, visible_height, col_heights, hole_mask),
            deepest_well=self.deepest_well(b_rows, width, visible_height, hole_mask),
        )