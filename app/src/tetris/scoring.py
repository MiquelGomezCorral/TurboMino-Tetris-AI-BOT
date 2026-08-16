import math
from enum import Enum
from typing import NamedTuple

class SpinType(Enum):
    NONE = 0
    MINI = 1
    REGULAR = 2


class PlacementEvent(NamedTuple):
    lines_cleared: int
    all_clear: bool
    regular_t_spin: bool


class ScoringSystem:
    def __init__(self):
        self.score = 0
        self.level = 1
        self.lines_cleared_total = 0
        self.combo = 0
        self.b2b_streak = 0
        self.last_move_name = ""
        self.last_placement_event = PlacementEvent(0, False, False)
        self.total_placements = 0
        self.total_all_clears = 0
        self.total_tetrises = 0
        self.total_t_spins = {
            "T-Spin": 0,
            "T-Spin Single": 0,
            "T-Spin Double": 0,
            "T-Spin Triple": 0,
            "T-Spin Mini": 0,
            "T-Spin Mini Single": 0,
            "T-Spin Mini Double": 0,
        }

    def _compute_move_name(self, lines: int, spin: SpinType, perfect_clear: bool) -> str:
        parts = []
        if spin == SpinType.REGULAR:
            if lines == 0:      parts.append("T-Spin")
            elif lines == 1:    parts.append("T-Spin Single")
            elif lines == 2:    parts.append("T-Spin Double")
            elif lines == 3:    parts.append("T-Spin Triple")
        elif spin == SpinType.MINI:
            if lines == 0:      parts.append("T-Spin Mini")
            elif lines == 1:    parts.append("T-Spin Mini Single")
            elif lines == 2:    parts.append("T-Spin Mini Double")
        else:
            if lines == 1:      parts.append("Single")
            elif lines == 2:    parts.append("Double")
            elif lines == 3:    parts.append("Triple")
            elif lines == 4:    parts.append("Tetris")
        if perfect_clear and parts:
            parts.append("All Clear")
        return " ".join(parts)

    def evaluate_drop(self, lines: int, spin: SpinType, perfect_clear: bool, drop_distance: int, hard_drop: bool):
        self.total_placements += 1
        tpin_attacks = {
            SpinType.REGULAR: {0: 0, 1: 2, 2: 4, 3: 6},
            SpinType.MINI: {0: 0, 1: 0, 2: 1},
            SpinType.NONE: {0: 0, 1: 0, 2: 1, 3: 2, 4: 4}
        }
        self.last_placement_event = PlacementEvent(
            lines_cleared=lines,
            all_clear=perfect_clear,
            regular_t_spin=spin == SpinType.REGULAR,
        )
        self.score += drop_distance * (2 if hard_drop else 1)
        self.last_move_name = self._compute_move_name(lines, spin, perfect_clear)
        if spin != SpinType.NONE:
            self.total_t_spins[self._compute_move_name(lines, spin, False)] += 1

        if lines == 0:
            # if spin != SpinType.NONE:
                # self.score += (400 if spin == SpinType.REGULAR else 100) * self.level
            self.combo = 0
            return 0

        self.combo += 1
        additional_combo_clears = self.combo - 1
        was_b2b_active = self.b2b_streak > 0
        base = 0
        is_difficult = False
        attack = tpin_attacks[spin].get(lines, 0)

        if spin == SpinType.REGULAR:
            is_difficult = True
            if lines == 1: base = 800
            elif lines == 2: base = 1200
            elif lines == 3: base = 1600
        elif spin == SpinType.MINI:
            is_difficult = True
            if lines == 1: base = 200
            elif lines == 2: base = 400
        else:
            if lines == 1: base = 100
            elif lines == 2: base = 300
            elif lines == 3: base = 500
            elif lines == 4:
                base = 800
                is_difficult = True

        if perfect_clear:
            is_difficult = True

        if is_difficult and was_b2b_active:
            base = int(base * 1.5)

        surge = 0
        if is_difficult:
            self.b2b_streak += 1
        else:
            surge = self.b2b_streak if self.b2b_streak >= 4 else 0
            self.b2b_streak = 0

        if perfect_clear:
            if lines == 1: base += 800
            elif lines == 2: base += 1200
            elif lines == 3: base += 1800
            elif lines == 4: base += 2000

        self.score += (base + (50 * additional_combo_clears)) * self.level
        self.lines_cleared_total += lines
        self.level = (self.lines_cleared_total // 10) + 1
        if perfect_clear:
            self.total_all_clears += 1
        if lines == 4:
            self.total_tetrises += 1

        if is_difficult and was_b2b_active:
            attack += 1
        if attack > 0:
            attack = int(attack * (1 + 0.25 * additional_combo_clears))
        elif additional_combo_clears >= 2:
            attack = int(math.log(1 + 1.25 * additional_combo_clears))
        if perfect_clear:
            attack += 5
        return attack + surge

    def get_combo(self):
        return self.combo

    def get_b2b_active(self):
        return self.b2b_streak > 0

    def get_b2b_streak(self):
        return self.b2b_streak
