import math
from enum import Enum

class SpinType(Enum):
    NONE = 0
    MINI = 1
    REGULAR = 2

class ScoringSystem:
    def __init__(self):
        self.score = 0
        self.level = 1
        self.lines_cleared_total = 0
        self.combo = -1
        self.b2b_streak = 0
        self.last_move_name = ""
        self.total_all_clears = 0
        self.total_tetrises = 0

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
        self.score += drop_distance * (2 if hard_drop else 1)
        self.last_move_name = self._compute_move_name(lines, spin, perfect_clear)

        if lines == 0:
            # if spin != SpinType.NONE:
                # self.score += (400 if spin == SpinType.REGULAR else 100) * self.level
            self.combo = -1
            return 0

        self.combo += 1
        was_b2b_active = self.b2b_streak > 0
        base = 0
        is_difficult = False
        attack = 0

        if spin == SpinType.REGULAR:
            is_difficult = True
            attack = (0, 2, 4, 6)[lines]
            if lines == 1: base = 800
            elif lines == 2: base = 1200
            elif lines == 3: base = 1600
        elif spin == SpinType.MINI:
            is_difficult = True
            attack = (0, 0, 1)[lines]
            if lines == 1: base = 200
            elif lines == 2: base = 400
        else:
            attack = (0, 0, 1, 2, 4)[lines]
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

        self.score += (base + (50 * self.combo)) * self.level
        self.lines_cleared_total += lines
        self.level = (self.lines_cleared_total // 10) + 1
        if perfect_clear:
            self.total_all_clears += 1
        if lines == 4:
            self.total_tetrises += 1

        if is_difficult and was_b2b_active:
            attack += 1
        if attack > 0:
            attack = int(attack * (1 + 0.25 * self.combo))
        elif self.combo >= 2:
            attack = int(math.log(1 + 1.25 * self.combo))
        if perfect_clear:
            attack += 5
        return attack + surge

    def get_combo(self):
        return self.combo

    def get_b2b_active(self):
        return self.b2b_streak > 0

    def get_b2b_streak(self):
        return self.b2b_streak
