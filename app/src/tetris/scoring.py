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
        self.b2b_active = False

    def evaluate_drop(self, lines: int, spin: SpinType, perfect_clear: bool, drop_distance: int, hard_drop: bool):
        # 1. Drop Score
        self.score += drop_distance * (2 if hard_drop else 1)

        if lines == 0:
            if spin != SpinType.NONE:
                # T-Spins without line clears still award minor points
                self.score += (400 if spin == SpinType.REGULAR else 100) * self.level
            self.combo = -1
            return

        # 2. Base Line Clear Score
        self.combo += 1
        base = 0
        is_difficult = False

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

        # 3. Apply Modifiers
        if is_difficult and self.b2b_active:
            base = int(base * 1.5)

        if is_difficult:
            self.b2b_active = True
        else:
            self.b2b_active = False

        # 4. Perfect Clear override/bonus
        if perfect_clear:
            if lines == 1: base += 800
            elif lines == 2: base += 1200
            elif lines == 3: base += 1800
            elif lines == 4: base += 2000

        # 5. Final Calculation
        self.score += (base + (50 * self.combo)) * self.level
        self.lines_cleared_total += lines
        self.level = (self.lines_cleared_total // 10) + 1
