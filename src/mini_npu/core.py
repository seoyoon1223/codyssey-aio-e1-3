"""Input validation, MAC arithmetic, and label normalization."""

from __future__ import annotations

import math
from collections.abc import Sequence

EPSILON = 1e-9
CROSS = "Cross"
X_LABEL = "X"
UNDECIDED = "UNDECIDED"

Matrix = Sequence[Sequence[float]]


def normalize_label(value: object) -> str | None:
    """Map supported dataset spellings to the two canonical labels."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    if normalized in {"cross", "+"}:
        return CROSS
    if normalized == "x":
        return X_LABEL
    return None


def validate_matrix(value: object, size: int, *, name: str = "matrix") -> list[list[float]]:
    """Return a finite numeric square matrix or raise ValueError."""
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name}: expected {size} rows")
    matrix: list[list[float]] = []
    for row_number, row in enumerate(value, start=1):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(f"{name}: row {row_number} must contain {size} values")
        parsed: list[float] = []
        for column_number, item in enumerate(row, start=1):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"{name}: value at {row_number},{column_number} is not numeric")
            numeric = float(item)
            if not math.isfinite(numeric):
                raise ValueError(f"{name}: value at {row_number},{column_number} is not finite")
            parsed.append(numeric)
        matrix.append(parsed)
    return matrix


def mac(pattern: Matrix, filter_matrix: Matrix) -> float:
    """Multiply matching cells and accumulate the products using plain loops."""
    if len(pattern) == 0 or len(pattern) != len(filter_matrix):
        raise ValueError("pattern and filter must have the same non-zero size")
    size = len(pattern)
    total = 0.0
    for row_index in range(size):
        if len(pattern[row_index]) != size or len(filter_matrix[row_index]) != size:
            raise ValueError(f"pattern and filter must both be {size}x{size}")
        for column_index in range(size):
            left = pattern[row_index][column_index]
            right = filter_matrix[row_index][column_index]
            if not isinstance(left, (int, float)) or isinstance(left, bool):
                raise ValueError("pattern values must be numeric")
            if not isinstance(right, (int, float)) or isinstance(right, bool):
                raise ValueError("filter values must be numeric")
            total += float(left) * float(right)
    return total


def classify(score_a: float, score_b: float, *, epsilon: float = EPSILON) -> str:
    """Return A/B, or UNDECIDED when the scores are within epsilon."""
    if abs(score_a - score_b) < epsilon:
        return UNDECIDED
    return "A" if score_a > score_b else "B"


def classify_labels(cross_score: float, x_score: float, *, epsilon: float = EPSILON) -> str:
    """Classify a pattern using canonical Cross and X filter scores."""
    if abs(cross_score - x_score) < epsilon:
        return UNDECIDED
    return CROSS if cross_score > x_score else X_LABEL
