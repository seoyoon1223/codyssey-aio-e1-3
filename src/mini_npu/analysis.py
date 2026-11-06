"""JSON dataset analysis and repeatable performance measurements."""

from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from .core import CROSS, EPSILON, UNDECIDED, X_LABEL, classify_labels, mac, normalize_label, validate_matrix

_PATTERN_KEY = re.compile(r"^size_(\d+)_(\d+)$")
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data.json"


def load_dataset(path: Path = DEFAULT_DATA) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON value must be an object")
    if not isinstance(data.get("filters"), dict) or not isinstance(data.get("patterns"), dict):
        raise ValueError("JSON must contain filters and patterns objects")
    return data


def analyze_dataset(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Analyze every keyed case independently so malformed cases do not stop the run."""
    filters_by_size = data.get("filters")
    patterns = data.get("patterns")
    if not isinstance(filters_by_size, dict) or not isinstance(patterns, dict):
        raise ValueError("JSON must contain filters and patterns objects")

    results: list[dict[str, Any]] = []
    for key, case in patterns.items():
        match = _PATTERN_KEY.fullmatch(key) if isinstance(key, str) else None
        if match is None:
            results.append(_failed(str(key), "pattern key must match size_{N}_{index}"))
            continue
        size = int(match.group(1))
        expected = normalize_label(case.get("expected")) if isinstance(case, dict) else None
        if expected is None:
            results.append(_failed(key, f"unsupported expected label: {case.get('expected')!r}" if isinstance(case, dict) else "case must be an object"))
            continue
        filter_group = filters_by_size.get(f"size_{size}")
        if not isinstance(filter_group, dict):
            results.append(_failed(key, f"missing filters for size_{size}"))
            continue
        try:
            cross_raw = _get_filter(filter_group, CROSS)
            x_raw = _get_filter(filter_group, X_LABEL)
            cross_filter = validate_matrix(cross_raw, size, name=f"{key} Cross filter")
            x_filter = validate_matrix(x_raw, size, name=f"{key} X filter")
            pattern = validate_matrix(case.get("input"), size, name=f"{key} input")
            cross_score = mac(pattern, cross_filter)
            x_score = mac(pattern, x_filter)
        except (KeyError, TypeError, ValueError) as error:
            results.append(_failed(key, str(error)))
            continue
        predicted = classify_labels(cross_score, x_score)
        passed = predicted == expected
        results.append(
            {
                "key": key,
                "size": size,
                "crossScore": cross_score,
                "xScore": x_score,
                "predicted": predicted,
                "expected": expected,
                "passed": passed,
                "reason": "" if passed else _failure_reason(predicted, expected),
            }
        )
    return results


def _get_filter(group: dict[str, Any], label: str) -> Any:
    for key, value in group.items():
        if normalize_label(key) == label:
            return value
    raise KeyError(f"missing {label} filter")


def _failed(key: str, reason: str) -> dict[str, Any]:
    return {
        "key": key,
        "size": None,
        "crossScore": None,
        "xScore": None,
        "predicted": UNDECIDED,
        "expected": None,
        "passed": False,
        "reason": reason,
    }


def _failure_reason(predicted: str, expected: str) -> str:
    if predicted == UNDECIDED:
        return "scores fall within epsilon; classification is UNDECIDED"
    return f"predicted {predicted}, expected {expected}"


def benchmark(size: int, pattern: list[list[float]], filter_matrix: list[list[float]], repeats: int = 10) -> dict[str, float | int]:
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    start = perf_counter_ns()
    for _ in range(repeats):
        mac(pattern, filter_matrix)
    elapsed = perf_counter_ns() - start
    return {
        "size": size,
        "averageMs": elapsed / repeats / 1_000_000,
        "operations": size * size,
        "repeats": repeats,
    }


def benchmark_rows(data: dict[str, Any], repeats: int = 10) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    cross_3 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    x_3 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    rows.append(benchmark(3, cross_3, x_3, repeats))
    for size in (5, 13, 25):
        group = data["filters"][f"size_{size}"]
        pattern = validate_matrix(
            data["patterns"][f"size_{size}_1"]["input"], size, name=f"benchmark size_{size}"
        )
        filter_matrix = validate_matrix(_get_filter(group, CROSS), size, name=f"benchmark filter_{size}")
        rows.append(benchmark(size, pattern, filter_matrix, repeats))
    return rows


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [result for result in results if not result["passed"]]
    return {
        "total": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "failures": failures,
    }
