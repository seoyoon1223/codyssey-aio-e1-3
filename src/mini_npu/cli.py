"""Interactive console entry points for the Mini NPU simulator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import analyze_dataset, benchmark, benchmark_rows, load_dataset, summarize
from .core import EPSILON, UNDECIDED, classify, classify_labels, mac, validate_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Mini NPU MAC simulator")
    parser.add_argument("--data", type=Path, help="path to a data.json file")
    parser.add_argument("--mode", choices=("1", "2"), help="skip the interactive mode prompt")
    args = parser.parse_args()
    mode = args.mode or _ask_mode()
    if mode == "1":
        return _manual_mode()
    return _data_mode(args.data)


def _ask_mode() -> str:
    while True:
        print("=== Mini NPU Simulator ===\n\n[모드 선택]\n1. 사용자 입력 (3x3)\n2. data.json 분석")
        choice = input("선택: ").strip()
        if choice in {"1", "2"}:
            return choice
        print("1 또는 2를 입력하세요.")


def _read_matrix(label: str, size: int) -> list[list[float]]:
    while True:
        print(f"{label} ({size}줄, 한 줄에 숫자 {size}개)")
        rows: list[list[float]] = []
        try:
            for row_number in range(1, size + 1):
                fields = input(f"  {row_number}: ").split()
                if len(fields) != size:
                    raise ValueError(f"각 줄에 숫자 {size}개를 입력해야 합니다")
                rows.append([float(field) for field in fields])
            return validate_matrix(rows, size, name=label)
        except ValueError as error:
            print(f"입력 형식 오류: {error}. 다시 입력하세요.")


def _show_matrix(label: str, matrix: list[list[float]]) -> None:
    print(f"{label}:")
    for row in matrix:
        print("  " + " ".join(f"{value:g}" for value in row))


def _timed_average(pattern: list[list[float]], matrix: list[list[float]], repeats: int = 10) -> float:
    from time import perf_counter_ns

    started = perf_counter_ns()
    for _ in range(repeats):
        mac(pattern, matrix)
    return (perf_counter_ns() - started) / repeats / 1_000_000


def _manual_mode() -> int:
    print("\n# [1] 필터 입력")
    filter_a = _read_matrix("필터 A", 3)
    filter_b = _read_matrix("필터 B", 3)
    print("필터 저장 완료.")
    print("저장 확인:")
    _show_matrix("필터 A", filter_a)
    _show_matrix("필터 B", filter_b)
    print("\n# [2] 패턴 입력")
    pattern = _read_matrix("패턴", 3)
    _show_matrix("입력 패턴", pattern)
    score_a, score_b = mac(pattern, filter_a), mac(pattern, filter_b)
    decision = classify(score_a, score_b)
    print("\n# [3] MAC 결과")
    print(f"A 점수: {score_a:.12g}")
    print(f"B 점수: {score_b:.12g}")
    print(f"연산 시간(평균/10회): {_timed_average(pattern, filter_a):.6f} ms")
    print(f"판정: {'판정 불가 (epsilon 동점)' if decision == UNDECIDED else decision}")
    print(f"동점 기준: abs(A-B) < {EPSILON:g}")
    print("\n# [4] 성능 분석 (평균/10회)")
    print("크기(N×N) | 평균 시간(ms) | 연산 횟수(N²)")
    row = benchmark(3, pattern, filter_a)
    print(f"3×3       | {row['averageMs']:.6f}      | {row['operations']}")
    return 0


def _data_mode(path: Path | None) -> int:
    try:
        data = load_dataset(path) if path else load_dataset()
    except (OSError, ValueError) as error:
        print(f"data.json을 읽을 수 없습니다: {error}", file=sys.stderr)
        return 2

    print("\n# [1] JSON 필터 및 패턴 분석")
    results = analyze_dataset(data)
    for result in results:
        print(f"--- {result['key']} ---")
        if result["crossScore"] is None:
            print(f"FAIL: {result['reason']}")
            continue
        print(f"Cross 점수: {result['crossScore']:.12g}")
        print(f"X 점수: {result['xScore']:.12g}")
        verdict = "PASS" if result["passed"] else f"FAIL ({result['reason']})"
        print(f"판정: {result['predicted']} | expected: {result['expected']} | {verdict}")

    print("\n# [2] 성능 분석 (평균/10회)")
    print("크기(N×N) | 평균 시간(ms) | 연산 횟수(N²)")
    try:
        for row in benchmark_rows(data):
            print(f"{row['size']}×{row['size']:<7} | {row['averageMs']:.6f}      | {row['operations']}")
    except (KeyError, TypeError, ValueError) as error:
        print(f"성능 측정 일부를 건너뜁니다: {error}")

    report = summarize(results)
    print("\n# [3] 결과 요약")
    print(f"총 테스트: {report['total']}개")
    print(f"통과: {report['passed']}개")
    print(f"실패: {report['failed']}개")
    if report["failures"]:
        print("실패 케이스:")
        for failed in report["failures"]:
            print(f"- {failed['key']}: {failed['reason']}")
    return 0 if report["failed"] == 0 else 1
