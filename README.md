# Mini NPU MAC Simulator

E1-3 implements the multiply-accumulate (MAC) step behind a small pattern classifier. It compares an input matrix with Cross and X filters using only Python loops and the standard library.

## Environment

| Item | Value |
| --- | --- |
| OS | Windows 11 |
| Shell | PowerShell |
| Python | 3.12.14 |
| Package manager | uv |
| Runtime dependencies | None |

The project uses PEP 621 metadata in pyproject.toml and a src/mini_npu package. data.json is the supplied challenge dataset.

## Run

    uv sync
    uv run python main.py

Choose 1 for 3×3 console input or 2 to analyze data.json. To run a mode directly:

    uv run python main.py --mode 1
    uv run python main.py --mode 2
    uv run python main.py --mode 2 --data .\data.json
    uv run python -m unittest discover -s tests -v

The input mode accepts three whitespace-separated numbers per row. A malformed row or nonnumeric value prints an error and restarts that matrix prompt.

## Dataset and decisions

The JSON has filters.size_5, filters.size_13, and filters.size_25, each with cross and x matrices. Pattern keys follow size_{N}_{index} and each case contains input and expected. The analyzer extracts N from the key, checks matrix dimensions, and isolates schema errors to the affected case.

Labels are normalized before comparison: cross and + become Cross; x becomes X. MAC computes sum(pattern[row][column] * filter[row][column]) with nested loops. The program classifies the larger score; if abs(cross_score - x_score) < 1e-9, it reports UNDECIDED. It never uses NumPy or another runtime package.

## Observed run

On the supplied dataset, the program reported:

    size_5_1   Cross 0.9, X 0.9  -> UNDECIDED / expected X     FAIL
    size_5_2   Cross 8.9, X 0.1  -> Cross / expected Cross     PASS
    size_13_1  Cross 0.3, X 14.7 -> X / expected X             PASS
    size_13_2  Cross 7.5, X 7.5  -> UNDECIDED / expected Cross FAIL
    size_25_1  Cross 4.9, X 4.9  -> UNDECIDED / expected X     FAIL
    size_25_2  Cross 52.9, X 0.1 -> Cross / expected Cross     PASS

Total: 6 | Passed: 3 | Failed: 3

The failure cases are reported instead of terminating the application. Their matrices give equal Cross and X scores at the precision used by the MAC loop, so the required epsilon policy cannot justify selecting either label. The expected label records the dataset target, but it does not override an ambiguous score. The other three cases have a strict score winner that matches their normalized expected label.

## Performance

Each row below is the measured mean of 10 calls to the MAC function. File I/O and printing are outside the timed section. These small timings vary with machine load.

| Matrix | Mean time (ms) | Multiply/add positions |
| --- | ---: | ---: |
| 3×3 | 0.002450 | 9 |
| 5×5 | 0.006040 | 25 |
| 13×13 | 0.036470 | 169 |
| 25×25 | 0.133910 | 625 |

For an N×N matrix, the nested loops visit every cell once, so the operation count is N² and the time complexity is O(N²). Increasing N from 5 to 25 multiplies the cell count by 25, from 25 to 625. The measured time also rises, though fixed Python call overhead and the very short 10-call sample mean the ratios do not exactly match the operation-count ratio. Larger repeat counts would reduce timing noise without changing the algorithm complexity.

## Extending labels and dataset cases

The current exercise intentionally has two canonical classes. To add another spelling for an existing class:

1. Add the spelling to normalize_label and map it to the existing canonical label.
2. Add a unit test for the new spelling and for an unsupported spelling that must still be rejected.
3. Add or update a data.json case using the spelling in expected or the filter key.
4. Run the focused unit tests, then run the complete JSON analysis.
5. Confirm that the normalized label, chosen filter, and PASS/FAIL summary agree.

A genuinely new class needs an explicit design change because the current classifier compares exactly Cross and X. Add the class filter to each relevant size group, classify all canonical labels, and report UNDECIDED when the best two scores are within epsilon. Add tests for a unique winner, an epsilon tie, missing filters, and malformed matrices before expanding the dataset.

## 1000×1000 scale estimate and optimization policy

The largest measured case is 25×25 at 0.133910 ms per MAC call in this run. A simple linear extrapolation from 625 cells estimates roughly 214 ms for one million cells, or roughly 428 ms to score one pattern against both filters. This is an estimate from a short local microbenchmark, not a measured 1000×1000 result; Python interpreter overhead, CPU load, and memory behavior can change it.

A Python float object and its list reference use roughly 32 bytes per cell before row and container overhead. Three normalized 1000×1000 matrices therefore need about 96 MB, and the original parsed JSON objects may remain alive at the same time. The actual peak can exceed 200 MB after parsed input, normalized copies, and runtime overhead are included. The current loop does not allocate a product matrix, which avoids another million-cell allocation.

The first in-scope optimization is to flatten each validated matrix once into one-dimensional lists and use a single index loop. This removes repeated row lookups while preserving the no-third-party-library rule. If inputs grow beyond available memory, process rows or tiles incrementally and release each tile before reading the next. Profile those changes against the existing implementation and keep them only when they improve the measured workload.

NumPy or a hardware vector backend could process many MAC operations in parallel, but this mission forbids external runtime libraries, so vectorization is not enabled here. If that constraint changes in a later project, vectorized CPU or NPU execution is a second-stage option after measuring data-transfer and setup costs.

## CPU and NPU bottlenecks

The Python CPU implementation executes the nested loop serially. At larger N, its bottlenecks are interpreter and loop overhead, repeated indexing, and moving matrix values through memory; the arithmetic itself is only one multiply and one addition per cell. A CPU can still be efficient for small matrices because it avoids accelerator setup and transfer costs.

An NPU can issue many MAC operations in parallel, but parallel arithmetic does not remove the need to move inputs and filters into accelerator memory. For large workloads, memory bandwidth, data reuse, launch overhead, and synchronization can become the bottleneck even when arithmetic units are available. A practical policy is to keep small cases on the CPU, profile first, then tile large matrices and reuse each input tile across filters. Consider NPU offload only when the workload is large enough to amortize data movement and launch overhead; verify output against the serial reference with the same epsilon rule.

## Result report and failure analysis

1. The JSON loader checks the top-level object and requires filters and patterns maps.
2. Each pattern key is parsed independently; a malformed key becomes a case-level failure.
3. The size encoded in a key determines which filter group is selected.
4. Expected labels and filter labels are normalized before score comparison.
5. Matrix row counts and column counts are checked against the encoded size.
6. Numeric values must be finite numbers; booleans and strings are rejected as matrix values.
7. A malformed case contributes one failure and does not prevent later cases from running.
8. The 3 failing supplied cases are score ties, not parser or matrix-shape failures.
9. UNDECIDED is intentional when the score difference is less than 1e-9.
10. Marking those cases PASS would require either changing the supplied expected labels or inventing a tie-break rule.
11. The implementation keeps the required neutral tie result and reports the mismatch transparently.
12. Each N×N MAC call performs N² multiplications and accumulations, giving O(N²) time and O(1) extra arithmetic space.
13. The benchmark excludes file loading and console output, then averages 10 measured calls per size.
14. Timing values are measurements from this Windows/Python environment, not performance guarantees.

## Verification

python -m unittest discover -s tests -v completed with 5 tests passing. The checks cover known Cross/X MAC scores, epsilon ties, label aliases, matrix-shape validation, and continuing after a malformed dataset case. Both the interactive 3×3 path and the JSON analysis path were also run with uv.
