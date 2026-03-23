# Benchmark Matrix

Generated: 2026-03-23T11:09:27.850710+00:00

| Profile | Source Path | Files (mean) | Index p50 (s) | Index p95 (s) | Query p50 (s) | Query p95 (s) | Peak Python Mem p95 (MB) | CPU Ratio p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| small | C:/Users/tusik/Documents/GitHub/Architext/src/tasks/analysis | 9 | 4.5209 | 5.835 | 0.1333 | 0.1373 | 25.34 | 7.8833 |
| medium | C:/Users/tusik/Documents/GitHub/Architext/src | 72 | 35.1885 | 36.141 | 0.1307 | 0.1364 | 7.1 | 7.7785 |

## Notes
- Query measurements use retrieval diagnostics (no external LLM synthesis latency).
- Memory metric is Python allocation peak via tracemalloc.
- CPU ratio is process CPU time divided by wall-clock time for each run.

JSON artifact: docs/benchmarks/metrics_2026-03-23.json