# costs 2026_09_29_10_43_20_united-snake

Kev/CLM: the shard's own timers (metrics.py) x Modal list price (GPU + reserved host RAM).
warm = loaded server to last answer; in-function = container function start to return.
prod / Jev rows are API token costs from the earlier run (not re-billed here).

USD = whole run (6 queries once); per 1k queries = linear extrapolation.

| ranker | gpu | shards | unmeasured | load s | warm GPU s | in-function s | warm USD | warm USD / 1k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| prod | - | 0 | - | - | - | - | (API / frozen) | |
| kev27b_noul | H100 | 1 | 0 | 29 | 115 | 144 | 0.16 | 26.40 |
| kev27b_rerun_noul | H100 | 1 | 0 | 45 | 115 | 160 | 0.16 | 26.42 |
| kev27b_sglang_noul | H100 | 1 | 0 | 68 | 3926 | 3994 | 5.42 | 903.72 |
| kev27b_sglang_floor_noul | H100 | 1 | 0 | 79 | 382 | 461 | 0.53 | 87.91 |
