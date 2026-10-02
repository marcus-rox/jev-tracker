# costs 2026_09_29_00_07_02_cuddly-horse

Kev/CLM: the shard's own timers (metrics.py) x Modal list price (GPU + reserved host RAM).
warm = loaded server to last answer; in-function = container function start to return.
prod / Jev rows are API token costs from the earlier run (not re-billed here).

USD = whole run (75 queries once); per 1k queries = linear extrapolation.

| ranker | gpu | shards | unmeasured | load s | warm GPU s | in-function s | warm USD | warm USD / 1k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| prod | - | 0 | - | - | - | - | (API / frozen) | |
| jev_noul | - | 0 | - | - | - | - | (API / frozen) | |
| jev_score | - | 0 | - | - | - | - | (API / frozen) | |
| kev27b_batched_c32_noul | H100 | 3 | 0 | 93 | 937 | 1030 | 1.29 | 17.26 |
| kev27b_FP8_w8a8_c32_noul | H100 | 3 | 0 | 88 | 978 | 1066 | 1.35 | 18.01 |
| kev27b_FP8_w8a8_c64_noul | H100 | 3 | 0 | 104 | 1049 | 1153 | 1.45 | 19.32 |
| kev27b_FP8_w8a8_c32_score | H100 | 3 | 0 | 110 | 1309 | 1419 | 1.81 | 24.11 |
| kev27b_FP8_w8a8_c64_score | H100 | 3 | 0 | 103 | 1354 | 1457 | 1.87 | 24.93 |
