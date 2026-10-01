# costs 2026_09_28_23_33_26_busy-sawfly

Kev/CLM: the shard's own timers (metrics.py) x Modal list price (GPU + reserved host RAM).
warm = loaded server to last answer; in-function = container function start to return.
prod / Jev rows are API token costs from the earlier run (not re-billed here).

USD = whole run (75 queries once); per 1k queries = linear extrapolation.

| ranker | gpu | shards | unmeasured | load s | warm GPU s | in-function s | warm USD | warm USD / 1k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| prod | - | 0 | - | - | - | - | (API / frozen) | |
| jev_noul | - | 0 | - | - | - | - | (API / frozen) | |
| jev_score | - | 0 | - | - | - | - | (API / frozen) | |
| kev27b_bf16_noul | H100 | 3 | 0 | 98 | 934 | 1032 | 1.29 | 17.20 |
| kev27b_bf16_score | H100 | 3 | 0 | 93 | 1158 | 1251 | 1.60 | 21.33 |
| kev27b_FP8_wo_noul | H100 | 3 | 0 | 93 | 2415 | 2508 | 3.34 | 44.47 |
| kev27b_FP8_wo_score | H100 | 3 | 0 | 102 | 2685 | 2787 | 3.71 | 49.45 |
| kev27b_FP8_w8a8_noul | H100 | 3 | 0 | 101 | 1021 | 1122 | 1.41 | 18.80 |
| kev27b_FP8_w8a8_score | H100 | 3 | 0 | 98 | 1284 | 1382 | 1.77 | 23.65 |
