# costs 2026_09_29_09_45_47_causal-urchin

Kev/CLM: the shard's own timers (metrics.py) x Modal list price (GPU + reserved host RAM).
warm = loaded server to last answer; in-function = container function start to return.
prod / Jev rows are API token costs from the earlier run (not re-billed here).

USD = whole run (6 queries once); per 1k queries = linear extrapolation.

| ranker | gpu | shards | unmeasured | load s | warm GPU s | in-function s | warm USD | warm USD / 1k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| prod | - | 0 | - | - | - | - | (API / frozen) | |
| kev27b_noul | H100 | 1 | 0 | 29 | 110 | 139 | 0.15 | 25.28 |
| kev27b_rerun_noul | H100 | 1 | 0 | 28 | 113 | 140 | 0.16 | 25.92 |
| kev27b_vllm_noul | H100 | 1 | 0 | 185 | 878 | 1063 | 1.21 | 202.07 |
| kev27b_vllm_floor_noul | H100 | 1 | 0 | 171 | 203 | 373 | 0.28 | 46.63 |
