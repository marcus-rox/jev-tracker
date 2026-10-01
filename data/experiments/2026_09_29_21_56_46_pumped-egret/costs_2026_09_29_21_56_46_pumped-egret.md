# costs 2026_09_29_21_56_46_pumped-egret

Kev/CLM: the shard's own timers (metrics.py) x Modal list price (GPU + reserved host RAM).
warm = loaded server to last answer; in-function = container function start to return.
prod / Jev rows are API token costs from the earlier run (not re-billed here).

USD = whole run (75 queries once); per 1k queries = linear extrapolation.

| ranker | gpu | shards | unmeasured | load s | warm GPU s | in-function s | warm USD | warm USD / 1k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| prod | - | 0 | - | - | - | - | (API / frozen) | |
| kev27b_noul | H100 | 1 | 0 | 30 | 906 | 936 | 1.25 | 16.69 |
| kev27b_rerun_noul | H100 | 1 | 0 | 32 | 930 | 962 | 1.28 | 17.13 |
| kev27b_vllm_noul | None | 1 | 1 | 0 | 0 | 0 | 0.00 | 0.00 |
| kev27b_vllm_floor_noul | H100 | 1 | 0 | 183 | 1946 | 2129 | 2.69 | 35.84 |
