# costs 2026_09_28_17_56_26_actual-clam

Kev/CLM: the shard's own timers (metrics.py) x Modal list price (GPU + reserved host RAM).
warm = loaded server to last answer; in-function = container function start to return.
prod / Jev rows are API token costs from the earlier run (not re-billed here).

USD = whole run (75 queries once); per 1k queries = linear extrapolation.

| ranker | gpu | shards | unmeasured | load s | warm GPU s | in-function s | warm USD | warm USD / 1k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| prod | - | 0 | - | - | - | - | (API / frozen) | |
| jev_noul | - | 0 | - | - | - | - | (API / frozen) | |
| jev_score | - | 0 | - | - | - | - | (API / frozen) | |
| kev4b_noul | L40S | 10 | 0 | 149 | 1127 | 1276 | 0.61 | 8.14 |
| kev4b_score | L40S | 10 | 0 | 142 | 1202 | 1344 | 0.65 | 8.69 |
| kev9b_noul | H100 | 10 | 0 | 451 | 880 | 1331 | 0.97 | 12.87 |
| kev9b_score | H100 | 10 | 0 | 310 | 939 | 1249 | 1.03 | 13.73 |
| kev27b_noul | H200 | 10 | 0 | 354 | 1984 | 2338 | 3.07 | 40.87 |
| kev27b_score | H200 | 10 | 0 | 302 | 2120 | 2422 | 3.28 | 43.68 |
