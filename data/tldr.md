Summary:
Jev is still the best ranker on the frozen 75-query benchmark: kept-mass 0.893 at k=50 and 0.961 at k=200. No open model matches it yet.
- The closest open model is Kev-27B at 0.887 / 0.953: 0.006 behind Jev at k=50, and about 8x the cost per run.
- Watch k=50: every Kev size tracks Jev at k=200 but loses most of its ground at k=50.

Key Points:
- Today's run (2026-10-02) added Kev-0.8B, Kev-9B v2 and an unofficial Kev-2B; all land well behind, best 0.753 / 0.889 for Kev-9B v2.
- Kev-27B v2 (weights of 9/30) is 0.013 below the pinned Kev-27B at k=50 and half its cost per run.
- Laya encoders (322M-421M) cost under $0.13 per run but rank far worse: 0.644 at k=50 at best.
- Production (gpt-5-mini, 3 passes) sits at 0.711 / 0.837 and is the most expensive run on the board.

Table:
| Model | kept-mass@50 | kept-mass@200 | $ / run | GPU |
|---|---|---|---|---|
| Jev | 0.893 | 0.961 | 0.46 | API |
| Kev-27B | 0.887 | 0.953 | 3.71 | H100 |
| Kev-27B v2 | 0.874 | 0.951 | 1.87 | H200 |
| Kev-9B v2 | 0.753 | 0.889 | 0.42 | H100 |
| Kev-4B | 0.726 | 0.881 | 0.65 | L40S |
| production | 0.711 | 0.837 | 4.73 | API |
| Kev-2B (per2021) | 0.684 | 0.868 | 0.18 | L4 |
| Kev-0.8B | 0.642 | 0.846 | 0.11 | L4 |
| Laya-421M typed | 0.644 | 0.850 | 0.12 | L4 |

Interesting Notes:
- Kev-9B v2 beats Kev-9B at k=50 (0.753 vs 0.732) at less than half the cost; at k=200 they are equal.
- Random order already keeps 0.749 at k=200, so k=200 separates models little; k=50 is the metric that does.
