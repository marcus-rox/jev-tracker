Jev is still the best ranker on the frozen 75-query benchmark (kept-mass 0.893 at k=50, 0.961 at k=200). Nothing open has matched it yet.
The closest alternatives are the Kev checkpoints: the pinned Kev-27B (9/24 weights) reaches 0.887 / 0.954 at about $0.90 per run of 75 queries, while the Kev-27B v2 weights published on 9/30 are about 0.01 lower.
Today's run (2026-10-02) added Kev-0.8B, Kev-9B v2 and an unofficial Kev-2B: all well behind (best 0.753 / 0.889 for Kev-9B v2 noul), each under $0.60 per run.
Laya (322M–421M encoders) is far cheaper but ranks much worse; production (gpt-5-mini, 3 passes) sits at 0.711 / 0.837.
