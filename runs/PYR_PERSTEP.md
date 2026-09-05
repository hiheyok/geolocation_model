# Does the blend weight want to change with the zoom step? No.

`scripts/pyr_perstep.py`, pyr47, 3,000 queries against a 38,009 bank. The
weighting for each row is chosen on 1,478 queries and reported on the other
1,522, split by sequence.

The agent uses the retrieval prior once per step, over a descent whose cell
goes 2504 km -> 156 km -> 9.8 km -> 611 m, and the components disagree about
scale. So `w_t` is the obvious next move. It does not survive.

**Headroom is zero or negative at three of five thresholds.** The best
weighting *for a threshold*, chosen on the selection half, is compared to the
single global weighting on the reporting half:

| threshold | headroom |
|---|---|
| 1 km | +0.46 pp |
| 25 km | +0.00 pp |
| 200 km | **-0.59 pp** |
| 750 km | +1.51 pp |
| 2500 km | **-0.07 pp** |

Scored without the held-out split the same sweep looked uniformly positive
(+0.53 / 0 / 0 / +0.40 / +0.27). That difference is the whole result: per-step
tuning fits the selection half and does not transfer.

**The plumbing would not have been the obstacle.** Where headroom exists it is
fully captured by retrieving once under the global weighting and re-ranking the
cached top-32 per step -- 100% of it at 1 km, 750 km and 2500 km. So four banks
were never needed, and if a better head ever makes the headroom real, the cheap
design is already known to reach it.

**Do not build per-step weighting.** Revisit only if a component is added whose
scale specialisation is much stronger than the head's.

---

```text
pyr47  47,646 rows, levels [ 3  6 24]
3,000 queries  38,009 bank
1,478 queries choose the weighting, 1,522 report it

components + similarities in 24s
swept 126 weightings in 120s

single global weighting, chosen on <25 km over the selection half:
  L 4:0:1  head 0.10

Every figure below is the REPORTING half. Each row's weighting was
chosen on the selection half, so the headroom is what per-step
tuning would actually transfer -- not what it fits.

threshold  best weighting for it               best    global  headroom
------------------------------------------------------------------------
1   km s3 L 3:1:1  head 0.10                18.53%    18.07%    +0.46 pp
25  km s2 L 4:0:1  head 0.10                33.90%    33.90%    +0.00 pp
200 km s1 L 1:2:2  head 0.05                52.17%    52.76%    -0.59 pp
750 km s1 L 2:1:2  head 0.10                73.78%    72.27%    +1.51 pp
2500km s0 L 1.5:0:1  head 0.10              89.03%    89.09%    -0.07 pp

--- one retrieval (global weighting, top-32), re-ranked per threshold ---
threshold     global re-ranked      gain of headroom
----------------------------------------------------
1   km       18.07%     18.53%    +0.46 pp      100%
25  km       33.90%     33.90%    +0.00 pp       n/a
200 km       52.76%     52.63%    -0.13 pp       22%
750 km       72.27%     73.78%    +1.51 pp      100%
2500km       89.09%     89.03%    -0.07 pp      100%

--- global weighting against the published level mean ---
  +0.53[-0.7,+1.7]~ +1.18[-0.6,+3.0]~ +3.22[+1.2,+5.2]  +2.30[+0.3,+4.3]  +2.56[+1.1,+4.1] 

~ marks an interval spanning zero. 125s total
```
