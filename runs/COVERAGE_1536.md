# Where the errors live: coverage, retrieval, or ranking

`knn_pool_bal_bank70_sequence_k32_bank_ext70.npz`, split `test`, 49,788 queries against a 3,400,180-row eligible bank.

Predictor: **rank-1 retrieval**, scoring **50.1% `<25 km`** on this cohort (49,788 queries, all of the `test` split). No checkpoint, no beam, no GPU. The agent's number is measured on a different cohort and is deliberately not quoted here: the comparison that would mean anything is a paired one on these same ids. Same-sequence rows are excluded from the coverage test as well as from retrieval.

Extension coordinates are the source parquets' true `lat`/`lon`, not z16 tile centres.


## A_r(K): oracle ceiling for returning a candidate's coordinates

| r | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | any eligible bank row |
|---|---|---|---|---|---|---|---|
| **1 km** | 10.5% | 13.9% | 17.7% | 21.6% | 25.8% | 30.1% | 81.5% |
| **25 km** | 50.1% | 58.5% | 66.0% | 72.5% | 78.3% | 83.3% | 99.8% |
| **200 km** | 69.9% | 77.5% | 83.9% | 89.0% | 92.6% | 95.5% | 100.0% |

## The partition

| r | coverage | retrieval | ranking | solved |
|---|---|---|---|---|
| **1 km** | 18.5% | 51.4% | 19.6% | 10.5% |
| **25 km** | 0.2% | 16.6% | 33.2% | 50.1% |
| **200 km** | 0.0% | 4.5% | 25.6% | 69.9% |

class 1 needs corpus; class 2 needs a better representation or a larger K; class 3 is the only one a reranker over the cached shortlist can reach.

## Conditional on being covered

* **1 km** — 81.5% of queries have a legal bank row within 1 km. Of those, retrieval puts one in the top-32 for 36.9%, and ranks it first for 12.9%.
* **25 km** — 99.8% of queries have a legal bank row within 25 km. Of those, retrieval puts one in the top-32 for 83.4%, and ranks it first for 50.2%.
* **200 km** — 100.0% of queries have a legal bank row within 200 km. Of those, retrieval puts one in the top-32 for 95.5%, and ranks it first for 69.9%.

## Nearest legal bank row

| percentile | km |
|---|---|
| p10 | 0.08 |
| p25 | 0.15 |
| p50 | 0.32 |
| p75 | 0.74 |
| p90 | 1.77 |
| p99 | 9.35 |
