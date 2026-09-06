# Where the errors live: coverage, retrieval, or ranking

`knn_pca768_bank70_sequence_k32_bank_ext70.npz`, split `test`, 49,788 queries against a 3,400,180-row eligible bank.

Predictor: **rank-1 retrieval**, scoring **49.7% `<25 km`** on this cohort (49,788 queries, all of the `test` split). No checkpoint, no beam, no GPU. The agent's number is measured on a different cohort and is deliberately not quoted here: the comparison that would mean anything is a paired one on these same ids. Same-sequence rows are excluded from the coverage test as well as from retrieval.

Extension coordinates are the source parquets' true `lat`/`lon`, not z16 tile centres.


## A_r(K): oracle ceiling for returning a candidate's coordinates

| r | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | any eligible bank row |
|---|---|---|---|---|---|---|---|
| **1 km** | 10.4% | 13.9% | 17.5% | 21.6% | 25.7% | 29.8% | 81.5% |
| **25 km** | 49.7% | 58.1% | 65.6% | 72.0% | 78.0% | 82.9% | 99.8% |
| **200 km** | 69.4% | 77.1% | 83.6% | 88.6% | 92.6% | 95.3% | 100.0% |

## The partition

| r | coverage | retrieval | ranking | solved |
|---|---|---|---|---|
| **1 km** | 18.5% | 51.6% | 19.4% | 10.4% |
| **25 km** | 0.2% | 16.9% | 33.2% | 49.7% |
| **200 km** | 0.0% | 4.7% | 25.8% | 69.4% |

class 1 needs corpus; class 2 needs a better representation or a larger K; class 3 is the only one a reranker over the cached shortlist can reach.

## Conditional on being covered

* **1 km** — 81.5% of queries have a legal bank row within 1 km. Of those, retrieval puts one in the top-32 for 36.6%, and ranks it first for 12.8%.
* **25 km** — 99.8% of queries have a legal bank row within 25 km. Of those, retrieval puts one in the top-32 for 83.0%, and ranks it first for 49.8%.
* **200 km** — 100.0% of queries have a legal bank row within 200 km. Of those, retrieval puts one in the top-32 for 95.3%, and ranks it first for 69.4%.

## Nearest legal bank row

| percentile | km |
|---|---|
| p10 | 0.08 |
| p25 | 0.15 |
| p50 | 0.32 |
| p75 | 0.74 |
| p90 | 1.77 |
| p99 | 9.35 |
