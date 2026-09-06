# Where the errors live: coverage, retrieval, or ranking

`knn_pool_bal_bank70_sequence_k32_bank_ext70.npz`, split `test`, 49,788 queries against a 3,400,180-row eligible bank.

The predictor is **rank-1 retrieval**, which scores 56.4% `<25 km` where the full agent scores 57.5% -- so this measures the system supplying almost all of the accuracy, with no checkpoint and no beam. Same-sequence rows are excluded from the coverage test as well as from retrieval.


## A_r(K): oracle ceiling for returning a candidate's coordinates

| r | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | any eligible bank row |
|---|---|---|---|---|---|---|---|
| **1 km** | 10.5% | 13.9% | 17.6% | 21.5% | 25.6% | 29.9% | 81.4% |
| **25 km** | 50.1% | 58.5% | 66.1% | 72.5% | 78.3% | 83.3% | 99.8% |
| **200 km** | 69.9% | 77.5% | 83.9% | 89.0% | 92.6% | 95.5% | 100.0% |

## The partition

| r | coverage | retrieval | ranking | solved |
|---|---|---|---|---|
| **1 km** | 18.6% | 51.5% | 19.5% | 10.5% |
| **25 km** | 0.2% | 16.6% | 33.2% | 50.1% |
| **200 km** | 0.0% | 4.5% | 25.6% | 69.9% |

class 1 needs corpus; class 2 needs a better representation or a larger K; class 3 is the only one a reranker over the cached shortlist can reach.

## Conditional on being covered

* **1 km** — 81.4% of queries have a legal bank row within 1 km. Of those, retrieval puts one in the top-32 for 36.8%, and ranks it first for 12.9%.
* **25 km** — 99.8% of queries have a legal bank row within 25 km. Of those, retrieval puts one in the top-32 for 83.4%, and ranks it first for 50.2%.
* **200 km** — 100.0% of queries have a legal bank row within 200 km. Of those, retrieval puts one in the top-32 for 95.5%, and ranks it first for 69.9%.

## Nearest legal bank row

| percentile | km |
|---|---|
| p10 | 0.12 |
| p25 | 0.19 |
| p50 | 0.34 |
| p75 | 0.75 |
| p90 | 1.76 |
| p99 | 9.24 |
