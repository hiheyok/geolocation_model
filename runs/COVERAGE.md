# Where the errors live: coverage, retrieval, or ranking

`knn_pca768_bank70_sequence_k32_bank_ext70.npz`, split `test`, 49,788 queries against a 3,400,180-row eligible bank.

The predictor is **rank-1 retrieval**, which scores 56.4% `<25 km` where the full agent scores 57.5% -- so this measures the system supplying almost all of the accuracy, with no checkpoint and no beam. Same-sequence rows are excluded from the coverage test as well as from retrieval.


## A_r(K): oracle ceiling for returning a candidate's coordinates

| r | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | any eligible bank row |
|---|---|---|---|---|---|---|---|
| **1 km** | 10.4% | 13.8% | 17.4% | 21.5% | 25.6% | 29.7% | 81.4% |
| **25 km** | 49.7% | 58.1% | 65.6% | 72.0% | 78.0% | 82.9% | 99.8% |
| **200 km** | 69.4% | 77.1% | 83.6% | 88.6% | 92.6% | 95.3% | 100.0% |

## The partition

| r | coverage | retrieval | ranking | solved |
|---|---|---|---|---|
| **1 km** | 18.6% | 51.7% | 19.3% | 10.4% |
| **25 km** | 0.2% | 16.9% | 33.2% | 49.7% |
| **200 km** | 0.0% | 4.7% | 25.8% | 69.4% |

class 1 needs corpus; class 2 needs a better representation or a larger K; class 3 is the only one a reranker over the cached shortlist can reach.

## Conditional on being covered

* **1 km** — 81.4% of queries have a legal bank row within 1 km. Of those, retrieval puts one in the top-32 for 36.5%, and ranks it first for 12.7%.
* **25 km** — 99.8% of queries have a legal bank row within 25 km. Of those, retrieval puts one in the top-32 for 83.0%, and ranks it first for 49.8%.
* **200 km** — 100.0% of queries have a legal bank row within 200 km. Of those, retrieval puts one in the top-32 for 95.3%, and ranks it first for 69.4%.

## Nearest legal bank row

| percentile | km |
|---|---|
| p10 | 0.12 |
| p25 | 0.19 |
| p50 | 0.34 |
| p75 | 0.75 |
| p90 | 1.76 |
| p99 | 9.24 |
