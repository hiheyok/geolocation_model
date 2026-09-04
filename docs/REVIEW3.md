# Codex code review, round three

Static review of the current `retrieval-dropout` branch on 2026-09-04. This
round found **10 confirmed bugs**: five high severity and five medium severity.
Several reopen guarantees marked fixed in `REVIEW.md` or `REVIEW2.md`; the
implementation covers one producer or one consumer, but not the full artifact
lifecycle.

No source files were changed. All 111 Python files parse successfully. The full
pytest suite could not be run because the only `python` on `PATH` is the MSYS2
3.12 interpreter and it has neither pytest nor the project's numeric packages.
The `file_stamp` collision below was reproduced directly with that interpreter.

## High severity

### 1. Any non-release-length street cache bypasses row provenance

[`GeoStepDataset`](../src/dataset.py#L160) checks the street sidecar only when
`len(street) == len(dataset.parquet)`. Every other length is accepted. That is
not limited to a correctly stacked bank: the repository contains standalone
750,000-row `bank_ext*_*.f16.npy` files which contain **no release rows at all**.
Passing one through `--street-file` makes the first 500,000 extension rows serve
as the 500,000 release images. Shapes and indices are valid, so training or
evaluation proceeds with every photograph paired to another photograph's
embedding.

The same bypass means a stacked bank is never checked at consumption time. A
stack can be replaced or reordered after its k-NN cache was built and the
dataset will combine the old neighbour indices with the new rows.

This reopens the row-provenance family closed in round one. The loader needs to
classify the row space: exact release length must match release ids; a stacked
file must match release ids followed by the recorded extension ids; every other
layout must be rejected. A regression test should pass an extension-only cache
whose width is valid and assert that construction fails before the first row is
read.

### 2. `--seed` still does not make sink training reproducible

Training constructs its dataset with the default `neg_random=True` at
[`train.py:438`](../src/train.py#L438). Each `__getitem__` then creates
`np.random.default_rng(None)` at
[`dataset.py:394`](../src/dataset.py#L394), which seeds from OS entropy and is
independent of both `np.random.seed(a.seed)` and `torch.manual_seed(a.seed)`.
Only the validation dataset sets `neg_random=False`; the statement in
`REVIEW2.md` that the training path does so is incorrect.

This affects every shipping-style run with `--neg 4`: two commands with the
same recorded seed see different off-path tiles, and with worker processes the
difference is also scheduling-dependent. The recently added seed field
therefore overstates reproducibility for the main training configuration.

A regression test should construct the same training sample twice under one
seed and compare `neg_step`, `neg_x0`, `neg_y0`, and `neg_tokens`, including with
more than one loader worker.

### 3. The production PCA basis is fitted on validation and test embeddings

[`project_street.py`](../scripts/project_street.py#L94) samples 200,000 rows
from the first `--fit-from` rows, whose default is 500,000. On the s10 release
that is the entire release, not its training split. The script never reads a
split column. With the repository's 80/10/10 split, roughly 20% of the PCA fit
sample is held-out data.

This is the same transductive PCA leakage that round one fixed in
`fuse_head.py` and `width_probe.py`, but it remains in the pipeline that built
`pca768_bank55_pca.npz` and the 768-dimensional retrieval arms. PCA is
unsupervised, so the likely effect is smaller than label leakage, but held-out
queries still influence the feature space in which their similarity to the bank
is measured. Reported comparisons against independently embedded external data
also use this transductively fitted basis.

The fit rows should be selected from the named training split and the split
mode/hash should be stored with the basis. A regression test should make held-out
rows numerically distinctive and prove that changing them cannot change a
training-only basis.

### 4. `stack_bank.py` cannot rebuild the documented multi-extension banks

The documented bank-growth workflow uses an already stacked base, for example
`--base pool_bal_bank25 --ext bank_ext2_pool --out pool_bal_bank40` in
[`merge_bank_meta.py:15`](../scripts/merge_bank_meta.py#L15). Current
`stack_bank.py` requires the base row count to equal the release row count at
[`stack_bank.py:67-73`](../scripts/stack_bank.py#L67). `pool_bal_bank25` has
1,250,000 rows while s10 has 500,000, so the documented command always exits.

Even if that guard were removed, the sidecar written at
[`stack_bank.py:85`](../scripts/stack_bank.py#L85) describes only
`release ++ new_extension`; it omits extensions already present in the base.
The current 2.0M, 2.75M, and 3.5M banks exist because they predate this guard,
but the checked-in tooling cannot reproduce them now.

A chain test should build `release ++ ext1`, then append `ext2`, and verify both
the output data and its digest describe `release ++ ext1 ++ ext2`.

### 5. Completion markers remain valid after inputs or outputs change

Round one item 70 included changed inputs, changed code, and deleted/corrupt
outputs. The replacement identity contains only stage name, argv, and release
at [`runlog.py:48-59`](../src/runlog.py#L48). `Stage.satisfied()` returns true
as soon as that marker matches at
[`overnight.py:203-218`](../scripts/overnight.py#L203); it does not run the
stage's `check`, examine `needs`, or verify an output.

Consequences include:

- rebuilding `dataset.parquet` or a street bank under the same path does not
  invalidate a matching k-NN marker;
- deleting a checkpoint while leaving its marker makes the training stage skip;
- changing implementation code without changing argv reuses results produced
  by the previous implementation.

This is especially dangerous for the same-sequence leak fix: a clean k-NN build
can be replaced by an older cache under the same name while the rebuild stage
continues to report satisfied. Marker identity needs explicit input stamps and
an output validator (or a separate artifact manifest), and tests should mutate
an input and delete an output while leaving argv unchanged.

## Medium severity

### 6. Row-preserving embedding transforms drop or invent provenance

`concat_street.py` and `pool_street.py` write new positional arrays but never
import or call `provenance`; fresh outputs have no sidecars at all
([`concat_street.py:57-77`](../scripts/concat_street.py#L57),
[`pool_street.py:69-95`](../scripts/pool_street.py#L69)). They also never verify
that their inputs describe the same rows. Equal row counts are not sufficient
for a positional concatenation.

`stack_bank.py` has the opposite failure: it writes a `basis="built"` sidecar
without checking either input sidecar first. A reordered base or an extension
embedding belonging to different metadata is copied and then certified as the
intended `release ++ extension` order. `build_knn.py` later trusts that newly
created sidecar, so the check cannot recover the truth.

Each transform should verify every input's row digest and only then carry or
compose those verified digests into the output. Tests should swap two rows in
one input while leaving its length unchanged.

### 7. `file_stamp()` discards the file size it claims to include

[`safeio.file_stamp`](../src/safeio.py#L42) concatenates hexadecimal size and
mtime, then keeps only the last 12 characters:

```python
"{:x}{:x}".format(st.st_size, st.st_mtime_ns)[-12:]
```

A contemporary nanosecond mtime is already more than 12 hex characters, so the
suffix contains only the low 48 bits of mtime. The size contributes nothing.
Using `README.md`'s current mtime, simulated sizes 1 and 999,999,999 both
produce `fa527e4b1f40`; this was reproduced directly.

This weakens both bootstrap error-cache invalidation and multiquery query-cache
invalidation. A rewrite whose timestamp is preserved can change length without
changing the key, contrary to the function's contract and the round-two fix.
Encode the two fields separately (or hash their structured representation) and
test equal-mtime/different-size files.

### 8. A corrupt JSON marker is treated as a trusted legacy success

[`marker_matches`](../src/runlog.py#L62) returns `None` for both a genuine old
plain-text marker and malformed JSON. `Stage.satisfied()` treats every `None`
as satisfied. The current test even classifies `"{not json}"` as a legacy marker
at [`test_runlog.py:99-106`](../tests/test_runlog.py#L99).

Marker writes use non-atomic `Path.write_text` at
[`overnight.py:306`](../scripts/overnight.py#L306), so interruption can create
exactly this malformed file. On restart the stage is skipped as complete.
Only a syntactically non-JSON legacy format should receive the compatibility
path; a string beginning like the new JSON format but failing to decode should
be a mismatch. New-format marker writes should use `safeio.write_text`.

### 9. Training accepts non-binary completion-mask values as complete

The producer now validates that `done.u8.npy` contains only zero or one, but the
training consumer does not. [`_check_fetched`](../src/dataset.py#L108) checks
only `len(done)` and defines missing rows as `done[used] == 0`. A value of 2,
255, or any other nonzero value therefore certifies an unwritten all-zero token
row as fetched. Beam inference is stricter (`done[rows] == 1`), so the same
cache can train on a blank tile and fetch that tile live at evaluation.

The consumer should require shape `(n,)`, dtype/values representing a binary
mask, and exact agreement with the token array's row count. A regression test
should put value 2 on a reachable row and require refusal.

### 10. Two extension consumers bypass the release-checking loader

`provenance.bank_ext()` is the central loader that rejects extension metadata
from a different release. `serve.py` bypasses it on the normal modern-cache
path and calls `np.load(config.bank_meta(ext))` directly at
[`serve.py:74-81`](../scripts/serve.py#L74). `occupancy_probe.py` does the same at
[`occupancy_probe.py:124-129`](../scripts/occupancy_probe.py#L124).

`check_split()` in the server validates the checkpoint against the release; it
does not validate the independently replaceable extension metadata. A metadata
file from another release can therefore attach wrong z16 addresses to valid
bank rows and return plausible but incorrect locations. Both consumers should
use `provenance.bank_ext(stem, config.RELEASE)`, with a regression test using a
matching-length metadata file stamped for another release.

## Review boundary

I did not repeat already documented, deliberately deferred model-definition
issues such as `quality()` seeing a dropped neighbour, `--save-opt` never being
restored, step-3 sink supervision, memory-dropout scaling, or multi-photo keyed
scoring. They remain open, but the findings above are additional bugs or
closures that do not hold under the current code.
