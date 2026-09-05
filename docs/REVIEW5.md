# Codex code review, round five

Static review of commit `e546c90` and the current working tree on 2026-09-05.
No source files were changed. `e546c90` landed in the shared workspace during
the review; its conditioning adapter, builders, and tests are included here.
The pre-existing untracked `runs/BOOTSTRAP_tiledrisk.md` was left untouched.

The full suite passes under the requested launcher:

```text
OSV_RELEASE=s10 py -3.13 -m pytest -q
494 passed in 8.77s
```

All 138 Python files under `src`, `scripts`, and `tests` also parse successfully.
The passing suite does not exercise the failure paths below. This round found
**ten new bugs** (six high severity and four medium severity). It also
confirmed that **twelve findings from REVIEW4 remain open**; those are listed
after the new findings so that an open bug is not mistaken for a new discovery.

## New high-severity findings

### 1. `embed_native` resumes a cache without proving which images or settings it contains

[`embed_native.py:94-116`](../scripts/embed_native.py#L94) opens an existing
output whenever its shape is `(n, 1536)` and loads its completion mask. It does
not read or validate the metadata written at lines 201-212: `rows_parquet`,
`rows_digest`, `crops`, encoder names, and input sizes are all ignored on resume.
The shape cannot distinguish any of those inputs because the script mean-pools
every encoder to one 768-dimensional vector regardless of crop count.

This creates two silent corruption paths:

- Reusing `--out cond_native` with a different same-length `--parquet` accepts
  completed rows from the old image list. If the mask is partial, old completed
  rows are retained while unfinished rows are filled from the new list, creating
  a mixed cache with a valid binary mask.
- Reusing the output with a different `--crops` returns `nothing to do` when the
  mask is complete, even though the requested representation was never built.

The early return at lines 120-122 occurs before any metadata comparison, so a
fully completed stale cache is the easiest case to accept. The resume path
should compare every result-defining setting and the ordered image-ID digest
before retaining a single row. Tests should cover a same-length image-list swap,
a crop-count change, and a partial mask that would otherwise mix generations.

### 2. `query_only` bypasses both checkpoint and k-NN validation

[`query_only.py:167-197`](../scripts/query_only.py#L167) loads an arbitrary
checkpoint, its bank, and `bank_rows` directly. It never calls
`evaluate.check_split`, `provenance.check`, or `knnmeta.check`. Consequently it
does not verify the checkpoint release, the live split hash, the k-NN
`street_file`, split mode/hash, query count, neighbour width, extension identity,
or bank-row bounds.

Negative `bank_rows` wrap from the end at lines 196-197. A same-shaped cache from
another split or extension therefore produces finite embeddings, sequences, and
coordinates for the wrong photographs. A checkpoint from another release is
also evaluated against the current release selected by `OSV_RELEASE`; the
script never rejects that cross-release benchmark.

This is a new direct consumer of the exact contract centralized in
[`knnmeta.py`](../src/knnmeta.py). It should use the shared validator and the
same checkpoint/release/split checks as `serve.py` and `eval_highres.py` before
indexing anything. Regression tests should include negative rows, a mismatched
street filename, a different split hash, an extension swap, and an s01
checkpoint under s10.

### 3. `pool_pyramid` can overwrite a good output and leave its old sidecar certifying the replacement

[`pool_pyramid.py:217-237`](../scripts/pool_pyramid.py#L217) truncates the final
destination and writes the new array before calling `prov.carry` at lines
254-257. `prov.carry` is the first operation that checks whether the crop
source's sidecar describes the expected row count. If that check fails, the
old output bytes are already gone. If the source has no sidecar, `prov.carry`
only warns and returns without writing a new one.

Neither path removes a sidecar already present beside `--out`. A previously
valid `pyr_l0.f16.npy.prov.json` can therefore remain beside newly written data
from an unverified or mismatched source and cause downstream consumers to accept
the replacement as the old provenanced artifact. This repeats the publication
bug fixed for concat and stack in REVIEW4, now on the new production path used
by `tilebig.py`.

All source/provenance checks should happen before opening the destination, and
the data should be published from a temporary file only after verification. A
regression should begin with a valid output and sidecar, use a source with a
wrong row record and then one with no sidecar, and prove the old generation is
either preserved or made unambiguously untrusted.

### 4. `pyr_levels` silently includes unfinished pyramid rows in its measurements

[`pyr_levels.levels`](../scripts/pyr_levels.py#L64) reads the full pyramid array
and normalizes every row. Unlike `pyr_blend.py`, `pyr_gem.py`, and
`pyr_perstep.py`, the script never opens `<stem>_done.u8.npy` or verifies that
the cache is complete. `pyramid_cache.py` creates the full-size array before it
finishes filling it, so interruption leaves well-shaped zero rows behind.

Those zero rows survive as zero vectors after the clamped normalization and
enter both the query and bank partitions at lines 137-146. The result remains
finite and looks like an ordinary retrieval degradation; nothing reports that
the experiment measured an unfinished artifact. The script should require a
valid complete mask (or an explicit verified legacy exception) before pooling.
A test should supply a full-shaped cache with one unwritten row and a zero in
the mask and require refusal.

### 5. `make_cond` certifies conditioning rows that were never matched to the retrieval rows

[`make_cond.py:68-89`](../scripts/make_cond.py#L68) proves only that retrieval
and conditioning arrays have the same row count and that the conditioning mask
is complete. It never compares the conditioning sidecar's `rows_digest` with
the retrieval sidecar. At lines 114-115 it calls `prov.carry` on the retrieval
file alone, so the output receives the retrieval file's authoritative row
digest even if row *i* of the conditioning file belongs to another photograph.

A reordered or same-length replacement conditioning cache therefore becomes a
fully provenanced joined artifact. Every array shape and completion check passes,
and training reads the wrong photograph's high-resolution features as if they
belonged to the current retrieval row. The output is also opened at lines 95-96
before `prov.carry` validates the retrieval source, leaving the same stale-output
sidecar hazard described in finding 3.

Both input sidecars must be present, valid, and equal before the destination is
opened; the conditioning metadata/done-mask identity should be bound into the
published sidecar as well. Tests should swap two conditioning rows while keeping
the same count and a complete mask, and should verify that a missing or corrupt
conditioning sidecar cannot be laundered through retrieval provenance.

### 6. The advertised old-checkpoint fine-tune is rejected by `train.py`

The conditioning adapter is explicitly designed to start from an existing
unconditioned checkpoint. The new test at
[`test_cond_adapter.py:122-133`](../tests/test_cond_adapter.py#L122) confirms
that loading such a state dict produces five expected missing adapter keys.
However, the actual `--init` path in
[`train.py:559-575`](../src/train.py#L559) permits only the older `street.gate`,
geography, and sink additions. It does not add `street.cond_gate`,
`street.cond_proj.{weight,bias}`, or `street.cond_norm.{weight,bias}` to the
allowed additive set.

Therefore the intended command—train a conditioned cache with `--init` pointing
at the incumbent checkpoint—loads with `strict=False`, sees those five missing
keys, and immediately exits as an architecture mismatch. The unit test exercises
`load_state_dict` directly and never passes the result through the production
filter, so the suite stays green while the main experiment cannot start.

The initialization policy needs to recognize exactly those zero-gated adapter
parameters as an allowed strict superset, and an integration test should invoke
the same compatibility helper used by `train.main`, not merely PyTorch's raw
loader. It should still reject a checkpoint missing any non-adapter parameter.

## New medium-severity findings

### 7. `query_only --tag` accepts checkpoints whose embedding recipe it cannot reproduce

The CLI exposes an unrestricted `--tag`, but
[`query_only.pipeline`](../scripts/query_only.py#L106) always constructs exactly
one representation: DINOv2 plus SigLIP, three-crop mean, SigLIP multiplied by
`scale_b`, and an optional PCA. At line 204 it obtains the scale from
`ck.get("scale_b", 4.03)`, but no checkpoint in the repository records that
field. Local inspection found checkpoints using `dual_c3`, `embeddings_c3`,
`dual_bal`, pooled, PCA, and pyramid street spaces, all with `scale_b=None`.

The default checkpoint happens to name `pca768_bank70`, whose basis was fitted
from a balanced input, but other accepted tags do not share that recipe. For an
unbalanced `dual_c3` checkpoint the script multiplies SigLIP by 4.03 anyway; for
a single-encoder or pyramid checkpoint it constructs a different semantic space
and may fail only at matrix multiplication, or produce same-width but
meaningless cosine scores.

The query recipe should be derived from structured street/basis provenance and
the script should refuse unsupported spaces. A filename-independent test should
cover balanced dual, unbalanced dual, single encoder, pyramid, and PCA bases
fitted from each.

### 8. `pyr_levels` tunes and reports L2 weights on the same query cohort

Weights supplied through `--w2` are appended to the arms at
[`pyr_levels.py:150-154`](../scripts/pyr_levels.py#L150), and every arm is scored
and compared on the same `te` rows at lines 166-193. There is no selection/report
split. The purpose stated by the module is to sweep the L2 weight, and the
commit-level conclusion selects the near-neutral low-weight region; choosing
that region and quoting its interval on the same observations includes the
noise that favored it.

Sibling scripts already split whole sequences into tuning and reporting halves
for this reason. `pyr_levels` should do the same and base all claimed comparisons
on the untouched reporting half. A test can use synthetic arms whose apparent
winner changes between halves and assert that selection never reads reporting
outcomes.

### 9. `gain_density` and `knn_gap` compare caches to each other, not to the requested live split

[`gain_density.py:76-85`](../scripts/gain_density.py#L76) chooses test rows from
the CLI's `--split-mode`, then reads neighbour arrays without checking either
cache's split metadata at all. [`knn_gap.py:81-100`](../scripts/knn_gap.py#L81)
checks that its two files agree with each other on `split_mode`, `split_hash`,
and `bank_n`, but never checks that either agrees with `a.split_mode` or with
the current dataset's split hash.

Two stale sequence caches therefore agree with each other and can be scored as
`--split-mode cell8`; their neighbours were selected from the sequence-training
bank, while the reported queries come from the cell8 test side. The result is a
plausible comparison for a benchmark that was never built. Both scripts should
validate each cache against the live dataset and requested split through
`knnmeta.check`. Tests should pair two mutually agreeing sequence caches with a
cell8 request and require refusal, then mutate the live split hash while leaving
both files unchanged.

### 10. `embed_native` labels a 224-pretrained SigLIP model as native at 512

The script describes both encoders as running at their native training
resolution, but [`embed_native.py:68-69`](../scripts/embed_native.py#L68) uses
`vit_base_patch16_siglip_224.v2_webli` with `img_size=512`. The model identifier
itself identifies the 224 input variant, and the existing shipping code also
documents SigLIP's native size as 224. Passing 512 to `timm.create_model` makes
the architecture accept a larger grid; it does not turn a 224-pretrained
position embedding into a model trained natively at 512.

Thus the proposed comparison is not “native SigLIP versus forced 224.” It is
224-pretrained SigLIP with positional interpolation and four times as many
patches versus its training resolution. That can still be a useful experiment,
but the mechanism and interpretation are different. The code should either use
a checkpoint actually pretrained at the requested resolution or record the
pretraining resolution and explicitly label the arm as interpolated high
resolution. A metadata test should reject a claim that runtime `img_size` alone
defines native resolution.

## Confirmed open findings from REVIEW4

These were rechecked against current `HEAD`. They remain reachable bugs; none of
the relevant source files changed enough to close them. Numbers refer to
[`REVIEW4.md`](REVIEW4.md).

| REVIEW4 | Severity | Current status |
|---:|:---:|---|
| 1 | High | Still open in [`dataset.py:376-384`](../src/dataset.py#L376): the dataset reads the k-NN extension coordinates without calling `knnmeta.check_ext`, so same-length extension embeddings and coordinates can still be crossed. |
| 2 | High | Still open in [`train.py:559-580`](../src/train.py#L559): `--init` checks state-dict compatibility but not release, split, street space, map cache, or k-NN identity before stamping the child as the current run. |
| 6 | High | Still open in [`runlog.py:75-114`](../src/runlog.py#L75): stage input discovery misses `--a`, `--b`, `--src`, `--parquet`, `--parts`, `--tokens`, `--pyr-stem`, and `--seed-from`, and resolves through the process release rather than `Stage.release`. |
| 7 | High | Still open in [`runlog.py:117-165`](../src/runlog.py#L117): inferred-output stages such as `build_knn` and tile fetching record no output, and an empty output map is treated as intact. |
| 10 | High | Still open in [`bootstrap.py:122-152`](../scripts/bootstrap.py#L122): cached errors fingerprint street/k-NN files but omit map tokens/index/mask and dataset/target authorities. |
| 13 | High | Still open across [`build_dataset.py:150-165`](../scripts/build_dataset.py#L150) and the bank-extension builder: paired artifacts are published separately and row identity binds IDs only, so same-ID coordinate/sequence corrections can mix generations. |
| 16 | High | Still open in [`dataset.py:302-310`](../src/dataset.py#L302) and [`beam.py:53-83`](../src/beam.py#L53): map index rows and keys are not validated for negativity, bounds, uniqueness, or one-to-one coverage. |
| 8 | Medium | Only the producer half changed. [`concat_street.py:104-112`](../scripts/concat_street.py#L104) now records `combined`, but [`provenance.encoder_of`](../src/provenance.py#L256) still discards it. PCA reuse and serving therefore cannot consume the second encoder/scale identity. |
| 9 | Medium | Still open in [`provenance.py:283-306`](../src/provenance.py#L283): malformed or unreadable sidecars collapse to `None` and are accepted as legacy absence. |
| 15 | Medium | Still open in [`summary_table.py:54-57`](../scripts/summary_table.py#L54): teacher-forced loss omits the checkpoint's soft targets and `sink_w`. |
| 18 | Medium | Still open in [`dataset.py:313-330`](../src/dataset.py#L313) and the server: targets are reshaped positionally without validating the explicit per-image `step` sequence. |
| 22 | Medium | Still open in `StreetProj`: `--enc-gate` infers two semantic encoder blocks from even width alone and accepts single-encoder multi-crop or PCA-mixed inputs. |

The REVIEW4 triage counted partial work on #1 and #8 toward its fix total, but
both remain actionable bugs. Including them, the current review boundary
contains **twenty-two open findings: ten new and twelve carried forward**.

## Review boundary

I reviewed the current core modules, recent remediation commits, newly added
pyramid/query/runner scripts, and the untracked native-resolution encoder. I did
not execute GPU training, multi-gigabyte cache builds, or live tile-server calls.
Existing data and checkpoints were inspected read-only where useful. Generated
and source artifacts were not modified; this report is the only file added.
