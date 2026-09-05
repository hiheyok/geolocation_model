# Codex code review, round six

Static review of commit `b2978bd` and the current working tree on 2026-09-05.
No source files were changed. The pre-existing untracked
`runs/BOOTSTRAP_tiledrisk.md` was left untouched.

The full suite passes with the requested launcher:

```text
OSV_RELEASE=s10 py -3.13 -m pytest -q
497 passed in 8.85s
```

All 138 Python files under `src`, `scripts`, and `tests` parse successfully.
This review also used two read-only executable probes. They confirmed that a
conditioned `dual` retrieval call crashes on a 12-wide neighbour tensor when
the retrieval projection is 8-wide, and that the deterministic 64-row prefix
check accepts a changed row outside its sample.

This round found **ten additional bugs: seven high severity and three medium
severity**. Every finding below includes proposed regression tests. The first
section records prior-review disposition so fixed items are not repeated as new.

## REVIEW5 disposition

| REVIEW5 | Status at `b2978bd` |
|---:|---|
| 1 | **Fixed.** `embed_native` now records result-defining settings before work and checks them before retaining completed rows. New findings 7 and 8 below cover different defects in that revised path. |
| 2 | **Open.** `query_only` still bypasses checkpoint, split, release, provenance, and shared k-NN validation. |
| 3 | **Open.** `pool_pyramid` still truncates the final output before `prov.carry` validates its source and can leave an old sidecar beside replacement bytes. |
| 4 | **Open.** `pyr_levels` still consumes the full pyramid without requiring a complete done-mask. |
| 5 | **Open.** `make_cond` still compares input lengths rather than row digests and launders the conditioning block through retrieval provenance. |
| 6 | **Fixed.** The production `init_from` policy now permits exactly the five zero-gated conditioning-adapter parameters and has non-vacuous tests. |
| 7 | **Open.** `query_only --tag` still applies one hard-coded balanced dual/PCA query recipe to checkpoints from other embedding spaces. |
| 8 | **Open.** `pyr_levels` still selects and reports L2 weights on the same cohort. |
| 9 | **Open.** `gain_density` and `knn_gap` still compare cache metadata only with each other, not with the requested live split and split hash. |
| 10 | **Fixed.** The native encoder now uses the actual `vit_base_patch16_siglip_512.v2_webli` checkpoint. |

The twelve carried-forward REVIEW4 findings listed in REVIEW5 also remain open,
including cross-benchmark `--init`, incomplete runner input/output identities,
bootstrap cache inputs, map-index validation, malformed-provenance fail-open,
target step ordering, and `--enc-gate` semantic validation.

## New high-severity findings

### 1. Conditioned `pos` and `dual` models slice the query prefix but not neighbour embeddings

[`GeoAgent.retr_prior`](../src/model.py#L234) correctly removes the conditioning
suffix from `street` at lines 250-257. It then passes `nbrs[3]` unchanged to
`RetrievalPrior.weights` at lines 261-262. The neighbour table is loaded from
the checkpoint's combined street file in
[`train.py:553-555`](../src/train.py#L553) and
[`evaluate.py:325-327`](../src/evaluate.py#L325), so each gathered neighbour is
`retrieval_dim + cond_dim` wide.

The positive and negative key projections were constructed for `retrieval_dim`.
Their linear layers therefore receive the wider combined neighbour tensor and
raise on the first batch. A direct probe reproduced the production shape:

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (6x12 and 8x4)
```

The current test only asserts that some retrieval parameter is retrieval-width;
it never calls the keyed prior with conditioned neighbour embeddings. Both the
query and every neighbour must be sliced to the retrieval prefix before keyed
weighting. Loading only the prefix into `street_table` would also avoid carrying
unused conditioning bytes through every gather.

Proposed tests:

- `test_conditioned_dual_prior_slices_query_and_neighbours`: construct an
  8-wide retrieval plus 4-wide conditioning model, pass 12-wide query and
  neighbour tensors, and assert the result has the requested policy width and
  equals a call made with explicit 8-wide prefixes.
- Parametrize the same test over `pos` and `dual`; include `scalar` and `cond`
  controls so the test covers all four retrieval modes.
- `test_conditioned_train_batch_with_keyed_retrieval`: execute one synthetic
  `run_epoch` batch through the actual neighbour-gather path and assert forward,
  backward, and optimizer step complete without a dimension error.

### 2. The 64-row retrieval-prefix check misses localized rebuilds by construction

[`_check_retrieval_prefix`](../src/dataset.py#L94) claims to establish that the
joined prefix *is* the current retrieval cache byte for byte, but lines 126-139
compare only a deterministic sample of at most 64 rows. On a 500,000-row cache,
499,936 rows are never examined. A partial rebuild, block corruption, or changed
row outside that fixed sample passes.

This was reproduced with 1,000 rows: row 0 was selected specifically because it
is outside the function's deterministic 60 distinct sampled rows, then changed
in the retrieval file. `_check_retrieval_prefix` returned successfully:

```text
mismatched row passed 0 sampled 60
```

This is silent, not probabilistic in repeated runs: the seed is fixed, so the
same rows are ignored forever. The joined file then supplies old query/key
vectors while the current retrieval cache and possibly its k-NN table describe
new ones.

Proposed tests:

- `test_prefix_check_rejects_a_change_outside_the_old_sample`: reproduce the
  1,000-row case above and require rejection.
- `test_join_records_full_retrieval_content_identity`: after `make_cond`, assert
  the sidecar contains a content identity for the entire retrieval prefix and
  that changing any block invalidates it.
- If full hashing is intentionally avoided, test a block-manifest design by
  changing the first and last row of every block, including blocks not touched
  by the old 64-row sample.

### 3. A k-NN cache is not bound to the bytes of the street cache that produced it

[`build_knn.py:283-289`](../scripts/build_knn.py#L283) records the street
filename, split metadata, and bank rows, but no street content stamp or digest.
Consumers compare only the filename at
[`dataset.py:418-427`](../src/dataset.py#L418) and in `knnmeta.check`.

This leaves a failure that even a complete prefix check cannot catch:

1. Build k-NN cache `K` from retrieval file `R_old`.
2. Rebuild the same path with `R_new` and rebuild the conditioned join from it.
3. The join prefix matches the current `R_new` exactly, the k-NN metadata still
   says the same filename, and every current check passes.
4. `idx` and `sim` in `K` still describe neighbours selected in `R_old` space.

Training and evaluation then use stale neighbour IDs and similarities with new
query and key vectors. All dimensions, row digests, split hashes, and filenames
remain plausible. A k-NN artifact needs to record the street generation/content
identity at construction, and every consumer needs to compare it.

Proposed tests:

- `test_knn_rejects_street_rebuilt_under_same_name`: create `R_old`, construct a
  small cache carrying its identity, replace the path with same-shaped `R_new`,
  and require `GeoStepDataset` and `knnmeta.check` to reject it.
- `test_conditioned_join_does_not_mask_stale_knn`: rebuild both retrieval and
  joined prefix while retaining the old k-NN file; verify the k-NN generation
  check still fails even though the prefix check passes.
- `test_knn_identity_changes_for_same_size_same_mtime_rewrite`: if the chosen
  identity is only size/mtime, preserve both and show why a content or immutable
  generation digest is required for an authoritative contract.

### 4. Training never verifies that cached neighbours belong to the recorded bank

The dataset reads `idx` and checks only that values fall inside the combined
address-table range at [`dataset.py:432-455`](../src/dataset.py#L432). It does
not validate `bank_rows`, `bank_n`, or membership of each neighbour in that
bank. It also does not recheck the same-sequence exclusion that is central to
the benchmark.

A cache can therefore record a perfectly valid train bank in `bank_rows` while
placing validation/test rows, the query itself, or another same-sequence row in
`idx`. Every index is non-negative and in range, so construction succeeds and
retrieval leaks held-out or self information into training/evaluation. The
shared `knnmeta.bank_rows` validator checks the row list's bounds and uniqueness,
but it does not check `idx` membership, and `GeoStepDataset` does not call it in
this path anyway.

Proposed tests:

- `test_dataset_rejects_neighbour_outside_bank_rows`: set `bank_rows=[0,1]`
  and place row 2 in an otherwise valid `idx`; require dataset construction to
  fail before the first item is read.
- `test_dataset_rejects_self_and_same_sequence_neighbours`: create a train
  query whose cached neighbour is itself, and an extension neighbour sharing
  its sequence; require both to fail even though each belongs to `bank_rows`.
- `test_idx_and_sim_contract`: require identical shapes, at least the requested
  width, finite similarities, descending rank order, and one legal bank member
  per slot.

### 5. Conditioned training derives the wrong default k-NN filename

When `--retr` is enabled without an explicit `--knn-file`,
[`train.py:478-481`](../src/train.py#L478) calls `config.knn_name` with
`a.street_file`. For a conditioned cache that is the joined filename, for
example `pyr768_mix_cond.f16.npy`. The k-NN was deliberately built on the
sidecar's `retrieval_file`, `pyr768_mix.f16.npy`, not on the combined file.

The dataset knows how to compare k-NN metadata against `self.retr_file`, but it
cannot reach that check because training first derives and attempts to open
`knn_pyr768_mix_cond_sequence_k32.npz`, a file that should not exist. Thus the
advertised decoupled path works only if the operator supplies a redundant,
easy-to-forget override. The default should be derived from the validated
retrieval prefix recorded by the street sidecar.

Proposed tests:

- `test_default_knn_name_uses_conditioned_retrieval_file`: give a joined
  sidecar `retrieval_file=r.f16.npy`, omit `--knn-file`, and assert training
  resolves `knn_r_sequence_k32.npz`, not `knn_joined_sequence_k32.npz`.
- `test_explicit_knn_still_must_match_prefix`: pass an explicit cache built on
  another street file and assert the normal metadata check rejects it.
- Add a parser-to-dataset smoke test for the exact documented make-cond/train
  sequence so filename derivation is tested outside isolated helpers.

### 6. `serve.py` cannot load or embed queries for a conditioned checkpoint

[`serve.load_everything`](../scripts/serve.py#L42) resolves the checkpoint's
combined street file and loads every full-width row into `STATE["bank"]` at
lines 106-115. That already violates the design: retrieval should search only
the recorded prefix, never the conditioning suffix.

The request encoder then targets `STATE["bank"].shape[1]` in
[`serve.py:278-290`](../scripts/serve.py#L278). The current design is 768
retrieval plus 1,536 conditioning = 2,304 dimensions. `embed` can produce only
the raw 4,608 dual-crop vector, its 1,536 pooled form, or a 768 PCA projection.
None is 2,304, so it raises `bank is 2304-d; no projection to match`. Even if a
combined width happened to match one branch, searching it would put conditioning
back into cosine, contradicting the experiment.

Serving also uses the 224 encoders for every request and has no path to build the
518/512 conditioning suffix expected by the model. A conditioned checkpoint
therefore needs two query encodings: the shipping retrieval recipe for neighbour
search and the recorded native-resolution recipe appended only for policy input.

Proposed tests:

- `test_server_conditioned_bank_uses_only_retrieval_prefix`: load a synthetic
  8+4 joined bank and assert `STATE["bank"]` is 8-wide while policy street
  tensors are 12-wide.
- `test_server_builds_conditioning_with_recorded_recipe`: use fake encoders to
  assert retrieval is produced by the retrieval recipe and conditioning by the
  separate recorded 518/512 recipe, in the correct concatenation order.
- `test_conditioning_never_changes_neighbour_ids`: alter only the conditioning
  suffix and assert server top-k retrieval is bit-identical.

### 7. Non-positive `embed_native --batch` publishes untouched rows as complete

`--batch` is accepted as any integer at
[`embed_native.py:136-139`](../scripts/embed_native.py#L136). With `--batch 0`
or a negative value, the queue fill condition at lines 229-234 is false from the
start and no image is ever decoded or embedded. Both encoder loops finish without
an exception. Lines 266-278 then mark every `todo` row done, write
`complete=True`, and publish provenance even though the memmap is still its
zero fill.

There is no final zero/finite-row scan to catch this. `--crops 0` is a sibling
input-validation hole that can produce an empty mean and NaNs or an incidental
backend failure rather than a clean refusal.

Proposed tests:

- Parametrize `test_embed_native_rejects_nonpositive_batch` over `0` and `-1`
  and assert refusal occurs before output or metadata creation.
- Parametrize an equivalent positive-integer guard over `--crops` and
  `--workers`.
- `test_embed_native_never_marks_unwritten_rows_done`: fake the encoder loop so
  it processes zero rows and assert the mask remains zero, `complete` remains
  false, and no provenance sidecar is published.
- `test_embed_native_final_scan_rejects_zero_and_nonfinite_rows`: inject one
  zero, NaN, and infinity row in separate cases and require failure.

## New medium-severity findings

### 8. A no-work resume rewrites `complete=True` metadata to `False`

On every invocation, [`embed_native.py:180-186`](../scripts/embed_native.py#L180)
writes metadata with `complete=False`. It calculates `todo` afterwards and, when
the done-mask is already full, returns at lines 187-192. The final rewrite to
`complete=True` at lines 271-273 is never reached.

Merely checking or resuming a completed cache therefore changes truthful
metadata to claim the cache is incomplete. Current `make_cond` happens to trust
the mask instead of this field, but the field was added precisely to describe
publication state and a future or external consumer can correctly reject the
cache after a harmless no-op invocation.

Proposed tests:

- `test_completed_native_resume_preserves_complete_metadata`: create a full
  binary mask and `complete=True` metadata, invoke the no-work path, and assert
  the field remains true.
- `test_partial_resume_sets_false_until_final_publish`: use a partial mask and
  assert false is written before work, then true only after all rows are flushed.
- `test_complete_field_agrees_with_mask`: parameterize complete, partial, and
  non-binary masks and require metadata/mask agreement or explicit refusal.

### 9. The `torch.compile` fallback catches setup errors but not lazy compilation failures

[`embed_native.py:205-210`](../scripts/embed_native.py#L205) wraps only the call
to `torch.compile(net)`. PyTorch compilation is lazy: backend/Triton/codegen
errors commonly occur on the first `net(x)` at line 245, outside the `try`.
Despite printing that compile unavailability falls back to eager, the common
failure point aborts the embedding pass instead.

This is recoverable on resume, but it defeats the advertised fallback and can
waste the first shard preload/decodes before failing. Either compilation should
be probed with a representative batch before the main loop, with controlled
eager fallback, or the CLI should state that compile failures are fatal and
require `--no-compile`.

Proposed tests:

- `test_lazy_compile_failure_falls_back_before_writing`: monkeypatch
  `torch.compile` to return a wrapper that raises a backend error on first call;
  assert eager execution is selected and no row is marked done twice.
- `test_model_error_is_not_misclassified_as_compile_error`: make eager and
  compiled execution both fail for a real shape error and assert it propagates
  rather than being hidden by fallback.

### 10. Other row-preserving transforms repeat REVIEW5's stale-sidecar publication bug

REVIEW5 #3 identified `pool_pyramid`, but the same ordering remains in two more
production transforms. [`pool_street.py:70-99`](../scripts/pool_street.py#L70)
and [`project_street.py:163-176`](../scripts/project_street.py#L163) truncate the
final destination before `prov.carry` validates source provenance. If that late
check fails—or the source has no sidecar and `carry` only warns—the old output
sidecar is left beside replacement bytes.

`project_street` now publishes its PCA basis carefully, but its projected array
and provenance still have this earlier failure window. Both utilities are used
to build the shipping bank, so fixing only `pool_pyramid` would leave the same
certification bug on the next and previous representation paths.

Proposed tests:

- Parametrize `test_transform_preserves_old_generation_on_bad_source` over
  pool-street, pool-pyramid, and project-street: start with a valid output plus
  sidecar, supply a source with a wrong row count/digest, and assert both old
  files remain byte-identical.
- Repeat with an absent and malformed source sidecar; require refusal rather
  than a new output accompanied by stale provenance.
- Simulate interruption after the first output block and assert the final path
  still names the previous complete generation while only a visibly temporary
  file is partial.

## Review boundary

The review covered all core `src` modules, the conditioning adapter/builders,
k-NN construction and consumption, training, evaluation, serving, provenance,
completion masks, and row-preserving transforms. GPU training, multi-gigabyte
cache rebuilds, and live tile-server calls were not executed. Proposed tests are
designs only; no tests or source files were added or modified.
