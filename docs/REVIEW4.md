# Codex code review, round four

Static review of commit `d4dfe26` on 2026-09-04. No source files were
changed. This round found **22 confirmed bugs**: fourteen high severity and eight
medium severity. Most are seam failures in guarantees added during rounds one
through three: each component has a provenance record, but two records are not
compared; or a marker records some inputs and outputs, but not the ones a
particular command actually uses.

The full suite passes under the requested launcher:

```text
OSV_RELEASE=s10 py -3.13 -m pytest -q
355 passed in 5.48s
```

All 111 Python files also compile. Passing tests do not cover the failure paths
below. Findings are ordered by impact, not by file.

> **Triage, 2026-09-04 13:45.** Read this before working the list.
>
> **UPDATE 2026-09-05 05:05. 11 fixed: #3, #4, #5, #11, #12, #14, #17, #19,
> #20, #21, and half of #1.**
>
> **#19 fixed.** `tile_cache` now digests the ids at the rows it covers, in
> cache order, and re-derives that from the current list on every resume;
> `pool_pyramid` re-checks it on consumption, reading the parquet the metadata
> names, because a digest that is written and never read is not a guard.
> Missing metadata is reported and allowed rather than refused -- `tile6`
> predates both fields and holds 500,000 finished rows -- but never treated as
> agreement. The regression the review asks for is in both places: reorder a
> same-length synthetic dataset, keep the cache files, require resume *and*
> consumption to fail. `tests/test_tile_rows_parquet.py`,
> `tests/test_pool_pyramid.py`; 445 pass.
>
> **#14 fixed, and it carries most of #1.** `src/knnmeta.py` is the shared
> validator the review asks for, wired into `serve.py`, `eval_highres.py` and
> `multiquery.py`. It adds two checks nothing had: a negative `bank_rows` entry
> wrapped to the end of the bank instead of being refused, and the cache's
> `bank_ext` is now compared against the street file's recorded extensions --
> **by ordered id digest, not by name**, because a bank recording four stems
> and a cache naming the combined `bank_ext70` are the same corpus and a name
> comparison would reject production. Verified against live artifacts.
> `tests/test_knnmeta.py`, 18 cases.
>
> **#21 fixed.** The query cache now aggregates one `file_stamp` per *selected*
> image into its stamp, so replacing a JPEG under an unchanged manifest
> re-embeds. The size-and-mtime blind spot is pinned by its own test rather
> than papered over. `tests/test_query_stamp.py`, 9 cases. 475 pass.
>
> **#1 half done. The remaining half is blocked until the current chain
> finishes.**
> `knnmeta.check_ext` implements the comparison, but `src/dataset.py` does not
> yet call it --
> `src/dataset.py:139` validates the *street file's* extension list through
> `prov.exts_of`, and `:314` reads the *k-NN cache's* `bank_ext`. Grep confirms
> the two are never compared, so the review's scenario stands exactly as
> written. It is directly on the path `scripts/tilebig.py` takes
> (`build_knn --bank-ext`), so it is next. `src/dataset.py` is imported by the
> training chain running now, and editing a file mid-chain is what voided
> `seqfix2`; the fix lands in the tiling window, hours before any rung starts.
>
> **7 fixed so far: #3, #4, #5, #11, #12, #17, #20** (`[x]` below). All are in
> files the running chain does not import. It guards the KartaView benchmark, which is
> the selection benchmark and which `seqfix3` is about to measure against, so
> it could not wait.
>
> **#20 was verified latent, not active, for everything published.** All twelve
> `runs/hr_*_n5k.npz` exports were checked directly and share **exactly the
> same 5,000 image ids**; the manifest has not grown since 2026-09-02, and
> today's hash rule reproduces the published cohort 5,000/5,000. Every KartaView
> comparison on record is therefore on one cohort and "select on KartaView"
> stands. The cohort is now frozen as
> `E:/data/kartaview_hr/cohort_seed0_n5000.json`, digest `14680b9ed911`,
> replayed by every later run and exported into each npz — pinning what was
> already true rather than changing it. The false claim in the source comment
> ("stable as the manifest grows") is corrected; hashing fixes the *ordering*,
> not the *membership*.
>
> **#15 checked, contained.** Six checkpoints carry `soft != 0`
> (`soft0.25`, `soft0.5`, `soft1.0`, `soft_fine`, `soft_fine2`, `soft_grad`);
> none carry `sink_w != 1`. They are the archived soft-label arms, already a
> recorded negative result. Teacher-forced accuracies are unaffected, so the
> conclusion stands — but any `summary_table` loss column printed for those six
> was hard CE, not their objective.
>
> **Everything else is latent.** Each remaining item needs an interruption, a
> rename, a swap, or a rebuild that has not happened. None of them says a
> current number is wrong. That is the opposite of the same-sequence leak and
> of round three's #3, both of which were active.
>
> **Ordering constraint.** `scripts/seqfix3.py` is training, and STATE.md §11
> forbids editing code a running chain has not finished importing — that rule
> exists because doing so voided the previous clean-bank result. Items whose
> files that chain does not touch (3, 4, 5, 8, 11, 12, 13, 15, 17, 21) can be
> worked while it runs; the rest (1, 2, 6, 7, 9, 10, 14, 16, 18, 19, 22) must
> wait for it to finish, because they touch `train.py`, `dataset.py`,
> `bootstrap.py`, `runlog.py` or `eval_highres.py`.
>
> **#8 is half done.** `concat_street` now records a structured
> `combined` identity naming both encoders and the exact scale. The other
> half -- teaching `provenance.encoder_of` to return it, so the PCA basis
> check and the server actually read it -- touches `provenance.py`, which
> the running chain imports, so it waits.
>
> **#6 and #7 are corrections to my round-three fix.** I made stage
> input/output discovery generic and rejected per-stage declaration as "a big
> refactor". The reviewer is right that generic discovery covers almost none of
> the real stages: `build_knn` records no outputs at all, so its marker still
> survives a deleted cache. The refactor is the fix.

## High severity

### 1. A k-NN cache can attach one extension's coordinates to another extension's embeddings

[`GeoStepDataset`](../src/dataset.py#L313) independently validates two things:
the street file's row space at construction, and the k-NN cache's
`bank_ext` metadata when it builds the coordinate arrays. It never checks that
they name the same extension. The street file may prove it is
`release ++ bank_ext`, while the k-NN file says `bank_ext2`; line 318 then loads
`bank_ext2_meta.npz` and lines 319-322 append those coordinates to the address
table for the first file's embedding rows.

This passes every current guard when the extensions have the same length. That
is not hypothetical: `bank_ext`, `bank_ext2`, `bank_ext3`, and `bank_ext4` on
disk each contain exactly 750,000 rows. The cache's `street_file` string can
still match, the query count is unchanged, and every neighbour index remains
in range. Retrieval then gives each visually matched embedding a different
image's z16 address, producing plausible but wrong priors and metrics.

The consumer should compare the k-NN cache's `bank_ext` against the street
sidecar's recorded extension (or, better, compare an ordered bank-row digest
stored in the cache). A regression test should pair a
`release ++ bank_ext` street sidecar with a same-length k-NN cache stamped
`bank_ext2` and require construction to fail.

### 2. `--init` can launder a checkpoint from another benchmark into the current one

Training loads `--init` directly at [`train.py:554-570`](../src/train.py#L554)
and checks only state-dict keys and tensor shapes. It does not check the source
checkpoint's release, split mode/hash, street embedding space, map cache, or
k-NN cache against the new run. The output checkpoint is then stamped with the
*current* release, split hash, street file, and k-NN file at
[`train.py:637-677`](../src/train.py#L637), retaining only the source tag in
`init_from`.

The split case breaks benchmark integrity. Initialising a `cell8` arm from a
`sequence` arm imports weights already trained on images that the cell8 run
calls validation/test. `evaluate.check_split()` correctly refuses that direct
cross-split evaluation, but `train.py --init` bypasses the check and the new
checkpoint subsequently looks like a clean cell8 model. Same-shaped but
different street representations are also accepted, so a model can be
continued with its learned input projection attached to another coordinate
space.

Training should validate the source checkpoint before loading it. Deliberate
transfer needs a separate explicit override that is recorded as contamination,
not the ordinary continuation path. Tests should cover a different split hash,
a different release, and a same-width/different-projection street cache.

### 3. `[x]` An interrupted street-embedding pass publishes a trusted partial artifact

[`embed_street.py`](../scripts/embed_street.py#L187) creates the full-sized
memmap, then writes its `basis="built"` provenance sidecar at lines 196-203
*before embedding the first image*. The exhaustive zero-row scan is only
reached after all shards finish at lines 256-273. If the process is killed,
the full-shaped, partially zero-filled array and authoritative-looking sidecar
remain on disk.

`GeoStepDataset` validates the row digest but never validates embedding
completion, so a direct training/evaluation command accepts this artifact.
The runner's `emb_ok` samples only 96 rows, which is not a completion record and
is not used outside that runner. Thus the all-row scan that closed review item
45 cannot detect the interruption it was intended to cover: the interrupted
process never executes it.

The builder needs a temporary destination plus atomic publish, or a durable
completion mask/manifest that consumers require. Provenance should be written
only after the data is flushed and verified. A regression test should simulate
termination after a middle block and prove the resulting path is not
consumable.

### 4. `[x]` Embedding transforms overwrite their destination before validating inputs

Both [`concat_street.py`](../scripts/concat_street.py#L58) and
[`stack_bank.py`](../scripts/stack_bank.py#L51) open the destination with
`mode="w+"` and copy all data before checking input provenance. Concat checks
at lines 83-95; stack does not begin reconstructing and checking row identity
until lines 72-124. Their zero-row checks are also after the destructive open.

If the destination already has a valid sidecar, a mismatched input causes the
command to fail only *after* replacing the data. The old sidecar is not removed.
A later consumer therefore sees newly written bad/partial bytes accompanied by
the previous successful run's valid row digest and accepts them. A check meant
to prevent bad publication instead corrupts an existing good artifact and
leaves it certified.

All input validation must precede opening the destination, and publication
should use a temporary data file and sidecar committed only after verification.
A regression test should start with a valid output and sidecar, run a transform
with deliberately mismatched input provenance, and verify the original output
is unchanged.

### 5. `[x]` Concatenation certifies the result when the second input has no provenance

[`concat_street.py:83-95`](../scripts/concat_street.py#L83) refuses only when
*both* sidecars exist and their digests differ. If A has a sidecar and B does
not, it prints a warning and calls `prov.carry(pa, out, ...)`, which writes A's
row digest onto the combined output. The result now looks fully provenanced
even though nothing established that B contains the same ordered rows.

The asymmetry makes this especially easy to miss: missing provenance on A
leaves the result unstamped, while missing provenance on B launders the result
through A. This directly reopens the positional-concatenation part of review
item 46.

Concat should write a sidecar only after both inputs have valid, matching row
records; otherwise it should refuse or leave the output explicitly unverified.
A regression test should exercise both one-sidecar permutations, not just two
matching or two mismatching sidecars.

### 6. Runner input stamps omit the actual inputs of common stages and can stamp the wrong release

[`runlog.stage_inputs`](../src/runlog.py#L79) recognises only
`--street-file`, `--knn-file`, `--basis`, `--bank`, and `--init`. It misses, among
others, concat's `--a/--b`, pool/project's `--src`, embedding's `--parquet`,
metadata merge's `--parts`, pyramid fusion's `--tokens/--pyr-stem`, and tile
fetching's `--seed-from`. A direct probe of current commands confirmed:

```text
concat inputs: code, dataset.parquet
pool inputs:   code, dataset.parquet
```

Rebuilding either concat component or the pool source under the same name does
not invalidate its marker, so the stage is skipped with stale output.

There is a second bug in the same helper: it resolves paths through the
process-global `config`, not `Stage.release`. `overnight.py` runs s10 prep and
then explicit s01 control stages, but an s01 stage's identity stamps the s10
dataset/street paths whenever the parent was launched under s10. The subprocess
gets the correct `OSV_RELEASE`; its marker does not describe those inputs.

Input discovery should be stage-specific (or supplied explicitly by each
`Stage`) and resolve paths under the stage's release. Tests should mutate every
declared input for concat, pool, projection, embedding, and the s01 control.

### 7. Runner output validation records no output for major successful stages

[`runlog.stage_outputs`](../src/runlog.py#L117) understands only `--out`,
`--export`, and training's `--tag`, and only a few path/name forms for those
flags. Outputs inferred by a script are invisible. For the current
`build_knn.py --street-file dual_c3.f16.npy --split-mode sequence` command, the
helper returns an empty dictionary. Fetching tiles is likewise unrecorded, and
`merge_bank_meta --out bank_ext40` writes `bank_ext40_meta.npz`, a form the
generic resolver does not consider.

[`outputs_intact`](../src/runlog.py#L146) explicitly treats an empty output map
as intact. Deleting or replacing one of these outputs therefore leaves its
completion marker satisfied, despite review round three claiming deleted or
changed outputs now invalidate a stage. The next dependent stage either fails
later or, if its own marker also exists, the whole stale chain is skipped.

Each stage should declare its expected output paths rather than infer them from
generic flags. At minimum, regression tests should delete the inferred k-NN
file, map token cache, and merged metadata while retaining their markers and
assert that each stage becomes unsatisfied.

## Medium severity

### 8. Combined-encoder provenance omits the second encoder and cannot supply the scale the server requests

`concat_street` carries A's metadata and adds `joined_with` plus `scale_b` at
[`concat_street.py:95`](../scripts/concat_street.py#L95). However,
[`provenance.encoder_of`](../src/provenance.py#L256) returns only `model`,
`crops`, and `size`. It discards both `joined_with` and `scale_b`. Consequently:

- `project_street`'s reused-basis check at lines 99-105 sees only encoder A.
  Two concatenations with the same DINO input but a different SigLIP model or
  scale compare equal, so a PCA basis fitted in one combined space is accepted
  for another.
- `serve.py` asks `encoder_of()` for `siglip_scale` at
  [`serve.py:133-136`](../scripts/serve.py#L133), but that key can never be
  returned and the producer writes it under a different name anyway. Serving
  therefore still selects the scale from `"bal"`/`"pca768"` filename
  substrings, reopening review item 20 for any renamed output.

Combined provenance needs a structured identity for both inputs and the exact
scale, with one field name shared by producer, PCA validator, and server. Tests
should show that changing only B or `--scale-b` rejects basis reuse and that a
renamed balanced cache still configures serving correctly.

### 9. A malformed provenance sidecar is treated exactly like no sidecar and can fail open

[`provenance.read`](../src/provenance.py#L283) returns `None` for missing JSON,
malformed JSON, and read errors. [`provenance.check`](../src/provenance.py#L293)
treats every `None` as the backward-compatible "absent" case: it warns and
returns success to the caller. For a release-length street cache, dataset
construction therefore accepts a corrupt sidecar. For a release-length base in
`stack_bank`, the same behavior allows the transform to continue and stamp a
new authoritative sidecar onto its output.

Round three already distinguished corrupt new-format completion markers from
legacy markers for this exact reason. Provenance needs the same three states:
absent may retain the migration warning, present-and-valid may pass, and
present-but-unreadable must refuse. Tests should cover truncated JSON and a
sidecar path that cannot be read.

### 10. Bootstrap error-cache keys omit map and benchmark artifacts

[`bootstrap.inputs_stamp`](../scripts/bootstrap.py#L122) says it fingerprints
everything outside the checkpoint that determines errors, but lines 147-151
hash only the checkpoint's street and k-NN files. Rollout also depends on the
map tokens/index/completion mask and on `dataset.parquet`/`targets.parquet` for
the sampled rows, ground-truth coordinates, and actions.

The pre-cache `provenance()` call validates release and the split-label hash
only. That hash contains labels in row order, not image IDs, coordinates,
targets, or map contents. Rebuilding/correcting a map cache therefore returns
the old per-image errors immediately. Reordering images within the same split
or correcting coordinates can do the same; on a cache miss the street-row
provenance might refuse the new dataset, but a cache hit bypasses dataset
construction entirely and serves the stale numbers.

The error key should include stamps for the selected map cache's three files
and the dataset/target authorities (ideally content manifests rather than only
mtime). A regression test should mutate each while keeping the checkpoint,
street file, and k-NN file unchanged and require a cache miss.

## Additional findings from the second pass

## High severity

### 11. `[x]` `tile_cache` can certify an incomplete cache when its resume mask is non-binary

The other completion-mask producers validate both shape and the exact value set
`{0, 1}`, but [`tile_cache.py:152-180`](../scripts/tile_cache.py#L152) loads its
old mask without either check. It copies `np.load(done_p)[:len(old)]` into the
new mask and defines work as only rows equal to zero. A value such as `2` is
therefore treated as already complete.

There are two silent success paths. If every entry is nonzero, lines 183-185
return before final verification or metadata publication. If a mask contains
one `2` and one missing `0`, the missing row is rebuilt as `1`, then
`done.sum()` is greater than or equal to `n`; the completion test at lines
287-298 passes even though the mask is invalid. The fusion consumer compounds
this at [`fuse_head.py:190-200`](../scripts/fuse_head.py#L190) by converting the
mask to `bool`, so `2` explicitly certifies a possibly zero-filled row.

`tile_cache` needs the same shape/binary validator already used by
`fetch_tiles` and `pyramid_cache`, and the no-work branch must still perform
final verification and rewrite metadata. A regression test should use a mask
with one `2` and one `0` and require the build and fusion load to refuse it.

### 12. `[x]` `fetch_tiles --seed-from` validates the destination mask but trusts the source cache

The normal resume path goes through `_load_mask`, which checks shape and binary
values. The seed path at [`fetch_tiles.py:168-185`](../scripts/fetch_tiles.py#L168)
instead loads the source index, mask, and tokens directly. Its only validation
is the token width. It does not check the source mask's shape or values, the
source index's row range/uniqueness, or that those rows address the source token
array.

This is silent for important corruptions: a negative source row indexes the
last mask/token row, and any nonzero mask value (including `2` or `255`) passes
`if sdone[sr]`. The copied token is then marked `done[row] = 1` in the
destination. A stale source index whose row is in range similarly copies a
different tile's valid-looking tensor and permanently blesses it in the new
cache.

The seed source should be opened through a shared cache validator that proves
index schema, unique tile keys, row bounds, token shape, and an exact binary
mask before copying anything. Tests should cover `row=-1`, a non-binary source
mask, and a same-shaped source index with duplicate or stale row assignments.

### 13. Same-ID rebuilds can mix new coordinates with stale targets or bank metadata

The image-ID checks added after the earlier reviews detect reorderings, but not
value changes for the same images. [`build_dataset.py:150-165`](../scripts/build_dataset.py#L150)
publishes `dataset.parquet` and its sidecar before it builds or writes
`targets.parquet`. If a rebuild corrects latitude/longitude for existing IDs
and is interrupted in between, [`dataset.py:260-273`](../src/dataset.py#L260)
accepts the old targets because it compares only image IDs. Training then uses
the old path/action/click target while evaluation measures against the new
latitude/longitude. Both files and both sidecars look valid.

The same failure exists for bank extensions. [`build_bank_ext.py:114-127`](../scripts/build_bank_ext.py#L114)
writes the parquet first and the address/sequence metadata second. An
interruption can leave a newly embedded parquet beside old metadata with the
same IDs. `provenance.bank_ext` checks only the metadata's release; the
embedding sidecar also identifies rows only by ID. Corrected coordinates or
sequences are therefore invisible, even though they determine neighbour votes,
held-cell filtering, and same-sequence exclusion.

Each paired build needs a generation/content manifest that binds all
result-defining columns, followed by atomic publication of that manifest only
after every temporary output is durable. Tests should rebuild the same IDs with
changed coordinates (and changed extension sequence), stop after the first
artifact, and prove consumers reject the mixed generation.

## Medium severity

### 14. `[x]` Serving and external evaluators bypass the k-NN cache checks used by the dataset

`GeoStepDataset` validates a k-NN cache's split mode/hash, street filename,
query count, width, and neighbour range before use. The direct consumers do
not use that path. [`serve.py:74-86`](../scripts/serve.py#L74) reads
`bank_rows` and ignores the cache's `split_mode`, `split_hash`, `street_file`,
and `bank_n`. [`eval_highres.py:286-297`](../scripts/eval_highres.py#L286) and
[`multiquery.py:376-387`](../scripts/multiquery.py#L376) likewise turn the rows
straight into a boolean search mask.

Consequently a stale or replaced cache of the right apparent name can select a
different split or bank while serving/evaluation continues normally. Negative
rows also wrap from the end in NumPy instead of being refused. `check_split` in
the server validates the checkpoint against the live dataset, not this
independently replaceable k-NN file, so it does not close the gap.

The k-NN metadata contract should have one validator shared by dataset,
server, and external evaluators. It should also bind the ordered bank rows to
the street artifact with a digest, not just a filename. Tests should swap in a
same-shaped cache with a different split hash/street file and supply negative
and out-of-range bank rows to every consumer.

### 15. `summary_table` reports the wrong loss for soft-target and custom sink-weight arms

Training and validation pass the parsed soft-target temperatures and
`--sink-w` into `run_epoch` at [`train.py:604-611`](../src/train.py#L604), and
both settings are recorded in checkpoints. The summary's
[`teacher_forced`](../scripts/summary_table.py#L54) calls `run_epoch` with
neither, so it silently uses hard cross-entropy and `sink_w=1.0`. The datasets
still include the checkpoint's configured number of negatives, which makes
the mismatch especially misleading: a sink loss is present, just with the
wrong coefficient.

Teacher-forced accuracies are unaffected, but `loss trn`, `loss tst`, and
`d-loss` at lines 153-164 are not the saved arm's training/validation objective
whenever `soft != 0` or `sink_w != 1`. This can reverse comparisons among the
ablation arms the table exists to compare.

The helper should parse the checkpoint's `soft` field exactly as `train.main`
does and pass it plus `sink_w` to `run_epoch`. A unit test can use fixed logits
where hard and soft CE differ, and a negative batch where changing `sink_w`
changes only the reported loss.

## Additional findings from the third pass

## High severity

### 16. Map-cache index rows are not validated and negative rows silently select other tiles

[`GeoStepDataset`](../src/dataset.py#L240) packs each index address and builds a
dictionary directly from the index's `row` column. It never checks that rows are
non-negative, unique, within the token array, or one-to-one with the index.
`_check_fetched` then indexes the completion mask with those same rows at line
193. A negative row therefore wraps from the end in both the completion mask
and the token memmap: the wrong tile is marked complete and returned as a
perfectly valid tensor.

Beam inference has the sibling gap at [`beam.py:53-82`](../src/beam.py#L53).
Its mask-length guard considers only `rows.max()`, so negative rows evade the
bound and again wrap in `done[rows]` and `self.tokens[row]`. Duplicate tile keys
also collapse silently through `dict(zip(...))`, while duplicate row values can
make several addresses share one token. A same-sized, permuted row column passes
every existing check and makes training and inference consistently read the
wrong maps.

The index/token/mask triplet needs one shared validator: legal unique tile
addresses, integer rows exactly covering the expected range (or at least unique
and bounded), token-row count, and binary mask count must agree. Regression
tests should cover `row=-1`, duplicate keys, duplicate rows, and a permutation
of otherwise valid rows in both dataset and beam consumers.

### 17. `[x]` `project_street` can publish a new PCA basis while leaving the old bank certified

When fitting rather than reusing a basis, [`project_street.py:139-154`](../scripts/project_street.py#L139)
writes `<out>_pca.npz` before it opens and rebuilds `<out>.f16.npy`. If the
process stops after the basis save, the old projected bank and its old valid
provenance sidecar remain, but the basis path named by that sidecar now contains
new components.

Serving and external evaluation load that path to project new queries. They
then compare vectors in the new coordinate system against bank vectors in the
old coordinate system. Dimensions match, all similarities are finite, and the
row-provenance check cannot notice because no row changed. Reusing the same
basis path to project a new extension can likewise stack new-space extension
vectors under an old-space release bank.

The basis, projected data, and sidecar must be one generation: write each to a
temporary name, verify the projection, then atomically publish a manifest that
binds content digests for both files. A regression test should stop after the
basis save and prove that consumers either retain the previous basis or reject
the mixed generation.

### 18. Target row order within an image is trusted even though an explicit `step` column exists

The target guard at [`dataset.py:260-281`](../src/dataset.py#L260) verifies that
each five-row group repeats the correct image ID, then reshapes every other
column positionally. It never reads `targets.parquet`'s `step` column. Swapping
two target rows within every image therefore passes the ID check: all five IDs
are equal. The loader assigns the swapped tile, action, and corner to
`self.step_ids = arange(steps + 1)`, training a different transition at that
step while every value remains in range.

The server has the same positional assumption at
[`serve.py:53-56`](../scripts/serve.py#L53): it treats array position four as
the z16 address without verifying that its recorded step is four. Thus an
inner-group reorder can corrupt both supervision and bank coordinates without
changing row counts, image IDs, or provenance digests.

The loader should require each image's step vector to equal
`[0, 1, ..., steps]`, validate the matching zoom progression, and select fields
by that verified order. A test should permute rows only within each image and
require both training and serving loaders to refuse the file.

### 19. `[x]` `tile_cache` resume identity is numeric row position, not image identity

[`tile_cache.py:144-178`](../scripts/tile_cache.py#L144) derives a seeded set of
dataset row numbers and stores only those integers. On resume it proves merely
that the old row-number set is a subset of the new one. Its metadata records
release, grid, seed, and count, but no ordered image IDs or digest of
`dataset.parquet`.

If a release is rebuilt or reordered at the same length, the seeded numeric
selection is identical and every old tile embedding is retained under a row
that now names another photograph. [`fuse_head.py:312-322`](../scripts/fuse_head.py#L312)
then reads current coordinates, split labels, and crop embeddings at `sel`, but
combines them with the old tile embeddings at the corresponding cache
positions. The completion mask proves only that *some* tensor was written; it
cannot prove whose tensor it is.

The cache needs to store selected image IDs and a digest of the dataset row
authority, validate them before any resume/grow operation, and expose that
identity to fusion. A regression test should reorder a same-length synthetic
dataset while retaining cache files and require resume and consumption to fail.

## Medium severity

### 20. `[x]` The external evaluator's hash-ranked sample is not stable as the manifest grows

[`eval_highres.py:204-212`](../scripts/eval_highres.py#L204) replaced a seeded
permutation with the `n` smallest CRC32 values and states that membership now
depends only on each ID. Membership actually depends on the global cutoff:
every newly harvested ID whose hash is below the old nth value enters and
evicts an old member. Two runs with the same `--n` and `--seed` but different
manifest sizes still score different benchmarks.

The scale of the error is large by construction. A synthetic reproduction at
the sizes cited in the source (18,812 growing to 47,646, `n=1000`) retained 390
old members and evicted 610, a 61% sample change. The exact real-ID overlap will
vary, but under uniform hashes its expectation is about 39.5%, not stability.
Exports carry IDs, so `parity_report` can refuse a paired comparison, but normal
point estimates are still printed under the same `n`/seed protocol.

A fixed-size cohort must be frozen as an explicit ID artifact (and its digest
reported), or the evaluator must pin a manifest generation. A hash threshold
is append-stable but produces a variable count; selecting the lowest fixed `n`
cannot provide both properties. Tests should append low-hash IDs and assert the
declared protocol does not silently change its cohort.

### 21. `[x]` Multiquery cache invalidation fingerprints the manifest but not the image bytes

The stamp in [`multiquery.py:219-240`](../scripts/multiquery.py#L219) includes
the data path, manifest file stamp, PCA basis, scale, and release. It does not
include any `img/<id>.jpg` content or generation. Replacing, correcting, or
re-downloading an image under the same manifest leaves `same_build=True`, so
the old embedding is returned even though a fresh run would read different
pixels.

This is narrower than the earlier missing-provenance bug: metadata and basis
changes now invalidate correctly, but the comment's claim that everything
deciding the vector is fingerprinted is still false. Because IDs and
coordinates are unchanged, the cache's remaining checks cannot distinguish
the stale vector.

The harvest manifest should record an image content digest (or a versioned
immutable object key), and the query-cache stamp should aggregate the selected
images' digests. A regression test should replace one selected JPEG without
editing the manifest and require re-embedding.

### 22. `--enc-gate` silently interprets any even-width street vector as two encoders

`StreetProj` checks only divisibility by two at
[`encoders.py:74-90`](../src/encoders.py#L74), while the CLI merely says the
flag assumes `[DINOv2 | SigLIP]` and never enforces that assumption. A
single-encoder multi-crop cache is split halfway through its crop layout; a PCA
cache is worse, because PCA mixes both encoders across every component and the
two halves have no encoder meaning at all. Both configurations train and save
normally, and the learned values are then reported as DINO versus SigLIP gates.

This can turn a mechanistic ablation into a mislabeled result without a shape
error. The training setup should require provenance proving a two-block
combined layout and refuse projected/mixed spaces, unless the projection itself
records a block-preserving structure. Tests should exercise a true dual cache,
a single-encoder even-width cache, and a PCA-derived cache.

## Review boundary

I did not repeat deliberately deferred model-definition items from earlier
reviews (`--save-opt` restoration, dropout scaling, step-3 sink supervision,
multi-photo learned scoring, or RoPE semantics). I also did not classify
non-atomic writes generally as separate bugs; findings 3, 4, and 13 are
narrower because they leave artifacts that current consumers specifically
accept as complete or provenanced. The already-recorded duplicate release IDs,
fixed-width multiquery search, and report-tag parsing issues were not counted
again.

The test suite was run, but GPU training, tile-server requests, and full
multi-gigabyte cache rebuilds were not executed during this review.
