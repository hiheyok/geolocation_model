# Codex code review, round two

A follow-up review of the fixes made on 2026-09-03: **11 items reopened**
and **11 new bugs**, plus a lower-priority list. **All 22 are now fixed.**
Its central finding is that
`split_hash` never distinguished train from test, which invalidated a guarantee
added earlier the same day.

The reviewer noted they could not run the test suite because the Python on
their PATH lacks NumPy — that is the MSYS2 interpreter this project keeps
tripping over. Everything below was verified by running it here.

**Markers.** `[x]` fixed and verified. `[~]` confirmed, not yet fixed. `[-]`
refuted or narrower than stated. `[.]` deliberately deferred.

---

## Reopened

1. `[x]` **`split_hash` did not distinguish train from test.** The worst item
   in either review, and it invalidated my own fix. It hashed `str(x)[0]`, so
   both labels became `"t"`. Measured on the live s10 sequence split: swapping
   **every** train row with **every** test row leaves the digest unchanged at
   `59e4b597281a`, and so does moving a single row. Only val moves showed up.
   So the bootstrap cross-arm check added for 78/79 — "same release, split mode
   and split hash, therefore the same rows" — pinned the mode and the val
   positions and nothing else. Now hashes the full labels; `split_hash_legacy`
   is kept so 122 existing checkpoints still load, with `check_split` printing
   what a legacy match does and does not prove. Covered by
   `tests/test_split_hash_and_cache.py`, which fails against the old digest.

2. `[x]` **`find_err_cache` could return stale results or crash.** Mine, from
   the same morning. It took the newest file for a tag, which after retraining
   is the previous model's errors. It now requires the current checkpoint's
   stamp, accepts an unstamped file only when it postdates the checkpoint, and
   returns `None` otherwise — which `marathon` now handles instead of calling
   `.exists()` on it.

3. `[x]` **`digest` reported every stale generation.** Five tags in this
   workspace have both a legacy and a stamped cache, so each appeared twice
   with the current checkpoint's metadata attached. Now one generation per arm:
   67 rows, down from 72.

4. `[x]` **`--overfit` still could not save a checkpoint.** It sets
   `va_ds=None`, so with the default `--select hit` the criterion stayed NaN
   and the run trained and exited having written nothing. The earlier fix only
   bounded the subset. It now falls back to `--select loss` and says so.

5. `[x]` **The beam completion-mask check failed open.** It applied the mask
   only when it was long enough, so a truncated mask skipped the check
   entirely — the case where it is least trustworthy. Now refuses.

6. `[x]` **The pyramid mask shape was checked on rows only.** `(n,1)` and
   `(n,3)` both passed. Now the exact shape; verified the real `pyr47` mask
   loads and both wrong shapes are refused.

7. `[x]` **Tile-cache grid geometry — I fixed the wrong file.** The review
   named `tile_cache.py`; I had changed `fetch_tiles.py`, a different script.
   `3x2` and `2x3` both hold six tiles, so every count-based check passes.
   `tile_cache.py` now compares the recorded geometry and refuses a transpose.

8. `[x]` **Fusion still ignores the tile-cache completion mask.**
   `fuse_head.py` loaded every `tile6` row without consulting
   `tile6_done.u8.npy`. The beam fix does not reach this separate consumer,
   and an unwritten row is the zero fill -- which, after the per-token L2 the
   head applies, is a unit-length nothing that trains as if it were a view.
   Extracted as `require_written` so it is testable, and it is: a hole, a mask
   from another build, and an absent mask are three distinct outcomes. The
   live cache passes -- 120,000 rows, 0 unwritten -- so the failures had to be
   constructed.

9. `[x]` **My top-k fix introduced an out-of-bounds crash.** The original
   `min(K, n-1)` assumed the query sits inside the bank, which is false for
   disjoint sets, so it dropped a legal neighbour. My `min(K, n)` then passed
   `k` as `argpartition`'s `kth`, which is an index — out of bounds exactly
   when K reaches the bank size. **Both versions shipped.** Corrected in all
   three consumers; `tile_pool` and `tile_probe` still had the original
   off-by-one. `tests/test_topk_bounds.py` fails against both.

10. `[x]` **The query-cache stamp fingerprints names, not contents.** It
    recorded the data path and PCA filename, so rebuilding the PCA or
    replacing images under the same path did not invalidate it -- and
    rebuilding a basis under the same name is how this project normally
    produces one. Now stamps the *bytes* of the basis and the manifest, via a
    single `safeio.file_stamp` (size and mtime) that `bootstrap.ckpt_stamp`
    now also uses rather than defining its own. A miss says why it is
    re-embedding instead of just looking slow.

11. `[x]` **`--save-dir` remains a no-op.** Recorded twice: the first fix
    only made the flag announce that it created a directory, which is honest
    and still useless. It downloads the thumbnails now -- the manifest already
    carries a signed URL per row, so this is a fetch, not a re-query. Skips
    ids already on disk, writes through a `.part` rename, and counts failures
    rather than discarding a harvest that cost API quota.

## Newly identified

1. `[x]` **Transfer evaluation skips the checkpoint hash comparison** when
   evaluating under a different mode, then loads current source-mode labels
   without proving they match training. Worse than it first reads: that branch
   uses those labels to compute the contamination figure the refusal is
   decided on, so moved labels do not merely go unchecked -- they decide
   whether the run is allowed. Now compares the training mode's hash too, with
   the same legacy-digest fallback and the same warning about what it proves.

2. `[x]` **External evaluators do not check `ck["release"]`.** Both force
   s10 and neither verifies the checkpoint agrees. They never call
   `check_split` -- there is no split to check against a foreign image set --
   but they do index s10 row-addressed artifacts. Moved the check into
   `load_model`, which is the one place a checkpoint becomes a model, so it
   covers every consumer at once. Deliberately asymmetric: a disagreement is
   refused, a missing field warns, because 44 of the 122 checkpoints on disk
   predate the field and inventing a release for them would refuse real arms
   on a guess.

3. `[x]` **Multiquery bank overrides reused the original bank's row
   restriction.** The same defect fixed in `eval_highres` today; the sibling
   consumer was missed. Now resolves the cache built over the bank in use.

4. `[x]` **Training was not reproducible.** No `--seed` and no seeding of
   torch or numpy, so weight init, loader shuffling and dropout differed run to
   run with nothing recorded to explain it. Added and stored in the checkpoint.
   *Partly refuted:* sink negatives were already deterministic — training
   passes `neg_random=False, neg_seed=11`, so the OS-entropy path the review
   cites is not the one training uses.

5. `[x]` **`--init` ignored the equals form.** `--lr=3e-4` was not recognised
   and was silently overwritten by the warm-restart default.

6. `[x]` **The fusion OneCycle schedule was too long.** It was built from
   `len(pairs)` while `--uniq-anchor` keeps one row per anchor and drops the
   remainder, so the scheduler planned for more steps than the run takes and
   never reached its final phase. **This affected the masked fusion runs
   reported today.**

7. `[x]` **Tile-cache decode failures still exit zero**, so a runner can
   mark an incomplete cache finished. `fetch_tiles` and `pyramid_cache` were
   fixed; `tile_cache.py` was not. It now exits non-zero on unreadable images
   *or* unfilled rows, which are two ways to reach the same zero-filled cache.

8. `[x]` **Pyramid completion verification can allocate over 10 GB** —
   fancy indexing across both encoders before selecting one. Same shape as the
   `embed_street` scan already chunked today. Chunked at 20,000 rows, with the
   encoder axis sliced before the float32 cast; a test pins the chunked count
   to the whole-array count at five chunk sizes and on an empty selection.

9. `[x]` **A complete pyramid mask returns before writing metadata**, so a
   crash between the final mask update and `_meta.npz` leaves a cache that
   permanently reports "nothing to do" — unrecoverable without deleting a
   cache that is, in fact, complete. It falls through now: the encoder loops
   are already no-ops when done, and the verification and metadata write are
   cheap and idempotent.

10. `[x]` **Fetch-tile masks were loaded without validation.** A short mask
    left the cache unrecoverable without deleting it by hand; values above one
    corrupt the completed count. Verified the real 626,284-row mask still loads
    and both failure shapes are refused.

11. `[x]` **Leak-screen coverage double-counted.** It summed `n` across runs
    without recording which images were seen, so re-running the seeded default
    sample counted them again and could report coverage above the corpus. Now
    stores ids and reports their union.

## Lower-priority, not yet addressed

`report.py` reads `ck["limit"]` into an unused variable and reconstructs from
obsolete tag syntax; its header claims median selection though the default is
hit rate. `standard_metrics.py` mixes checkpoint generations and protocols.
`config.knn_name` rounds bank limits to thousands, so 1,000 and 1,999 collide —
which narrows the earlier refutation of item 11 in round one rather than
overturning it. `build_dataset.py` does not reject duplicate image ids across
shards. `multiquery` infers width from a filename and caps its search at 32
regardless of `retr_k`. `retrieval.py` hardcodes four zoom bits per step, so it
is correct only at `g=16`. Argument validation still accepts negative retrieval
temperatures, negative sink weights and invalid memory-drop probabilities.

## Where this leaves the code

All 22 items in this round are closed. What remains open is `docs/REVIEW.md`'s
thirty, and twelve of those are two structural problems rather than twelve
independent ones: #40 with 43, 44, 46, 47, 62, 63 and 66 is a single
issue — every artifact addressed by row position, with nothing proving that
two of them describe the same rows — and one provenance manifest retires all
eight. #70 through #73 are the runner's marker and log design.

Seven items also wait on a decision that is not mine to make: fixing any of them
changes what a trained arm *is*, so every one of the 122 checkpoints on file
becomes incomparable and the parity table has to be rebuilt. They are listed in
`docs/STATE.md` — #29, the weight-decay rule that exempts exactly the learned
retrieval keys, is the one most likely to have moved a published result.

## What this round changes about the first one

Round one closed with "40 fixed, 30 open". Three of those forty were wrong:
`split_hash` (1), the cache resolver (2, 3) and the top-k bound (9) — and the
last two were defects I introduced while fixing something else. One was fixed
in the wrong file (7).

The pattern is worth naming: **every one of those was a fix I did not test
against the failure it was meant to prevent.** The split hash was never fed a
swapped assignment; the resolver was never given a retrained checkpoint; the
top-k was never run at `K == bank size`. Both new test files exist to close
exactly that gap, and both were checked by reverting the fix and watching them
fail.
