# Code-quality backlog

From a static review on 2026-09-02, ordered by leverage. Two items are done;
the rest are deferred because the experiment pipeline was mid-flight and every
one of them touches a path that was executing.

**Read this together with the bug list at the bottom of PR #13.** That list is
behavioural defects; this one is structure. They share a root cause, which is
worth stating once because it is what most of these items are really about:

> Ten of the twelve defects found on 2026-09-02 were the same shape — a value
> that could have been derived from an artifact was instead restated in code or
> taken from an ambient default. None raised. Every one completed and reported
> a plausible number.

Items 1–5 and 13–16 attack that directly. The others are ordinary hygiene.

## Done

* **23 — non-ASCII in `summary_table.py`.** The reviewer reported mojibake
  (`Î”loss`); the file was in fact correct UTF-8 (`Δ`, bytes `CE 94`) and their
  tooling read it as cp1252. But there was a real adjacent bug: printing that
  character to this machine's cp1252 console raises `UnicodeEncodeError` at
  runtime. Replaced with `d-loss`; the file is now pure ASCII.
* **5 — fail fast on invalid combinations.** `train.check_args` rejects `--retr`
  with `--retr-k 0`, `--retr-drop` outside `[0, 1)` or without `--retr`,
  `--sink-k < 1`, `--sink-k > 1` without `--neg`, and `--soft` with `--neg`
  (256-wide targets against 257 logits). `policy_logits` already raises when
  `step` is omitted while extra sink keys exist.

## Highest leverage, deferred

**1 — One checkpoint runtime factory.** `evaluate`, `bootstrap`, `serve`,
`summary_table`, `error_profile` and `diag_beam` each reproduce a different
subset of: load model, validate release/split, resolve street/map/kNN
artifacts, build `TokenSource`, load the neighbour table. Every parity bug this
week was one of those subsets being wrong in one file. `source_for(ck)` was the
first slice of this; the rest belongs with it.
*Touches every entry point. Do it when no ladder is queued.*

**2 — Versioned checkpoint schema.** The checkpoint is a loose dict read with
`.get(key, default)` at ~40 sites. A missing key is indistinguishable from a
legacy checkpoint, so a typo silently takes the default — which is how
`map_sub` would have been dropped at inference. A `CheckpointConfig` dataclass
with `schema_version` and one migration function would make that a load-time
error.

**3 — Artifact provenance manifests.** Caches are identified by filename and
validated by width. `pca768_bank70.f16.npy` says nothing about which release,
split, or PCA basis produced it, and a same-width file passes every check we
have. Each map/embedding/PCA/bank/kNN artifact should carry release, split
hash, dimensions, row count, tokenizer settings, source artifacts and
completion state, validated before use. **This subsumes items 9 and 20**, and
would have caught the `OSV_RELEASE` fetch bug, the bank mismatch, and the
`--seed-from` width mismatch by construction rather than by three separate
guards.

**4 — One public scoring method.** Production callers assemble `fuse_flat` →
`retr_prior` → `_add_geo` → `policy_logits` by hand. The sink parity bug was
exactly one caller omitting one argument. A `model.score_policy(...)` that owns
the sequence makes the omission unrepresentable rather than merely tested.

## Structure and readability, deferred

**6 — Split `train.py`.** `main()` owns argument parsing, dataset construction,
model construction, checkpoint init, the epoch loop, selection and
serialisation. It is the file most often edited and the hardest to review.

**7 — Make scripts an importable package.** Every script opens with
`sys.path.insert`. A `pyproject.toml` with console entry points would remove
that and make scripts testable, which is a precondition for item 15.

**8 — Centralise external-query preparation.** `serve`, `eval_highres` and
`multiquery` each independently choose encoders, the 4.03 scaling, pooling, the
PCA basis, bank metadata and dimensionality. `multiquery` still hardcodes the
768-d projection, which is why its own default 1536-d tag cannot run
(review item 8, deferred with it). A shared `QueryEncoder`/`RetrievalBank`
would fix both.

**9 — Replace filename inference.** `"768" in name`, `"bank70" in name`,
`"bal" in name` decide dimensions, metadata stems and preprocessing. Folded
into item 3.

**10 — Descriptive names in core paths.** `a`, `b`, `m`, `st`, `sp`, `p`, `k`,
`r` in beam and training. Worth doing *with* item 6 rather than as a separate
diff, so the rename lands in already-restructured code.

**11 — Comments about invariants, not experiments.** Several docstrings carry
experiment narratives and numbers. Some of that is load-bearing — the "why" of
a surprising operation — but results belong in `docs/` and go stale in source.
Prune when touching each file; do not do a sweep.

**12 — Shared result types.** `{"err", "radius", "step_acc"}` dicts with units
implied by convention. Dataclasses would prevent field drift between
evaluators.

## Testing, deferred

**13 — Checkpoint-contract integration tests.** Build a tiny checkpoint and run
it through training forward, beam, bootstrap and serve, asserting identical
logits. Component tests did not catch the sink parity failure and by
construction could not: the bug was in how a caller assembled the pieces.
**Highest value of the testing items.**

**14 — Cache tests with a fake tile client.** Widths, incomplete caches,
mismatched indices, release metadata, seed caches, live-fetch fallback. Note
`pq.write_table` faults under pytest on this machine, so fixtures must avoid
parquet or the client must be faked at a higher level.

**15 — CLI smoke tests.** `--help` on every script would have caught the
`diag_beam` signature rot. Cheap once item 7 lands.

**16 — Checkpoint round-trip tests.** Save every architecture option, reload
through the public loader, compare logits. Covers sink, retrieval modes, map
sub, positional modes, memory, encoder gates.

**17 — Static checks in CI.** Ruff and Pyright on `src/`, plus compile, tests
and smoke checks.

**18 — Mark stale scripts.** A script's presence implies it understands current
checkpoints; several did not. A status header (supported / historical /
deprecated) or `scripts/archive/`.

## Smaller, deferred

**19** Domain exceptions in library code; `SystemExit` only at entry points.
`evaluate.street_file_for` raising `SystemExit` is wrong for a library.
**20** Structured cache keys. Folded into item 3.
**21** Atomic writes for checkpoints and reports (temp file, then rename).
**22** Standard logging instead of scattered `print`, with a consistent
artifact/config summary at startup.
**24** Shape annotations and type aliases for image batches, policy rows, map
tokens, neighbour tensors.

## Behavioural items deferred from PR #13

Not code quality, but they belong on the same list:

* `config.RELEASE` defaults to `s01`. **This is the root cause of four silent
  failures.** The fix is no default at all, which touches every script and
  every runner — worth doing as its own change, with item 7.
* `dataset.py` never reads `done.u8.npy`, so a partially-fetched cache yields
  zero-filled token rows that look valid.
* kNN caches are validated on `split_mode` only, not split hash, street file,
  query count or `knn_k`.
* The bootstrap error cache is keyed by tag without a checkpoint hash, so
  retraining under the same tag returns stale numbers.
* `--save-opt` saves optimizer state that no code path loads; scheduler and RNG
  state are not saved at all.
* `quality()` in the retrieval prior sees the unmasked top-1 similarity even
  when `--retr-drop` hid that neighbour, so the conditioning gate is told about
  evidence the prior no longer contains. **Deliberately not fixed yet**:
  changing it alters training semantics and would make new arms incomparable to
  the p-curve measured on 2026-09-02. Fix with a re-baseline, not silently.
* `eval_highres` hash sampling takes the lowest N, so a growing manifest can
  displace members. Stable for a fixed manifest, which the harvest now is;
  revisit when it next grows.
