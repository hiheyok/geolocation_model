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

---

# Performance and gradient backlog

From a second static review on 2026-09-02. Same rule: applied where safe with a
ladder running, recorded where not. The efficiency items are worth real time —
training is GPU-bound at 94%, so saved work is saved wall-clock, not just tidiness.

## Done

* **Blocker — `--neg` was rejected for every run.** `check_args` tested
  `args.soft` for truthiness, but `--soft` is a comma-separated string whose
  default `"0"` is truthy, so every sink-training command failed validation.
  Introduced and fixed the same day; now parses the temperatures and tests
  `any(t > 0)`. Verified against the actual queued commands rather than
  synthetic ones, which is how it was caught.
* **Per-step sink gate.** The extra sink keys had one shared scalar gate across
  four step-specific key sets. Combined with the item below — negatives are
  sampled only at steps 1 and 2 — the unsupervised steps contribute nothing but
  "do not fire" and can drive a shared gate negative, which is not the
  interpolation the formula claims. Now one gate per step.

## Highest value, deferred

**Step-3 sink keys have no positive supervision.** `dataset._negatives` draws
`t = rng.integers(1, 3)`, so sink positives exist only at steps 1 and 2. Step 3
is where a wrong beam descendant most needs rejecting, and step 0 cannot be
off-path at all and arguably should not expose a sink action. Fixing needs a
bounded cache of z12 negative siblings, or mining off-path z12 tiles already on
disk. **Interpret any sink-capacity result with this in mind: half the
step-specific keys are currently unsupervised.**

**A ladder is three optimizer restarts, not six epochs.** Each 2-epoch stage
warm-starts weights only, discards Adam moments, and runs a fresh
warmup-plus-cosine cycle that decays to zero. So "e6" is three independent
optimisations, and "e4 vs e6 is inside noise" partly reflects that rather than
convergence. Wants distinct `--init-weights` and `--resume`, the latter
restoring optimizer, scheduler, epoch and RNG state. Changes training, so it
needs a re-baseline.

**Optimizer parameter classification is wrong.** The substring rule puts
`retr.q_pos.weight` and `retr.k_pos.weight` in the no-decay group because their
names contain `"pos"`; `mem.weight` gets decay despite being an embedding;
matrix-shaped `sink_ext_b` gets decay despite being a bias; `street.gate` gets
decay because it is 2-D. Classify by owning module and explicit role. Changes
training, so it needs a re-baseline.

**Instrument the objective.** Four policy losses, a click loss and a sink loss,
averaged independently, then one global clip at 1.0. Record pre-clip total and
per-module gradient norms, clip frequency, gradient norms behind each gate, each
loss component's contribution, and the retrieval temperatures and gates. If
clipping is frequent, normalise the components before touching learning rates.

## Efficiency, deferred

Ordered by expected saving. None applied: all touch the training or beam path
that was executing.

1. **Street projection runs nine times per image** — five main steps plus four
   negatives, for one 1536→512 layer. Project once per block, then apply the
   cheap per-step encoder gates. Removes roughly 8/9 of that layer.
2. **Retrieval weights recomputed everywhere** — `q_pos`/`k_pos`/`q_neg`/`k_neg`
   and the neighbour softmax are recomputed for the main forward, again for
   negatives, and again at every beam step. An image-level `RetrievalContext`
   holding projected neighbours, weights and the dropout mask would be computed
   once. (It would also fix the dropout/quality inconsistency for free, by
   giving both paths the same mask.)
3. **Fuse the negative pass into the main one** — one model call concatenating
   policy rows, sharing street and retrieval context.
4. **Per-batch CUDA syncs in metric collection** — `float(loss)`, four per-step
   conversions, sink metrics and a `.cpu().numpy()` every batch. Accumulate
   detached tensors on device, transfer one packed tensor per epoch.
5. **Beam ranking is a NumPy round trip** — the whole `B×K×256` log-prob tensor
   goes to host, then nested Python loops rank it. Do `scores + logp`, sink
   masking and top-k over flattened `K×A` in torch; transfer only the chosen
   `B×K` actions and scores.
6. **Beam-invariant state recomputed per step** — street projections and
   retrieval weights depend on image and step, not the live beam.
7. **FP16 caches upcast in the worker** — `dataset` converts FP16 street and map
   rows to FP32 per item, doubling host traffic and pinned memory before BF16
   autocast converts them again. Ship FP16 and let autocast handle it.
8. **`expand().reshape()` materialises copies** of coordinates, similarities and
   weights per step and beam. Teach `child_prior` to take
   `(batch, rows, neighbours)` directly.
9. **Map tokens re-encoded per row** — the world tile is identical for every
   image and z4 tiles repeat constantly. Return token row ids, encode unique
   tiles once, gather back. Keep dropout after the deterministic projection.
10. **Dataset worker duplication** — the kNN `.npz` cannot be memmapped and is
    copied into every spawned Windows worker. Store indices and similarities as
    separate `.npy` memmaps reopened in `__setstate__`, and replace the per-item
    `np.random.default_rng()` with worker-local generators.
11. **`zero_grad` after the next forward** — correctness is unaffected, but the
    previous batch's gradients occupy memory across the next forward.

## Smaller

* Region-memory dropout zeros vectors without dividing survivors by `1-p`, so
  training sees a weaker expected signal than inference. Use inverted dropout
  unless the shift is deliberate.
* The click head is `SmoothL1(sigmoid(logit), target)`, whose gradient vanishes
  near tile boundaries once the sigmoid saturates. Compare against unconstrained
  regression with inference-time clamping, or `BCEWithLogitsLoss` on continuous
  targets.
* Cache sinusoidal and grid tensors as buffers; use `inference_mode` rather than
  `no_grad` in evaluation; try fused AdamW; drop `torch.cuda.empty_cache()` from
  repeated loops.
