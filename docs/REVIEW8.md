# Code review, round eight: bugs

Reviewed commit `eaa1b76` on 2026-09-05. Only this report was added; no implementation or test files were changed.

The review focused on training, rollout, retrieval-cache validation, serving, and map-cache construction, with a continuation covering external and multi-photo evaluation. There are **11 findings: four P1 and seven P2**. P1 means high priority because the defect can silently change the benchmark or attach incorrect coordinates; P2 means medium priority because the defect affects a narrower configuration, numerical contract, or execution path. No P0 issue was established. These are reachable code defects, not claims that existing published measurements are affected.

## Severity ranking

Finding numbers remain stable across the two review passes; this table gives the recommended repair order. Within a severity, the ranking weighs likely reach and impact rather than implying a measured production effect. Findings 6-11 were added in the continuation.

| Rank | Severity | Finding | Bug and consequence |
|---|---|---|---|
| 1 | P1 / High | 1 | A cache can declare held-out photographs to be valid bank rows, allowing evaluation leakage. |
| 2 | P1 / High | 2 | Serving skips k-NN identity checks and can attach one extension's coordinates to another's embeddings. |
| 3 | P1 / High | 6, new | Missing named k-NN files silently expand external evaluation to the entire embedding corpus. |
| 4 | P1 / High | 7, new | External evaluators call the shared validator without supplying the split identity it needs to check. |
| 5 | P2 / Medium | 8, new | Default multi-photo calibration changes the learned quality signal even for the one-photo baseline. |
| 6 | P2 / Medium | 9, new | Padded neighbours receive nonzero negative-branch weight in dual retrieval. |
| 7 | P2 / Medium | 10, new | `fetch_tiles --out-cache` validates and stamps the default cache's renderer instead of the output cache's. |
| 8 | P2 / Medium | 3 | Seed copying mixes renderer identities before validation and publishes the mixed rows as complete. |
| 9 | P2 / Medium | 4 | Rollout combines cached and live tiles without enforcing renderer identity. |
| 10 | P2 / Medium | 11, new | Different restricted-bank sizes resolve to the same k-NN filename and overwrite each other. |
| 11 | P2 / Medium | 5 | Valid `--limit` plus `--overfit` arguments crash before training starts. |

## 1. [P1] Bank membership validation never establishes that the bank is allowed by the split

Location: [`src/dataset.py:301-307`](../src/dataset.py#L301), called at [`src/dataset.py:560`](../src/dataset.py#L560); shared row validation in [`src/knnmeta.py`](../src/knnmeta.py).

The new `_check_neighbours` verifies that each neighbour occurs in the cache's own `bank_rows`, then excludes self and same-sequence matches. However, neither it nor `knnmeta.bank_rows` checks that release bank rows have the live `train` label. Extension rows are also not checked against the held-out cells of a spatial split. The split hash validates the recorded label assignment, but does not prove the bank was selected from that assignment.

Trigger: a builder regression or an incorrectly assembled cache includes a held-out row in both `bank_rows` and `idx`, while retaining the correct split metadata. All current membership, range, self-match, sequence, and embedding-digest checks can pass. Other queries then retrieve held-out photographs and their target coordinates, contaminating evaluation.

Confirmed with the actual validator, entirely in memory:

```python
class Cache:
    files = ["bank_rows"]
    def __getitem__(self, key):
        return np.array([0, 1, 2])

# Live labels would be [train, train, test]. Row 2 must not be in the bank.
# Every query retrieves a different row and every sequence is different.
_check_neighbours(
    np.array([[2], [0], [1]]), Cache(), "probe", 3,
    np.array(["a", "b", "c"]), None,
)
# Returns successfully; the API has no live labels to check.
```

Recommended correction: validate bank eligibility against the live split before checking neighbour membership. Allow intentional bank subsets, but require every release row to be training data and every extension row to satisfy the split's spatial exclusions. Apply the same eligibility check to direct consumers of `bank_rows`.

Regression check: use correct split metadata with a held-out row present in both the declared bank and a different-sequence query's neighbours; require rejection. Also cover an extension row in a held-out spatial cell and a legitimate restricted training bank.

Relationship to earlier reviews: a remaining gap after REVIEW6 #4. The new tests cover neighbours outside the declared bank, self matches, and same-sequence matches; they do not establish that the declared bank itself is valid.

## 2. [P1] Serving still bypasses the shared k-NN metadata and content checks

Location: [`scripts/serve.py:76-107`](../scripts/serve.py#L76).

`load_everything` reads the checkpoint's k-NN file and calls only `knnmeta.bank_rows`, which checks row bounds and uniqueness. It never calls `knnmeta.check`, `check_ext`, or `check_bytes`. `check_split` immediately above validates the checkpoint against the dataset, not the k-NN artifact that is opened afterwards.

Trigger: replace the named k-NN file with one carrying a different split, embedding filename, or same-sized extension. Serving accepts its bank rows and extension coordinates while loading embeddings from the checkpoint's street file. A different split silently changes the searchable corpus; a different same-sized extension attaches the wrong coordinates to retrieved images. The new street-content stamp is also ignored by this consumer.

Evidence: the only `knnmeta` call in this startup path is the bounds/uniqueness check at line 92. `eval_highres.py` and `multiquery.py` call the shared `knnmeta.check` validator, but the continuation found that those callers omit its split arguments; see finding 7. No GPU execution or server startup was needed to establish the missing serving call.

Recommended correction: validate the loaded k-NN artifact against the resolved street file, live split and hash, release count, and full bank size before allocating the bank or loading encoders. Include extension identity and street-content validation. For conditioned checkpoints, resolve the retrieval file explicitly.

Regression check: mock startup inputs with valid row indices but a wrong split hash, wrong street filename, and different same-sized extension, in separate cases. Each must fail before encoder creation.

Relationship to earlier reviews: REVIEW4 #14 remains unresolved for serving. Using the shared validator has also only partially resolved that finding for external evaluation, as detailed in finding 7. This finding is independent of the conditioned-serving failure in REVIEW6 #6.

## 3. [P2] `--seed-from` can mix renderers before the renderer guard runs

Location: [`scripts/fetch_tiles.py:158-224`](../scripts/fetch_tiles.py#L158), especially the copy at line 204 and early return at line 215.

The seed path validates source token width, row indices, and completion masks, but never reads the source's `renderer.json`. It copies source tokens into the destination and publishes their done bits before `T.check_renderer` runs. That later call compares only the destination sidecar with the live server.

Trigger: resume an incomplete destination stamped renderer A, using a same-shaped seed cache stamped renderer B. B's rows are copied into A's cache. If seeding completes the destination, the function returns without any renderer check. If work remains and the live server is A, the later check succeeds and the mixed cache remains labelled A. A fresh destination can likewise acquire B's seed rows and then be stamped A.

This defeats the renderer binding even when both source and destination have explicit identities. It also means a later refusal cannot undo the already-published seed rows.

Recommended correction: compare source and destination renderer identities before copying or publishing anything. Carry the seed identity into a fresh destination, and require any subsequent live fetches to match it. Make handling of unknown legacy identities explicit.

Regression check: two tiny caches with matching shapes and different renderer sidecars must be rejected before any destination token or done bit changes. Cover both a seed that completes the destination and one that leaves live work outstanding.

Validation: static control-flow review; no cache was copied or modified.

## 4. [P2] Rollout ignores renderer identity when combining cached and live tokens

Location: [`src/beam.py:84`](../src/beam.py#L84) and [`src/beam.py:109-112`](../src/beam.py#L109).

`TokenSource` opens the map cache, selects a live server through `T.connect`, and fetches misses without comparing that server's renderer identity with the cache's `renderer.json`. The only production call to `T.check_renderer` is in `fetch_tiles.py`.

Trigger: the primary server is unavailable and the backup uses another renderer, or the configured server changes after cache construction. Cached hits retain renderer A's tokens while misses receive renderer B's tokens. The same model can therefore change predictions as cache coverage changes. This affects evaluation, serving, and training's rollout-based checkpoint selection.

An in-memory probe instantiated the real `TokenSource` with mocked cache reads and a client whose `health()` raises if called. Construction succeeded: the injected client was accepted without any identity inspection. Source inspection confirms that the normal `T.connect` path checks availability but does not compare the result against the cache sidecar.

Recommended correction: enforce the cache's recorded renderer identity for live fallback before serving misses. A read-only inference path should validate an existing record without silently stamping a legacy cache as though its origin were known.

Regression check: a cache stamped A and client reporting B must be rejected before a miss is fetched; matching identities must still permit live fallback. This requires a separate fix from seed-copy validation because rollout does not call `fetch_tiles`.

## 5. [P2] Combining `--limit` and `--overfit` crashes during model construction

Location: [`src/train.py:528`](../src/train.py#L528), [`src/train.py:541`](../src/train.py#L541), and [`src/train.py:591-592`](../src/train.py#L591).

When `--limit` reduces the training set, `tr` becomes a `Subset`. A valid `--overfit` then wraps that subset in another `Subset`. Model construction unwraps only one level and accesses `_ds.dim_street`, but `_ds` is still a `Subset`, so it raises `AttributeError`. The repeated one-level accesses to `.tokens` have the same assumption.

Trigger: with more than 100 training images available, run training with `--limit 100 --overfit 10`. The size guard accepts this combination, and overfit correctly switches selection to loss, but construction fails before the first epoch.

Confirmed in memory with PyTorch's real `Subset` and the exact unwrapping expression: `Subset(Subset(base, ...), ...)` unwraps to `Subset`, which has no `dim_street` attribute. This does not require loading the production dataset.

Recommended correction: retain the original `GeoStepDataset` for artifact/model metadata before wrapping it, or unwrap recursively in one helper and use that consistently.

Regression check: exercise model construction with both flags active and a valid overfit count; separately cover either flag alone. No full training run is necessary.

## 6. [P1] External evaluation silently searches the whole corpus when a named k-NN cache is missing

Locations: [`scripts/eval_highres.py:325-334`](../scripts/eval_highres.py#L325) and [`scripts/multiquery.py:406-415`](../scripts/multiquery.py#L406).

Both evaluators read `kf = ck.get("knn_file")`, but validate it only if the file exists. If a checkpoint names a file that is unavailable, `keep` remains `None`, and the fallback selects `np.arange(bank.shape[0])`. The message then says the checkpoint records no bank rows, concealing the difference between a legacy checkpoint with no record and a missing required artifact.

Trigger: move or delete the checkpoint's k-NN file while keeping the checkpoint and street bank. A run without an explicit `--bank` override continues over every release and extension embedding, including rows excluded from the checkpoint's training bank. Restricted-bank and spatial-holdout experiments therefore acquire a different searchable corpus without an explicit experimental override. External queries need not themselves be in the release for this to invalidate a comparison of bank coverage.

Confirmed by executing the actual bank-selection statements extracted from each entry point with AST, mocking only file existence. With a checkpoint recording `named-missing.npz` and six embedding rows, both selected `[0, 1, 2, 3, 4, 5]` and printed `checkpoint records none`.

Recommended correction: if a checkpoint explicitly names a k-NN file, require it to exist and contain the bank definition needed by the evaluator. Treat deliberate full-corpus evaluation as a separate explicit option with a different reported protocol.

Regression check: a named-but-missing file must fail before retrieval; a present valid restricted bank must select only its declared rows. Test the normal checkpoint path, since the distinct `--bank` override path already refuses a missing derived cache.

## 7. [P1] External callers omit the split arguments that activate shared k-NN validation

Locations: [`scripts/eval_highres.py:330-331`](../scripts/eval_highres.py#L330), [`scripts/multiquery.py:411-412`](../scripts/multiquery.py#L411), and [`src/knnmeta.py:199`](../src/knnmeta.py#L199).

Both callers pass only `n_bank` and `street_path` to `knnmeta.check`. Its `split_mode`, `split_hash`, and `street_file` arguments default to `None`; their checks therefore do not execute. Extension identity, content digest, and row bounds are checked, but they do not establish that the bank was selected for this checkpoint's benchmark. Neither external evaluator calls `check_split` to establish that the current split still matches the trained checkpoint.

Trigger: replace the expected k-NN file with one built over the same embeddings and extension but a different split. Its street-content digest remains correct, and its bank rows remain in range. External evaluation accepts that bank for the original checkpoint. The same problem occurs when the dataset's split assignments change while the file still carries old assignments.

An in-memory call with the exact keyword set used by these consumers accepted a cache declaring `split_mode=cell8`, an unrelated split hash, and another street filename. The independent extension/content functions were mocked in this probe to isolate argument dispatch; a real different-split cache over unchanged embedding bytes can satisfy both of those independent checks.

Recommended correction: resolve and validate the checkpoint's training split against the live release, then supply the expected split mode, canonical hash, live labels for legacy-hash handling, and retrieval filename to the shared validator. Preserve deliberate corpus overrides, but validate their bank definition against the selected protocol.

Regression check: a same-content, same-extension cache from another split must fail in each external entry point. A call-presence assertion is insufficient; test which metadata comparisons actually run.

Relationship to earlier reviews: this refines REVIEW4 #14 and the initial assessment in finding 2. Calling the shared function did not activate all of its checks.

## 8. [P2] Default multi-photo calibration changes the one-photo baseline's learned quality input

Locations: [`scripts/multiquery.py:120-122`](../scripts/multiquery.py#L120), [`src/retrieval.py:113`](../src/retrieval.py#L113), and [`src/retrieval.py:221`](../src/retrieval.py#L221).

`--calib top1`, the default, subtracts each photograph's best cosine from all its candidate scores. For a single photograph this leaves softmax weights unchanged, which is what the existing regression test verifies. However, `quality()` also passes the first similarity directly into the learned conditional gate. Calibration changes that feature from the raw best cosine to zero for every one-photo query. `cond`, `pos`, and `dual` checkpoints were trained with the raw cosine in this slot.

Consequently, even the one-photo reference measures a changed gate input, and the multi-photo result combines neighbour aggregation with removal of a learned confidence signal. This is not the intended constant-shift invariance asserted by the test and merge-function documentation. Scalar retrieval has no conditional quality gate and is unaffected by this particular defect.

Confirmed using the real merge function and a small conditional prior with an explicitly nonzero weight on the quality feature. For similarities `[0.9, 0.8]`, raw versus calibrated neighbour weights differed by only `2.98e-08`, while the additive policy bias differed by **3.3818 logits**. This is a constructed counterexample demonstrating the bug, not a measured delta on a shipping checkpoint.

Recommended correction: carry raw retrieval quality separately from the calibrated logits used for neighbour weighting. Specify how quality is aggregated for multiple photographs, and preserve the trained single-photo input when `N=1`.

Regression check: compare complete prior outputs before and after one-photo calibration with a nonzero conditional gate. Testing only score differences or softmax weights misses the affected path.

## 9. [P2] Padded neighbours still vote in the dual retrieval prior's negative branch

Locations: [`scripts/multiquery.py:153-160`](../scripts/multiquery.py#L153) and [`src/retrieval.py:198-203`](../src/retrieval.py#L198). A concrete entry-point trigger comes from the fixed search width at [`scripts/multiquery.py:424`](../scripts/multiquery.py#L424) versus the checkpoint width at line 470.

When fewer than `K` distinct candidates are available, `merge_candidates` repeats the final row with similarity `-10000`, intending to give padding no weight. This suppresses the positive branch. The dual negative branch computes its scores solely from learned query/neighbour cosine and its own temperature, so it ignores the similarity sentinel. Its only mask is random training dropout, which is absent during evaluation. Repeated padding rows therefore receive ordinary negative-branch probability and disproportionately suppress their geographic cell when `g_neg` is nonzero.

Trigger: a valid checkpoint trained with `retr_k=64` evaluated by `multiquery` at one photo. The script searches only 32 candidates per photo but asks the merger for 64, so it pads 32 repeated rows even when the bank contains enough genuine neighbours. The issue also affects any smaller candidate pool requiring padding.

Confirmed through the real merger and `RetrievalPrior.weights` with two distinct rows padded to four. With identical learned negative keys:

```text
row IDs:          [7, 8, 8, 8]
similarities:     [0.9, 0.8, -10000, -10000]
positive weights: [0.8067, 0.1933, 0, 0]
negative weights: [0.25,   0.25,   0.25, 0.25]
```

Row 8 receives 75% of the negative probability instead of the 50% it would receive without duplicate padding.

Recommended correction: represent candidate validity explicitly and mask invalid slots in both branches and quality calculations. Retrieve at least the checkpoint's requested neighbour count per photo so supported larger-k checkpoints are not padded unnecessarily.

Regression check: padded and unpadded versions of the same candidate set must produce equal dual priors with `g_neg` enabled; also exercise a checkpoint requesting more than 32 neighbours. The current padding test checks only that the emitted similarities are sufficiently negative.

## 10. [P2] `fetch_tiles --out-cache` binds renderer metadata to the wrong directory

Locations: [`scripts/fetch_tiles.py:87-89`](../scripts/fetch_tiles.py#L87) and [`scripts/fetch_tiles.py:224`](../scripts/fetch_tiles.py#L224).

`--out-cache` redirects `TOKENS`, `INDEX`, and `DONE` to the requested directory, but `T.check_renderer(client, config.MAP_CACHE)` still uses the release's default cache. It can stamp that unrelated default directory while leaving the actual output without a renderer record. On resume, it ignores an explicit mismatching renderer record in the output directory as long as the default directory agrees with the live server.

Trigger: default cache and server use renderer A, while an incomplete custom output is stamped B. `fetch_tiles --out-cache <custom>` passes the check against the default A cache and fills B's remaining rows with A's tokens. Conversely, a fresh independent custom build can be rejected because the unrelated default cache is stamped differently from the server.

Confirmed statically and by inspecting the actual call expression: the guard argument remains `config.MAP_CACHE`; `a.out_cache` changes only the three artifact paths. This requires no seeding and is distinct from finding 3.

Recommended correction: resolve one output-cache directory and use it for every artifact, including renderer identity. Run validation before changing existing output contents.

Regression check: custom output B plus default A plus live A must reject the custom-cache mismatch. A fresh custom build must neither inspect nor write the default cache's sidecar.

## 11. [P2] Restricted-bank cache names truncate the exact bank size

Location: [`src/config.py:101`](../src/config.py#L101), used for publication at [`scripts/build_knn.py:294`](../scripts/build_knn.py#L294).

The cache name encodes `bank_limit // 1000`, although `--bank-limit` accepts an arbitrary integer and the builder samples the exact requested count. Therefore different restricted banks within the same thousand-row interval share one output path.

Confirmed with the real naming function:

```text
bank_limit=1000 -> knn_x_sequence_k32_bank1k.npz
bank_limit=1999 -> knn_x_sequence_k32_bank1k.npz
```

Trigger: build those two valid restricted banks sequentially. The second `np.savez` overwrites the first bank's cache. A checkpoint or subsequent training command retaining the first path now uses the second bank, even though embedding content and split metadata still agree. The metadata records the actual rows, but consumers do not compare their count with the bank restriction intended by the old run.

Recommended correction: encode the exact limit, or require multiples of 1000 before doing any build work. Existing rounded names need an explicit compatibility policy so old checkpoints do not silently change banks.

Regression check: distinct positive accepted bank limits must produce distinct names, including 1 versus 999 and 1000 versus 1999. Unrestricted naming must remain separate.

Relationship to earlier reviews: REVIEW.md #11 correctly refuted a collision between restricted and unrestricted caches. This is a narrower, different collision between two restricted banks; the earlier refutation does not cover it.

## Verification and limits

During the initial pass, the existing suite passed with bytecode generation and pytest's persistent cache disabled:

```text
OSV_RELEASE=s10 PYTHONDONTWRITEBYTECODE=1
py -3.13 -m pytest -q -p no:cacheprovider
535 passed in 9.35s
```

Three additional in-memory probes exercised bank membership validation, nested `Subset` unwrapping, and `TokenSource` client acceptance. They used synthetic arrays and mocked cache reads, wrote no repository files, and contacted no services. The installed Python required execution outside the sandbox because the sandboxed launcher could not discover it.

The continuation added in-memory probes for calibration through a conditional prior, padding through the dual negative branch, the actual missing-cache selection branches in both external evaluators, metadata-check argument dispatch, and exact-limit filename collisions. These confirmed the outputs quoted in findings 6-11. The renderer-directory finding was also checked against the parsed call expression. The code and tests remained unchanged at `eaa1b76`, so the full suite was not repeated; the continuation's evidence is the targeted probes rather than another identical suite run.

No training, live serving, GPU workload, network request, or production artifact rebuild was performed during the bug-review passes. The report proposes corrections and regression cases only; it does not implement them. The research addition below subsequently used public paper websites, not project services.

---

## Research-backed recommendations: general accuracy and high-resolution detail

Research date: 2026-09-05; workspace HEAD at final verification: `7a54a72`. That commit adds only `docs/PROJECT_BRIEF.md` relative to the reviewed `eaa1b76`; implementation files are unchanged between them. This section is a proposed roadmap, not additional confirmed bugs or measured improvements. The objective is better geolocation generally, including external images and large errors, with a separate investigation into whether genuine high-resolution detail improves fine localization.

### Evidence and strategic recommendation

The [project brief](PROJECT_BRIEF.md) was read first. Recommendations account for its negative experiments, frozen encoders, offline retrieval, and single RTX 3070 with 8 GB VRAM. Local checks also covered [the tiled bootstrap report](../runs/BOOTSTRAP_tiledrisk.md), [current state](STATE.md), and the embedding, retrieval, conditioning, and beam interfaces. Some experiment reports referenced by the brief, including `runs/RESMATCH.md` and `runs/PYR_LEVELS.md`, were not present; their conclusions here are attributed to the brief, not independently reproduced.

The strongest direction is to separate three jobs: **finding plausible regions, comparing candidate-specific visual evidence, and refining a location using maps**. Keep the current retrieval representation stable while testing the second job. Add an independent geographic proposal path to address failures that no reranker can repair.

Reasons, grounded in this repository:

- The brief reports 57.5% within 25 km on OSV versus 13.4% on KartaView for the shipping model. This strongly motivates a coverage audit, but the aggregate gap alone does not causally exclude domain shift, geographic composition, or preprocessing differences. Measure these before treating coverage as the sole explanation.
- Top-1 retrieval already reaches 56.4% versus the agent's 57.5% on the reported OSV comparison. Every new agent component must beat both the current agent and a retrieval-only baseline at the same bank and compute budget.
- L0+L1 has positive agent-level evidence: the tiled report's paired 95% interval is +2.22 to +4.64 percentage points within 25 km against L0 at epoch 6. That experiment used the documented 400,180-row tiled-bank setting; transfer to the full bank and external corpus remains an experiment, not an established gain.
- The native-resolution evidence is mixed and includes a SigLIP checkpoint/resolution confound. The brief's small within-1-km gains justify a detail experiment, not replacing the full retrieval bank with higher-resolution vectors.
- Retrieval dropout already provides evidence for improving the non-retrieval path. Conversely, query-only changes to retrieval geometry, richer map tokens alone, learned query-only blend gates, GeM, and soft tile-label KL have negative or null evidence. They are not default recommendations here.

### Priority order and decision gates

These are research priorities, separate from the P1/P2 bug severity ranking above. Cost labels are relative, not runtime promises. The brief's approximately 26-minute training rung, 5-7-minute kNN build, and 19-hour full embedding pass are reference measurements; new shapes and models must be profiled.

| Order | Proposal | Main benefit sought | Cost / uncertainty | Advance only if |
|---|---|---|---|---|
| R0 | Correct the relevant review defects, lock artifact identity, and measure coverage | Trustworthy comparisons across every image type | Low compute; prerequisite | All consumers use the intended eligible bank and identical evaluation IDs |
| R1 | Larger cached shortlist, simple candidate scorer, geographically diverse proposals | General accuracy and recovery from wrong top-1 predictions | Low to medium; benefit unknown | Coverage or realized accuracy improves at a measured latency cost |
| R2 | Separate native-detail conditioning, then compact local-token reranking | Fine detail plus candidate discrimination on ordinary images | Medium; promising mechanism, unproven transfer | Detail beats a matched low-detail control, not merely a different checkpoint |
| R3 | Independent image-to-geography proposal head | Sparse-bank and external generalization | Medium; high uncertainty | Rescues queries whose retrieval shortlist lacks a nearby candidate |
| R4 | Frozen descriptor aggregation pilot: AnyLoc, then SALAD | Better global candidate coverage without encoder fine-tuning | Medium pilot; expensive full rollout | Matched-bank oracle coverage and downstream accuracy improve |
| R5 | On-policy hard-state supervision and map counterfactuals | Make map descent add value beyond the prior | Medium; first diagnose the failure | Gains survive prior-only and shuffled-map controls |
| R6 | Local image-to-map alignment | Sub-kilometer refinement with a distinct role for maps | High uncertainty and integration cost | A small correctly localized-region pilot fits memory and beats the existing click head |

R1 and R2 are the first capability experiments; R3 is the strategic generalization experiment. Do not wait for a complete architecture rewrite to test them. Do not run GPU experiments concurrently.

### R0. Measure the actual bottleneck before changing representations

For each query, export its stable ID, sequence/group, source dimensions, candidate IDs, raw similarities, final prediction, error, and artifact hashes. Keep a clean OSV protocol and an external protocol with separate development and final evaluation groups. The old external `hr_*` versus corrected `hrfix_*` distinction documented in [STATE.md](STATE.md) must not disappear in an aggregate comparison. Generate paired per-image outputs for any new comparison rather than joining unrelated summary metrics.

Compute the shortlist coverage diagnostic at 1, 25, and 200 km:

```text
A_r(K) = fraction of queries for which
         min(distance(query GPS, candidate GPS) over top K candidates) < r
```

This is an oracle ceiling for a method that must output one of those candidate coordinates. It is NOT an upper bound on the complete agent, which can predict a new coordinate, or on the proposed independent geographic branch.

Split errors into three measurable classes:

1. No eligible bank image lies within the target radius: a corpus-coverage limitation for image-neighbor selection.
2. A nearby bank image exists but is absent from the shortlist: a retrieval/representation limitation.
3. The shortlist contains a nearby image but the final prediction is wrong: a ranking, prior, search, or refinement limitation.

Use a geographic index over eligible bank coordinates for the first diagnostic; evaluation GPS is allowed for analysis, never as an inference feature or candidate-selection rule. Stratify by source, country/region, bank density, sequence, and native resolution. This separates geographic coverage from visual domain mismatch more directly than comparing two aggregate benchmarks.

Report within-1/25/200/750/2500-km rates, median and p90 error, rescue/regression counts, shortlist coverage, and success conditional on coverage. Use paired comparisons on identical IDs and bootstrap sequence groups where queries are correlated. Reserve an untouched final set; a small development pilot is a screening tool, not confirmation. Use multiple matched seeds for training finalists, ideally three, because the brief shows seed effects comparable to treatment effects. Keep historical leaky-training checkpoints explicitly labeled and do not recreate leakage as a training strategy.

Suggested, not mandated, promotion rule: predeclare a practical improvement target such as +1 percentage point within 25 km for an expensive full-bank experiment. Confirm a positive paired interval on the primary target, report both corpora, and predeclare a regression budget on the other target. A high-resolution arm might target within-1-km improvement with no more than 0.5 percentage points loss within 25 km. These thresholds are engineering choices to settle on development data, not paper-derived constants.

### R1. Improve candidate use before paying for a new bank

Start with the cached top-32. Train a small candidate scorer using existing frozen query/candidate vectors, their elementwise interactions, raw cosine, and candidate-to-candidate geographic agreement. Use legal training queries and their actual clean retrieval lists for mining. Include hard negatives from visually similar but geographically distant candidates. Audit false negatives and label quality before training; the brief already documents a failed fusion head that was not rescued merely by fixing false negatives.

Use a candidate-level geographic objective, initially a binary within-25-km label, with within-1-km and within-200-km auxiliary targets as an ablation. This is not the failed distance-smoothed per-step tile KL objective. A geographically positive pair need not depict the same physical object, so do not use these kilometer-scale labels as supervision for exact patch correspondences. Queries with no positive candidate need an explicit none-of-the-above target or exclusion from a relative-ranking loss, not an invented positive.

Compare cosine-only, the simple scorer alone, and scorer-plus-agent. Start with a single globally tuned residual weight. A query-only gate over existing scores is already a negative result; candidate interactions or local observations are the new evidence that would justify revisiting a learned decision.

Next search the existing bank vectors for K=64 and K=128, with identical eligibility and same-sequence exclusions. This rebuilds the neighbor cache, not the image embeddings. Re-ranking the existing 32 requires neither rebuild, although a learned reranker still costs training and inference time. Pass the best 16 onward as a first integration experiment; consuming all 128 in the prior would confound candidate quality with a changed prior distribution.

For search diversity, compare allocating a fixed beam budget across distinct candidate geographic clusters with allocating it to the highest individual scores. Preserve separate modes; do not average coordinates across continents. Cluster scale and allocation are development-set choices. Keep raw cosine, learned ranking score, prior weight, and validity mask as different quantities; findings 8 and 9 demonstrate why conflating them changes semantics.

Stop enlarging K if oracle coverage plateaus, or if additional coverage cannot be converted into better predictions at acceptable cost. Missing candidates cannot be rescued by a stronger reranker restricted to the original shortlist.

### R2. Use high-resolution information beside retrieval, not inside its contract

The code already has the right first seam: [embed_native.py](../scripts/embed_native.py) creates native-resolution conditioning, and [StreetProj](../src/encoders.py) has a separate conditioning projection. This is not an entirely new architectural proposal. First validate that path end to end through training, rollout, and serving, including missing-sidecar behavior and the review's provenance issues. Compare retrieval-only conditioning against the same checkpoint initialization plus the native sidecar. Its pooled vector is a useful cheap baseline, but it does not preserve individual local observations.

For a decisive resolution test, use the SAME genuinely high-resolution photographs in paired arms: original-detail input versus controlled downsampled input, passed through the SAME encoder checkpoint and output interface. Keep the retrieval vector and candidate list identical. Do not compare different source corpora and call the difference a resolution effect. Separately test checkpoint choice; a 224-trained SigLIP with interpolated positions is not the same experiment as the 512 checkpoint selected by the current script.

Preserve aspect ratio and record crop coverage. OSV's 512-pixel height cannot support the proposed 6x4 grid of distinct 224-pixel tiles without resampling. Genuine larger external originals may support additional tiles, but that is a new cohort and token-budget experiment. Upsampling or generative super-resolution cannot establish that additional real scene evidence was available. Screen burned-in GPS overlays consistently across arms without indiscriminately removing ordinary scene text.

If pooled conditioning is inadequate, retain a small set of region or patch features and compare them against candidate-specific detail. R2Former provides a relevant mechanism: a reranker consumes local feature correlations and spatial/attention information rather than relying only on the global descriptor. Its full training setup is not this project's frozen-backbone setup, so the proposed small frozen-feature head is an adaptation, not a reproduction. [Zhu et al., R2Former, CVPR 2023](https://arxiv.org/abs/2304.03410).

Proposed pilot, not a paper-prescribed configuration:

- Reuse unpooled crop/tile features if compatible caches retain them; pooled means cannot be inverted into their original regions. Start with one encoder.
- Otherwise extract 16-64 selected tokens per image, retain normalized image coordinates, and project to 128 dimensions. Fit PCA/codebooks only on training data. Spatially balanced selection is a useful control against selecting only sky or repeated texture.
- Precompute query details and the union of their retrieved candidate details for a bounded pilot. A 5,000-query top-32 evaluation needs at most 160,000 distinct candidate rows; the training union can be much larger and needs its own cap. Never choose the cached subset using evaluation GPS.
- Test low-detail versus high-detail features with identical tokens/head capacity. Compare a pooled-detail baseline against candidate-local interactions to distinguish pixel benefit from architecture benefit.
- Preserve the base retrieval fallback when detail is unavailable. In the controlled pilot, ensure all compared candidates have detail; in deployment, use an explicit missing-data mask and report availability rather than silently penalizing uncached candidates.

Do not assume reranking necessarily improves a strong retriever: a 2025 study finds that matching-based reranking can degrade modern VPR systems and investigates matching for verification instead. This is direct counterevidence to an unconditional reranking recommendation. [Sferrazza et al., To Match or Not to Match, 2025](https://arxiv.org/abs/2504.06116).

As a later, narrow within-1-km experiment, verify only a few top candidates with a compatible local-feature extractor and LightGlue. LightGlue is an adaptive sparse correspondence matcher, not a global geolocator or a drop-in consumer of arbitrary DINO tokens. No geometric match is inconclusive when views do not overlap; do not hard-reject every such candidate. Correspondences alone do not produce a new GPS fix without additional geometric information. [Lindenberger et al., LightGlue, ICCV 2023](https://arxiv.org/abs/2306.13643).

Optional encoder research: SigLIP 2's NaFlex variant explicitly supports variable resolution and aspect ratio, and the work targets improved dense features. That makes it a relevant frozen detail-branch pilot, not evidence that replacing SigLIP will improve geolocation. Evaluate a small checkpoint at equal pixel/token budget first; any use as the retrieval representation requires a separately matched bank. [Tschannen et al., SigLIP 2, 2025](https://arxiv.org/abs/2502.14786).

### R3. Add a geographic proposal path that is not limited to retrieved photographs

GeoCLIP aligns image features with a continuous multiscale GPS representation using random Fourier features. Its image-to-location formulation is relevant because candidate coordinates need not each have a stored matching photograph. It still learns from geographically distributed training data; it does not guarantee accurate localization in unseen regions. [Vivanco Cepeda et al., GeoCLIP, NeurIPS 2023](https://arxiv.org/abs/2309.16020).

Proposed adaptation: freeze existing image features and train a small image projection plus geographic encoder. This changes the paper's image-feature setup and needs its own validation. First compare against a cheaper coarse-cell classification head on those same features. The classifier is an auxiliary proposal baseline, not a requirement to replace the map agent with geocell classification.

Score a geographically distributed coordinate grid, retain several modes, and inject their ancestor cells as additional search proposals. Keep retrieval-derived proposals alongside them under a fixed total beam budget. Calibrate the two score sources on development data; independently normalized logits are not automatically comparable. Never collapse distinct geographic modes into one mean coordinate.

Train with legal location labels and capped geographic balancing, then compare against an image-independent occupancy prior. Evaluate on queries lacking a within-25-km retrieval candidate and on held-out geographic groups. A gain only on already-covered dense-bank queries does not demonstrate the desired generalization. Preserve and retune retrieval dropout per width rather than assuming the shipping 0.7 remains optimal after adding a branch.

Cost: cached-vector head training and a reusable grid embedding, without re-encoding the visual bank. Success would provide a principled path to general improvement beyond high-resolution matching; failure would show that this frozen feature/training regime still does not provide enough geographic information.

### R4. Test aggregation that preserves useful local evidence

AnyLoc uses off-the-shelf self-supervised features with unsupervised aggregation, including VLAD, for place recognition across environments. It is unusually compatible with the frozen-encoder constraint. [Keetha et al., AnyLoc, 2023](https://arxiv.org/abs/2308.00688).

Pilot an AnyLoc-style descriptor on a fixed eligible bank subset, using a vocabulary and optional compression fitted on training rows only. Encode every pilot query and bank row identically. Compare against both L0 and the proven L0+L1 recipe using the same bank membership. This tests a materially different aggregation of patch evidence, not another GeM exponent. Report candidate coverage, retrieval-only accuracy, and agent accuracy; a small-bank win is only a gate to a larger trial.

SALAD uses optimal-transport-based feature aggregation with a dustbin for uninformative features. Its published method includes DINOv2 fine-tuning, so freezing the backbone and training only its aggregator is an extrapolation. A frozen public checkpoint is another pilot option, subject to upstream data-overlap and licensing checks; neither option is a proven upgrade here. [Izquierdo and Civera, Optimal Transport Aggregation for Visual Place Recognition, CVPR 2024](https://arxiv.org/abs/2311.15937).

Try AnyLoc first because it does not require supervised encoder adaptation. Do not run a 19-hour bank pass for either method before measuring a matched pilot. New aggregation changes retrieval geometry and usually vector size: it needs a separate artifact generation and fully compatible query encoding, not a partial overwrite of the shipping bank. The brief's DINOv3 result is a useful warning that newer representations are not automatically better.

### R5. Train on the search states the agent actually encounters

Before changing map architecture, investigate the brief's below-chance `cell8` result with cheap counterfactuals: correct map, shuffled map, constant map, coordinate/occupancy-only prior, and retrieval disabled. Hold query features and evaluation IDs fixed. Check action labels and token order. If map changes barely affect predictions, richer rendering is unlikely to address the demonstrated failure mechanism.

DAgger motivates collecting states induced by a learner and obtaining expert supervision there, addressing the distribution mismatch of teacher-forced imitation. [Ross, Gordon and Bagnell, A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning, AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html).

A bounded project adaptation is offline beam-state mining on training queries: retain difficult incorrect tiles, supervise their existing sink/reject action, and mix them with correct-prefix examples. Compare against the current off-path-negative implementation instead of claiming negative-state training is absent. Label mined states from training GPS only and bind them to the generating checkpoint and renderer. This is DAgger-inspired hard-state augmentation, not a claim that the paper's guarantees apply.

Important limitation: descent cannot recover a ground-truth tile outside its current ancestor. For such a state the valid target is rejection, not a fabricated child action. If every retained branch is wrong, actual recovery needs reopening an ancestor or admitting independent proposals such as R3. Measure rejected wrong branches, retained correct branches, and final error together; a good sink AUC alone is insufficient.

If map-sensitive learning emerges, try balanced correct-versus-hard-map compatibility training with candidates of similar occupancy. Use spatial holdouts to prevent learned tile IDs becoming another occupancy table. Do not increase map-token resolution or add a large zoom-history transformer until these controls show that map evidence is being used.

### R6. Give maps a local geometric role, only after coarse localization works

OrienterNet demonstrates image-to-2D-map localization through neural bird's-eye-view matching, using calibrated image geometry and a local search region. This is more directly map-grounded than matching a global photograph vector to class-fraction histograms. It does not solve this project's worldwide candidate-generation problem. [Sarlin et al., OrienterNet, CVPR 2023](https://arxiv.org/abs/2304.02009).

Consider a later pilot restricted to correctly proposed local regions with usable camera metadata. Compare a compact orientation-aware local map matcher against the current final click head. Report both the conditional refinement gain and end-to-end gain, including failures to propose the region. OSM class masks must retain categorical semantics; do not interpolate class IDs.

This is a high-risk option, not the next implementation task. The official reference evaluation reports an 11 GB GPU requirement at its standard rotation count, already above this project's card; fewer rotations are suggested by the authors but still require profiling. The released model/code also carry noncommercial terms. A smaller search area, fewer rotations, or a distilled design would be new experiments, and full reference training is outside the stated budget. [OrienterNet official implementation](https://github.com/facebookresearch/OrienterNet).

### Pipeline and architectural optimizations

These are engineering proposals inferred from the code review, not claims of paper-established speedups.

**Separate data contracts, share execution logic.** Expose retrieval features, optional detail features, and geographic proposals explicitly at the artifact boundary. Internally they may still be packed for compatibility; do not introduce independent feature-concatenation logic in every consumer. Preserve a shared model assembly/forward path for training, beam evaluation, external evaluation, and serving, with parity tests. Candidate records should carry stable row IDs, raw similarity, ranking score, and validity independently.

The proposed relationship is:

```text
image -> fixed retrieval recipe -> eligible bank search -> candidates ------+
image -> optional detail encoder -> query/candidate detail -> reranker -----+-> bounded proposals
image -> independent geographic head -> geographic modes -----------------+        |
                                                                                  v
                                                               shared map descent/refinement
                                                                                  |
                                                                                  v
                                                              coordinate + calibrated uncertainty
```

The geographic branch expands the proposal set; the detail branch compares existing candidates. This distinction is essential: only the former can suggest a region absent from the visual shortlist without changing retrieval.

**Publish immutable artifact bundles.** Extend the existing provenance machinery rather than creating a second one. A manifest should bind encoder checkpoint, transform/crops/normalization, numerical backend/precision, PCA or aggregator fit identity, ordered row IDs, exact bank membership, split/exclusion policy, renderer, and dependent artifact digests. Validate completed outputs before atomically publishing a generation manifest. Refuse missing named dependencies rather than falling back to a broader bank. Validate immutable data once per load, not by repeatedly hashing multi-gigabyte files per batch. Findings 1-4, 6-7, and 10-11 are concrete reasons to prioritize these contracts.

**Budget local features explicitly.** Approximate payload sizes below use decimal GB, float16 features, and one encoder. They exclude coordinates, masks, indexes, filesystem overhead, and backups; two independent detail encoders approximately double detail storage.

| Representation / scope | Calculation | Payload |
|---|---|---|
| 768-d globals, 3.4M rows | 3.4M x 768 x 2 bytes | 5.22 GB |
| 1536-d globals, 3.4M rows | 3.4M x 1536 x 2 | 10.44 GB |
| 16 local tokens x 128-d, 3.4M rows | 3.4M x 16 x 128 x 2 | 13.93 GB |
| 64 local tokens x 128-d, 3.4M rows | 3.4M x 64 x 128 x 2 | 55.71 GB |
| 64 local tokens x 128-d, 100k-row pilot | 100k x 64 x 128 x 2 | 1.64 GB |
| Top-128 cache for 500k queries, int64 IDs + float16 scores | 500k x 128 x 10 | 0.64 GB |

Stream memory-mapped detail shards and microbatch query/candidate pairs; disk payload is not a promise about training VRAM. The top-128 example is about 0.48 GB more than top-32 for those 500k queries. A token cache should grow only after measured benefit, not immediately to the full bank. Compression must itself be ablated for geographic recall, and PCA must not be fitted on final evaluation images.

**Optimize retrieval only if it is a measured bottleneck.** Faiss provides exact and approximate vector-search approaches with memory/speed/accuracy tradeoffs. An optional IVF/PQ candidate stage followed by exact reranking of original normalized vectors could reduce search cost; approximation is not inherently an accuracy improvement. [Douze et al., The Faiss library, 2024](https://arxiv.org/abs/2401.08281).

Keep an exact-search reference on a representative development set. Measure neighbor recall and geographic coverage after all exclusions, not just unfiltered index recall. Oversample/refill when filtering self or same-sequence neighbors. A suggested gate is at least 99% exact-neighbor recall plus geographic non-inferiority; select the actual threshold against latency and memory needs. Do not assume a compatible Windows GPU package is already available, or replace the current working search merely to introduce a library.

**Profile the existing embedding producer/consumer path.** The native-embedding script itself labels its newer feeding changes as not yet A/B verified under load. Measure decode time, queue starvation, host-to-device transfer, encoder time, peak pinned memory, and end-to-end images/second on a fixed shard. Compare bounded prefetch and reusable pinned staging under the same numerical recipe. The brief already rules out larger batches beyond saturation and `channels_last` as useful defaults. Preserve per-encoder completion masks and resumability, and never mix numerical recipes within a retrieval-bank generation to gain throughput.

**Calibrate confidence before adaptive compute.** Beam dispersion or a confidence radius is not automatically the probability of being within 25 km. Fit and evaluate calibration on held-out groups, then report risk-versus-coverage and latency-versus-accuracy. Adaptive verification or abstention is useful only if its saved cost or reduced risk is measured; always retain full-cohort accuracy so dropping difficult images cannot manufacture an apparent gain. The brief's failed confidence gates make this a later step, not an assumed easy win.

### Concrete next experiment sequence

1. **Protocol and coverage report, no re-encoding:** after correcting the relevant P1 paths, lock the shipping checkpoint/bank and export paired OSV/external predictions. Measure eligible-bank proximity and A_r(16/32/64/128), using existing embeddings for any larger-k search. This decides whether candidate selection or absent coverage deserves the next budget.
2. **Detail experiment, fixed retrieval:** compare no sidecar, pooled native conditioning, and a bounded compact-detail candidate scorer on identical IDs and candidate lists. Include same-image downsample controls and the existing agent baseline. Profile one encoder and a small token cache before allocating a full corpus pass.
3. **Generalization experiment, no new visual bank:** compare a coarse geographic head and a GeoCLIP-inspired coordinate head using cached features, then admit their proposals under the same total beam budget. Judge especially the external and retrieval-uncovered subsets. Promote only if rescues outweigh regressions beyond seed noise.

If coverage is the limiting factor and R3 fails, prioritize R4 or targeted acquisition of genuinely new geographic coverage over more reranker depth. Any future acquisition should follow training/development geography and a predefined deployment objective, not select images around final-test GPS. If coverage is good but ranking fails, prioritize R1/R2. If the agent still does not beat a simple reranker at matched cost, keep it optional until R5/R6 establishes a distinct map-based benefit.

No recommended experiment, dependency installation, model download, artifact rebuild, or implementation change was performed for this addition. The paper links support the described mechanisms; all project-specific architectures, budgets, and expected benefits remain proposals to test.

---

## Follow-up: answers after the corrected coverage and ranking measurements

Written 2026-09-05 against workspace `96779a9`, answering the five questions in the supplied `astra_prompt.md`. This addendum supersedes the earlier roadmap where its premises conflict with the new measurements; the earlier text is retained as history. Only this document was edited. The supplied measurements were compared with [COVERAGE.md](../runs/COVERAGE.md), [COVERAGE_1536.md](../runs/COVERAGE_1536.md), and the current reporting code, not rerun. The consensus/query-expansion numbers below remain user-supplied evidence.

**Bottom line:** prioritize a small, training-free, per-candidate visual-verification experiment, but test whether usable visual overlap exists before building a token bank. Also allow one inexpensive descriptor-specificity control. Do not prioritize more geographic coverage for OSV at 25 km, and do not infer that the agent contributes only one percentage point. Keep the 1-km candidate-generation problem separate.

### 1. Is per-candidate verification the right survivor? What else survives?

It is the leading new-information hypothesis, not the only survivor and not yet a demonstrated solution. There are four important qualifications to the argument in the prompt.

1. **Geographically correct is not necessarily visually matchable.** A candidate 20 km away can satisfy the metric while sharing no facade, road segment, or sign with the query. Conversely, two views of one distant landmark may overlap while their camera positions differ. The 33.2-percentage-point ranking headroom measures candidate-coordinate availability, not recoverable correspondence evidence. Measure how much of it has visible overlap.
2. **Cosine does not exhaust the information in pooled vectors.** It reduces two vectors to one scalar. A learned pairwise metric can distinguish pairs with identical cosine using different feature components or semantic compatibility. A small diagonal/bilinear scorer on existing features is therefore a legitimate control without local matching or candidate-set pooling. Given the failed fusion heads, keep it a control, not another large search over architectures.
3. **A candidate-independent score is not immune to wrong candidates.** The score of a fixed pair can be invariant to the other 31 images, but its chance of winning the maximum is not. Illustratively, independent 1% false-accept events across 31 negatives produce about a 27% chance of at least one false accept. Real candidates are correlated; evaluate all 32 jointly rather than interpreting balanced-pair AUC as retrieval performance. Local verification also aggregates correspondences within a pair; its distinction is avoiding cross-candidate voting, not literally performing no pooling.
4. **The negative results do not prove the entire positional family impossible.** Candidate coordinates alone can encode a prior; query-derived visual features can score a location hypothesis without knowing query GPS. The latter is conditional image-to-geography compatibility, not merely bank density. The existing prior already illustrates this distinction. However, the supplied consensus results are sufficient to demote another hand-designed density vote.

The observed oracle increment from K=16 to K=32 is not an estimate of the fraction of those sixteen slots that are correct. Correct but redundant candidates add zero oracle coverage. Before attributing both failed aggregations to a mostly-wrong set, measure the number of within-radius candidates per query and their geographic-mode composition, separately when rank-1 is right and wrong. Also, consensus at K=8 is reported positive (+0.76 pp), and query expansion at K=2 is almost unchanged; neither table establishes a strictly decreasing curve starting from rank-1 or a statistically established loss for every setting. This does not justify a parameter sweep, but the distinction matters when generalizing the elimination.

#### The cheapest remaining alternative: visual specificity / reverse-neighborhood density

Ask whether a high cosine is unusual **for that candidate**, rather than whether many candidates agree geographically. A generic road image may be similar to many unrelated queries. This is a property of the visual feature distribution, not a geographic vote over the current top-32.

CSLS subtracts local-neighborhood similarity terms to address embedding hubs. Its original evidence is cross-lingual word retrieval, not geolocation, so this is a mechanism-inspired control. For a fixed query, its query-side term is constant across candidates; the candidate-side correction can change the ordering. [Conneau et al., Word Translation Without Parallel Data, ICLR 2018](https://arxiv.org/abs/1710.04087).

Concrete bounded adaptation:

```text
U = union of candidate IDs for the pilot queries
T = fixed, sequence-balanced training reference set; no development/test queries
b(c) = mean of the 10 largest cosine(c, t), t in T,
       excluding c itself and same-sequence references
score(q, c) = cosine(q, c) - lambda * b(c)
```

Use 4,096 reference rows initially and restrict scoring to the unchanged top-32. Predeclare a small development-only coefficient set, including zero; lambda=0.5 corresponds to the candidate-dependent part of CSLS up to a positive rescaling, but this training-reference approximation is not exact CSLS. For 1,024 pilot queries, U has at most 32,768 rows. At 768 dimensions this is about 103 billion multiply-accumulates, streamed in blocks, plus only a scalar per candidate to store. It needs no image decoding and no full bank-to-bank pass. Benchmark rather than labeling this computation free. First check whether b(c) predicts errors within narrow cosine bins; stop if it contains no useful conditional signal or merely suppresses correctly dense regions.

Reciprocity is related but different. A held-out query is absent from a bank-only index, so searching for its ID in a candidate's stored bank neighbors will always fail. Define the query's reverse rank by hypothetically inserting that query into the candidate's legal comparison set. A candidate's k-th-neighbor similarity provides a reverse-neighborhood threshold, with explicit self/sequence exclusions and tie handling. Only compute such thresholds for the pilot candidate union before considering a full bank-to-bank index. Reciprocity can penalize dense genuinely correct places too; it is not new visual evidence. Full k-reciprocal/Jaccard reranking additionally reintroduces neighborhood-set aggregation. Its established evidence includes person re-identification, not this kilometer-scale task. [Zhong et al., Re-Ranking Person Re-Identification With k-Reciprocal Encoding, CVPR 2017](https://arxiv.org/abs/1701.08398).

One other category survives: **image-conditioned semantic compatibility with a location**. A frozen-feature geographic head can distinguish regional visual attributes without requiring shared physical landmarks or neighbor consensus. GeoCLIP supplies an image-to-GPS precedent, not a guarantee for the current features. Defer this behind the cheap pair tests for OSV ranking; reconsider it if the overlap audit finds that most 25-km positives are non-overlapping. [Vivanco Cepeda et al., GeoCLIP, NeurIPS 2023](https://arxiv.org/abs/2309.16020). Generic OCR is not a priority given the project's negative evidence and overlay leakage.

### 2. Cheapest local verification: exact design, budget, and falsification pilot

**First, spend zero encoder passes if compatible tile caches exist.** [tile_cache.py](../scripts/tile_cache.py) stores six individual 1536-d tile vectors, not just their mean. Compare the six DINO blocks per query against the six per candidate with late interaction, while leaving retrieval fixed. This is coarse region matching, not dense patch verification. It tests whether discarding regional identity was harmful; a null does not reject finer correspondences. Use only a cohort defined before checking cache availability, or clearly label a cache-complete subset as a diagnostic with its own baseline.

**The first genuine patch pilot should use one frozen DINOv2-B encoder, 24 tokens per image, and no trained reranker.** AnyLoc provides evidence that frozen DINO-family local features contain useful correspondence information and that layer/facet choice matters; its preferred large-model layer number must not be copied into the 12-block ViT-B. R2Former supplies a local-correlation reranking precedent, but its published trained system does not validate this much smaller training-free adaptation. [AnyLoc](https://arxiv.org/abs/2308.00688), [R2Former](https://arxiv.org/abs/2304.03410).

Proposed fixed recipe, chosen for a cheap test rather than claimed optimality:

- Reuse the six 224-pixel views of the documented 3x2 tile recipe, recording its resize geometry. Encode query and candidate details identically. Do not alter the shipping global features, cosine, or top-32 identities.
- Extract patch tokens after block 11 of 12, apply the model's normalization, and discard all prefix tokens. Keep final-block features on the small diagnostic set as a control. Each tile has a 16x16 patch grid.
- Select the token nearest each of the four quadrant centers: four per tile, 24 per image. This deterministic spatial rule is deliberately simple and label-free. It can miss thin signage; that is a budget limitation to test, not a reason to assume the mechanism absent.
- Fit an unwhitened 768-to-128 PCA using at most 100,000 patch tokens from training images only; freeze it. Subtract its training mean, project, L2-normalize each token, and store float16. Keep original 768-d selected tokens on the small diagnostic set to test whether compression destroys the signal.
- Store each token's original-image normalized x/y coordinates. Do not require equal query/candidate x/y locations: viewpoint changes move correspondences. Do not select tokens by largest raw feature norm. High-norm ViT artifacts can occur in uninformative regions; inspect this model instead of assuming norm means useful detail. [Darcet et al., Vision Transformers Need Registers, ICLR 2024](https://arxiv.org/abs/2309.16588).

The current [embed_street.py patch-grid branch](../scripts/embed_street.py#L236) average-pools final patch maps. It is a possible extraction starting point, not this exact selection recipe, and an existing pooled grid cannot recover the original selected tokens.

For each pair, form the 24x24 cosine matrix S. Keep mutual row/column nearest matches and assign each the positive margin over the strongest alternative in its row or column. Use the sum of those margins divided by 24 as a local score L. Then compare:

```text
baseline:       score(q,c) = original cosine(q,c)
local-only:     score(q,c) = L(q,c)                  # diagnostic, not default
residual:       score(q,c) = cosine(q,c) + lambda * L(q,c)
```

Use a predeclared small lambda set such as {0, 0.1, 0.3, 1}, selected on development groups only. No pair-specific coefficient, query-confidence gate, GPS feature, or geometric fit in this first test. Empty/ambiguous matches give zero local evidence rather than a hard rejection. The score is a correspondence-ambiguity heuristic, not a calibrated same-place probability. Initially evaluate its candidate-coordinate output directly; then test the best fixed rule alongside the actual agent on identical queries. Preserve raw cosine separately from the new score when integrating the prior.

#### Storage and compute accounting

These are arithmetic estimates, not measured timings. Decimal MB/GB; one encoder; 24 tokens with 128 float16 components and two float16 coordinates cost **6,240 bytes/image**. Masks, IDs, images, provenance, and filesystem overhead are extra.

| Scope | Maximum distinct detail images | Detail payload |
|---|---:|---:|
| 1,024 queries plus their full top-32 union | 33,792 | 210.9 MB |
| 5,000 queries plus their full top-32 union | 165,000 | 1.03 GB |
| Full 3,400,180-row bank | 3,400,180 | 21.22 GB |

One pair costs 24 x 24 x 128 = 73,728 multiply-accumulates for the similarity matrix; 32 candidates cost about 2.36 million per query, excluding extraction and reductions. All 32 float32 similarity matrices occupy about 72 KiB. Six 224-pixel ViT forwards per distinct image dominate extraction; selecting 24 tokens after a forward does not reduce encoder compute.

Run inference-only extraction in small tile batches, initially eight tiles, and profile peak VRAM before increasing it. Stream image shards and detail arrays; do not keep the bank on the GPU. Extraction time is unique_images / measured_images_per_second. For the 33,792-image maximum, 20 images/s would mean about 28 minutes and 5 images/s about 113 minutes, excluding cold shard reads. Those rates are illustrative, not forecasts. A full patch pass is not automatically the previously measured 19-hour global-embedding job.

No partial change to retrieval geometry is involved: detail is a separately versioned pair-scoring channel. Require a complete detail union for the controlled comparison. If an image cannot be decoded or located, keep the query in the denominator and apply the documented fallback; do not silently remove difficult examples. A separate detail cache still requires identical extraction semantics on both sides.

#### The smallest pilot that can kill a proposal without pretending to prove a small gain

**Stage A: mechanism/implementation diagnosis.** On development groups, visually inspect approximately 128 rank-1-wrong/top-32-covered queries and whether any of their geographically correct candidates visibly overlap. Separately assemble 64 clearly overlapping legal cross-drive pairs and 64 visually similar non-overlapping negatives, plus synthetic transformed-image pairs. This outcome-stratified set is diagnostic only, never an accuracy estimate. Use image coordinates/correspondence labels for this audit, not as inference GPS.

Compare the 24-token scorer against an uncompressed control and a denser, 96-token version on these pairs; 96 means sixteen spatially distributed tokens per tile. Retain dense features only for this small diagnostic, not the corpus. Synthetic pairs must pass but are insufficient: they do not test viewpoint and time variation. If sparse tokens fail but dense tokens succeed, the budget/selection rule failed. If frozen DINO tokens fail real overlap but an independent local matcher succeeds, this feature/scoring choice failed. Neither is evidence that all verification is impossible.

An independent frozen XFeat extractor is a useful bounded control, not another bank upgrade: the paper targets efficient sparse/semi-dense matching, and the official implementation provides 64-d descriptors. Try 256 spatially distributed detected points on the diagnostic pairs only, with mutual descriptor matching. This count is our budget choice, not the paper's reported benchmark setting. It requires a new dependency/checkpoint but no fine-tuning; nothing is downloaded as part of this review. [Potje et al., XFeat, CVPR 2024](https://arxiv.org/abs/2404.19174), [official implementation](https://github.com/verlab/accelerated_features).

**Stage B: small end-to-end screening.** Freeze 512 development and 512 separate screening queries by stable IDs and sequence groups, sampled before inspecting outcomes. Score every cached candidate, not just one known positive and one easy negative. This gives the 1,024-query budget above. Compare baseline, coarse six-region matching if available, local-only, and one development-selected residual. Export rescues, regressions, and errors on all cases, including queries with no positive candidate. Reserve external screening groups separately; OSV results do not establish KartaView transfer.

Stop before a full-bank pass if controls work but the screening result is materially harmful, or the upper paired confidence limit is below a predeclared useful effect. An interval spanning a useful positive effect is **inconclusive**, not a null. With 10% paired outcome discordance, 1,024 independent queries have an approximate 95% half-width of 1.9 pp; detecting a +1-pp effect with 80% power needs roughly 7,840 independent queries before sequence clustering. Thus this small pilot can reject gross failure, not reliably rule out +1 pp. Expand only a surviving fixed recipe, retain grouped uncertainty estimates, and do not tune on the expansion's final set.

### 3. How much should the counter-evidence lower confidence?

Substantially for **unconditional replacement of cosine by a matching score**; less decisively for testing genuinely new pairwise observations. I would budget for a possible modest gain, not assume that a large fraction of the 33.2-pp oracle gap is recoverable. There is no defensible numeric probability of success from seven experiments: they share data, design assumptions, and failure modes and are not seven independent Bernoulli trials of patch verification.

The 2025 counter-paper is more specific than “matching does not work.” It reranks MegaLoc's top-100 by geometric inlier counts, generally using a 25-**meter** success radius and 512x512 matcher inputs. It finds degradation on saturated datasets, but gains on difficult night, occlusion, and indoor settings. Its 100-meter check retains some negative results, so the threshold does not explain everything. This is neither a universal rejection nor direct validation of frozen DINO patch scores at 25 **kilometers**. [Sferrazza et al., To Match or Not to Match: Revisiting Image Matching for Reliable Visual Place Recognition, 2025](https://arxiv.org/abs/2504.06116).

Here, rank-1 at about 49.7% is not saturated, but that alone does not make verification appropriate: its wrong predictions may lack visually matchable alternatives. That is why the overlap audit is more informative than another citation count or oracle-gap calculation.

Use the following decision logic, with every score and control fixed before examining final evaluation results:

| Observation | Interpretation / next decision |
|---|---|
| Identical-image/transformed-image controls fail, or cached versus direct features disagree | Extraction, normalization, coordinate mapping, or matching implementation is not validated; no scientific null yet |
| Scores depend on candidate ordering or batch partitioning | An allegedly pairwise implementation is coupling candidates; diagnose before evaluating the hypothesis |
| 768-d or 96-token controls work, compressed 24-token version fails | Compression/token selection is the failed design; measure whether a larger affordable version is worthwhile |
| DINO fails visually overlapping cross-drive pairs, an independent matcher succeeds | Current frozen features or scoring rule are unsuitable; do not indict local correspondence generally |
| Overlap pairs score well, but few ranking-limited queries contain any overlap | Local verification has limited applicability to the 25-km metric; try semantic compatibility rather than deeper matching |
| Pair tests look good, full top-32 decisions regress | Extreme false positives, score calibration, or near-positive label ambiguity defeat ranking; pair AUC was insufficient |
| Reranker beats cosine, but reranker-plus-agent does not beat the existing agent | Evidence is redundant with what the agent already uses, or integration changes the prior; no end-to-end gain established |
| Correct controls, frozen protocol, and sufficiently narrow upper effect bound below the useful target | Reject this mechanism/configuration at this budget; do not repeatedly relabel it an implementation problem |

Additional checks: permuting token storage while moving x/y with it must preserve a set-based score; breaking descriptor-to-coordinate alignment must affect a geometric verifier, but need not affect the non-geometric first pilot. Shuffling candidate detail identities should remove any genuine advantage on average. For a frozen model in evaluation mode, changing batch size should not change outcomes beyond documented numerical tolerance. Avoid claiming success merely because a new score has a nonzero variance or a synthetic positive scores highly.

The seven failed rules justify a small, preregistered search and harsh accounting of regressions. They do not justify assuming a null in advance, and they do not license an endless sequence of post-hoc rescue explanations after a well-controlled null.

### 4. Is a separate sub-kilometer mechanism justified?

Yes, as a separate candidate-generation/refinement experiment, not automatically a second full model. At 1 km the supplied numbers imply three distinct budgets:

| Limitation for returning a bank candidate's coordinate | Fraction | What could address it |
|---|---:|---|
| Correct candidate already in top-32, not rank-1 | 19.3% | Pairwise verification may help if views overlap |
| Legal within-1-km row exists outside top-32 | 51.7% | Larger or different retrieval, or a second-stage geographically restricted search |
| No legal within-1-km row anywhere in the bank | 18.6% | New geographic coverage or predicting a new coordinate; reranking cannot fix it |

Even a perfect top-32 selector is capped at the reported 29.7% within 1 km. Do not build a sub-kilometer strategy solely around rearranging that list. Also, these are coordinate-policy-dependent figures: the extension's tile-center quantization discussed in question 5 is material at this radius.

**Proposed second-stage mechanism:** use broad candidates or agent branches to propose a few geographic search regions, then retrieve finer candidates from the existing eligible bank *within those regions*, before applying local verification. This uses candidate position as a hypothesis specifying where to look, not as proof that the location is correct.

For a bounded pilot, propose at most four distinct centers from the unchanged broad shortlist, without GPS-based choice. Search a predeclared radius such as 25 km around each center. Use existing visual vectors to rank the eligible rows in each region and retain, for example, 16 per region, with self/same-sequence filtering and deterministic deduplication. Score these new fine candidates with the validated detail branch. Tune region radius and allocation only on development data; the first 4 x 16 budget is an experiment, not an optimum.

This changes candidate membership and therefore requires a new staged-retrieval artifact/workflow. It does not require globally re-encoding 3.4M images, because the initial regional search uses their existing vectors. It is not a claim that a head can discover uncached candidates inside the current fixed top-32 pipeline. Do not benchmark it by secretly centering the region on query GPS; an oracle-centered arm is diagnostic and must be labeled as such.

Before extracting more details, measure whether the proposed regions contain any legal within-1-km row, and whether the regional visual search retrieves it. Keep the original broad prediction as the fallback outside successful refinement, under a fixed development-selected combination rule. Report full-cohort within-1-km and within-25-km accuracy, especially how often refinement turns a correct coarse prediction into a wrong one. The hard part may remain finding the correct region, not matching inside it.

Evaluate additional real pixels within this fine stage using same-photograph, same-checkpoint downsampling controls. Training-free local correspondence is a plausible consumer of those pixels, but the reported resolution gains remain a hypothesis until deconfounded. A sparse frozen matcher such as XFeat can test whether actual correspondences benefit without a large DINO-token cache; it cannot infer accurate camera GPS merely from matching descriptors.

For cases with no sufficiently close legal bank coordinate, the existing map/click path is still relevant because it can output a new coordinate. A local image-to-map method such as OrienterNet is a research precedent, but it needs usable geometric assumptions and a local prior and is not an 8-GB drop-in solution. Do not promise sub-meter or sub-kilometer results from map semantics alone. [Sarlin et al., OrienterNet, CVPR 2023](https://arxiv.org/abs/2304.02009). First determine whether true bank GPS can be recovered from source metadata; finer output coordinates may fix a measurement/representation limitation without another network.

### 5. What might still be measured incorrectly?

There are already concrete reporting/implementation hazards in the new diagnostic, in addition to prospective protocol risks. These observations concern the code at `96779a9`; they do not assert that a listed branch affected the published default run.

#### Directly observed in the current coverage code

**M1 — The disproved baseline is still generated as prose.** [coverage.py:185](../scripts/coverage.py#L185) hardcodes 56.4% and the near-equivalence-to-agent conclusion into every report. Both current coverage Markdown files contain that prose above tables showing 49.7% or 50.1%. This is precisely how a corrected computation can continue propagating an incorrect strategic conclusion. Generate headline statements from the actual counted outcomes and attach the cohort/predictor ID. Do not calculate a component's causal contribution by dividing two accuracies.

**M2 — A finite geographic probe is not exact coverage, and the stated bound is backwards.** [nearest_legal](../scripts/coverage.py#L113) checks only 64 geographic neighbors by default. If they are all the query's sequence, it returns infinity. The caller counts that query as uncovered and describes the result as a “floor on class 1” at [line 168](../scripts/coverage.py#L168). It is instead an **upper bound on the uncovered fraction** under otherwise correct eligibility/coordinates: a legal 65th neighbor may still be inside the radius.

A concrete counterexample needs no model: the first 64 geographic neighbors are same-drive, a different-drive row at rank 65 is 200 m away, and the cached top-1 is that legal row. The geographic probe says uncovered while the retrieval calculation says solved. Since [the partition](../scripts/coverage.py#L209) computes these flags independently, one query can enter both classes. The diagnostic also computes distance percentiles only over finite results at [line 234](../scripts/coverage.py#L234), excluding unresolved queries. The reported nearest-distance distribution is therefore conditional if any probe was exhausted.

Required evidence before calling the measurements exact: record the exhausted-query count in the artifact, then expand the search for unresolved queries until a legal neighbor is found or all rows are exhausted. For a fixed radius, a radius search can stop once all within-radius candidates are known. Assert per query that `top1_hit implies topK_hit implies covered` and that the four partition indicators sum to exactly one. Rounded aggregate percentages cannot substitute for these checks. If the original run exhausted zero probes, this hazard does not alter that run's coverage numbers.

**M3 — The spatial-split guard is incomplete.** [coverage.py:152](../scripts/coverage.py#L152) calls `sp.ineligible` with zero x/y arrays and only handles `bad_rel`, ignoring `bad_ext`. Real extension tile coordinates are required to validate a geographic holdout. This is not evidence against the default sequence-split result, but the script accepts other kNN files and must not silently label those banks eligible under a cell split.

**M4 — The diagnostic trusts more cache identity than it establishes.** At [lines 143-159](../scripts/coverage.py#L143), it reads live labels and cached neighbor IDs without checking the cached split hash against the live hash or establishing each neighbor's current eligibility through the full shared validation path. Checking `bank_rows` and release membership is not equivalent to verifying the cached `idx` rows, same-sequence exclusion, and row-order identity. Require the same manifest/neighbor checks as production evaluation and record hashes in the report. No malformed current cache was demonstrated in this follow-up.

**M5 — Physical bank coverage and tile-center coverage are conflated at small radii.** [bank_coords](../scripts/coverage.py#L87) uses true lat/lon for release rows but z16 tile centers for extension rows. A roughly 611-m-wide equatorial tile has a half-diagonal of about 432 m; this is not negligible beside a 1-km threshold or a reported 340-m median nearest distance. Ground scale varies with latitude. A center can fall inside the threshold while the original photograph falls outside, or vice versa.

If candidate prediction really returns that center, its candidate-coordinate oracle remains a valid diagnostic of that output policy. It is not exact physical coverage of the original bank photographs. Recover original GPS where possible and report both definitions. Otherwise, for each tile derive a conservative center-to-point distance bound delta: `d_center + delta < r` is definitely inside, `d_center - delta >= r` is definitely outside, and the remainder is ambiguous. Quantify that ambiguity before attributing small within-1-km differences to higher resolution. Also align whether rank-1, agent, and coverage evaluation use true coordinates, tile centers, or learned offsets.

These issues are documented here only; no reporting script or artifact was corrected during this task.

#### Next protocol checks, in priority order

1. **Put every baseline on identical query IDs.** The relevant agent comparison is a paired evaluation of rank-1, reranker, agent, and agent-plus-reranker on the same bank and same images. Numerically, 57.5 minus 49.7 is 7.8 pp, not 7.4; 57.5 minus the 1536-bank 50.1 gives 7.4 but changes the retrieval arm. Neither establishes a paired treatment effect across different cohorts. Repartition the agent's actual errors; the 33.2% ranking bucket belongs to rank-1, not automatically to the agent.
2. **Freeze cohorts by stable identity, not a seed alone.** A seed against a differently ordered or extended parquet file selects different images. Export a sorted ID manifest, source/shard/sequence composition, and inclusion probabilities. If balancing geography or sequence length, report that target population separately from the natural per-image average. Do not tune on a repeatedly inspected test cohort and continue calling it untouched.
3. **Publish every denominator and missingness count.** Account for decode failures, absent detail rows, invalid GPS, absent maps, exhausted probes, and skipped sequences. Paired intersection-only evaluation can favor easy, cache-complete images; report the original cohort and fallback outcomes. The median over finite geographic probes is a concrete instance of this hazard.
4. **Audit capture identity across release/extension boundaries.** Same-sequence filtering is not cross-sequence duplicate detection. Check canonical IDs, duplicate frames across sources, null sequence IDs, and sequence truncation: the diagnostic casts IDs to `U40`, so verify that truncation does not merge distinct real sequences. Do not declare a new leak without those checks, but do not infer independence merely from different sequence strings.
5. **Check coordinates and thresholds with independent fixtures.** Latitude/longitude order, degrees/radians, Earth-radius convention, strict `<` versus `<=`, projection limits, antimeridian behavior, and extension row offsets should have small analytical tests. Report boundary counts near 1 km and use original GPS when interpreting physical proximity. A 25-meter VPR metric and a 25-kilometer geolocation metric are different tasks, not interchangeable recalls.
6. **Quantify churn in the PCA control.** A net 0.4-pp difference does not imply that only 0.4% of queries changed or that PCA has no subgroup effect. Export paired top-1 flips, correct-candidate gains/losses, and oracle-set overlap. The similar aggregate ceiling makes PCA an unlikely explanation for the entire gap, not a proof that projection is irrelevant.
7. **Keep repeated-search and uncertainty accounting honest.** Paired/grouped intervals, preregistered primary radius, and a fresh confirmation cohort matter more than another best-of-many number. Count how many lambdas, token counts, layers, seeds, and metrics were tried. A paired interval on the same biased first-N cohort estimates the effect on that cohort; pairing does not repair its geographic selection bias.
8. **Decompose KartaView independently.** The corrected OSV coverage result neither proves nor disproves an external coverage bottleneck. Use the external query IDs and the exact bank the external evaluator searched. Stratify genuine original resolution separately from source, camera, compression, and crop field of view. Remove GPS overlays under one documented policy and keep a sensitivity report rather than crediting them as localization skill.

### Recommended commitment, revised from the earlier roadmap

Authorize no corpus-scale encoding on the evidence available. The next useful sequence is: repair/validate the coverage diagnostic and establish a same-cohort agent baseline; audit overlap; run one descriptor-specificity control and the bounded frozen-feature verification pilot; expand only a surviving fixed recipe. Investigate regional fine retrieval separately for 1 km. Keep the image-to-geography branch as an alternative if overlap is too rare, rather than treating it as the immediate answer to a nonexistent OSV 25-km coverage shortage.

The falsifiable central claim is: **on unchanged legal candidate lists, local pair evidence produces more correct replacements than incorrect replacements beyond what the current agent already achieves, at an affordable extraction/storage cost.** The proposed controls distinguish an invalid implementation, insufficient token representation, missing visual overlap, and a genuine failure to improve decisions. None of those outcomes is yet measured by this document.
