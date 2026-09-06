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
