No files were changed. The most consequential finding is that the fusion-head experiment is not an apples-to-apples test of learned fusion versus mean pooling.

<!-- annotated 2026-09-03 -->

**Status markers.** `[x]` fixed and verified. `[~]` confirmed by measurement,
not yet fixed. `[!]` confirmed and it changed a conclusion already published.
`[-]` refuted or narrower than stated. `[.]` already known and deliberately
deferred. Unmarked items are not yet triaged.


## Critical: fusion conclusions

1. `[!]` **The fusion residual does not initialize to the reported baseline.**
   [fuse_head.py](/C:/Users/longd/Programming/geolocation_model/scripts/fuse_head.py:103) projects the 1536-d mean through a fixed random 1536→768 matrix before adding the learned residual. At initialization it is a normalized random projection, not the measured mean-pooled representation. The −11.30 pp result therefore combines:
   - random dimensionality reduction,
   - learned fusion,
   - and the changed output width.

   It establishes that this particular 768-d system is bad, but not that a learned identity-preserving fusion head is bad.

   > **Confirmed - conclusion retracted.** Premise confirmed: `base_w` is a fixed random 1536->768 map, so the head does not start at the measured baseline. **Quantified**: scoring the head at init (`scripts/fuse_init_probe.py`) gives 33.1% within 25 km against the 1536-d mean's 33.7% -- the projection is worth **-0.63 pp of the -11.30 pp**, and a train-fitted PCA to 768 gives +0.53 pp. So the confound is real but small, and the trained head is 10.7 pp below its own starting point: training destroys the representation rather than failing to improve it. The recorded conclusion "the pyramid fusion head is negative, do not re-open" is **retracted** -- what was measured was the objective below.


2. `[~]` **The printed baseline and the head’s internal baseline differ.**
   The head averages normalized level vectors without normalizing their mean at [fuse_head.py:123](/C:/Users/longd/Programming/geolocation_model/scripts/fuse_head.py:123). The displayed baseline normalizes again at [fuse_head.py:397](/C:/Users/longd/Programming/geolocation_model/scripts/fuse_head.py:397). The code explicitly says they should match.

   > **Confirmed.** Confirmed. `FuseHead.baseline` omits the per-encoder renormalisation that the printed baseline applies, so the two differ by a dino/siglip reweighting -- the same shape as the recorded "encoder blend was accidental" bug.


3. `[x]` **The contrastive batches contain false negatives.**
   [fuse_head.py:266](/C:/Users/longd/Programming/geolocation_model/scripts/fuse_head.py:266) creates directed near pairs, then applies diagonal symmetric cross-entropy without deduplicating anchors or masking other valid positives. Two nearby positives in the same batch are trained against one another.

   > **Fixed.** Confirmed and **far worse than stated**. Measured over 60 real batches of 256: **100% contain a duplicate anchor**, and **50.3% of off-diagonal cells are true positives trained as negatives** -- 128 of every 255. Half the gradient was pushing apart images within 5 km of each other, which explains item 1's 10.7 pp collapse. Fixed: the loss masks off-diagonal true positives by great-circle distance and batches at most one row per anchor (`--mask-fn`, `--uniq-anchor`, both default on; `--mask-fn 0` reproduces the old runs). Mask unit-tested to kill co-located cross-terms, preserve the diagonal and keep the loss finite. Corrected arm re-running; epoch-1 loss 3.36 against 5.03 before.


4. `[~]` **The “mean + head, PCA” comparison leaks test data.**
   PCA is fitted over every row, including test queries, at [fuse_head.py:416](/C:/Users/longd/Programming/geolocation_model/scripts/fuse_head.py:416), before the train/test retrieval evaluation.

   > **Confirmed.** Confirmed. `combo` spans every row and `pca_to` fits on a sample of it, so the `mean + head, PCA to 1536` arm -- the one with the best median -- is fitted transductively on the test queries.


5. **The exported combined representation is incomplete.**
   The PCA-transformed vectors are saved, but the corresponding PCA mean and basis are not. New queries cannot be embedded in the exported coordinate system.

6. **Hard-negative scheduling is calculated from the wrong number of batches.**
   In `--hard` mode, OneCycle uses the full pair count, while training drops small buckets and bucket remainders. Consequently, the schedule may never approach its final phase. The latest pyr47 run did not use `--hard`, so this did not affect that result.

7. **Fusion ignores pyramid completion masks.**
   Missing pyramid rows would silently become zero tokens. I checked the actual `pyr47_done.u8.npy`: all 95,292 encoder-row entries are complete, so this bug did not affect the reported pyr47 run.

## External evaluation and leakage

8. **`eval_highres.py` does not implement the claimed shipping preprocessing.**
   [eval_highres.py:90](/C:/Users/longd/Programming/geolocation_model/scripts/eval_highres.py:90) omits the PIL decoder hint used by cache generation and omits the BF16 autocast used by cache generation and serving. External images are therefore embedded by a slightly different pipeline.

9. **PCA identity is not validated.**
   [eval_highres.py:143](/C:/Users/longd/Programming/geolocation_model/scripts/eval_highres.py:143) defaults to the bank55 PCA and checks only output width. Any unrelated 768-d PCA is accepted silently.

10. **Bank metadata is inferred from filenames.**
    Renaming a bank or using a custom bank can cause the evaluator to load the wrong extension metadata while still passing dimensional checks.

11. **Restricted-bank k-NN cache names omit the bank limit.**
    Different bank restrictions can collide or reuse incompatible caches.

12. `[~]` **`eval_highres.py` and `multiquery.py` restore an implicit `s10` default.**
    This bypasses the repository’s newer “release must be explicit” rule and reintroduces wrong-release execution.

   > **Confirmed.** Confirmed. Both scripts set `OSV_RELEASE=s10` when unset, before importing config, which bypasses the no-default rule in exactly the two tools that produce every external number. Narrower than stated in one respect: they only set it when unset, so an explicit choice is never overridden.


13. **The external KartaView corpus has no enforced OSV-overlap exclusion.**
    The harvest is seeded near OSV images and KartaView is an OSV source, but neither image IDs nor sequences are checked against OSV. The assumption that external queries need no same-sequence masking is therefore unproven. This is a high-confidence leakage risk; an actual collision requires a data audit.

14. **Leak screening does not gate downstream use.**
    `screen_leak.py` is the only writer of `leak_blocklist.json`; no training or evaluation path reads it. The existing offline run screened only 2,000 of 47,646 images and did not write a blocklist.

15. **Leak-screen failures are counted as successfully screened.**
    Unreadable images are skipped, but the denominator remains the complete manifest size.

16. **Partial screening can overwrite a previous blocklist.**
    A sampled run replaces rather than merges exclusions.

17. **Resolution-probe decode failures remain in evaluation.**
    [res_probe.py:159](/C:/Users/longd/Programming/geolocation_model/scripts/res_probe.py:159) leaves failed images as zero embeddings while retaining their IDs, coordinates and sequence metadata.

18. **Resolution probing can buffer too many decoded images.**
    Unbounded executor mapping can retain many completed image variants simultaneously and cause avoidable memory spikes.

## Serving and multi-photograph retrieval

19. **Serving does not validate checkpoint split/release provenance.**
    [serve.py:40](/C:/Users/longd/Programming/geolocation_model/scripts/serve.py:40) reads positional artifacts without calling the repository’s split check.

20. **Serving selects preprocessing from filename substrings.**
    Any filename containing `pca768` receives the bank55 basis, whether or not it was produced with that basis.

21. **Serving assumes every checkpoint contains `knn_file`.**
    Non-retrieval checkpoints fail with an incidental `KeyError` instead of being supported or rejected explicitly.

22. **`topk` can exceed the available bank size.**

23. **“Equal-photo” reciprocal-rank fusion is not actually equal-photo.**
    Candidate slots are allocated equally, but raw similarities from different photographs are pooled into one softmax. Scores are not calibrated across images, so one view can still dominate.

24. **Learned multi-photo modes score secondary candidates with the primary photo.**
    For `pos` and `dual`, the primary street embedding produces the learned scores for candidates retrieved by every secondary photograph.

25. **The default multiquery configuration is dimensionally inconsistent.**
    The default tag expects 1536 dimensions, while cached queries are always projected with the bank55 PCA to 768 dimensions.

26. **Multiquery query caches lack provenance.**
    They validate IDs/count only—not image root, preprocessing, encoder, PCA basis, scale or release.

## Core training and model behavior

27. `[.]` **Optimizer resume is not implemented.**
    [train.py:435](/C:/Users/longd/Programming/geolocation_model/src/train.py:435) can save optimizer state, but `--init` restores only model weights. Scheduler and RNG state are also absent. A 2+2+2 ladder is three Adam/cosine restarts, not a continuous six-epoch run.

   > **Known, deferred.** Already recorded and deliberately deferred: fixing it changes training and would break comparability with every arm on file. Now also pinned in `tests/test_checkpoint_contract.py` as a known orphan, so it cannot be quietly forgotten.


28. `[.]` **Conditional quality gating sees a neighbor hidden by dropout.**
    [retrieval.py:184](/C:/Users/longd/Programming/geolocation_model/src/retrieval.py:184) masks retrieval neighbors, but the quality gate receives the original top similarity.

   > **Known, deferred.** Already recorded and deliberately deferred, for the same reason as 27: it would alter training semantics and break comparability with the measured p-curve.


29. **Weight-decay parameter grouping uses unsafe substring matching.**
    For example, `q_pos.weight` receives no decay because its name contains `pos`; memory embeddings and some bias-like tensors receive decay incorrectly.

30. `[.]` **Step-3 sink keys receive no positive sink supervision.**
    Sink-positive sampling in [dataset.py:283](/C:/Users/longd/Programming/geolocation_model/src/dataset.py:283) is limited to steps 1 and 2.

   > **Known, deferred.** Already recorded in `docs/STATE.md`; it is why the sink capacity experiment was retired as contaminated rather than reported as a null.


31. **The sink “interpolation” coefficient is unconstrained.**
    [model.py:212](/C:/Users/longd/Programming/geolocation_model/src/model.py:212) allows negative values and values above one, so the operation can extrapolate or invert rather than interpolate.

32. **Memory dropout lacks inverted-dropout scaling.**
    Its magnitude distribution changes between training and evaluation.

33. **`--overfit` is broken with the default hit-based selector.**
    Validation is disabled, selection metrics stay `NaN`, and no best checkpoint is written. An overfit size larger than the limited training set also creates invalid subset indices.

34. **Negative `--soft` temperatures are silently treated as hard CE.**

35. **Map subdivision width is inferred by rounded square root without validating `width == 12 × sub²`.**
    Invalid dimensions can be accepted and misinterpreted.

36. **Checkpoints omit many result-defining hyperparameters.**
    Memory dropout, sink weight, embedding noise/dropout, smoothing, learning rate, weight decay, warmup and batch size are among the missing fields.

37. **Evaluation ignores checkpoint grid geometry.**
    [evaluate.py:21](/C:/Users/longd/Programming/geolocation_model/src/evaluate.py:21) reconstructs models using current `G` and `STEPS`, not the saved checkpoint values.

38. **Explicit street-file overrides are not checked for provenance or dimensions.**

39. `[~]` **The RoPE implementation does not provide the claimed relative-position behavior.**
    [encoders.py:159](/C:/Users/longd/Programming/geolocation_model/src/encoders.py:159) rotates token embeddings before LayerNorm and arbitrary Q/K projections. Those operations do not generally commute with RoPE rotations.

   > **Confirmed.** Independently confirmed earlier the same day by measurement, before this review was read. `scripts/rope_probe.py`: with identical content at all 256 positions, the attention logit's within-offset spread is 0.845 of its overall spread as built, against 0.000 for the textbook arrangement. Relative position is absent. Live for 108 of 122 checkpoints including the shipping arm.

## Positional artifact correctness

40. **Dataset artifacts are trusted purely by row position.**
    [dataset.py](/C:/Users/longd/Programming/geolocation_model/src/dataset.py:141) does not prove that street embeddings, target files, image-ID sidecars and map indices describe the same ordered rows.

41. **k-NN indices are not fully validated.**
    Negative or out-of-range indices can survive; negative NumPy indices read from the end and silently reference the wrong sample.

42. **Beam token loading ignores map completion masks.**

43. **Dataset and target files are published separately.**
    An interruption can leave two valid-looking files from different generations.

44. **Generic street embedding can overwrite the canonical cache.**
    A custom parquet/model invocation without `--out` writes to the normal release embedding path.

45. **Street embedding has no completion mask.**
    An interrupted full-shaped memmap appears complete. Its final sanity check examines only the first 512 rows.

46. **Stacking, concatenation, pooling and projection utilities do not verify row IDs.**
    Same-shaped artifacts from different orders can be combined without error.

47. **Reused PCA bases are validated by shape only.**

## Cache construction

48. **Pyramid `--n` extension actually destroys and recomputes the cache.**

49. **Pyramid progress is recorded only after an entire encoder pass.**
    A mid-pass interruption loses all resumable progress from that pass.

50. **Pyramid decode failures are swallowed and the command can exit successfully with incomplete output.**

51. **Completion bits can be persisted before embedding data is flushed.**
    A crash can leave rows marked complete whose token data was not durable.

52. **Existing pyramid completion arrays are not validated for shape and legal values.**

53. **Tile-cache reuse ignores grid geometry.**
    Configurations such as 3×2 and 2×3 with the same tile count can reuse incompatible data.

54. **Tile-cache completion can also be saved before the memmap is flushed.**

55. **Tile-cache consumers are inconsistent.**
    Pooling and matching consult completion data; OSV token loading in the fusion script does not.

56. **Tile-fetch recreation does not immediately reset the previous completion mask.**
    A crash between truncating the cache and rewriting the mask can bless new zero rows on restart.

57. **The tile “GeM” implementation is missing the p-th root.**
    It computes a signed p-th moment, not generalized-mean pooling.

58. **Tile matching chooses `min(K, bank_size - 1)` for disjoint query and bank sets.**
    It unnecessarily drops one valid neighbor and breaks on a one-row bank.

59. **Tile-match oracle statistics are computed before same-sequence exclusion.**
    The reported oracle/headroom can include forbidden near-duplicate matches—the evidence used to motivate fusion is therefore inflated.

## k-NN and bank extensions

60. **Held-region filtering is wrong for cell12/cell16 extensions.**
    [build_knn.py:130](/C:/Users/longd/Programming/geolocation_model/scripts/build_knn.py:130) reads held cells from the dataset’s z8 field and compares them with extension z12/z16 cells.

61. **Sequences are forcibly treated as disjoint across release and extension banks.**
    If one real drive crosses the corpus boundary, same-sequence exclusion fails.

62. **Extension release metadata is ignored.**

63. **Embedding validation permits extra rows and does not establish row identity/order.**

64. **When fewer than K legal neighbors exist, excluded or placeholder rows can remain in top-k results.**

65. **Extension duplicate image IDs are not rejected.**

66. **Merged bank metadata checks total length but not whether part order matches embedding-stack order.**

67. **KartaView quotas are consumed before download success is known.**
    Failed or undersized images suppress later valid candidates.

68. **Mapillary `--save-dir` does not save images.**
    The directory is created, but the implementation writes only metadata/URLs.

69. **Mapillary harvesting can exceed `--max-images`.**

## Orchestration and reporting

70. **Overnight completion markers are keyed only by stage name.**
    Changed arguments, code, release, inputs or deleted/corrupt outputs do not invalidate them.

71. **Retries append to the same log.**
    Calibration and reporting can read stale attempts; the calibration path takes the first regex match. The pyr47 log itself contains multiple appended attempts and duplicate successes.

72. **Report parsing can mix epochs from separate attempts.**

73. **The report epoch regex is not multiline-aware.**
    Logs beginning with an attempt header may produce an empty epoch list and `NaN` seconds-per-epoch.

74. **A later successful retry does not clear the stage’s stale failure record.**

75. **Dependency readiness checks only file existence.**
    A metadata file from an incomplete cache can launch downstream work.

76. **Retry deadlines are checked before, rather than during, the retry loop.**

77. **Marathon looks for an obsolete unstamped bootstrap filename.**
    It can fall back to validation data or stale legacy errors instead of the current test bootstrap.

78. `[x]` **Bootstrap cache hits bypass model and split validation.**
    The key does not include dataset/split identity.

   > **Fixed.** Confirmed and fixed. `check_split` ran only on a cache miss, so a cached arm was returned with no proof of its release or split. Provenance now runs before the cache and reads only the checkpoint dict, so a hit does not pay for a model build.


79. `[x]` **Bootstrap error caches contain no image IDs.**
    Same-length results with different image ordering can be treated as paired.

   > **Fixed.** Confirmed. Addressed by pinning the row set rather than storing ids: `main()` now requires every arm to share release, split mode and split hash, which with the seeded sample and equal `n` fixes the rows exactly. Verified both ways -- two `sequence` arms still pair from cache, and a `sequence` arm against a `cell8` arm now refuses instead of returning a tight interval over unrelated images.


80. **The summary table reconstructs training limits from obsolete tag syntax.**
    Current tags can report evaluation over all training rows, including unseen rows.

81. **The report glob omits current `d768`/`d1536`-style runs.**

82. **Digest cannot parse current stamped bootstrap filenames correctly.**

83. `[-]` **Several runners now fail standalone because they import `config` before setting their declared release.**
    Conversely, another group silently hardcodes `s10`; together these undermine the explicit-release invariant.

   > **Partly refuted.** Half confirmed, half refuted. The 16 scripts that hardcode a release are real, and they do undercut the explicit-release invariant. But the scripts that "fail standalone" do not fail for that reason: `after`, `b70`, `half`, `keys`, `more`, `offline`, `tiles2`, `tonight_0902` and `w768b70` take positional arguments and simply reject `--help`. All 72 scripts import cleanly.


84. **Width-probe PCA is fitted on train and held-out queries together.**

85. **Map-structure probing splits patches rather than source tiles.**
    Patches from one map tile can occur in both train and test, overstating recoverability. Its orientation label also names the gradient direction as the road direction, although those are perpendicular.

86. **Threshold semantics differ between evaluators.**
    Core baselines use `<=`; external/bootstrap evaluation uses `<`.

The bottom-N external selection issue remains a known growth-stability limitation, not a current fixed-manifest correctness problem, so I would not prioritize it.

This was a static review. All 95 Python files parse successfully, but I could not run the test suite because the available shell has no pytest-capable Python installation.