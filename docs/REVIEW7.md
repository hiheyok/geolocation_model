# Code review, round seven: structure, modularity, and code quality

Review of commit `aa75aea` and the tracked working tree on 2026-09-05. This
round deliberately reviews maintainability rather than searching for another
set of runtime defects. Existing bug findings in REVIEW5 and REVIEW6 are not
repeated unless they expose a structural cause.

No source files were changed. The pre-existing untracked files
`runs/BOOTSTRAP_tiledrisk.md` and `scripts/audit_knn.py` were left untouched and
are outside this review.

The full suite passes with the requested launcher:

```text
OSV_RELEASE=s10 py -3.13 -m pytest -q
507 passed in 9.00s
```

## Executive assessment

The numerical core is more disciplined than the repository layout suggests.
`tile_math`, `splits`, `knnmeta`, `provenance`, `maskio`, and `safeio` each own a
recognisable concern, and the regression suite is unusually attentive to
silent data mismatches. Those are strong foundations.

The main maintainability risk is at the boundaries around that core. The
repository is not an importable package; configuration is process-global and
performs I/O at import time; `GeoStepDataset` and `train.main` assemble many
subsystems at once; executable scripts double as shared libraries; and tensor
contracts are carried in loosely structured dictionaries and tuples. This
makes a change easy to implement in one execution path and easy to omit from
another. Several comments already describe exactly that history.

The highest-leverage work is therefore not a broad rewrite. It is to make the
existing boundaries real: package the library, pass an explicit release
context, give artifacts and batches typed representations, and reduce scripts
to thin entry points over reusable modules.

## High-priority findings

### 1. The repository has no stable Python package boundary

There is no `pyproject.toml`, `setup.py`, or `setup.cfg`. Instead, library
modules modify `sys.path` to import their siblings, for example
[`train.py:19`](../src/train.py#L19), [`dataset.py:19`](../src/dataset.py#L19),
and [`evaluate.py:11`](../src/evaluate.py#L11). Scripts add both `src` and
`scripts`, and tests repeat the same setup. Across tracked Python files there
are 188 `sys.path.insert`/`append` calls.

This is already complex enough to require a bespoke static regression test.
[`test_import_order.py:1-17`](../tests/test_import_order.py#L1) explains that an
import placed before the path mutation broke the unattended runner, while the
normal test suite and compilation checks did not notice. The test preserves a
fragile convention rather than removing the convention.

Consequences:

- Import meaning depends on the entry point and current path order.
- A local file can shadow an installed module without an obvious failure.
- IDEs, type checkers, test tools, and external callers cannot consume one
  canonical package.
- Moving a function between `src` and `scripts` forces unrelated path edits.

Recommendation: create an installable package, for example
`src/geolocation/`, use relative imports inside it, and expose console entry
points for training and evaluation. During migration, a minimal `pyproject.toml`
plus an editable install is enough; files do not need to be reorganised all at
once. Delete `test_import_order.py` only after imports work without path
mutation, replacing it with a clean-process package-import smoke test.

### 2. `config.py` is global state, validation, path construction, and I/O in one import

[`config.py:8-50`](../src/config.py#L8) resolves environment variables into
module globals. Importing the module exits the process if `OSV_RELEASE` is
missing or unknown at lines 34-44, and importing it creates four directories at
[`config.py:126-127`](../src/config.py#L126). Many scripts must therefore set
the environment before importing anything else; `overnight.py` even changes
`OSV_RELEASE` after modules may already have captured their configuration at
[`overnight.py:465-474`](../scripts/overnight.py#L465).

This design makes the code non-reentrant: one process cannot safely work with
two releases, tests cannot construct an isolated configuration without module
reloads, and a read-only import mutates the filesystem. It also contributes to
the import-order problem above.

Recommendation: introduce an immutable `ReleaseConfig`/`Paths` object built by
the CLI boundary and passed to loaders, datasets, and runners. Parsing and
validating environment variables belongs in a `from_env()` constructor;
directory creation belongs in an explicit `ensure_directories()` call made by
commands that write. Keep domain constants such as split fractions in their
own modules rather than on the mutable deployment object.

### 3. `GeoStepDataset` is both a dataset and the repository's artifact composition root

The constructor at [`dataset.py:270-483`](../src/dataset.py#L270) does all of
the following:

- reads the release and target Parquets;
- computes and validates the split;
- opens street and map caches;
- validates provenance and conditioning layout;
- constructs the tile lookup;
- checks completion masks;
- computes coordinate targets;
- opens and validates k-NN data;
- extends bank address tables;
- stores training-ready arrays.

The class then owns multiprocessing memmap reopening, negative sampling, and
batch-schema construction through line 583. Its constructor has configuration
for splitting, caches, retrieval, and negative sampling, so merely creating a
dataset performs a large, failure-prone integration operation.

The validation is valuable; the problem is ownership. It cannot be reused
without constructing a training dataset, and a unit test for one concern must
fabricate inputs for several others. Other consumers consequently reload and
revalidate overlapping artifacts themselves.

Recommendation: split this into small, validated stores such as
`ReleaseTable`, `StreetStore`, `MapTokenStore`, `TargetStore`, and `KnnStore`.
Each loader should return a validated object with its row-space identity.
`GeoStepDataset` should receive those objects and focus on selecting rows and
producing samples. A separate factory can retain today's convenient
"construct everything from paths" behaviour for CLI callers.

### 4. Training orchestration is concentrated in one large procedural entry point

[`train.main`](../src/train.py#L460) spans 279 lines. It parses CLI state,
applies restart policy, resolves artifacts, seeds randomness, creates datasets
and loaders, constructs the model, handles checkpoint compatibility, builds
the optimizer and schedule, compiles the model, runs epochs, chooses the best
checkpoint, and serialises run metadata. `run_epoch` at
[`train.py:130-221`](../src/train.py#L130) also combines device transfer,
augmentation, forward logic, two loss families, optimisation, and metric
aggregation.

This makes configuration changes and training-engine changes inseparable. It
also encourages external tools to reproduce fragments of training logic
rather than call a stable API.

Recommendation: parse arguments into a `TrainConfig`, resolve them into a
`TrainPlan`, and give an `ExperimentRunner` explicit collaborators for data,
model construction, optimisation, checkpointing, and metrics. Extract the
off-path loss into a named loss component and metric accumulation into a small
state object. `main()` should ideally be only: parse, build context, run, map a
domain error to a user-facing exit.

### 5. Executable scripts are an informal shared-library layer

`scripts/` contains 95 tracked Python files. Many import implementation from
other executables: `fuse_head.py` imports metrics from `tile_pool.py` and
`tile_match.py`; `gain_density.py` imports `great_circle` from
`query_only.py`; `osv_pyramid.py` imports `load_tokens` from `fuse_head.py`;
and more than twenty experiment launchers import mutable runner machinery from
`overnight.py`. Examples are
[`fuse_head.py:67-68`](../scripts/fuse_head.py#L67),
[`gain_density.py:46-47`](../scripts/gain_density.py#L46), and
[`after.py:63-64`](../scripts/after.py#L63).

This reverses the usual dependency direction: command-line programs have
become libraries, but their public surface is neither declared nor protected.
It also produces duplication. Great-circle distance has implementations in
`src/baselines.py` and at least five scripts; `paired` appears in multiple
scripts; and several runners define their own `train_argv` variants.

The experiment launchers (`after.py`, `b70.py`, `half.py`, `more.py`,
`seqfix2.py`, `tonight_0902.py`, and others) encode plans in imperative,
historically named files. Understanding which one is current requires project
history rather than a discoverable registry.

Recommendation:

- Move reusable metrics, embedding transforms, candidate ranking, and runner
  primitives under the package.
- Leave each script with argument parsing and one call into the package.
- Express experiment matrices as declarative records consumed by one runner.
- Give experiment plans descriptive IDs and archive completed plans under
  `runs/` or `docs/` rather than keeping every historical plan in the command
  namespace.

### 6. Core model interfaces use positional tuples, string-keyed dictionaries, and a private method

[`GeoAgent.forward`](../src/model.py#L296) expects an undocumented combination
of required and optional dictionary keys. It constructs a neighbour tuple at
lines 306-309. [`retr_prior`](../src/model.py#L234) then assigns meaning by
position (`nbrs[0]` through `nbrs[3]`) and switches shape semantics with
`len(nbrs)`. `run_epoch` independently reconstructs the same tuple at
[`train.py:182-189`](../src/train.py#L182).

The inference layer also calls the model's private `_add_geo` method directly
at [`beam.py:168`](../src/beam.py#L168). Its surrounding comment says a prior
implementation diverged from training and silently omitted learned terms. The
current centralisation fixed that behaviour, but the underscore API and manual
pipeline (`fuse`, `retr_prior`, `_add_geo`, `policy_logits`) still let callers
skip a stage.

Recommendation: define `GeoBatch` and `NeighborBatch` dataclasses or typed
named tuples with shape documentation. Give the model one public method for
policy evaluation from arbitrary tiles, used by both training and beam search,
and keep composition of retrieval and geographic priors inside the model. The
compiler-facing tensor core can remain positional internally if necessary;
conversion should happen once at its boundary.

### 7. Artifact contracts are improving, but access is still decentralised

`provenance.py`, `knnmeta.py`, `maskio.py`, and `safeio.py` are good moves
toward single ownership. However, tracked `src` and `scripts` still contain 127
direct `np.load` calls and 15 direct `torch.load` calls. Consumers decide for
themselves whether to request mmap, allow pickle, inspect dimensions, verify a
sidecar, check a completion mask, or bind an artifact to a release.

The result is a protocol spread across filenames, NumPy keys, JSON sidecars,
checkpoint dictionaries, and comments. A new consumer can load valid bytes
without loading the evidence that gives those bytes meaning. REVIEW5 and
REVIEW6 contain several examples of that exact class of failure.

Recommendation: make format-specific repositories the normal access path:
`CheckpointStore.load`, `StreetArtifact.open`, `KnnArtifact.open`, and
`MapArtifact.open`. Return typed metadata with an explicit schema version and
row-space identifier. Low-level raw loading can remain in those modules, but a
code search outside them should find very few `np.load`/`torch.load` calls.

## Medium-priority findings

### 8. Error policy is mixed into reusable code

Library-level validation frequently raises `SystemExit`, including artifact
checks throughout [`dataset.py:313-476`](../src/dataset.py#L313) and model
construction in [`evaluate.py:101-148`](../src/evaluate.py#L101). Tests must
therefore assert process-exit exceptions for what are really domain validation
errors. Conversely, runner code has broad `except Exception` blocks, such as
[`overnight.py:79-85`](../scripts/overnight.py#L79) and
[`overnight.py:336-342`](../scripts/overnight.py#L336), which can erase useful
failure distinctions.

Recommendation: raise typed exceptions such as `ArtifactMismatch`,
`IncompleteArtifact`, and `CheckpointCompatibilityError` below the CLI layer.
CLI entry points can catch the common base exception and render the same concise
messages with a non-zero exit code. Catch only expected exception types when a
fallback is genuinely safe; log unexpected exceptions with context.

### 9. Some tests lock source spelling instead of behaviour

The suite has excellent behavioural coverage in newer subsystem tests, but
`test_review3.py` and `test_review4.py` are organised by review round and mix
unrelated domains. They also contain many assertions against source strings,
for example [`test_review3.py:393-408`](../tests/test_review3.py#L393) and
[`test_review4.py:66-77`](../tests/test_review4.py#L66). A semantics-preserving
refactor can fail these tests, while a differently broken implementation can
pass if it retains the expected text.

Static architecture tests can be appropriate, but implementation-substring
tests should be rare and explicit. This suite currently makes modularisation
harder precisely where modularisation is most needed.

Recommendation: move each regression into the subsystem it protects
(`test_provenance`, `test_training_config`, `test_artifact_publication`, and so
on) and assert through public behaviour. Add reusable factories for tiny
release tables, artifact manifests, checkpoints, and batches so such tests stay
cheap. Retain AST/static checks only for rules that cannot reasonably be
observed at runtime.

### 10. Historical rationale often overwhelms the local contract

The code records valuable operational history, but some docstrings narrate
individual incidents and experiment results at much greater length than the
function's current contract. Examples include `param_groups` beginning at
[`train.py:65`](../src/train.py#L65), the release discussion at
[`config.py:13-32`](../src/config.py#L13), and many validation branches in the
dataset constructor. This makes already-large modules harder to scan and
increases the chance that comments become stale after the next change.

Recommendation: keep invariants, shapes, units, and non-obvious reasons beside
the code. Move incident histories, benchmark deltas, and migration narratives
to architecture decision records or run reports, linked from a short comment.
A reader of a function should be able to identify its inputs, outputs, and
invariants before reading its history.

### 11. Automated quality controls cover regressions but not consistency

The repository has no checked-in formatter, linter, type-checker, pre-commit
configuration, or CI workflow. The strong test suite catches known semantics,
but inconsistent interfaces, unused imports, accidental broad exceptions, and
new dependency-direction violations rely on review.

Recommendation: add a deliberately small baseline rather than attempting a
repository-wide style rewrite: Ruff for import and correctness rules, a
formatter with one pinned version, and type checking initially limited to new
typed boundary modules. Run these plus the existing suite in CI. Ratchet rules
forward as touched modules are cleaned up.

## Suggested target shape

One possible end state, intended as a dependency map rather than a required
directory naming scheme:

```text
geolocation/
  config.py              immutable runtime/release configuration
  errors.py              typed domain errors
  domain/
    tiles.py             addressing and coordinate math
    splits.py            split construction and identity
    metrics.py           geographic and paired metrics
  artifacts/
    schema.py            versioned identities and metadata types
    street.py            street/conditioning cache loader
    maps.py              token/index/completion loader
    knn.py               neighbour cache loader
    checkpoints.py       model/run checkpoint loader
  data/
    batches.py           GeoBatch and NeighborBatch
    dataset.py           sample assembly from validated stores
  modeling/
    agent.py
    encoders.py
    retrieval.py
    beam.py
  training/
    config.py
    losses.py
    metrics.py
    runner.py
  experiments/
    stage.py             reusable runner engine
    plans.py             declarative experiment definitions
  cli/
    train.py
    evaluate.py
    serve.py
```

Dependencies should point downward: CLIs and experiment plans may depend on
library modules; library modules should never import a CLI or mutate import
paths.

## Incremental refactoring order

1. Add package metadata and canonical imports without changing behaviour.
2. Introduce explicit `ReleaseConfig` while leaving a compatibility adapter for
   current environment variables.
3. Add typed artifact and batch objects at existing function boundaries.
4. Extract loaders from `GeoStepDataset`, preserving its public factory.
5. Extract `TrainConfig`, loss calculation, metrics, and checkpointing from
   `train.main` one seam at a time.
6. Move shared script utilities into the package, then replace imperative
   experiment scripts with declarative plans.
7. Relocate review-round tests by subsystem and replace source-text assertions
   as the affected code is touched.

Each step can be required to keep the existing 507 tests green. This ordering
first removes import and configuration hazards, then adds types at the seams,
and only afterward decomposes the largest modules; that keeps the migration
reviewable and avoids a high-risk rewrite.

## Review boundary

This was a structural and static quality review plus the existing test suite.
It did not run training, evaluation, serving, artifact-generation pipelines,
network harvesters, or GPU workloads. Passing tests demonstrate that the
current covered behaviour is stable; they do not resolve the architectural
coupling described above.
