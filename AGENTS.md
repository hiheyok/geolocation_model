# Working in this repository

For any agent — human or otherwise — making changes here. These are not style
preferences. Every rule below is written down because breaking it cost this
project measurable time or a wrong result, and the incident is named so you can
judge whether the rule still applies to your case.

---

## 1. One change, one branch, one PR

Branch from `main`. One reviewable idea per PR. Do not stack unrelated work on
a branch because it is already checked out.

PR #26 accumulated nine distinct changes — a k-NN validation chain, a project
brief, a conditioning harness, a coverage diagnostic, and the removal of a
state doc — and became impossible to analyse or partially revert.

Legitimate exceptions, both narrow:

* **The same lines.** Two findings in the same ten lines of the same two files
  ship together; splitting them creates a stacked dependency and conflict churn
  without helping review. Say so in the PR body.
* **A genuine chain.** If B cannot exist without A, stack B on A and say which
  PR it depends on.

Results are their own PR. A measurement is a thing to weigh; a code change is a
thing to review. Do not bundle them.

### What must not be in git

`docs/STATE.md` and `docs/NEXT.md` are session state, rewritten wholesale each
time, and they conflicted on every parallel branch. They are ignored. Findings
*about the code* — `docs/REVIEW*.md`, `docs/NAMING.md`, design notes, anything
in `runs/` — stay tracked.

---

## 2. Measure before you diagnose

Three bottleneck diagnoses were made in one evening without measuring. All
three were wrong:

| claimed | actual |
|---|---|
| "I/O-bound" (from a sawtooth shape) | single-core string comparison, 80x available |
| "model loading" (a 10 s stall) | the same string comparison |
| "not I/O" (after the first correction) | HDD seeks at 178 reads/s |

Each was settled by one command. Run it.

* **Sample a GPU faster than you think you need to.** 2 s polling showed a
  steady 99–100% while the card sawtoothed; at 100 ms the shape was a producer/
  consumer oscillation. `utilization` means "a kernel is resident", not "busy".
* **Distinguish starved from saturated.** Zero throttling + maxed clocks + high
  memory-controller duty + no sawtooth = bandwidth-bound, and more workers do
  nothing. A sawtooth = starved, and the producer is the thing to fix.
* **Check which device.** `E:` is a 5900 RPM HDD holding the image zips; `C:`
  is an SSD holding the caches. Random member reads run at 178/s; a sequential
  pass runs at 100–300 MB/s.
* **Know the ceiling before optimising.** Duty cycle 45% caps perfect overlap
  at 2.2x. Sixteen idle cores does not mean 16x.

---

## 3. Prefer the access pattern to the parallelism

An algorithmic or access-pattern fix usually dominates threading, and threading
a bad implementation **hides** the real defect rather than fixing it.

* Comparing `U40` strings instead of factorised ints: **80x**. Thread-pooling
  it would have given 16x and buried the finding.
* Reading a zip sequentially instead of by member name: **~46x**. Queue depth
  gives ~2x.
* `resmatch` already had a thread pool and still stalled, because the
  bottleneck was the access pattern.

---

## 4. Share the loop, not just the leaf

The recurring defect in this codebase is a shared leaf function with the loop
around it copied per caller.

| shared | duplicated | cost |
|---|---|---|
| `fuse_flat` | the scoring composition, 3 callers | beam dropped `nbr_emb`; 81.6 → 55.8 km, silent |
| `knnmeta.check` | the call-site arguments, 2 evaluators | split/hash/street checks never ran |
| `embed_street.preprocess` | the loading loop, ~10 scripts | GPU starved; the pool left behind |
| `sp.read` | the `[:n]` cohort slice, 4 probes | a different continent, 7 pp |

When you fix one of these, put the **whole loop** behind one entry point, and
add a test that fails when a new module writes its own. `src/shards.py` and
`tests/test_one_reader.py` are the worked example: an exported helper is a
suggestion, and suggestions get left behind.

---

## 5. Assert through the thing, not near it

A test that exercises something adjacent to the production path will pass while
the production path is broken. This happened three times in one session:
`load_state_dict` instead of the `--init` filter; "some parameter is
retrieval-width" instead of calling the prior; a union over call sites that the
*val* dataset satisfied alone.

* Call what `main` calls.
* **Prove the test fails against the pre-fix code.** Neuter the guard, watch
  the suite go red, restore it. If it stays green the test is decorative.
* Watch for tests that are vacuous by construction. Every gate on the retrieval
  prior is zero-initialised, so a freshly built model scores identically with
  and without its neighbours — parity tests on one prove nothing until the
  gates are given values.
* A call-presence assertion is not a behaviour assertion. Test *which*
  comparisons actually run.

---

## 6. Experiments

* **Paired bootstrap, or it is not a comparison.** The median error CI is
  ~16 km and two seeds differ by 18 km. Most arm differences here are not
  separable unpaired.
* **Random cohorts.** `flatnonzero(labels == "test")[:n]` is not a sample —
  `dataset.parquet` is shard-ordered, so the head is a different continent
  (BR/AR/ZA vs US/DE/RU) and 35% denser. It was worth 7 pp and shaped a
  strategic conclusion for weeks. Any `[:n]` over an ordered artifact is a
  stratified sample by whatever the order encodes.
* **Controls are not optional.** A conditioning arm at +2 epochs must be
  compared against a +2-epoch arm without it, not against the base. The base
  drifted downward and the null would have read as neutral.
* **Select on hit rate, not the median**, which swings 300 km between epochs.
  The 2,000-image selection set picks checkpoints; it does not measure them.
* **Never generalise a null from arms near-identical by construction.**
* **Never compare two numbers from different embedding spaces**, or from
  different cohorts. `57.5 - 49.7` across two cohorts is not a treatment effect.
* State absolute numbers with their bank size. Corpus has moved this metric
  more than any architectural change.

---

## 7. Artifacts must be bound by content, not by name

Six separate defects share one shape: **a key that names an identifier rather
than everything the value depends on** — a tag, a filename, a length, a prefix,
a substring, a size that gets sliced off.

* A k-NN cache records the content digest of the street file it searched.
* A joined cache records the digest of its retrieval prefix.
* A map cache is bound to the renderer that filled it.
* A missing named dependency is an **error**, never a fallback to something
  broader. Both external evaluators silently searched the whole corpus when a
  named cache was absent, and logged it as "checkpoint records none".
* Absence of a stamp is a warning; do not backfill one. Stamping an existing
  artifact with today's digest asserts the one thing that cannot be checked.

---

## 8. Long-running work

* **Never stack GPU jobs** — 2.6x measured cost.
* **Never edit code a running chain has not yet imported.** A `train.py` edit
  mid-ladder voided a whole experiment. If a later stage imports the file, it
  will run different code than the earlier stages.
  * The exception, and it must be *demonstrated* not asserted: a provably
    equivalent change, with a test showing bit-identical output. And check
    whether it is worth it first — the expensive stage may already have run.
* **A completion signal that only fires on the happy path is not one.** A
  nonexistent `Sampler.stop()` raised after every stage succeeded, so the
  runner died instead of logging the line the monitor watched for, and the GPU
  sat idle for six hours. Verify the harness, not just the payload:
  `--help`-check every flag *and* the runner's own API.
* Estimate from prior runs of the same stage, and say what the estimate is
  based on.

---

## 9. Environment

* **Use `py`.** `python` and `python3` are MSYS2's 3.12 with none of the
  project stack. `py` is 3.13 with numpy 2.3.5, torch 2.6.0+cu126 and CUDA.
  `py -m pytest tests/ -q` runs the suite.
* `OSV_RELEASE` is required and has no default. It selects the dataset release
  everything downstream is indexed against.
* Tile servers: use `tiles.connect(config.TILE_SERVERS)`, never a bare
  `TileClient`, so the fallback happens and announces itself. The two
  deployments are **different renderers**, and a map cache is bound to one.
* RTX 3070, 8 GB. A training rung is ~26 min; a k-NN rebuild 5–7 min; a full
  bank embedding pass ~19 h.
* **Never type a backslash inside a Bash heredoc** — it arrives as a real
  newline and produces an unterminated string literal. Use the Write/Edit
  tools, or `chr(10)`.

---

## 10. Reporting

* Report outcomes faithfully. If tests fail, say so with the output. If a step
  was skipped, say that.
* **Get the test count right.** It has been wrong in commit messages five
  times. Run the suite, read the number, paste the number.
* Retract clearly and in place. When a published claim turns out to be wrong,
  correct the document that carries it — do not leave the disproved number in
  a report template above a table that contradicts it.
* Claims must not outrun the measurement that produced them. Every correction
  on record comes from that one shape: a claim made from a smaller or cheaper
  measurement than the decision it was feeding.
