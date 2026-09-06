# Approved regression baselines

Validation is an additive engineering-review layer. A take is an immutable acquisition and a run is an immutable processing interpretation. An approved reference is a human decision about a run output; its baseline is a copied snapshot under `data/validation/baselines`, so run cleanup cannot invalidate it. It is not ground truth unless externally measured truth is explicitly recorded elsewhere.

Approving again creates a new version and leaves prior versions immutable. Activation and deactivation only alter the selected baseline record; they never remove history. A baseline records its source run, contract and recipe fingerprints, parameters, calibration context, reviewer and notes.

Suites live in `data/validation/suites`; executions in `data/validation/executions`, with incremental progress. A case connects one take to one or more baseline expectations. The current initial runner evaluates existing candidate runs (`latest` or an explicit run id), preserving all run semantics. Pipeline dispatch/rerun remains owned by the existing processing workflow, which keeps validation free from implicit recipe or run mutation.

Statuses have distinct meanings: `pass` is within tolerance; `changed` differs without a failure policy; `regression` violates an expectation; `not_comparable` lacks a compatible semantic contract; `blocked` is invalidated by upstream failure; and `needs_review` is evidence requiring an engineer. Studio owns baseline controls next to artifact review; suite governance and execution belong in the Validation workspace.

Initial 2.5D focus is reference support masks, belt plane/residual summaries, authoritative normalized height rasters, final object masks, measurements, classification fields, and quality flags. Overlay rendering is retained as review evidence.

## Choosing a comparator

An artifact resolves to a comparator through progressively weaker evidence, and the decision is recorded on the baseline as `comparator_source` so a reviewer can tell a declared decision from a guessed one:

1. `metadata.comparator` — a stage stating the comparator outright. Nothing overrides it.
2. `metadata.semantic_type`, mapped through the reviewed `COMPARATOR_BY_SEMANTIC_TYPE` table.
3. `metadata.role`, where it settles comparability alone (`difference_overlay`, `added_pixels`, `removed_pixels` are `visual_only`).
4. The artifact `kind` and the on-disk representation: `overlay` is `visual_only`, a `.npy` path is a `numeric_raster`.
5. Identifier heuristics, kept so nothing approvable stops being approvable, but reported as `heuristic`.

Anything unresolved is `not_comparable` and is skipped at approval time with that source recorded; it never passes. Extend the table rather than the heuristics — an entry there is a reviewed decision. Some declared types are deliberately absent: component id maps are label rasters, and comparing them as binary masks would silently under-report.

Note that `kind` is a closed literal (`image`, `point_cloud`, `table`, `json`, `metric`, `overlay`, `video`, `text`, `file`). It has no `mask`, `raster` or `npy` member, so rules written against those values can never fire.

Masks use IoU/Dice/add/remove fractions and connected-component counts; numeric rasters use error statistics over the mutually finite pixels; JSON-backed plane artifacts are structural comparisons after unstable run fields are excluded.

## Case governance

Cases are managed in `/validation`. The table shows take, enabled state, resolution, tags and notes, with add, edit, enable, disable, remove and history actions per row. Archived suites keep every case visible and disable every action; the service rejects `add_case`, `update_case` and `delete_case` on them.

Removing a case removes suite membership only. Its baselines, source runs and historical executions are all preserved, and the confirmation says so rather than implying deletion.

An edit sends only the fields the user changed. The server preserves omitted fields, so sending a whole record would overwrite anything another session edited meanwhile.

## Baseline resolution

A case resolves its expectations in one of three modes. Legacy cases carry no `baseline_resolution` and behave as `active`.

- **`active`** follows whichever version is currently active, not the version the case was configured against. When no active baseline covers the case the workspace says so explicitly and offers baseline history; it never silently uses a historical version.
- **`pinned`** always compares against one chosen version. A pinned baseline that is later deactivated stays pinned — the case never falls back to the active version — and the workspace labels it `Pinned · inactive vN`. A pin whose record is missing or fails integrity keeps the configured id and reports the specific problem.
- **`allowed_versions`** passes when the candidate matches any one of the versions in `allowed_baseline_ids`. The version that matched is persisted on the execution as `matched_baseline_id` and is read from there, never recomputed against today's history. Newly promoted versions are not added automatically.

Versions are chosen from approved history in every mode; no baseline id is ever typed. A selection that is present but empty is rejected rather than substituted — an absent key still defaults to the case's configured baselines. Duplicates are rejected, and a baseline from another take, pipeline or artifact family is rejected on creation and on edit alike.

## Activation is not exclusive

`set_active` flips one record. It does not deactivate anything else, and a family may hold several active versions at once — this is deliberate, and supports legitimate multi-outcome cases. An `active`-mode case resolves to the **highest-numbered** active version.

The consequence is worth stating plainly: activating an older version while a newer one is still active changes nothing about what cases compare against. The history drawer says which versions are active, which one cases will resolve to afterwards, and names the version to deactivate when the activation would otherwise have no effect.

Activation and deactivation refresh case resolution and coverage. Neither rewrites a stored execution: past executions keep the versions they used.

Before either action, `GET /api/validation/baselines/{id}/impact` reports which cases follow the change and which are deliberately unaffected — pinned, allowed-version, disabled, and archived-suite cases are bucketed separately. Deactivating never mutates a case configuration, but it can leave `active`-mode cases with no version to resolve to; the drawer warns when that is what would happen.

## Baseline history

The history drawer lists every approved version of an artifact newest first, with state, source run, reviewer, date, integrity and notes, plus the version that superseded each one. Expanding a row shows the baseline id, checksum, source artifact, stage and processing unit, comparator and its source, recipe and contract fingerprints, comparison policy, and promotion provenance where the version came from a promotion.

Integrity is recomputed on every read of the history endpoint, so an absent value means "not reported", which is not the same as verified. A version whose snapshot fails verification cannot be activated.

Deletion is never offered. Versions are activated or deactivated.

## Promotion

A candidate is promoted from an exact execution, case and comparison. The candidate artifact is re-resolved and re-checksummed, and the recorded provenance checksum must match the stored snapshot — a candidate rewritten during promotion fails rather than producing a baseline whose provenance disagrees with its own bytes.

Promotion is inactive by default: the new version is created but the previously active one keeps resolving until someone activates the new one explicitly. `carry_forward_policy` controls whether the source comparison policy is copied forward.

Retrying the same promotion reuses the existing baseline and reports `already_promoted` rather than creating a version. Idempotency is keyed on execution, case, comparison and candidate checksum, so a changed candidate is correctly treated as a new promotion.

The response describes state *after* the promotion: `previous_active_baseline` is re-read rather than reported as it was before, and `resulting_active_baseline` is the family's real active version, or null when none is active.

The comparison drawer exposes this as "Promote candidate". Confirming shows the current and expected version, the candidate run, the comparison status, the carry-forward and activate-now checkboxes, an optional reviewer and notes, and the same impact list `resolution_impact()` would report for the source baseline — computed before the new version exists, since impact is matched by artifact identity rather than by baseline instance. The outcome panel distinguishes a genuine creation ("Created baseline vN.") from a reused promotion ("This candidate was already promoted as baseline vN. No new version was created.") so a retry never reads like a fresh success; when the result is inactive, it offers an "Activate baseline" button straight into the existing history drawer.

## Studio integration

Studio's "Mark as reference" already creates a baseline; "Add to regression suite" (next to it, gated on that baseline existing) is the other half — it puts the take into a suite without leaving Studio. Picking an existing suite or naming a new one calls `add_case()` with the just-approved baseline. `add_case()` itself has no dedupe — two calls for the same take create two cases — so Studio checks the target suite's existing cases by `take_id` first: a suite that already tracks this take reports the existing case rather than creating a sibling one. A brand-new suite skips the check, since it cannot already have a case.

Either way the outcome links straight into `/validation` at the suite (and case, if the case already existed or was just created), via a `validationDeepLink` parallel to Studio's own `studioDeepLink`. The Validation page reads `suite_id`, `execution_id`, `case_id` and `comparison_id` (a baseline id, per the comparisons-have-no-identifier note below) from the URL once on load: a linked suite wins over "most recently viewed", a linked execution wins over "latest", and a linked case+comparison auto-opens the comparison drawer exactly once, guarded the same way Studio's own run-id deep link is.

## Current versus historical state

The case table describes present governance. A selected historical execution reports the versions it actually used, the version it matched, its original statuses and its original first divergence. Editing a case or activating a baseline afterwards does not alter any of it — executions are written once and never revisited.

## Stage summary

A matrix cell answers "did this one case pass" per stage; it says nothing about the sample as a whole. `stage_summary(execution_id)` closes that gap: for one execution, it groups every comparison by `(stage_id, comparator)` and summarises each numeric field the comparator's `metrics` actually reports — count, mean, median, P95, min, max — plus a status breakdown. A drift that never fails any single case (a mean IoU sliding from 0.97 to 0.94 while every case individually still passes) is visible here even though it is invisible cell by cell.

Numeric fields are discovered, not named. Every key in a comparison's `metrics` whose value is a number is summarised on its own; a comparator that reports a boolean (`json_fields`'s `{"equal": bool}`) is summarised the same way, and the mean of a boolean is exactly its pass fraction. Nothing here assumes what fields a given comparator emits, so a future comparator is summarised automatically.

Fields are never pooled across comparators — an IoU and an MAE mean nothing averaged together — so grouping is by `(stage_id, comparator)`, never by stage alone. Disabled and skipped cases contribute nothing, matching `matrix()`. Every stage the pipeline's own contract declares appears in the output, including ones with zero comparisons, which report an empty comparator list rather than being silently omitted; the workspace renders those as "no comparisons yet". Stage identity is resolved the same way `matrix()` resolves it, sharing the same registry-derived stage order and fallback list, so the two views can never disagree about which stages exist.

`GET /api/validation/executions/{id}/stage-summary` exposes it; the workspace renders it as a "Stage summary" panel below the matrix.

## Stage trend

A single execution's stage summary can still hide a slow slide: mean IoU at 0.97, then 0.96, then 0.95, never once crossing a pass/fail line on any individual case. `stage_trend(suite_id, limit=5)` lines up `stage_summary()` across a suite's most recent executions so that slide becomes visible, keyed the same way -- `(stage_id, comparator)` -- and never pooling an IoU with an MAE.

Computed on demand, not persisted: one `stage_summary()` call per execution in the window, no new storage. A suite whose history grows long enough that this is visibly slow to compute is the signal to reconsider persisting a rollup at write time instead; nothing about this shipped version requires or rules out that change.

A `(stage, comparator)` pair absent from a given execution -- no case exercised it that run, or a baseline's comparator was reconfigured afterward -- is a real gap and is reported as one (`null`), never backfilled as a zero. `limit` is clamped to at least 1 (Python's `list[-0:]` is the whole list, not an empty one, and 0 would otherwise silently mean "everything").

`GET /api/validation/suites/{id}/stage-trend?limit=5` exposes it; the workspace renders a "Stage trend" panel next to the stage summary, one table per stage with data, one column per execution.

## Operations

The CLI is intentionally usable in CI. Twenty-one commands cover the governance family: suite lifecycle (`list-suites`, `inspect-suite`, `update-suite`, `duplicate-suite`, `archive-suite`, `restore-suite`, `coverage`), cases (`add-case`, `update-case`, `remove-case`), baselines (`baseline-history`, `activate-baseline`, `deactivate-baseline`, `compare`, `promote-comparison`), and execution (`run-suite`, `list-executions`, `inspect-execution`, `rebuild-indexes`, `verify-integrity`, `smoke`). All emit JSON.

`run-suite` and `compare` return `1` for regressions and execution failures, `0` otherwise; their JSON retains `not_comparable` and `needs_review` rather than treating either as a silent pass. Integrity returns `2` for missing snapshots, checksum corruption, unsafe snapshot paths, or broken suite references.

The CLI is currently the most complete surface: it is the only one exposing `verify-integrity` and `rebuild-indexes`.

Indexes are caches and can be regenerated from `baseline.json`, `suite.json`, and `execution.json`. Snapshot paths are restricted to the validation baseline root and checksums are verified before an integrity report passes.

## Errors

The API answers refusals with a status code and a plain message, not typed error codes. The workspace matches on those messages to produce actionable sentences — an archived suite, an incompatible family, an empty allowed selection — and falls back to showing what the server said, minus the status prefix. Adding `VALIDATION_*` codes would require a backend change; nothing emits them today.

## Current limitations

Stage summary and stage trend both read through the same registry-derived stage list `matrix()` uses, so an artifact whose `stage_id` is not one the pipeline's own contract registers is invisible to all three, the same way `matrix()` already omitted it before either existed.

`coverage()` and `_case_baselines()` were fixed to resolve each case's active mode/pinned/allowed_versions setting rather than iterating raw `baseline_ids`. Two related things were true before this: coverage double-counted a case pinned to an inactive version, and active mode could return the same resolved baseline twice when `baseline_ids` held more than one version of the same family — which `execute_suite()` would then compare against redundantly. Both are covered by tests now.

- Stage trend labels each execution column by its date to a day's precision; several executions run within the same day (or the same minute, as in development) render identical column headers with nothing to disambiguate them beyond hovering the underlying execution id.
- The Studio-side duplicate-case check is per suite, by `take_id` only. The same take can still end up with a separate case in each of several suites (by design — a take can belong to more than one suite), and two same-named suites are not disambiguated in the picker beyond their id.
- Comparisons carry no identifier of their own. `matrix()` reports `baseline_id` in `comparison_ids`, so the `comparison_id` a validation deep link carries is a baseline id, not an independent comparison id.
- `execute_suite` accepts archived suites. Case mutation is blocked on them; running is not, on the grounds that an execution does not mutate the suite.
- Four routes have no client binding: single-baseline read, execution progress polling, integrity and index rebuild. The last two are exposed by the CLI.
- Frontend coverage is model-level. `validationCaseModel`, `validationHistoryModel`, `validationComparisonModel`, `validationStageSummaryModel`, `validationPromotionModel`, `validationDeepLink` and `addToSuiteModel` are tested; the panel, drawer and detail-view components are not.
