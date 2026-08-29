# Approved regression baselines

Validation is an additive engineering-review layer. A take is an immutable acquisition and a run is an immutable processing interpretation. An approved reference is a human decision about a run output; its baseline is a copied snapshot under `data/validation/baselines`, so run cleanup cannot invalidate it. It is not ground truth unless externally measured truth is explicitly recorded elsewhere.

Approving again creates a new version and leaves prior versions immutable. Activation and deactivation only alter the selected baseline record; they never remove history. A baseline records its source run, contract and recipe fingerprints, parameters, calibration context, reviewer and notes.

Suites live in `data/validation/suites`; executions in `data/validation/executions`, with incremental progress. A case connects one take to one or more baseline expectations. The current initial runner evaluates existing candidate runs (`latest` or an explicit run id), preserving all run semantics. Pipeline dispatch/rerun remains owned by the existing processing workflow, which keeps validation free from implicit recipe or run mutation.

Comparators are selected from artifact contracts where supplied, with conservative fallbacks: masks use IoU/Dice/add/remove fractions; numeric rasters use error statistics; JSON-backed plane, measurement, and classification artifacts are initially structural comparisons after unstable run fields are excluded. PNG overlays are `visual_only` and therefore `needs_review`, not failures. Unknown artifacts are `not_comparable`, never pass.

Statuses have distinct meanings: `pass` is within tolerance; `changed` differs without a failure policy; `regression` violates an expectation; `not_comparable` lacks a compatible semantic contract; `blocked` is invalidated by upstream failure; and `needs_review` is evidence requiring an engineer. Studio should use baseline controls next to artifact review; suite governance and execution belong in the Validation workspace.

Initial 2.5D focus is reference support masks, belt plane/residual summaries, authoritative normalized height rasters, final object masks, measurements, classification fields, and quality flags. Overlay rendering is retained as review evidence. Future extensions should add per-artifact `validationSpec` metadata (eligibility, comparator, tolerances, unstable/required fields, and downstream blocking) rather than filename-specific batch-runner logic.

## Operations

The CLI is intentionally usable in CI: `validation list-suites`, `run-suite`,
`compare`, `inspect-execution`, `rebuild-indexes`, and `verify-integrity` all
emit JSON. `run-suite` and `compare` return `1` for regressions/execution
failures, `0` otherwise; their JSON retains `not_comparable` and
`needs_review` rather than treating either as a silent pass. Integrity returns
`2` for missing snapshots, checksum corruption, unsafe snapshot paths, or
broken suite references.

## Semantic comparators and governance

Measurement tables use stable object IDs before conservative centroid matching.
Configured metrics pass when `abs(delta) <= max(absolute_tolerance,
relative_tolerance * abs(baseline))`; missing required metrics and unmatched
objects regress, while ambiguous geometric matches require review. Classification
uses the same matcher and treats superclass or detailed-label changes as
regressions unless an explicit allowed-label policy permits them. New configured
critical warnings also regress.

The Validation workspace (`/validation`) is the suite execution surface. Its
matrix keeps status cells separate from coverage: `pass`, `changed`,
`regression`, `not_comparable`, `blocked`, `needs_review`, and skipped states
are never collapsed. Matrix detail opens the persisted comparison context;
binary-mask and numeric-raster diff assets are retained below the execution
directory. The changed-run smoke proves the lifecycle: baseline v1 pass,
deterministic segmentation regression, then explicit promotion to v2 and pass.

Indexes are caches and can be regenerated from `baseline.json`, `suite.json`,
and `execution.json`. Snapshot paths are restricted to the validation baseline
root and checksums are verified before an integrity report passes. Deactivating
or superseding a baseline leaves its snapshot and historical execution reports
intact. Suite deletion must similarly only remove suite records, never a
baseline.
