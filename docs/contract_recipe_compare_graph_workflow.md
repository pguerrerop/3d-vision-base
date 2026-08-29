# Contract / Recipe / Compare / Graph Workflow

This document explains the 25D engineering workflow in Sensor Studio: processing-unit contracts, recipes, comparisons, runtime trace, and the read-only graph workspace.

## What is a processing unit?

A **processing unit** is the smallest contract-defined step in the 25D pipeline. Each unit declares:

- tunable parameters
- input and output artifacts
- views for Studio rendering
- optional diagnostics and metrics

Stages and substages in Processing Lab map back to these units. The **Contract Graph** is generated from the unit registry — it is read-only for now.

## What is a recipe?

A **recipe** is a named, versioned snapshot of stage/unit parameters for a pipeline. Recipes let you:

- save a known-good tuning baseline
- clone and iterate on variants
- compare a run or current edits against a saved recipe

Recipes are stored under `data/recipes/{pipeline_id}/{recipe_id}/recipe.json`.

## What does “dirty vs recipe” mean?

- **Matches selected recipe** — current form values equal the selected recipe for visible units.
- **Differs from recipe (dirty)** — you edited parameters after loading a recipe. Runs can still use the recipe plus your overrides.

The tune panel and graph workspace show dirty state per unit when recipe diff data is available.

## Runtime trace vs inferred trace

After a run, Studio builds a **processing-unit trace**:

| Badge | Meaning |
| --- | --- |
| **runtime traced** | Unit executed through a runtime callback; parameters, metrics, and timing are authoritative. |
| **inferred** | No runtime callback; status reconstructed from artifacts/registry fallback. |
| **mixed trace** | Pipeline-level summary when some units are runtime-traced and others inferred. |

**Runtime trace coverage** is shown as `runtime_traced_units / total_units` with a percentage. Low coverage is expected until more units have callbacks — inferred units are still useful for navigation and artifact inspection.

## Mask IoU / changed pixels

When comparing runs or recipes, diffable mask artifacts get a pixel summary:

- **changed %** — fraction of pixels that differ
- **IoU** — intersection-over-union between left and right masks
- **added / removed pixels** — directional change counts

If diff is unavailable, the UI explains why (missing artifact, not diffable, single-sided presence, etc.).

## Comparison history

Each compare action can persist an index entry under `data/comparisons/index.json` with:

- comparison id
- left/right source types (current edits, selected recipe, last run)
- summary of what changed most
- paths to full comparison JSON and mask diff artifacts

Use **Reopen** in the Compare tab or pass `comparison_id` in a Studio deep link.

## Graph workspace deep links

Studio URLs can include:

| Param | Purpose |
| --- | --- |
| `graph_workspace=1` | Open expanded Graph / Compare Workspace |
| `unit_id` | Focus a processing unit |
| `recipe_id` | Select a recipe |
| `comparison_id` | Reopen a saved comparison |
| `run_id` | Hint for run-scoped navigation |

Example (from demo CLI):

```
/studio?take_id=take_smoke_compare&pipeline_id=mining_steel_ball_classification_25d&graph_workspace=1&recipe_id=...&comparison_id=...
```

## Why is the graph read-only?

Graph editing and partial reruns are planned follow-ups. The current graph:

- reflects the contract registry and execution order
- overlays runtime/inferred trace, recipe dirty state, and comparison highlights
- links to tune, compare, and artifact inspection

Editing the graph would require dependency-safe mutation and rerun scheduling — out of scope for this milestone.

## Demo workflow command

```bash
python scripts/sensor_studio_cli.py 25d demo-contract-workflow --data-dir data
```

This validates contracts, creates demo recipes, runs comparable snapshots, computes mask diff, writes the comparison index, and prints Studio URLs/IDs for a known-good scenario.

## Suggested checklist

1. Select take
2. Select recipe
3. Tune unit parameters
4. Run with recipe + overrides
5. Compare against previous run
6. Inspect changed units in graph
7. Save recipe version
