from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_3d_acquisition.debug.contract_recipe_compare_smoke import run_25d_contract_recipe_compare_smoke

PIPELINE_ID = "mining_steel_ball_classification_25d"
DEMO_TAKE_ID = "take_smoke_compare"


def run_25d_demo_contract_workflow(data_dir: Path) -> dict[str, Any]:
    summary = run_25d_contract_recipe_compare_smoke(data_dir)
    base_recipe_id = str((summary.get("recipes") or {}).get("base_recipe_id") or "")
    run_compare_id = str(summary.get("run_compare_id") or "")
    query = (
        f"take_id={DEMO_TAKE_ID}"
        f"&pipeline_id={PIPELINE_ID}"
        f"&graph_workspace=1"
        f"&recipe_id={base_recipe_id}"
        f"&comparison_id={run_compare_id}"
        f"&run_id=smoke_run_b"
    )
    return {
        **summary,
        "demo_take_id": DEMO_TAKE_ID,
        "studio_open_url": f"/studio?{query}",
        "graph_workspace_url": f"/studio?{query}",
        "comparison_json_path": f"comparisons/{run_compare_id}/comparison.json",
        "comparison_index_path": "comparisons/index.json",
        "recipe_paths": {
            "base": f"recipes/{PIPELINE_ID}/{base_recipe_id}/recipe.json",
            "clone": f"recipes/{PIPELINE_ID}/{(summary.get('recipes') or {}).get('clone_recipe_id')}/recipe.json",
        },
        "next_steps": [
            "Open Processing Lab with studio_open_url",
            "Expand Graph / Compare Workspace to inspect changed units",
            "Reopen the saved comparison from Compare tab if needed",
        ],
    }
