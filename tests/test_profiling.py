from __future__ import annotations

import time

from vision_3d_acquisition.utils.profiling import ProcessingProfiler


def test_profiler_records_stage_duration() -> None:
    profiler = ProcessingProfiler()

    with profiler.stage("example", category="preprocessing", input_points=10) as stage:
        time.sleep(0.001)
        stage.output_points = 7

    payload = profiler.as_dict()
    assert payload["stages"][0]["name"] == "example"
    assert payload["stages"][0]["input_points"] == 10
    assert payload["stages"][0]["output_points"] == 7
    assert payload["stages"][0]["duration_ms"] >= 0


def test_profiler_aggregates_by_category() -> None:
    profiler = ProcessingProfiler()

    with profiler.stage("load", category="io"):
        pass
    with profiler.stage("render", category="debug_artifacts"):
        pass
    with profiler.stage("cluster", category="clustering"):
        pass

    payload = profiler.as_dict()
    assert payload["io_ms"] >= 0
    assert payload["debug_artifacts_ms"] >= 0
    assert payload["production_ms"] == round(profiler.production_ms(), 3)
    assert payload["production_ms"] < payload["total_ms"] or payload["debug_artifacts_ms"] == 0
