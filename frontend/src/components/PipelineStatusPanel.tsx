import { useState } from "react";
import type { PipelineInfo, ProcessingResult, RuntimeState, TakeDetail } from "../api/client";
import { activePipelineName, incompatibleModalityMessage, modalityCompatible, pipelineCompatibilityGroups } from "./pipelineModel";

type Props = {
  result: ProcessingResult | null;
  runtimeState?: RuntimeState | null;
  detail?: TakeDetail | null;
  pipelines?: PipelineInfo[];
  selectedPipelineId?: string;
  onSelectPipeline?: (pipelineId: string) => void;
  pipelineSearch?: string;
  onPipelineSearchChange?: (value: string) => void;
  pipelineFamilyFilter?: "all" | "2d" | "3d" | "generic";
  onPipelineFamilyFilterChange?: (value: "all" | "2d" | "3d" | "generic") => void;
  compact?: boolean;
};

export default function PipelineStatusPanel({
  result,
  runtimeState = null,
  detail = null,
  pipelines = [],
  selectedPipelineId,
  onSelectPipeline,
  pipelineSearch = "",
  onPipelineSearchChange,
  pipelineFamilyFilter = "all",
  onPipelineFamilyFilterChange,
  compact = false,
}: Props) {
  const activeId = selectedPipelineId ?? result?.processing_pipeline?.id ?? "3d_ball_inspection";
  const pipeline = pipelines.find((item) => item.id === activeId) ?? pipelines[0] ?? null;
  const required = pipeline?.required_modalities ?? result?.processing_pipeline?.required_modalities ?? ["point_cloud"];
  const compatible = pipeline ? modalityCompatible(pipeline, detail?.modalities ?? result?.input_modalities) : required.every((modality) => (detail?.modalities ?? result?.input_modalities ?? []).includes(modality));
  const incompatibleMessage = pipeline ? incompatibleModalityMessage(pipeline, detail?.modalities ?? result?.input_modalities) : null;
  const currentStage = result?.algorithm_stage ?? "idle";
  const groups = pipelineCompatibilityGroups(pipelines, detail?.modalities ?? result?.input_modalities);
  const [openSelector, setOpenSelector] = useState(false);
  const discovered = pipelines.filter((item) => {
    if (pipelineFamilyFilter !== "all" && (item.pipeline_family ?? "3d") !== pipelineFamilyFilter) return false;
    const query = pipelineSearch.trim().toLowerCase();
    if (!query) return true;
    return [item.display_name, item.name, item.id, item.description ?? ""].join(" ").toLowerCase().includes(query);
  });

  return (
    <section className={compact ? "concept-panel compact pipeline-context-card" : "concept-panel pipeline-context-card"}>
      <div className="concept-heading">
        <span className="eyebrow">Active processing pipeline</span>
      </div>
      <div className="pipeline-compact-context">
        <label className="field-label">
          Pipeline
          <button className="pipeline-picker-button" onClick={() => setOpenSelector((item) => !item)} type="button">
            {pipeline?.display_name ?? "Select pipeline"} ▼
          </button>
        </label>
        {openSelector && (
          <div className="pipeline-selector-popover">
            <div className="pipeline-selector-toolbar">
              <input type="text" value={pipelineSearch} onChange={(event) => onPipelineSearchChange?.(event.target.value)} placeholder="Search pipelines..." />
              <select value={pipelineFamilyFilter} onChange={(event) => onPipelineFamilyFilterChange?.(event.target.value as "all" | "2d" | "3d" | "generic")}>
                <option value="all">All</option>
                <option value="2d">2D</option>
                <option value="3d">3D</option>
                <option value="generic">Generic</option>
              </select>
            </div>
            {groups.compatible.length > 0 && <small className="pipeline-group-title">Compatible pipelines</small>}
            {discovered.filter((item) => groups.compatible.some((good) => good.id === item.id)).map((item) => (
              <button key={`pick_${item.id}`} className="pipeline-popover-item" onClick={() => { onSelectPipeline?.(item.id); setOpenSelector(false); }} type="button">
                <strong>{item.display_name}</strong>
                <small>{item.pipeline_family ?? "3d"} • {item.execution_backend ?? "filesystem"} • {(item.supported_modalities ?? item.required_modalities ?? []).join(", ")}</small>
                <small>Compatible</small>
              </button>
            ))}
            {groups.incompatible.length > 0 && <small className="pipeline-group-title">Incompatible pipelines</small>}
            {discovered.filter((item) => groups.incompatible.some((bad) => bad.pipeline.id === item.id)).map((item) => (
              <button key={`pick_bad_${item.id}`} className="pipeline-popover-item incompatible" onClick={() => { onSelectPipeline?.(item.id); setOpenSelector(false); }} type="button">
                <strong>{item.display_name}</strong>
                <small>{item.pipeline_family ?? "3d"} • {item.execution_backend ?? "filesystem"} • {(item.supported_modalities ?? item.required_modalities ?? []).join(", ")}</small>
                <small>Incompatible: {incompatibleModalityMessage(item, detail?.modalities ?? result?.input_modalities)}</small>
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="pipeline-chip-row">
        <span className="status-badge">{String(pipeline?.pipeline_family ?? "3d").toUpperCase()}</span>
        <span className="status-badge">{pipeline?.execution_backend ?? "filesystem_queue"}</span>
        <span className={`status-badge ${compatible ? "status-connected" : "status-failed"}`}>{compatible ? "Compatible" : "Incompatible"}</span>
        {(pipeline?.supported_modalities ?? pipeline?.required_modalities ?? []).map((item) => (
          <span className="status-badge" key={`mod_${item}`}>{item}</span>
        ))}
      </div>
      <small>Selected take has {(detail?.modalities ?? result?.input_modalities ?? []).join(", ") || "no modalities"}.</small>
      {!compatible && <p className="section-note">{incompatibleMessage ?? "This pipeline cannot consume the selected take modalities."}</p>}
      {!!runtimeState?.warnings?.length && <p className="section-note">{runtimeState.warnings.join(" | ")}</p>}
    </section>
  );
}
