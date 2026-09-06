import { useEffect, useState } from "react";
import {
  api,
  type ValidationBaseline,
  type ValidationComparisonResult,
  type ValidationExecution,
  type ValidationExecutionCase,
  type ValidationPromotionResponse,
  type ValidationResolutionImpact,
  type ValidationSuiteCase,
  type ValidationSuiteDetail,
} from "../../api/client";
import { buildStudioDeepLink } from "../../studioDeepLink";
import { validationStatus, type ValidationStatus } from "./validationStatus";
import { describeApiError } from "./validationCaseModel";
import {
  describePromotionOutcome,
  expectedNextVersion,
  summarisePromotionImpact,
} from "./validationPromotionModel";
import ValidationBaselineHistoryDrawer from "./ValidationBaselineHistoryDrawer";
import ValidationMaskDetail from "./ValidationMaskDetail";
import ValidationRasterDetail from "./ValidationRasterDetail";
import ValidationMeasurementTable from "./ValidationMeasurementTable";
import ValidationClassificationTable from "./ValidationClassificationTable";

interface Props {
  suite: ValidationSuiteDetail;
  execution: ValidationExecution;
  executionCase: ValidationExecutionCase;
  /** The comparisons the clicked matrix cell resolved to — usually one, more than one under allowed_versions. */
  comparisons: ValidationComparisonResult[];
  onClose: () => void;
  onChanged: () => Promise<void>;
}

const badge = (status: string) =>
  validationStatus[status as ValidationStatus] ?? {
    label: status,
    severity: "muted" as const,
    icon: "?",
    tooltip: status,
  };

export default function ValidationComparisonDrawer({
  suite,
  execution,
  executionCase,
  comparisons,
  onClose,
  onChanged,
}: Props) {
  const [selected, setSelected] = useState(0);
  const [baseline, setBaseline] = useState<ValidationBaseline | null>(null);
  const [error, setError] = useState("");
  const [historyOpen, setHistoryOpen] = useState(false);
  const [promoting, setPromoting] = useState(false);
  const [carryForwardPolicy, setCarryForwardPolicy] = useState(true);
  const [activateNow, setActivateNow] = useState(false);
  const [notes, setNotes] = useState("");
  const [reviewedBy, setReviewedBy] = useState("");
  const [impact, setImpact] = useState<ValidationResolutionImpact | null>(null);
  const [promotionBusy, setPromotionBusy] = useState(false);
  const [promotionResult, setPromotionResult] = useState<ValidationPromotionResponse | null>(null);

  const comparison = comparisons[selected];
  const baselineId = comparison?.baseline_ref.baseline_id ?? null;

  useEffect(() => {
    setBaseline(null);
    setPromoting(false);
    setPromotionResult(null);
    setImpact(null);
    if (!baselineId) return;
    let cancelled = false;
    api
      .validationBaseline(baselineId)
      .then((record) => {
        if (!cancelled) setBaseline(record);
      })
      .catch((caught: unknown) => {
        if (!cancelled) setError(describeApiError(caught));
      });
    return () => {
      cancelled = true;
    };
  }, [baselineId]);

  if (!comparison) return null;

  const suiteCase = suite.cases.find((item) => item.id === executionCase.case_id);
  const status = badge(comparison.status);

  const openPromotion = () => {
    setPromoting(true);
    setImpact(null);
    if (baselineId) {
      api
        .validationBaselineImpact(baselineId)
        .then((result) => setImpact(result))
        .catch((caught: unknown) => setError(describeApiError(caught)));
    }
  };

  const confirmPromotion = async () => {
    if (!baselineId) return;
    setPromotionBusy(true);
    setError("");
    try {
      const result = await api.promoteValidationComparison(execution.id, baselineId, {
        case_id: executionCase.case_id,
        activate: activateNow,
        carry_forward_policy: carryForwardPolicy,
        reviewed_by: reviewedBy.trim() || undefined,
        notes: notes.trim() || undefined,
      });
      setPromotionResult(result);
      setPromoting(false);
      await onChanged();
    } catch (caught: unknown) {
      setError(describeApiError(caught));
    } finally {
      setPromotionBusy(false);
    }
  };

  return (
    <div className="control-panel-section" role="dialog" aria-label="Comparison detail">
      <div className="control-panel-button-row">
        <h3 style={{ margin: 0 }}>Comparison detail</h3>
        <button type="button" onClick={onClose}>
          Close
        </button>
      </div>
      {error ? <p role="alert">{error}</p> : null}

      {comparisons.length > 1 ? (
        <div className="control-panel-button-row">
          <span>
            <small>This case allows several versions. {comparisons.length} were compared:</small>
          </span>
          {comparisons.map((entry, index) => (
            <button
              key={entry.baseline_ref.baseline_id ?? index}
              type="button"
              disabled={index === selected}
              onClick={() => setSelected(index)}
            >
              v{entry.baseline_ref.version ?? "?"} · {badge(entry.status).label}
            </button>
          ))}
        </div>
      ) : null}

      <div className="table-wrap">
        <table>
          <tbody>
            <tr>
              <th scope="row">Regression suite</th>
              <td>{suite.name}</td>
              <th scope="row">Execution</th>
              <td>{execution.id}</td>
            </tr>
            <tr>
              <th scope="row">Case</th>
              <td>{executionCase.case_id}</td>
              <th scope="row">Take</th>
              <td>{executionCase.take_id}</td>
            </tr>
            <tr>
              <th scope="row">Baseline version</th>
              <td>
                {baseline ? `v${baseline.version}` : (comparison.baseline_ref.version ?? "—")}
              </td>
              <th scope="row">Baseline source run</th>
              <td>
                {baseline ? (
                  <a
                    href={buildStudioDeepLink({
                      take_id: baseline.take_id,
                      pipeline_id: baseline.pipeline_id,
                      run_id: baseline.source_run_id,
                    })}
                  >
                    {baseline.source_run_id}
                  </a>
                ) : (
                  "—"
                )}
              </td>
            </tr>
            <tr>
              <th scope="row">Candidate run</th>
              <td>
                <a
                  href={buildStudioDeepLink({
                    take_id: executionCase.take_id,
                    pipeline_id: execution.pipeline_id,
                    run_id: execution.candidate_run_id,
                  })}
                >
                  {execution.candidate_run_id}
                </a>
              </td>
              <th scope="row">Pipeline</th>
              <td>{execution.pipeline_id}</td>
            </tr>
            <tr>
              <th scope="row">Recipe fingerprint</th>
              <td>{baseline?.recipe_fingerprint ?? "—"}</td>
              <th scope="row">Calibration</th>
              <td>
                {baseline?.calibration_context && Object.keys(baseline.calibration_context).length
                  ? JSON.stringify(baseline.calibration_context)
                  : "—"}
              </td>
            </tr>
            <tr>
              <th scope="row">Stage</th>
              <td>{baseline?.stage_id ?? "—"}</td>
              <th scope="row">Processing unit</th>
              <td>{baseline?.processing_unit_id ?? "—"}</td>
            </tr>
            <tr>
              <th scope="row">Substage</th>
              <td>{baseline?.substage_id ?? "—"}</td>
              <th scope="row">View</th>
              <td>{baseline?.view_id ?? "—"}</td>
            </tr>
            <tr>
              <th scope="row">Artifact</th>
              <td>{comparison.candidate_ref.artifact_id ?? baseline?.artifact_id ?? "—"}</td>
              <th scope="row">Comparator</th>
              <td>
                {comparison.comparator}
                {baseline?.comparator_source ? ` (${baseline.comparator_source})` : ""}
              </td>
            </tr>
            <tr>
              <th scope="row">Status</th>
              <td colSpan={3}>
                <span className={`status-badge ${comparison.status}`} title={status.tooltip}>
                  {status.icon} {status.label}
                </span>{" "}
                {comparison.summary}
                {comparison.reasons.length ? ` — ${comparison.reasons.join(", ")}` : ""}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="control-panel-button-row">
        {baseline ? (
          <a
            href={buildStudioDeepLink({
              take_id: baseline.take_id,
              pipeline_id: baseline.pipeline_id,
              run_id: baseline.source_run_id,
              stage: baseline.stage_id,
            })}
          >
            Open baseline source run in Studio
          </a>
        ) : null}
        <a
          href={buildStudioDeepLink({
            take_id: executionCase.take_id,
            pipeline_id: execution.pipeline_id,
            run_id: execution.candidate_run_id,
            stage: baseline?.stage_id,
          })}
        >
          Open candidate run in Studio
        </a>
        {suiteCase ? (
          <button type="button" onClick={() => setHistoryOpen(true)}>
            Open baseline history
          </button>
        ) : null}
        {baselineId && !promoting && !promotionResult ? (
          <button type="button" onClick={openPromotion}>
            Promote candidate
          </button>
        ) : null}
      </div>

      {promoting && baseline ? (
        <div className="control-panel-section" role="alertdialog" aria-label="Confirm promotion">
          <h4>Promote candidate as new baseline?</h4>
          <div className="table-wrap">
            <table>
              <tbody>
                <tr>
                  <th scope="row">Current version</th>
                  <td>v{baseline.version}</td>
                  <th scope="row">Expected next version</th>
                  <td>v{expectedNextVersion(baseline)}</td>
                </tr>
                <tr>
                  <th scope="row">Candidate run</th>
                  <td>{execution.candidate_run_id}</td>
                  <th scope="row">Comparison status</th>
                  <td>{status.label}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p>
            <small>
              The candidate artifact is re-resolved and re-checksummed at promotion time; if it no
              longer matches what this comparison ran against, promotion is refused rather than
              recording provenance that disagrees with the file.
            </small>
          </p>
          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={carryForwardPolicy}
              onChange={(event) => setCarryForwardPolicy(event.target.checked)}
            />
            Carry forward this baseline&apos;s comparison policy
          </label>
          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={activateNow}
              onChange={(event) => setActivateNow(event.target.checked)}
            />
            Activate immediately
          </label>
          <label>
            Reviewer
            <input value={reviewedBy} onChange={(event) => setReviewedBy(event.target.value)} />
          </label>
          <label>
            Notes
            <input value={notes} onChange={(event) => setNotes(event.target.value)} />
          </label>
          <p>
            <small>Impact</small>
          </p>
          {impact ? (
            <ul>
              {summarisePromotionImpact(impact, activateNow).map((line) => (
                <li key={line}>{line}</li>
              ))}
              {!summarisePromotionImpact(impact, activateNow).length ? (
                <li>No case references this artifact yet.</li>
              ) : null}
            </ul>
          ) : (
            <p>
              <small>Checking which cases this would affect…</small>
            </p>
          )}
          <div className="control-panel-button-row">
            <button
              type="button"
              disabled={promotionBusy || !impact}
              onClick={() => void confirmPromotion()}
            >
              Promote candidate
            </button>
            <button type="button" onClick={() => setPromoting(false)}>
              Cancel
            </button>
          </div>
        </div>
      ) : null}

      {promotionResult ? (
        <div className="control-panel-section" role="status" aria-label="Promotion result">
          {(() => {
            const outcome = describePromotionOutcome(promotionResult);
            return (
              <>
                <h4>{outcome.headline}</h4>
                <p>{outcome.detail}</p>
                <ul>
                  {summarisePromotionImpact(
                    promotionResult.affected_case_resolution,
                    promotionResult.baseline.status === "active",
                  ).map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
                {outcome.needsActivation ? (
                  <div className="control-panel-button-row">
                    <button type="button" onClick={() => setHistoryOpen(true)}>
                      Activate baseline
                    </button>
                  </div>
                ) : null}
              </>
            );
          })()}
        </div>
      ) : null}

      {historyOpen && suiteCase ? (
        <ValidationBaselineHistoryDrawer
          caseItem={suiteCase}
          suiteId={suite.id}
          pipelineId={suite.pipeline_id}
          editable={suite.status !== "archived"}
          onClose={() => setHistoryOpen(false)}
          onChanged={onChanged}
        />
      ) : null}

      <ComparatorDetail comparison={comparison} />
    </div>
  );
}

function ComparatorDetail({ comparison }: { comparison: ValidationComparisonResult }) {
  switch (comparison.comparator) {
    case "binary_mask":
      return <ValidationMaskDetail result={comparison} />;
    case "numeric_raster":
      return <ValidationRasterDetail result={comparison} />;
    case "measurement_table":
      return <ValidationMeasurementTable result={comparison} />;
    case "classification":
      return <ValidationClassificationTable result={comparison} />;
    default:
      return (
        <div className="control-panel-section">
          <p>
            <small>
              No semantic detail view exists for comparator <code>{comparison.comparator}</code>.
              Metrics, thresholds and reasons are shown above.
            </small>
          </p>
        </div>
      );
  }
}
