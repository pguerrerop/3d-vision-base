import { useEffect, useMemo, useState } from "react";
import {
  api,
  type ValidationBaseline,
  type ValidationResolutionImpact,
  type ValidationSuiteCase,
} from "../../api/client";
import { describeApiError } from "./validationCaseModel";
import {
  canAttachToCase,
  describeIntegrity,
  orderHistory,
  previewActivation,
  previewDeactivation,
  summariseImpact,
  supersededByLabel,
} from "./validationHistoryModel";

interface Props {
  /** The case whose family is being reviewed. Pin and allow actions target it. */
  caseItem: ValidationSuiteCase;
  suiteId: string;
  pipelineId: string;
  /** False for archived suites: history stays readable, actions do not apply. */
  editable: boolean;
  onClose: () => void;
  onChanged: () => Promise<void>;
}

const shortDate = (value: string | undefined): string =>
  value ? new Date(value).toISOString().slice(0, 16).replace("T", " ") : "—";

export default function ValidationBaselineHistoryDrawer({
  caseItem,
  suiteId,
  pipelineId,
  editable,
  onClose,
  onChanged,
}: Props) {
  const [history, setHistory] = useState<ValidationBaseline[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [pending, setPending] = useState<{
    baseline: ValidationBaseline;
    activate: boolean;
  } | null>(null);
  const [impact, setImpact] = useState<ValidationResolutionImpact | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const load = async () => {
    const rows = await api.validationBaselineHistory({
      take_id: caseItem.take_id,
      pipeline_id: pipelineId,
    });
    setHistory(orderHistory(rows));
  };

  useEffect(() => {
    void load().catch((caught: unknown) => setError(describeApiError(caught)));
    // Reloading is driven by the case identity, not by every render.
  }, [caseItem.id, caseItem.take_id, pipelineId]);

  const ordered = useMemo(() => orderHistory(history), [history]);

  const guarded = async (action: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await action();
      await load();
      await onChanged();
    } catch (caught: unknown) {
      setError(describeApiError(caught));
    } finally {
      setBusy(false);
    }
  };

  /** Ask the server who follows this change before offering to make it. */
  const propose = async (baseline: ValidationBaseline, activate: boolean) => {
    setError("");
    setPending({ baseline, activate });
    setImpact(null);
    try {
      setImpact(await api.validationBaselineImpact(baseline.id));
    } catch (caught: unknown) {
      setError(describeApiError(caught));
    }
  };

  const commit = () =>
    guarded(async () => {
      if (!pending) return;
      if (pending.activate) await api.activateValidationBaseline(pending.baseline.id);
      else await api.deactivateValidationBaseline(pending.baseline.id);
      setPending(null);
      setImpact(null);
    });

  const pinToCase = (baseline: ValidationBaseline) =>
    guarded(async () => {
      await api.updateValidationCase(suiteId, caseItem.id, {
        baseline_resolution: { mode: "pinned", baseline_id: baseline.id },
      });
    });

  const addToAllowed = (baseline: ValidationBaseline) =>
    guarded(async () => {
      const resolution = caseItem.baseline_resolution;
      const current =
        resolution?.mode === "allowed_versions"
          ? (resolution.allowed_baseline_ids ?? [])
          : caseItem.baseline_ids;
      if (current.includes(baseline.id)) return;
      await api.updateValidationCase(suiteId, caseItem.id, {
        baseline_resolution: {
          mode: "allowed_versions",
          allowed_baseline_ids: [...current, baseline.id],
        },
      });
    });

  const activation = pending?.activate ? previewActivation(pending.baseline, ordered) : null;
  const deactivation =
    pending && !pending.activate && impact
      ? previewDeactivation(pending.baseline, ordered, impact)
      : null;

  return (
    <section className="control-panel-section" aria-label="Baseline history">
      <h3>Baseline history · {caseItem.take_id}</h3>
      <p>
        <small>
          Every approved version of this artifact, newest first. Versions are never deleted; they
          are activated or deactivated.
        </small>
      </p>
      {error ? <p role="alert">{error}</p> : null}

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Version</th>
              <th>State</th>
              <th>Source run</th>
              <th>Reviewer</th>
              <th>Date</th>
              <th>Integrity</th>
              <th>Notes</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {ordered.map((baseline) => {
              const integrity = describeIntegrity(baseline);
              const superseded = supersededByLabel(baseline, ordered);
              const attachable = canAttachToCase(baseline, caseItem, ordered);
              return (
                <tr key={baseline.id}>
                  <td>
                    v{baseline.version}
                    {superseded ? <small> · superseded by {superseded}</small> : null}
                  </td>
                  <td>{baseline.status === "active" ? "Active" : "Inactive"}</td>
                  <td>{baseline.source_run_id}</td>
                  <td>{baseline.review?.reviewed_by ?? "—"}</td>
                  <td>{shortDate(baseline.created_at)}</td>
                  <td>
                    <span className={`status-badge ${integrity.severity}`} title={integrity.detail}>
                      {integrity.label}
                    </span>
                  </td>
                  <td>{baseline.review?.notes || "—"}</td>
                  <td>
                    <div className="control-panel-button-row">
                      <button
                        type="button"
                        onClick={() => setExpanded(expanded === baseline.id ? null : baseline.id)}
                      >
                        {expanded === baseline.id ? "Hide details" : "Inspect"}
                      </button>
                      {baseline.status === "active" ? (
                        <button
                          type="button"
                          disabled={!editable || busy}
                          onClick={() => void propose(baseline, false)}
                        >
                          Deactivate
                        </button>
                      ) : (
                        <button
                          type="button"
                          disabled={!editable || busy || !integrity.usable}
                          title={integrity.usable ? undefined : integrity.detail}
                          onClick={() => void propose(baseline, true)}
                        >
                          Activate
                        </button>
                      )}
                      <button
                        type="button"
                        disabled={!editable || busy || !attachable}
                        onClick={() => void pinToCase(baseline)}
                      >
                        Pin to case
                      </button>
                      <button
                        type="button"
                        disabled={!editable || busy || !attachable}
                        onClick={() => void addToAllowed(baseline)}
                      >
                        Add to allowed
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
            {!ordered.length ? (
              <tr>
                <td colSpan={8}>No approved versions exist for this take yet.</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      {expanded ? <BaselineDetails baseline={ordered.find((b) => b.id === expanded)!} /> : null}

      {pending ? (
        <div
          className="control-panel-section"
          role="alertdialog"
          aria-label="Confirm activation change"
        >
          <h4>
            {pending.activate ? "Activate" : "Deactivate"} v{pending.baseline.version}?
          </h4>
          {activation?.noop ? <p>This version is already the active one.</p> : null}
          {activation?.blocked ? <p role="alert">{activation.blocked}</p> : null}
          {activation && !activation.noop ? (
            <p>
              {activation.currentlyActive.length
                ? `Active now: ${activation.currentlyActive.map((b) => `v${b.version}`).join(", ")}. Activating does not deactivate them.`
                : "No version of this artifact is active right now."}
              {` Active-resolution cases will compare against v${activation.resolvesToAfter.version} afterwards.`}
            </p>
          ) : null}
          {activation?.ineffective ? <p role="status">{activation.ineffective}</p> : null}
          {deactivation?.message ? <p>{deactivation.message}</p> : null}

          {impact ? (
            <ul>
              {summariseImpact(impact).map((line) => (
                <li key={line}>{line}</li>
              ))}
              {!summariseImpact(impact).length ? (
                <li>No case references this artifact yet.</li>
              ) : null}
            </ul>
          ) : (
            <p>
              <small>Checking which cases this would affect…</small>
            </p>
          )}
          <p>
            <small>
              Past executions keep the versions they used. Only future runs are affected.
            </small>
          </p>

          <div className="control-panel-button-row">
            <button
              type="button"
              disabled={
                busy || !impact || Boolean(activation?.blocked) || Boolean(activation?.noop)
              }
              onClick={() => void commit()}
            >
              {pending.activate ? "Activate this version" : "Deactivate this version"}
            </button>
            <button
              type="button"
              onClick={() => {
                setPending(null);
                setImpact(null);
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      ) : null}

      <div className="control-panel-button-row">
        <button type="button" onClick={onClose}>
          Close history
        </button>
      </div>
    </section>
  );
}

function BaselineDetails({ baseline }: { baseline: ValidationBaseline }) {
  const rows: Array<[string, string]> = [
    ["Baseline id", baseline.id],
    ["Checksum", baseline.artifact_checksum ?? "—"],
    ["Source artifact", String(baseline.artifact_id ?? "—")],
    ["Stage", String(baseline.stage_id ?? "—")],
    ["Processing unit", String(baseline.processing_unit_id ?? "—")],
    ["Comparator", `${baseline.comparator ?? "—"} (${baseline.comparator_source ?? "unknown"})`],
    ["Semantic type", String(baseline.artifact_semantic_type ?? "—")],
    ["Recipe fingerprint", String(baseline.recipe_fingerprint ?? "—")],
    ["Contract fingerprint", String(baseline.pipeline_contract_fingerprint ?? "—")],
    ["Supersedes", String(baseline.supersedes_baseline_id ?? "—")],
  ];
  const promotion = baseline.promotion;
  if (promotion) {
    rows.push(
      ["Promoted from execution", promotion.source_execution_id],
      ["Promoted from case", promotion.source_case_id],
      ["Promoted from comparison", promotion.source_comparison_id],
      ["Candidate run", promotion.candidate_run_id],
      ["Promoted by", promotion.promoted_by ?? "—"],
    );
  }
  return (
    <div className="table-wrap">
      <table>
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label}>
              <th scope="row">{label}</th>
              <td>{value}</td>
            </tr>
          ))}
          <tr>
            <th scope="row">Comparison policy</th>
            <td>
              <code>{JSON.stringify(baseline.comparison_policy ?? {})}</code>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
