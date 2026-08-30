import { useMemo, useState } from "react";
import {
  api,
  type ValidationBaseline,
  type ValidationResolutionMode,
  type ValidationSuiteCase,
  type ValidationSuiteDetail,
} from "../../api/client";
import {
  buildCaseRequest,
  buildCaseUpdate,
  canMutateCases,
  caseRemovalWarning,
  compatibleBaselines,
  describeApiError,
  describeResolution,
  draftFromCase,
  draftProblems,
  parseTags,
  type ValidationCaseDraft,
} from "./validationCaseModel";
import ValidationBaselineHistoryDrawer from "./ValidationBaselineHistoryDrawer";

const EMPTY_DRAFT: ValidationCaseDraft = {
  take_id: "",
  baseline_ids: [],
  mode: "active",
  pinned_baseline_id: "",
  allowed_baseline_ids: [],
  enabled: true,
  tags: [],
  notes: "",
};

const MODES: Array<{ value: ValidationResolutionMode; label: string; hint: string }> = [
  {
    value: "active",
    label: "Active",
    hint: "Always compares against whichever version is active.",
  },
  {
    value: "pinned",
    label: "Pinned",
    hint: "Always compares against one chosen version, active or not.",
  },
  {
    value: "allowed_versions",
    label: "Allowed versions",
    hint: "Passes when the candidate matches any one of the chosen versions.",
  },
];

const describeBaseline = (baseline: ValidationBaseline): string => {
  const parts = [`v${baseline.version}`, baseline.status === "active" ? "active" : "inactive"];
  if (baseline.integrity_state === "invalid") parts.push("integrity failed");
  if (baseline.artifact_id) parts.push(String(baseline.artifact_id));
  parts.push(`from ${baseline.source_run_id}`);
  return parts.join(" · ");
};

interface Props {
  suite: ValidationSuiteDetail | null;
  baselines: ValidationBaseline[];
  /** Which allowed version each case matched in the selected execution, if any. */
  matchedByCase?: Record<string, string | null>;
  onChanged: () => Promise<void>;
}

export default function ValidationCasePanel({
  suite,
  baselines,
  matchedByCase = {},
  onChanged,
}: Props) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState<ValidationCaseDraft | null>(null);
  const [tagsText, setTagsText] = useState("");
  const [confirming, setConfirming] = useState<ValidationSuiteCase | null>(null);
  const [historyFor, setHistoryFor] = useState<ValidationSuiteCase | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const editable = canMutateCases(suite?.status);
  const pipelineId = suite?.pipeline_id ?? "";

  const takes = useMemo(
    () =>
      [
        ...new Set(baselines.filter((b) => b.pipeline_id === pipelineId).map((b) => b.take_id)),
      ].sort(),
    [baselines, pipelineId],
  );

  const selectable = useMemo(
    () =>
      draft
        ? compatibleBaselines(
            baselines,
            { take_id: draft.take_id, baseline_ids: draft.baseline_ids },
            pipelineId,
          )
        : [],
    [baselines, draft, pipelineId],
  );

  const problems = draft ? draftProblems(draft) : [];

  const openEditor = (target: ValidationSuiteCase | "new") => {
    setError("");
    if (target === "new") {
      const take = takes[0] ?? "";
      setEditing("new");
      setDraft({ ...EMPTY_DRAFT, take_id: take });
      setTagsText("");
      return;
    }
    setEditing(target.id);
    setDraft(draftFromCase(target));
    setTagsText(target.tags.join(", "));
  };

  const closeEditor = () => {
    setEditing(null);
    setDraft(null);
    setError("");
  };

  const guarded = async (action: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await action();
      await onChanged();
    } catch (caught: unknown) {
      setError(describeApiError(caught));
    } finally {
      setBusy(false);
    }
  };

  const save = () =>
    guarded(async () => {
      if (!suite || !draft) return;
      if (editing === "new") {
        await api.addValidationCase(suite.id, buildCaseRequest(draft));
      } else {
        const original = suite.cases.find((item) => item.id === editing);
        if (!original) return;
        const update = buildCaseUpdate(draft, original);
        if (Object.keys(update).length)
          await api.updateValidationCase(suite.id, original.id, update);
      }
      closeEditor();
    });

  const toggleEnabled = (item: ValidationSuiteCase) =>
    guarded(async () => {
      if (!suite) return;
      await api.updateValidationCase(suite.id, item.id, { enabled: !item.enabled });
    });

  const remove = (item: ValidationSuiteCase) =>
    guarded(async () => {
      if (!suite) return;
      await api.deleteValidationCase(suite.id, item.id);
      setConfirming(null);
      if (editing === item.id) closeEditor();
    });

  const patchDraft = (change: Partial<ValidationCaseDraft>) =>
    setDraft((current) => (current ? { ...current, ...change } : current));

  const toggleIn = (list: string[], id: string): string[] =>
    list.includes(id) ? list.filter((other) => other !== id) : [...list, id];

  return (
    <section className="control-panel-section">
      <h2>Cases</h2>
      {!editable ? (
        <p>Archived suites are read-only. Restore the suite to edit its cases.</p>
      ) : null}
      {error ? <p role="alert">{error}</p> : null}

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Take</th>
              <th>Enabled</th>
              <th>Resolution</th>
              <th>Tags</th>
              <th>Notes</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {suite?.cases.map((item) => {
              const summary = describeResolution(item, baselines, matchedByCase[item.id]);
              return (
                <tr key={item.id}>
                  <td>{item.take_id}</td>
                  <td>{item.enabled ? "Enabled" : "Disabled"}</td>
                  <td>
                    <span className={`status-badge ${summary.severity}`} title={summary.detail}>
                      {summary.label}
                    </span>
                  </td>
                  <td>{item.tags.join(", ") || "—"}</td>
                  <td>{item.notes || "—"}</td>
                  <td>
                    <div className="control-panel-button-row">
                      <button
                        type="button"
                        disabled={!editable || busy}
                        onClick={() => openEditor(item)}
                      >
                        Edit
                      </button>
                      <button
                        type="button"
                        disabled={!editable || busy}
                        onClick={() => void toggleEnabled(item)}
                      >
                        {item.enabled ? "Disable" : "Enable"}
                      </button>
                      <button type="button" onClick={() => setHistoryFor(item)}>
                        History
                      </button>
                      <button
                        type="button"
                        disabled={!editable || busy}
                        onClick={() => setConfirming(item)}
                      >
                        Remove
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
            {!suite?.cases.length ? (
              <tr>
                <td colSpan={6}>No cases yet. Add one to start protecting a take.</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      <div className="control-panel-button-row">
        <button
          type="button"
          disabled={!editable || busy || !takes.length}
          onClick={() => openEditor("new")}
        >
          Add case
        </button>
        {!takes.length ? <small>No approved references exist for this pipeline yet.</small> : null}
      </div>

      {historyFor && suite ? (
        <ValidationBaselineHistoryDrawer
          caseItem={suite.cases.find((item) => item.id === historyFor.id) ?? historyFor}
          suiteId={suite.id}
          pipelineId={suite.pipeline_id}
          editable={editable}
          onClose={() => setHistoryFor(null)}
          onChanged={onChanged}
        />
      ) : null}

      {confirming ? (
        <div className="control-panel-section" role="alertdialog" aria-label="Confirm case removal">
          <p>{caseRemovalWarning(confirming)}</p>
          <div className="control-panel-button-row">
            <button type="button" disabled={busy} onClick={() => void remove(confirming)}>
              Remove from suite
            </button>
            <button type="button" onClick={() => setConfirming(null)}>
              Keep it
            </button>
          </div>
        </div>
      ) : null}

      {draft ? (
        <div className="control-panel-section">
          <h3>{editing === "new" ? "Add case" : "Edit case"}</h3>

          <div className="control-panel-form-grid">
            <label>
              Take
              <select
                value={draft.take_id}
                disabled={editing !== "new"}
                onChange={(event) =>
                  patchDraft({
                    take_id: event.target.value,
                    baseline_ids: [],
                    allowed_baseline_ids: [],
                    pinned_baseline_id: "",
                  })
                }
              >
                {takes.map((take) => (
                  <option key={take} value={take}>
                    {take}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Tags
              <input
                value={tagsText}
                placeholder="nightly, smoke"
                onChange={(event) => {
                  setTagsText(event.target.value);
                  patchDraft({ tags: parseTags(event.target.value) });
                }}
              />
            </label>

            <label>
              Notes
              <input
                value={draft.notes}
                onChange={(event) => patchDraft({ notes: event.target.value })}
              />
            </label>

            <label className="checkbox-field">
              <input
                type="checkbox"
                checked={draft.enabled}
                onChange={(event) => patchDraft({ enabled: event.target.checked })}
              />
              Enabled
            </label>
          </div>

          <h4>Approved references</h4>
          <p>
            <small>
              Which approved artifacts this case compares. Only references from the selected take
              are offered.
            </small>
          </p>
          {selectable.map((baseline) => (
            <label key={baseline.id} className="checkbox-field">
              <input
                type="checkbox"
                checked={draft.baseline_ids.includes(baseline.id)}
                onChange={() =>
                  patchDraft({ baseline_ids: toggleIn(draft.baseline_ids, baseline.id) })
                }
              />
              {describeBaseline(baseline)}
            </label>
          ))}

          <h4>Baseline resolution</h4>
          {MODES.map((mode) => (
            <label key={mode.value} className="checkbox-field" title={mode.hint}>
              <input
                type="radio"
                name="resolution-mode"
                checked={draft.mode === mode.value}
                onChange={() => patchDraft({ mode: mode.value })}
              />
              {mode.label} — {mode.hint}
            </label>
          ))}

          {draft.mode === "pinned" ? (
            <label>
              Pin to version
              <select
                value={draft.pinned_baseline_id}
                onChange={(event) => patchDraft({ pinned_baseline_id: event.target.value })}
              >
                <option value="">Choose a version…</option>
                {selectable.map((baseline) => (
                  <option key={baseline.id} value={baseline.id}>
                    {describeBaseline(baseline)}
                  </option>
                ))}
              </select>
            </label>
          ) : null}

          {draft.mode === "pinned" &&
          selectable.find((b) => b.id === draft.pinned_baseline_id)?.status === "inactive" ? (
            <p role="status">
              This version is inactive. The case will stay pinned to it and will not fall back to
              the active version.
            </p>
          ) : null}

          {draft.mode === "allowed_versions"
            ? selectable.map((baseline) => (
                <label key={baseline.id} className="checkbox-field">
                  <input
                    type="checkbox"
                    checked={draft.allowed_baseline_ids.includes(baseline.id)}
                    onChange={() =>
                      patchDraft({
                        allowed_baseline_ids: toggleIn(draft.allowed_baseline_ids, baseline.id),
                      })
                    }
                  />
                  {describeBaseline(baseline)}
                </label>
              ))
            : null}

          {problems.length ? (
            <ul role="alert">
              {problems.map((problem) => (
                <li key={problem}>{problem}</li>
              ))}
            </ul>
          ) : null}

          <div className="control-panel-button-row">
            <button
              type="button"
              disabled={busy || problems.length > 0}
              onClick={() => void save()}
            >
              {editing === "new" ? "Add case" : "Save changes"}
            </button>
            <button type="button" onClick={closeEditor}>
              Cancel
            </button>
          </div>
        </div>
      ) : null}
    </section>
  );
}
