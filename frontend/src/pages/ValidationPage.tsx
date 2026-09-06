import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type ValidationBaseline,
  type ValidationCoverageArea,
  type ValidationExecution,
  type ValidationSuiteDetail,
} from "../api/client";
import { validationStatus, type ValidationStatus } from "../components/validation/validationStatus";
import ValidationCasePanel from "../components/validation/ValidationCasePanel";
import ValidationComparisonDrawer from "../components/validation/ValidationComparisonDrawer";
import ValidationStageSummaryPanel from "../components/validation/ValidationStageSummaryPanel";
import ValidationStageTrendPanel from "../components/validation/ValidationStageTrendPanel";
import { parseValidationDeepLink } from "../validationDeepLink";

type Matrix = Awaited<ReturnType<typeof api.validationMatrix>>;

/** Read the versions an execution actually matched, as recorded at the time. */
function matchedFromExecution(execution: ValidationExecution): Record<string, string | null> {
  const matched: Record<string, string | null> = {};
  for (const entry of execution.cases) {
    matched[entry.case_id] = entry.matched_baseline_id ?? null;
  }
  return matched;
}

export default function ValidationPage() {
  const [suites, setSuites] = useState<
    Array<{ id: string; name: string; pipeline_id: string; status: string }>
  >([]);
  const [suiteId, setSuiteId] = useState("");
  const [suite, setSuite] = useState<ValidationSuiteDetail | null>(null);
  const [coverage, setCoverage] = useState<ValidationCoverageArea[]>([]);
  const [history, setHistory] = useState<Array<Record<string, unknown>>>([]);
  const [executionId, setExecutionId] = useState("");
  const [matrix, setMatrix] = useState<Matrix | null>(null);
  const [message, setMessage] = useState("");
  const [newName, setNewName] = useState("");
  const [baselines, setBaselines] = useState<ValidationBaseline[]>([]);
  const [matchedByCase, setMatchedByCase] = useState<Record<string, string | null>>({});
  const [executionDetail, setExecutionDetail] = useState<ValidationExecution | null>(null);
  const [selectedCell, setSelectedCell] = useState<{
    caseId: string;
    comparisonIds: Array<string | null>;
  } | null>(null);

  // Read once: a link from Studio or elsewhere names a suite/execution/case/comparison
  // to land on directly, rather than the usual "most recent suite, latest execution".
  const deepLink = useMemo(
    () => parseValidationDeepLink(typeof window === "undefined" ? "" : window.location.search),
    [],
  );
  const deepLinkComparisonAppliedRef = useRef(false);

  // Workspace state is parent-owned so every panel reads one consistent snapshot.
  const reloadSuite = async (id: string) => {
    if (!id) return;
    const [detail, report] = await Promise.all([
      api.validationSuite(id),
      api.validationCoverage(id),
    ]);
    setSuite(detail);
    setCoverage(report.areas);
    setBaselines(await api.validationBaselines(undefined, detail.pipeline_id));
  };

  const refresh = async () => {
    const rows = await api.validationSuites();
    setSuites(rows);
    const linkedSuiteId =
      deepLink.suite_id && rows.some((row) => row.id === deepLink.suite_id) ? deepLink.suite_id : "";
    const id = suiteId || linkedSuiteId || rows[0]?.id || "";
    setSuiteId(id);
    if (id) {
      const [detail, report, history] = await Promise.all([
        api.validationSuite(id),
        api.validationCoverage(id),
        api.validationExecutions(),
      ]);
      setSuite(detail);
      setCoverage(report.areas);
      setBaselines(await api.validationBaselines(undefined, detail.pipeline_id));
      const suiteHistory = history.filter((x) => x.suite_id === id);
      const linked = deepLink.execution_id
        ? suiteHistory.find((x) => String(x.id) === deepLink.execution_id)
        : null;
      const latest = linked ?? suiteHistory.at(-1);
      if (latest?.id) {
        const executionId = String(latest.id);
        setExecutionId(executionId);
        const [grid, execution] = await Promise.all([
          api.validationMatrix(executionId),
          api.validationExecution(executionId),
        ]);
        setMatrix(grid);
        setExecutionDetail(execution);
        setMatchedByCase(matchedFromExecution(execution));
        if (
          !deepLinkComparisonAppliedRef.current &&
          deepLink.case_id &&
          deepLink.comparison_id &&
          execution.cases.some(
            (item) =>
              item.case_id === deepLink.case_id &&
              (item.comparisons ?? []).some((c) => c.baseline_ref.baseline_id === deepLink.comparison_id),
          )
        ) {
          deepLinkComparisonAppliedRef.current = true;
          setSelectedCell({ caseId: deepLink.case_id, comparisonIds: [deepLink.comparison_id] });
        }
      }
    }
  };

  useEffect(() => {
    void refresh().catch((error) =>
      setMessage(error instanceof Error ? error.message : "Validation load failed"),
    );
  }, []);

  const selectSuite = async (id: string) => {
    setSuiteId(id);
    setExecutionId("");
    setMatrix(null);
    setMatchedByCase({});
    const [detail, report, executions] = await Promise.all([
      api.validationSuite(id),
      api.validationCoverage(id),
      api.validationSuiteExecutions(id),
    ]);
    setBaselines(await api.validationBaselines(undefined, detail.pipeline_id));
    setSuite(detail);
    setCoverage(report.areas);
    setHistory(executions);
  };

  const run = async () => {
    if (!suiteId) return;
    setMessage("Running suite…");
    const execution = await api.runValidationSuite(suiteId);
    setExecutionId(execution.id);
    setExecutionDetail(execution);
    setMatchedByCase(matchedFromExecution(execution));
    setSelectedCell(null);
    setMatrix(await api.validationMatrix(execution.id));
    setMessage(execution.status);
  };

  const create = async () => {
    if (!newName.trim()) return;
    const created = await api.createValidationSuite({
      name: newName,
      pipeline_id: "mining_steel_ball_classification_25d",
    });
    setNewName("");
    await refresh();
    await selectSuite(created.id);
  };

  const lifecycle = async (action: "duplicate" | "archive" | "restore") => {
    if (!suiteId) return;
    const updated =
      action === "duplicate"
        ? await api.duplicateValidationSuite(suiteId)
        : action === "archive"
          ? await api.archiveValidationSuite(suiteId)
          : await api.restoreValidationSuite(suiteId);
    setMessage(`${action}: ${updated.name}`);
    await refresh();
  };

  // Selecting a historical execution must not disturb suite configuration.
  const selectedCells = useMemo(
    () =>
      matrix?.rows.flatMap((row) =>
        row.cells
          .filter((cell) => cell.first_divergence)
          .map((cell) => `${row.take_id}: ${cell.artifact_id ?? cell.column_id}`),
      ) ?? [],
    [matrix],
  );

  const loadExecution = async (id: string) => {
    setExecutionId(id);
    setSelectedCell(null);
    const [grid, execution] = await Promise.all([
      api.validationMatrix(id),
      api.validationExecution(id),
    ]);
    setMatrix(grid);
    setExecutionDetail(execution);
    setMatchedByCase(matchedFromExecution(execution));
  };

  const openComparison = (caseId: string, comparisonIds: Array<string | null>) => {
    if (!comparisonIds.length) return;
    setSelectedCell({ caseId, comparisonIds });
  };

  const badge = (status: string) =>
    validationStatus[status as ValidationStatus] ?? {
      label: status,
      severity: "muted",
      icon: "?",
      tooltip: status,
    };
  return (
    <main className="page-shell">
      <section className="page-header">
        <div>
          <p className="eyebrow">Regression governance</p>
          <h1>Validation</h1>
          <p>Approved references, suite coverage, historical executions, and first divergence.</p>
        </div>
      </section>
      <section className="control-panel-section">
        <div className="control-panel-button-row">
          <label>
            Regression suite{" "}
            <select value={suiteId} onChange={(event) => void selectSuite(event.target.value)}>
              {suites.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name} · {item.status}
                </option>
              ))}
            </select>
          </label>
          <input
            value={newName}
            placeholder="New regression suite name"
            onChange={(event) => setNewName(event.target.value)}
          />
          <button type="button" onClick={() => void create()}>
            Create regression suite
          </button>
          <button
            type="button"
            disabled={!suite || suite.status === "archived"}
            onClick={() => void run()}
          >
            Run regression suite
          </button>
          <button type="button" disabled={!suite} onClick={() => void lifecycle("duplicate")}>
            Duplicate
          </button>
          {suite?.status === "archived" ? (
            <button type="button" onClick={() => void lifecycle("restore")}>
              Restore regression suite
            </button>
          ) : (
            <button type="button" disabled={!suite} onClick={() => void lifecycle("archive")}>
              Archive regression suite
            </button>
          )}
        </div>
        <small>{message}</small>
      </section>
      <ValidationCasePanel
        suite={suite}
        baselines={baselines}
        matchedByCase={matchedByCase}
        onChanged={() => reloadSuite(suiteId)}
      />
      <section className="control-panel-section">
        <h2>Coverage</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Area</th>
                <th>Comparable</th>
                <th>Missing</th>
                <th>Stale</th>
                <th>Unsupported</th>
                <th>Visual-only</th>
              </tr>
            </thead>
            <tbody>
              {coverage.map((area) => (
                <tr key={area.area}>
                  <td>{area.area}</td>
                  <td>{area.approved}</td>
                  <td>{area.missing}</td>
                  <td>{area.stale}</td>
                  <td>{area.unsupported}</td>
                  <td>{area.visual_only}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="control-panel-section">
        <h2>Execution history</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Execution</th>
                <th>Started</th>
                <th>Status</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {history.map((item) => (
                <tr key={String(item.id)}>
                  <td>{String(item.id)}</td>
                  <td>{String(item.created_at ?? "—")}</td>
                  <td>{String(item.status)}</td>
                  <td>
                    <button type="button" onClick={() => void loadExecution(String(item.id))}>
                      Open
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="control-panel-section">
        <h2>Execution analysis {executionId ? `· ${executionId}` : ""}</h2>
        {selectedCells.length ? <p>First divergence: {selectedCells.join("; ")}</p> : null}
        {!matrix ? (
          <p>Run or select a suite execution to load the backend-provided contract matrix.</p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Take</th>
                  {matrix.columns.map((column) => (
                    <th key={column.id}>{column.label}</th>
                  ))}
                  <th>Overall</th>
                </tr>
              </thead>
              <tbody>
                {matrix.rows.map((row) => (
                  <tr key={row.case_id}>
                    <td>{row.take_id}</td>
                    {row.cells.map((cell) => {
                      const presentation = badge(cell.status);
                      const openable = cell.comparison_ids.some(Boolean);
                      const content = (
                        <span
                          title={presentation.tooltip}
                          className={`status-badge ${cell.status}`}
                        >
                          {cell.first_divergence ? "◆ " : presentation.icon + " "}
                          {presentation.label}
                        </span>
                      );
                      return (
                        <td key={cell.column_id}>
                          {openable ? (
                            <button
                              type="button"
                              onClick={() => openComparison(row.case_id, cell.comparison_ids)}
                            >
                              {content}
                            </button>
                          ) : (
                            content
                          )}
                        </td>
                      );
                    })}
                    <td>
                      <strong>{badge(row.overall_status).label}</strong>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      <ValidationStageSummaryPanel executionId={executionId || null} />
      {suiteId ? <ValidationStageTrendPanel suiteId={suiteId} /> : null}
      {selectedCell && executionDetail && suite
        ? (() => {
            const executionCase = executionDetail.cases.find(
              (item) => item.case_id === selectedCell.caseId,
            );
            const comparisons = (executionCase?.comparisons ?? []).filter((comparison) =>
              selectedCell.comparisonIds.includes(comparison.baseline_ref.baseline_id),
            );
            if (!executionCase || !comparisons.length) return null;
            return (
              <ValidationComparisonDrawer
                suite={suite}
                execution={executionDetail}
                executionCase={executionCase}
                comparisons={comparisons}
                onClose={() => setSelectedCell(null)}
                onChanged={() => reloadSuite(suiteId)}
              />
            );
          })()
        : null}
    </main>
  );
}
