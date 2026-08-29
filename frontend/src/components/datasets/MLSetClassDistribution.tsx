import type { MLSetSummaryResponse } from "../../api/client";

function totalOf(values: Record<string, number>): number {
  return Object.values(values).reduce((sum, value) => sum + Number(value || 0), 0);
}

export default function MLSetClassDistribution({
  classDistribution,
  superclassDistribution,
  onSelectClass,
  onSelectSuperclass,
}: {
  classDistribution: MLSetSummaryResponse["class_distribution"];
  superclassDistribution: MLSetSummaryResponse["superclass_distribution"];
  onSelectClass?: (value: string) => void;
  onSelectSuperclass?: (value: string) => void;
}) {
  const total = totalOf(classDistribution);
  return (
    <section className="entity-section">
      <h4>Class Distribution</h4>
      <div className="ml-set-distribution-grid">
        <div>
          <strong>Normalized classes</strong>
          <table className="datasets-table compact-table">
            <thead><tr><th>class</th><th>count</th><th>%</th></tr></thead>
            <tbody>
              {Object.entries(classDistribution).map(([label, count]) => (
                <tr key={label}><td>{onSelectClass ? <button type="button" className="link-button" onClick={() => onSelectClass(label)}>{label}</button> : label}</td><td>{count}</td><td>{total ? `${Math.round((count / total) * 100)}%` : "0%"}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <strong>Superclasses</strong>
          <table className="datasets-table compact-table">
            <thead><tr><th>superclass</th><th>count</th></tr></thead>
            <tbody>
              {Object.entries(superclassDistribution).map(([label, count]) => (
                <tr key={label}><td>{onSelectSuperclass ? <button type="button" className="link-button" onClick={() => onSelectSuperclass(label)}>{label}</button> : label}</td><td>{count}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
