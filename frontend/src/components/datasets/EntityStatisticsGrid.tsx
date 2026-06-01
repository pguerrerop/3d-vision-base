export default function EntityStatisticsGrid({ rows }: { rows: Array<{ label: string; value: string }> }) {
  return (
    <dl className="entity-stats-grid">
      {rows.map((row) => (
        <div key={row.label} className="entity-stats-row">
          <dt>{row.label}</dt>
          <dd>{row.value}</dd>
        </div>
      ))}
    </dl>
  );
}
