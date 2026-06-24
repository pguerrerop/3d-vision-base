import type { MLSetSummaryResponse } from "../../api/client";

export default function MLSetReadinessCards({ readiness }: { readiness: MLSetSummaryResponse["readiness"] }) {
  const leakageLabel = readiness.split_status === "unassigned" && readiness.leakage_risk === "NONE"
    ? "ready / no split assigned yet"
    : readiness.leakage_risk;
  const cards = [
    ["Takes", readiness.take_count],
    ["Objects", readiness.physical_object_count],
    ["Default Trainable", readiness.trainable_object_count],
    ["Review", readiness.review_required_count],
    ["Excluded", readiness.excluded_count],
    ["Leakage", leakageLabel],
    ["Split", readiness.split_status],
  ];
  return (
    <div className="entity-summary-cards ml-set-summary-cards">
      {cards.map(([label, value]) => (
        <article key={String(label)}>
          <small>{label}</small>
          <strong>{String(value)}</strong>
        </article>
      ))}
    </div>
  );
}
