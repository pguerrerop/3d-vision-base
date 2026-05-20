import type { ProcessingResult } from "../api/client";
import ProcessingModeBadge from "./ProcessingModeBadge";

type Props = {
  result: ProcessingResult | null;
};

export default function DecisionCard({ result }: Props) {
  const decision = result?.summary.decision ?? "review";
  const label = decision.toUpperCase();
  const confidenceLabel =
    result?.summary.confidence == null ? "Classification confidence not available" : `${Math.round(result.summary.confidence * 100)}% confidence`;
  return (
    <section className={`decision-card decision-${decision}`}>
      <ProcessingModeBadge mode={result?.processing_mode} />
      <div className="decision-label">Latest decision</div>
      <div className="decision-value">{label}</div>
      <div className="decision-subtext">
        {result?.processing_mode === "mock"
          ? `${confidenceLabel} from demonstration data`
          : result?.algorithm_stage === "segmentation"
            ? "Segmentation pipeline active; classification pending"
            : confidenceLabel}
      </div>
    </section>
  );
}
