import type { ProcessingResult } from "../api/client";

type Props = {
  result: ProcessingResult | null | undefined;
};

export default function CalibrationStatusBadge({ result }: Props) {
  const mode = result?.plane_mode ?? (result?.calibration_id ? "calibrated" : "auto");
  const label = mode === "calibrated" ? "Calibrated" : mode === "disabled" ? "Invalid calibration" : "Automatic";
  return <span className={`calibration-status-badge calibration-status-${mode}`}>{label}</span>;
}
