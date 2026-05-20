import type { ActiveCalibration, ProcessingResult } from "../api/client";
import { calibrationStateLabel } from "./calibrationStateModel";

type Props = {
  active?: ActiveCalibration | null;
  result?: ProcessingResult | null;
};

export default function CalibrationStateBadge({ active, result }: Props) {
  const status = active?.status ?? (result?.calibration_active ? "valid" : "none");
  return <span className={`calibration-state-badge calibration-state-${status}`}>{calibrationStateLabel(active, result)}</span>;
}
