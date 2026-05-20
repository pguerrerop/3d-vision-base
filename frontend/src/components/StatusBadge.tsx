import type { Decision, TakeStatus } from "../api/client";

type Props = {
  value: Decision | TakeStatus | string | null | undefined;
};

export default function StatusBadge({ value }: Props) {
  const label = value ?? "unknown";
  return <span className={`status-badge status-${label}`}>{label}</span>;
}
