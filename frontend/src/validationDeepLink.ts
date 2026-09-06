export type ValidationDeepLinkParams = {
  suite_id?: string | null;
  execution_id?: string | null;
  case_id?: string | null;
  comparison_id?: string | null;
};

function cleaned(value: string | null | undefined): string | null {
  const next = String(value ?? "").trim();
  return next ? next : null;
}

export function buildValidationDeepLink(params: ValidationDeepLinkParams): string {
  const qs = new URLSearchParams();
  const ordered: Array<keyof ValidationDeepLinkParams> = [
    "suite_id",
    "execution_id",
    "case_id",
    "comparison_id",
  ];
  for (const key of ordered) {
    const value = cleaned(params[key]);
    if (value) qs.set(key, value);
  }
  return `/validation${qs.toString() ? `?${qs.toString()}` : ""}`;
}

export function parseValidationDeepLink(search: string): ValidationDeepLinkParams {
  const params = new URLSearchParams(search.startsWith("?") ? search : `?${search}`);
  return {
    suite_id: cleaned(params.get("suite_id")),
    execution_id: cleaned(params.get("execution_id")),
    case_id: cleaned(params.get("case_id")),
    comparison_id: cleaned(params.get("comparison_id")),
  };
}
