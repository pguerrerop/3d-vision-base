type FieldMeta = {
  group?: string;
  visible_when?: Record<string, unknown>;
};

export function visibleStageParameterKeys(
  fields: Record<string, FieldMeta>,
  values: Record<string, unknown>,
  showAdvanced: boolean
): string[] {
  return Object.keys(fields).filter((key) => {
    const meta = fields[key] ?? {};
    if (meta.group === "advanced" && !showAdvanced) return false;
    if (!meta.visible_when) return true;
    return Object.entries(meta.visible_when).every(([dep, expected]) => values?.[dep] === expected);
  });
}
