export type StageParameterFieldMeta = {
  group?: string;
  advanced?: boolean;
  visible_when?: Record<string, unknown>;
};

export function isStageParameterVisible(
  meta: StageParameterFieldMeta,
  values: Record<string, unknown>,
  showAdvanced: boolean,
): boolean {
  if ((meta.group === "advanced" || meta.advanced === true) && !showAdvanced) return false;
  if (!meta.visible_when) return true;
  return Object.entries(meta.visible_when).every(([dep, expected]) => values?.[dep] === expected);
}

export function visibleStageParameterKeys(
  fields: Record<string, StageParameterFieldMeta>,
  values: Record<string, unknown>,
  showAdvanced: boolean
): string[] {
  return Object.keys(fields).filter((key) => {
    const meta = fields[key] ?? {};
    return isStageParameterVisible(meta, values, showAdvanced);
  });
}
