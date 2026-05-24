export type StudioDebugMode = "operator" | "engineering";

export function isEngineeringMode(mode: StudioDebugMode): boolean {
  return mode === "engineering";
}
