export type ProductArea = "operations" | "studio" | "superclass_histograms" | "runtime" | "calibration" | "diagnostics" | "take";

export const PRODUCT_NAV_ITEMS: Array<{ id: Exclude<ProductArea, "take">; label: string; href: string }> = [
  { id: "operations", label: "Operations", href: "/operations" },
  { id: "studio", label: "Studio", href: "/studio" },
  { id: "superclass_histograms", label: "Superclass Hist", href: "/superclass-histograms" },
  { id: "runtime", label: "Runtime", href: "/runtime" },
  { id: "calibration", label: "Calibration", href: "/calibration" },
  { id: "diagnostics", label: "Diagnostics", href: "/diagnostics" },
];

export function productAreaForPath(path: string): ProductArea {
  if (path.startsWith("/takes/")) return "take";
  if (path === "/operator" || path === "/operator/inspection") return "operations";
  if (path === "/operations" || path.startsWith("/operations/")) return "operations";
  if (path === "/studio" || path === "/processing-lab") return "studio";
  if (path === "/superclass-histograms") return "superclass_histograms";
  if (path === "/runtime") return "runtime";
  if (path === "/calibration") return "calibration";
  if (path === "/diagnostics" || path === "/debug") return "diagnostics";
  return "operations";
}
