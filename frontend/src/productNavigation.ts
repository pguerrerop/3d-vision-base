export type ProductArea = "operations" | "studio" | "validation" | "datasets" | "classifiers" | "feature_analytics" | "superclass_histograms" | "runtime" | "calibration" | "diagnostics" | "take";

export const PRODUCT_NAV_ITEMS: Array<{ id: Exclude<ProductArea, "take">; label: string; href: string }> = [
  { id: "operations", label: "Operations", href: "/operations" },
  { id: "studio", label: "Studio", href: "/studio" },
  { id: "validation", label: "Validation", href: "/validation" },
  { id: "datasets", label: "Datasets", href: "/datasets" },
  { id: "classifiers", label: "Classifiers", href: "/classifiers" },
  { id: "feature_analytics", label: "Feature Analytics", href: "/feature-analytics" },
  { id: "runtime", label: "Runtime", href: "/runtime" },
  { id: "calibration", label: "Calibration", href: "/calibration" },
  { id: "diagnostics", label: "Diagnostics", href: "/diagnostics" },
];

export function productAreaForPath(path: string): ProductArea {
  if (path.startsWith("/takes/")) return "take";
  if (path === "/operator" || path === "/operator/inspection") return "operations";
  if (path === "/operations" || path.startsWith("/operations/")) return "operations";
  if (path === "/studio" || path === "/processing-lab") return "studio";
  if (path === "/validation" || path.startsWith("/validation/")) return "validation";
  if (path === "/datasets" || path.startsWith("/datasets/")) return "datasets";
  if (path === "/classifiers") return "classifiers";
  if (path === "/feature-analytics") return "feature_analytics";
  if (path === "/superclass-histograms") return "superclass_histograms";
  if (path === "/runtime") return "runtime";
  if (path === "/calibration") return "calibration";
  if (path === "/diagnostics" || path === "/debug") return "diagnostics";
  return "operations";
}
