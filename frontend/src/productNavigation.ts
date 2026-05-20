export type ProductArea = "operations" | "studio" | "calibration" | "diagnostics" | "take";

export const PRODUCT_NAV_ITEMS: Array<{ id: Exclude<ProductArea, "take">; label: string; href: string }> = [
  { id: "operations", label: "Operations", href: "/operations" },
  { id: "studio", label: "Studio", href: "/studio" },
  { id: "calibration", label: "Calibration", href: "/calibration" },
  { id: "diagnostics", label: "Diagnostics", href: "/diagnostics" },
];

export function productAreaForPath(path: string): ProductArea {
  if (path.startsWith("/takes/")) return "take";
  if (path === "/studio" || path === "/processing-lab") return "studio";
  if (path === "/calibration") return "calibration";
  if (path === "/diagnostics" || path === "/debug") return "diagnostics";
  return "operations";
}
