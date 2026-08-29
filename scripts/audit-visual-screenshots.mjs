#!/usr/bin/env node
/**
 * Captures screenshots of all Sensor Studio UI views for visual/functional audit.
 * Usage: node scripts/audit-visual-screenshots.mjs [--base-url http://127.0.0.1:5074]
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");

const args = process.argv.slice(2);
const baseUrlArg = args.find((a) => a.startsWith("--base-url="))?.split("=")[1];
const baseUrl = baseUrlArg ?? process.env.AUDIT_BASE_URL ?? "http://127.0.0.1:5074";
const apiUrl = process.env.AUDIT_API_URL ?? baseUrl.replace(/:\d+$/, ":8380");

const outDir = path.join(repoRoot, "tmp", "audit-screenshots");
const pdfPath = path.join(repoRoot, "docs", "sensor-studio-visual-audit.pdf");

const VIEWPORT = { width: 1600, height: 1000 };

async function fetchSampleTakeId() {
  try {
    const res = await fetch(`${apiUrl}/api/takes`);
    if (!res.ok) return "2026-05-29T224708_151";
    const takes = await res.json();
    const processed = takes.find((t) => t.status === "processed" && (t.has_done || t.has_ready));
    return processed?.take_id ?? takes[0]?.take_id ?? "2026-05-29T224708_151";
  } catch {
    return "2026-05-29T224708_151";
  }
}

function slugify(label) {
  return label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

/** @type {Array<{ id: string; title: string; path: string; waitMs?: number; interact?: (page: import('playwright').Page) => Promise<void> }>} */
function buildTargets(takeId) {
  const encTake = encodeURIComponent(takeId);
  return [
    { id: "01-operations", title: "Operations — 25D Runtime Monitoring", path: "/operations" },
    { id: "02-operator-alias", title: "Operations (alias /operator)", path: "/operator" },
    { id: "03-operator-inspection-alias", title: "Operations (alias /operator/inspection)", path: "/operator/inspection" },
    { id: "04-studio-empty", title: "Studio — default landing", path: "/studio", waitMs: 2500 },
    {
      id: "05-studio-take-3d",
      title: `Studio — take loaded (3D pipeline) · ${takeId}`,
      path: `/studio?take_id=${encTake}&pipeline_id=3d_ball_inspection`,
      waitMs: 5000,
    },
    {
      id: "06-studio-take-25d",
      title: `Studio — take loaded (25D pipeline) · ${takeId}`,
      path: `/studio?take_id=${encTake}&pipeline_id=mining_steel_ball_classification_25d`,
      waitMs: 5000,
    },
    {
      id: "07-studio-graph-workspace",
      title: `Studio — graph workspace · ${takeId}`,
      path: `/studio?take_id=${encTake}&pipeline_id=mining_steel_ball_classification_25d&graph_workspace=1`,
      waitMs: 5000,
    },
    {
      id: "08-studio-report",
      title: `Studio report / walkthrough · ${takeId}`,
      path: `/studio/report?take_id=${encTake}&pipeline_id=3d_ball_inspection`,
      waitMs: 4000,
    },
    { id: "09-processing-lab-alias", title: "Studio (alias /processing-lab)", path: "/processing-lab", waitMs: 2500 },
    { id: "10-validation", title: "Validation — regression governance", path: "/validation", waitMs: 3000 },
    { id: "11-datasets-overview", title: "Datasets — Overview tab", path: "/datasets", waitMs: 3500 },
    {
      id: "12-datasets-takes",
      title: "Datasets — Takes tab",
      path: "/datasets",
      waitMs: 2000,
      interact: async (page) => {
        await page.locator('.datasets-tabs button:has-text("Takes")').click();
        await page.waitForTimeout(1500);
      },
    },
    {
      id: "13-datasets-physical-objects",
      title: "Datasets — Physical Objects tab",
      path: "/datasets",
      interact: async (page) => {
        await page.locator('.datasets-tabs button:has-text("Physical Objects")').click();
        await page.waitForTimeout(1500);
      },
    },
    {
      id: "14-datasets-objects",
      title: "Datasets — Objects tab",
      path: "/datasets",
      interact: async (page) => {
        await page.getByRole("button", { name: "Objects", exact: true }).click();
        await page.waitForTimeout(1500);
      },
    },
    {
      id: "15-datasets-labels",
      title: "Datasets — Labels tab",
      path: "/datasets",
      interact: async (page) => {
        await page.locator('.datasets-tabs button:has-text("Labels")').click();
        await page.waitForTimeout(1500);
      },
    },
    {
      id: "16-datasets-ml-sets",
      title: "Datasets — ML Sets tab",
      path: "/datasets",
      interact: async (page) => {
        await page.locator('.datasets-tabs button:has-text("ML Sets")').click();
        await page.waitForTimeout(1500);
      },
    },
    {
      id: "17-datasets-splits",
      title: "Datasets — Splits tab",
      path: "/datasets",
      interact: async (page) => {
        await page.locator('.datasets-tabs button:has-text("Splits")').click();
        await page.waitForTimeout(1500);
      },
    },
    { id: "18-classifiers", title: "Classifiers", path: "/classifiers", waitMs: 2500 },
    { id: "19-feature-analytics", title: "Feature Analytics", path: "/feature-analytics", waitMs: 3500 },
    { id: "20-superclass-histograms", title: "Superclass Feature Histograms", path: "/superclass-histograms", waitMs: 3000 },
    { id: "21-runtime", title: "Runtime", path: "/runtime", waitMs: 2500 },
    { id: "22-calibration", title: "Calibration — source alignment", path: "/calibration", waitMs: 2500 },
    { id: "23-diagnostics", title: "Diagnostics — runtime health console", path: "/diagnostics", waitMs: 2500 },
    { id: "24-debug-alias", title: "Diagnostics (alias /debug)", path: "/debug", waitMs: 2500 },
    {
      id: "25-take-detail",
      title: `Take detail · ${takeId}`,
      path: `/takes/${encTake}`,
      waitMs: 4000,
    },
  ];
}

async function captureTarget(page, target) {
  const url = `${baseUrl}${target.path}`;
  await page.goto(url, { waitUntil: "load", timeout: 120_000 });
  if (target.interact) {
    await target.interact(page);
  }
  await page.waitForTimeout(target.waitMs ?? 1500);
  await page.waitForLoadState("load", { timeout: 15_000 }).catch(() => {});
  const file = path.join(outDir, `${target.id}.png`);
  await page.screenshot({ path: file, fullPage: true });
  return { ...target, file, url };
}

function buildHtml(captures, generatedAt, takeId) {
  const sections = captures
    .map(
      (c, idx) => `
    <section class="audit-page">
      <header>
        <p class="index">${String(idx + 1).padStart(2, "0")} / ${String(captures.length).padStart(2, "0")}</p>
        <h1>${c.title}</h1>
        <p class="meta"><strong>URL:</strong> <code>${c.url}</code></p>
        <p class="meta"><strong>Archivo:</strong> <code>${path.basename(c.file)}</code></p>
      </header>
      <figure>
        <img src="file://${c.file}" alt="${c.title.replace(/"/g, "&quot;")}" />
      </figure>
    </section>`,
    )
    .join("\n");

  return `<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>Sensor Studio — Auditoría visual</title>
  <style>
    @page { size: A4 landscape; margin: 12mm; }
    * { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111; margin: 0; background: #f5f5f5; }
    .cover { min-height: 100vh; display: flex; flex-direction: column; justify-content: center; padding: 48px; page-break-after: always; background: linear-gradient(135deg, #0f172a, #1e293b); color: #f8fafc; }
    .cover h1 { font-size: 36px; margin: 0 0 12px; }
    .cover p { margin: 6px 0; opacity: 0.9; }
    .audit-page { page-break-after: always; background: white; padding: 18px 22px 28px; min-height: 100vh; }
    header { border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; margin-bottom: 14px; }
    .index { font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; margin: 0 0 6px; }
    h1 { font-size: 22px; margin: 0 0 8px; }
    .meta { font-size: 12px; color: #475569; margin: 2px 0; word-break: break-all; }
    code { background: #f1f5f9; padding: 1px 4px; border-radius: 4px; font-size: 11px; }
    figure { margin: 0; }
    img { width: 100%; height: auto; border: 1px solid #cbd5e1; border-radius: 6px; display: block; }
    .toc { page-break-after: always; background: white; padding: 36px 42px; }
    .toc h2 { margin-top: 0; }
    .toc ol { columns: 2; column-gap: 32px; padding-left: 20px; }
    .toc li { margin: 6px 0; font-size: 13px; }
  </style>
</head>
<body>
  <section class="cover">
    <p>DevAI 3D Acquisition · Sensor Studio</p>
    <h1>Auditoría visual / funcional</h1>
    <p>Generado: ${generatedAt}</p>
    <p>Base URL: ${baseUrl}</p>
    <p>Take de referencia: ${takeId}</p>
    <p>${captures.length} vistas capturadas</p>
  </section>
  <section class="toc">
    <h2>Índice de vistas</h2>
    <ol>
      ${captures.map((c) => `<li>${c.title}</li>`).join("\n      ")}
    </ol>
  </section>
  ${sections}
</body>
</html>`;
}

async function main() {
  fs.mkdirSync(outDir, { recursive: true });
  fs.mkdirSync(path.dirname(pdfPath), { recursive: true });

  const takeId = await fetchSampleTakeId();
  const targets = buildTargets(takeId);
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: VIEWPORT, deviceScaleFactor: 1 });
  const page = await context.newPage();

  const captures = [];
  console.log(`Capturing ${targets.length} views from ${baseUrl} (take: ${takeId})`);

  for (const target of targets) {
    process.stdout.write(`  · ${target.title}... `);
    try {
      const capture = await captureTarget(page, target);
      captures.push(capture);
      console.log("ok");
    } catch (error) {
      console.log("FAILED");
      console.error(`    ${error instanceof Error ? error.message : error}`);
    }
  }

  await browser.close();

  const generatedAt = new Date().toLocaleString("es-CL", { dateStyle: "full", timeStyle: "short" });
  const htmlPath = path.join(outDir, "audit-report.html");
  fs.writeFileSync(htmlPath, buildHtml(captures, generatedAt, takeId), "utf8");

  const pdfBrowser = await chromium.launch({ headless: true });
  const pdfPage = await pdfBrowser.newPage();
  await pdfPage.goto(`file://${htmlPath}`, { waitUntil: "load" });
  await pdfPage.pdf({
    path: pdfPath,
    format: "A4",
    landscape: true,
    printBackground: true,
    margin: { top: "10mm", right: "10mm", bottom: "10mm", left: "10mm" },
  });
  await pdfBrowser.close();

  const manifestPath = path.join(outDir, "manifest.json");
  fs.writeFileSync(
    manifestPath,
    JSON.stringify({ generatedAt, baseUrl, takeId, pdfPath, captures: captures.map(({ id, title, url, file }) => ({ id, title, url, file })) }, null, 2),
    "utf8",
  );

  console.log(`\nDone.`);
  console.log(`  PDF: ${pdfPath}`);
  console.log(`  PNGs: ${outDir}`);
  console.log(`  HTML: ${htmlPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
