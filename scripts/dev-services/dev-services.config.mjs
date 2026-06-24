import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "..");
const tmpDir = path.join(repoRoot, "tmp", "dev-services");

function readEnvFile(filePath) {
  if (!fs.existsSync(filePath)) {
    return {};
  }
  const payload = fs.readFileSync(filePath, "utf8");
  const entries = {};
  for (const rawLine of payload.split(/\r?\n/u)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    const eqIndex = line.indexOf("=");
    if (eqIndex <= 0) {
      continue;
    }
    const key = line.slice(0, eqIndex).trim();
    const value = line.slice(eqIndex + 1).trim();
    entries[key] = value;
  }
  return entries;
}

function envPort(value, fallback) {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 && parsed <= 65535 ? parsed : fallback;
}

const backendEnv = readEnvFile(path.join(repoRoot, ".env"));
const frontendEnv = readEnvFile(path.join(repoRoot, "frontend", ".env"));
const backendExampleEnv = readEnvFile(path.join(repoRoot, ".env.example"));
const frontendExampleEnv = readEnvFile(path.join(repoRoot, "frontend", ".env.example"));

const apiPort = envPort(
  process.env.API_PORT ?? backendEnv.API_PORT ?? backendExampleEnv.API_PORT,
  8000,
);
const apiHost = process.env.API_HOST ?? backendEnv.API_HOST ?? backendExampleEnv.API_HOST ?? "127.0.0.1";
const frontendPort = envPort(
  process.env.VITE_PORT
    ?? frontendEnv.VITE_PORT
    ?? process.env.FRONTEND_PORT
    ?? backendEnv.FRONTEND_PORT
    ?? frontendExampleEnv.VITE_PORT
    ?? backendExampleEnv.FRONTEND_PORT,
  5173,
);

function pythonExecutable() {
  const venvPython = path.join(repoRoot, ".venv", "bin", "python");
  if (fs.existsSync(venvPython)) {
    return venvPython;
  }
  return "python3";
}

export function getDevServicesConfig() {
  const python = pythonExecutable();
  const platform = os.platform();
  const npmCmd = platform === "win32" ? "npm.cmd" : "npm";

  return {
    repoRoot,
    tmpDir,
    apiPort,
    frontendPort,
    services: [
      {
        key: "frontend",
        name: "Frontend",
        port: frontendPort,
        hosts: ["127.0.0.1", "localhost"],
        healthPaths: ["/studio", "/operations"],
        filter: ["vite", path.join(repoRoot, "frontend")],
        cwd: path.join(repoRoot, "frontend"),
        command: [npmCmd, "run", "dev", "--", "--host", "127.0.0.1"],
        install: {
          cwd: path.join(repoRoot, "frontend"),
          command: [npmCmd, "install"],
          requiredPath: path.join(repoRoot, "frontend", "node_modules"),
        },
      },
      {
        key: "api",
        name: "API",
        port: apiPort,
        hosts: [apiHost, "127.0.0.1", "localhost"],
        healthPaths: ["/api/health"],
        filter: ["scripts/run_api.py", "vision_3d_acquisition.api.main:app"],
        cwd: repoRoot,
        command: [python, "scripts/run_api.py"],
      },
      {
        key: "trispector_ftp",
        name: "TriSpector FTP",
        port: 2121,
        healthPaths: [],
        filter: ["vision_3d_acquisition.runtime.process_runner", "trispector_ftp_runtime"],
        cwd: repoRoot,
        command: [python, "scripts/runtime.py", "start", "trispector_ftp", "--foreground"],
      },
      {
        key: "worker_25d",
        name: "25D Worker",
        port: null,
        healthPaths: [],
        filter: ["scripts/run_25d_worker.py"],
        cwd: repoRoot,
        command: [python, "scripts/run_25d_worker.py"],
      },
      {
        key: "rgb_worker",
        name: "RGB Worker",
        port: null,
        healthPaths: [],
        filter: ["scripts/run_rgb_worker.py"],
        cwd: repoRoot,
        command: [python, "scripts/run_rgb_worker.py"],
      },
      {
        key: "fusion_publisher_worker",
        name: "Fusion Publisher",
        port: null,
        healthPaths: [],
        filter: ["scripts/run_fusion_publisher_worker.py"],
        cwd: repoRoot,
        command: [python, "scripts/run_fusion_publisher_worker.py"],
      },
    ],
  };
}
