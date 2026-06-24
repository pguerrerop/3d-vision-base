import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

function formatCommand(command) {
  return command.map((part) => (/\s/u.test(part) ? JSON.stringify(part) : part)).join(" ");
}

function runForeground(command, cwd) {
  return new Promise((resolve, reject) => {
    const child = spawn(command[0], command.slice(1), {
      cwd,
      stdio: "inherit",
      env: process.env,
    });
    child.once("error", reject);
    child.once("exit", (code) => resolve(code ?? 0));
  });
}

export async function ensureInstall(service, { skipInstall = false, yes = false } = {}) {
  if (!service.install || fs.existsSync(service.install.requiredPath)) {
    return { changed: false, skipped: false };
  }
  if (skipInstall) {
    return { changed: false, skipped: true, reason: `missing ${path.relative(service.cwd, service.install.requiredPath)}` };
  }
  if (!yes && !process.stdin.isTTY) {
    throw new Error(`Cannot install dependencies for ${service.key} without --yes in non-interactive mode.`);
  }
  if (process.stdin.isTTY && !yes) {
    process.stdout.write(`Install dependencies for ${service.name}? [Y/n] `);
    const answer = await new Promise((resolve) => {
      process.stdin.resume();
      process.stdin.once("data", (chunk) => resolve(String(chunk).trim().toLowerCase()));
    });
    if (answer === "n" || answer === "no") {
      return { changed: false, skipped: true, reason: "declined" };
    }
  }
  const exitCode = await runForeground(service.install.command, service.install.cwd);
  if (exitCode !== 0) {
    throw new Error(`Install failed for ${service.key} with exit code ${exitCode}.`);
  }
  return { changed: true, skipped: false };
}

export function metadataPath(tmpDir, serviceKey) {
  return path.join(tmpDir, `${serviceKey}.json`);
}

export function logPath(tmpDir, serviceKey) {
  return path.join(tmpDir, `${serviceKey}.log`);
}

export async function startDetachedService(service, tmpDir) {
  fs.mkdirSync(tmpDir, { recursive: true });
  const logFile = logPath(tmpDir, service.key);
  const out = fs.openSync(logFile, "a");
  const err = fs.openSync(logFile, "a");
  const child = spawn(service.command[0], service.command.slice(1), {
    cwd: service.cwd,
    env: process.env,
    detached: true,
    stdio: ["ignore", out, err],
  });
  child.unref();
  fs.writeFileSync(
    metadataPath(tmpDir, service.key),
    JSON.stringify(
      {
        key: service.key,
        pid: child.pid,
        logPath: logFile,
        cwd: service.cwd,
        command: service.command,
        commandText: formatCommand(service.command),
        startedAt: new Date().toISOString(),
      },
      null,
      2,
    ) + "\n",
    "utf8",
  );
  return { pid: child.pid, logPath: logFile };
}

export function readServiceMetadata(tmpDir, serviceKey) {
  const filePath = metadataPath(tmpDir, serviceKey);
  if (!fs.existsSync(filePath)) {
    return null;
  }
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}
