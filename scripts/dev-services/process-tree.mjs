import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

async function safeExec(file, args) {
  try {
    return await execFileAsync(file, args, { encoding: "utf8", maxBuffer: 1024 * 1024 * 8 });
  } catch (error) {
    if (typeof error?.stdout === "string" || typeof error?.stderr === "string") {
      return { stdout: error.stdout ?? "", stderr: error.stderr ?? "" };
    }
    throw error;
  }
}

export async function listProcesses() {
  let stdout = "";
  try {
    ({ stdout } = await safeExec("ps", ["-axo", "pid=,ppid=,command="]));
  } catch {
    return [];
  }
  return stdout
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const match = line.match(/^(\d+)\s+(\d+)\s+(.*)$/u);
      if (!match) {
        return null;
      }
      return {
        pid: Number(match[1]),
        ppid: Number(match[2]),
        command: match[3],
      };
    })
    .filter(Boolean);
}

export function matchServiceProcesses(service, processes) {
  const filters = Array.isArray(service.filter) ? service.filter : [service.filter].filter(Boolean);
  return processes.filter((processInfo) => filters.every((token) => processInfo.command.includes(token)));
}

export function resolveDescendantPids(rootPid, processes) {
  const childrenByParent = new Map();
  for (const processInfo of processes) {
    const siblings = childrenByParent.get(processInfo.ppid) ?? [];
    siblings.push(processInfo.pid);
    childrenByParent.set(processInfo.ppid, siblings);
  }
  const discovered = new Set([rootPid]);
  const queue = [rootPid];
  while (queue.length > 0) {
    const current = queue.shift();
    const children = childrenByParent.get(current) ?? [];
    for (const childPid of children) {
      if (discovered.has(childPid)) {
        continue;
      }
      discovered.add(childPid);
      queue.push(childPid);
    }
  }
  return [...discovered];
}

export async function killProcessTree(rootPid, { signal = "SIGTERM", graceMs = 1500 } = {}) {
  const pidList = [rootPid];
  let groupKillAttempted = false;
  try {
    process.kill(-rootPid, signal);
    groupKillAttempted = true;
  } catch {}
  if (!groupKillAttempted) {
    try {
      process.kill(rootPid, signal);
    } catch {}
  }
  if (signal === "SIGKILL" || graceMs <= 0) {
    return pidList;
  }
  await new Promise((resolve) => setTimeout(resolve, graceMs));
  const stubborn = pidList.filter((pid) => isPidAlive(pid));
  for (const pid of stubborn) {
    try {
      process.kill(-pid, "SIGKILL");
    } catch {}
    try {
      process.kill(pid, "SIGKILL");
    } catch {}
  }
  return pidList;
}

export function isPidAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return false;
  }
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
