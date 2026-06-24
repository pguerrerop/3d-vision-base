import fs from "node:fs";
import { checkServiceHealth } from "./health-checks.mjs";
import { observePort } from "./port-observer.mjs";
import { isPidAlive, listProcesses, matchServiceProcesses } from "./process-tree.mjs";
import { logPath, readServiceMetadata } from "./start-service.mjs";

function statusFrom({ matchedProcesses, portInfo, health, metadataPidAlive }) {
  if (health?.ok === true) {
    return "healthy";
  }
  if (portInfo?.listening || matchedProcesses.length > 0 || metadataPidAlive) {
    return "running";
  }
  return "stopped";
}

export async function collectServiceStatus(config) {
  const processes = await listProcesses();
  const services = [];

  for (const service of config.services) {
    const matchedProcesses = matchServiceProcesses(service, processes);
    const portInfo = await observePort(service.port);
    const health = await checkServiceHealth(service);
    const meta = readServiceMetadata(config.tmpDir, service.key);
    const metadataPidAlive = isPidAlive(Number(meta?.pid));
    const logfile = logPath(config.tmpDir, service.key);

    services.push({
      ...service,
      status: statusFrom({ matchedProcesses, portInfo, health, metadataPidAlive }),
      matchedProcesses,
      pid: matchedProcesses[0]?.pid ?? portInfo.pids[0] ?? (metadataPidAlive ? meta?.pid : null) ?? null,
      pids: Array.from(new Set([
        ...matchedProcesses.map((item) => item.pid),
        ...(Array.isArray(portInfo.pids) ? portInfo.pids : []),
        ...(metadataPidAlive ? [Number(meta.pid)] : []),
      ])),
      portInfo,
      health,
      meta,
      metadataPidAlive,
      logFile: logfile,
      hasLogFile: fs.existsSync(logfile),
    });
  }

  return services;
}
