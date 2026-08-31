#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { getDevServicesConfig } from "./dev-services/dev-services.config.mjs";
import { killProcessTree } from "./dev-services/process-tree.mjs";
import { startDetachedService, ensureInstall } from "./dev-services/start-service.mjs";
import { collectServiceStatus } from "./dev-services/status.mjs";
import { runInteractiveUi } from "./dev-services/ui.mjs";
import { reportCatalogDrift } from "./dev-services/catalog-drift.mjs";

const config = getDevServicesConfig();
fs.mkdirSync(config.tmpDir, { recursive: true });

function parseArgs(argv) {
  const flags = new Set();
  const positionals = [];
  for (const arg of argv) {
    if (arg.startsWith("--")) {
      flags.add(arg);
    } else {
      positionals.push(arg);
    }
  }
  return {
    command: positionals[0] ?? null,
    target: positionals[1] ?? null,
    flags,
  };
}

function findService(serviceKey) {
  const service = config.services.find((item) => item.key === serviceKey);
  if (!service) {
    throw new Error(`Unknown service: ${serviceKey}`);
  }
  return service;
}

async function getStatuses() {
  return collectServiceStatus(config);
}

function printStatuses(statuses) {
  for (const service of statuses) {
    const health = service.health.ok === null
      ? "-"
      : service.health.ok
        ? "ok"
        : "fail";
    console.log(
      [
        service.key.padEnd(24, " "),
        service.status.padEnd(8, " "),
        `port=${String(service.port ?? "-").padEnd(5, " ")}`,
        `pid=${String(service.pid ?? "-").padEnd(8, " ")}`,
        `health=${health}`,
      ].join("  "),
    );
  }
}

async function startService(serviceKey, options) {
  const service = findService(serviceKey);
  const current = (await getStatuses()).find((item) => item.key === serviceKey);
  if (current?.status !== "stopped") {
    return { skipped: true, reason: `${serviceKey} is already ${current.status}` };
  }
  const installResult = await ensureInstall(service, options);
  if (installResult.skipped) {
    return { skipped: true, reason: `Skipped ${serviceKey}: ${installResult.reason}` };
  }
  const result = await startDetachedService(service, config.tmpDir);
  return { ...result, serviceKey };
}

async function killService(serviceKey) {
  const status = (await getStatuses()).find((item) => item.key === serviceKey);
  if (!status || status.pids.length === 0) {
    return { skipped: true, reason: `${serviceKey} is not running` };
  }
  const killed = new Set();
  for (const pid of status.pids) {
    if (killed.has(pid)) {
      continue;
    }
    await killProcessTree(pid);
    killed.add(pid);
  }
  return { killed: [...killed] };
}

async function restartService(serviceKey, options) {
  await killService(serviceKey);
  return startService(serviceKey, options);
}

async function startMissing(options) {
  const statuses = await getStatuses();
  const results = [];
  for (const service of statuses.filter((item) => item.status === "stopped")) {
    results.push(await startService(service.key, options));
  }
  return results;
}

async function killAll() {
  const statuses = await getStatuses();
  const results = [];
  for (const service of [...statuses].reverse()) {
    results.push({ serviceKey: service.key, ...(await killService(service.key)) });
  }
  return results;
}

function printLogs(serviceKey, lines = 60) {
  const service = findService(serviceKey);
  const logFile = path.join(config.tmpDir, `${service.key}.log`);
  if (!fs.existsSync(logFile)) {
    console.log(`No log file yet for ${service.key}: ${logFile}`);
    return;
  }
  const content = fs.readFileSync(logFile, "utf8");
  const tail = content.split(/\r?\n/u).slice(-lines);
  console.log(tail.join("\n"));
}

async function main() {
  const parsed = parseArgs(process.argv.slice(2));
  const options = {
    yes: parsed.flags.has("--yes"),
    skipInstall: parsed.flags.has("--skip-install"),
  };

  if (!parsed.command && process.stdin.isTTY && process.stdout.isTTY) {
    await runInteractiveUi({
      config,
      getStatuses,
      actions: {
        start: (key) => startService(key, { ...options, yes: true }),
        restart: (key) => restartService(key, { ...options, yes: true }),
        kill: killService,
        startMissing: () => startMissing({ ...options, yes: true }),
        killAll,
      },
    });
    return;
  }

  switch (parsed.command) {
    case "status": {
      printStatuses(await getStatuses());
      reportCatalogDrift(config);
      return;
    }
    case "start": {
      if (!parsed.target) {
        throw new Error("Usage: npm run dev:services -- start <service-key>");
      }
      console.log(JSON.stringify(await startService(parsed.target, options), null, 2));
      reportCatalogDrift(config);
      return;
    }
    case "restart": {
      if (!parsed.target) {
        throw new Error("Usage: npm run dev:services -- restart <service-key>");
      }
      console.log(JSON.stringify(await restartService(parsed.target, options), null, 2));
      return;
    }
    case "kill": {
      if (!parsed.target) {
        throw new Error("Usage: npm run dev:services -- kill <service-key>");
      }
      console.log(JSON.stringify(await killService(parsed.target), null, 2));
      return;
    }
    case "logs": {
      if (!parsed.target) {
        throw new Error("Usage: npm run dev:services -- logs <service-key>");
      }
      printLogs(parsed.target);
      return;
    }
    case "start-missing": {
      console.log(JSON.stringify(await startMissing(options), null, 2));
      return;
    }
    case "kill-all": {
      console.log(JSON.stringify(await killAll(), null, 2));
      return;
    }
    case null: {
      printStatuses(await getStatuses());
      return;
    }
    default:
      throw new Error(`Unknown command: ${parsed.command}`);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
