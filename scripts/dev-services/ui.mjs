import fs from "node:fs";
import readline from "node:readline";

const CLEAR = "\u001bc";
const DIM = "\u001b[2m";
const RESET = "\u001b[0m";
const BOLD = "\u001b[1m";
const REVERSE = "\u001b[7m";

function pad(value, width) {
  return String(value).padEnd(width, " ");
}

function readLogTail(filePath, lines = 20) {
  if (!filePath || !fs.existsSync(filePath)) {
    return [];
  }
  const content = fs.readFileSync(filePath, "utf8");
  return content.split(/\r?\n/u).slice(-lines).filter(Boolean);
}

function renderRow(service, selected) {
  const port = service.port ?? "-";
  const pid = service.pid ?? "-";
  const health = service.health.ok === null ? "-" : service.health.ok ? "ok" : "fail";
  const text = [
    pad(service.key, 24),
    pad(service.status, 9),
    pad(port, 6),
    pad(pid, 8),
    pad(health, 6),
    service.name,
  ].join("  ");
  return `${selected ? REVERSE : ""}${text}${RESET}`;
}

function footer(service) {
  const logTail = readLogTail(service.logFile, 12);
  const hint = "Keys: arrows move, s start, r restart, k kill, l logs, m start-missing, x kill-all, enter refresh, q quit";
  const lines = [
    "",
    `${BOLD}Selected${RESET}: ${service.key}  ${DIM}${service.command.join(" ")}${RESET}`,
    `${BOLD}Health${RESET}: ${service.health.checks.map((item) => `${item.status ?? "ERR"} ${item.url}`).join(" | ") || "-"}`,
    `${BOLD}Log${RESET}: ${service.logFile}`,
    `${BOLD}Tail${RESET}:`,
  ];
  if (logTail.length === 0) {
    lines.push(`${DIM}(no log output yet)${RESET}`);
  } else {
    lines.push(...logTail.map((line) => line.slice(0, 160)));
  }
  lines.push("");
  lines.push(`${DIM}${hint}${RESET}`);
  return lines.join("\n");
}

export async function runInteractiveUi({ config, getStatuses, actions }) {
  let selectedIndex = 0;
  let statusRows = await getStatuses();

  async function refresh(message = "") {
    statusRows = await getStatuses();
    selectedIndex = Math.max(0, Math.min(selectedIndex, statusRows.length - 1));
    const selected = statusRows[selectedIndex];
    const header = [
      `${CLEAR}${BOLD}Dev Services${RESET}`,
      "",
      `${pad("KEY", 24)}  ${pad("STATUS", 9)}  ${pad("PORT", 6)}  ${pad("PID", 8)}  ${pad("HEALTH", 6)}  NAME`,
      ...statusRows.map((service, index) => renderRow(service, index === selectedIndex)),
      footer(selected),
    ];
    if (message) {
      header.splice(1, 0, `${DIM}${message}${RESET}`);
    }
    process.stdout.write(header.join("\n"));
  }

  await refresh();
  readline.emitKeypressEvents(process.stdin);
  if (process.stdin.isTTY) {
    process.stdin.setRawMode(true);
  }

  return await new Promise((resolve, reject) => {
    let busy = false;
    const onKeypress = async (_str, key) => {
      if (busy) {
        return;
      }
      try {
        if (key.name === "q" || (key.ctrl && key.name === "c")) {
          cleanup();
          resolve();
          return;
        }
        if (key.name === "up") {
          selectedIndex = Math.max(0, selectedIndex - 1);
          await refresh();
          return;
        }
        if (key.name === "down") {
          selectedIndex = Math.min(statusRows.length - 1, selectedIndex + 1);
          await refresh();
          return;
        }
        if (key.name === "return") {
          await refresh("Refreshed.");
          return;
        }
        const selected = statusRows[selectedIndex];
        busy = true;
        if (key.name === "s") {
          await actions.start(selected.key);
          await refresh(`Started ${selected.key}.`);
        } else if (key.name === "r") {
          await actions.restart(selected.key);
          await refresh(`Restarted ${selected.key}.`);
        } else if (key.name === "k") {
          await actions.kill(selected.key);
          await refresh(`Stopped ${selected.key}.`);
        } else if (key.name === "l") {
          await refresh(`Showing latest logs for ${selected.key}.`);
        } else if (key.name === "m") {
          await actions.startMissing();
          await refresh("Started missing services.");
        } else if (key.name === "x") {
          await actions.killAll();
          await refresh("Stopped all services.");
        } else {
          busy = false;
          return;
        }
        busy = false;
      } catch (error) {
        busy = false;
        await refresh(error instanceof Error ? error.message : String(error));
      }
    };

    function cleanup() {
      process.stdin.off("keypress", onKeypress);
      if (process.stdin.isTTY) {
        process.stdin.setRawMode(false);
      }
      process.stdout.write("\n");
    }

    process.stdin.on("keypress", onKeypress);
    process.once("SIGINT", () => {
      cleanup();
      reject(new Error("Interrupted"));
    });
  });
}
