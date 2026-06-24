import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

async function safeExec(file, args) {
  try {
    return await execFileAsync(file, args, { encoding: "utf8", maxBuffer: 1024 * 1024 * 4 });
  } catch (error) {
    return { stdout: error?.stdout ?? "", stderr: error?.stderr ?? "" };
  }
}

export async function observePort(port) {
  if (!port) {
    return { listening: false, pids: [], raw: "" };
  }
  const { stdout } = await safeExec("lsof", ["-nP", `-iTCP:${port}`, "-sTCP:LISTEN", "-Fpctn"]);
  const pids = new Set();
  let command = null;
  let name = null;
  for (const line of stdout.split(/\r?\n/u)) {
    if (line.startsWith("p")) {
      pids.add(Number(line.slice(1)));
    } else if (line.startsWith("c")) {
      command = line.slice(1);
    } else if (line.startsWith("n")) {
      name = line.slice(1);
    }
  }
  return {
    listening: pids.size > 0,
    pids: [...pids].filter(Number.isFinite),
    command,
    name,
    raw: stdout,
  };
}
