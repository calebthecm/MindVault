#!/usr/bin/env node

const { spawnSync } = require("node:child_process");

const args = process.argv.slice(2);
const env = { ...process.env, MINDVAULT_NPM_WRAPPER_ACTIVE: "1" };

function tryRun(command, commandArgs) {
  const result = spawnSync(command, commandArgs, { stdio: "inherit", env });
  if (result.error) {
    return { ok: false, code: 1, error: result.error };
  }
  if (typeof result.status === "number") {
    return { ok: true, code: result.status };
  }
  return { ok: false, code: 1 };
}

const python = tryRun("python3", ["-m", "mindvault", ...args]);
if (python.ok) {
  process.exit(python.code);
}

console.error(
  "MindVault requires the Python app to be installed. " +
  "Install it with `pip install mindvault` or `pip install -e .`, then rerun this command."
);
process.exit(1);
