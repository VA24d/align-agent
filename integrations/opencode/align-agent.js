// OpenCode plugin (V1 API, OpenCode 1.x): write-through through the align-agent-wt command.
// Copy to .opencode/plugins/ or ~/.config/opencode/plugins/. Needs `pip install git+https://github.com/VA24d/align-agent@v0.4.2`.
import { spawnSync } from "node:child_process";

const WT = process.env.ALIGN_WT || "align-agent-wt";

function wt(event) {
  const r = spawnSync(WT, [], { input: JSON.stringify(event), encoding: "utf8" });
  try { return JSON.parse(r.stdout || "{}"); } catch { return {}; }
}

export const Align = async ({ directory }) => ({
  // a sed -i on line numbers the model has not seen as they are now is refused once
  "tool.execute.before": async (input, output) => {
    if (input.tool !== "bash") return;
    const r = wt({ event: "before", root: directory, session: input.sessionID, command: output.args.command });
    if (r.deny) throw new Error(r.deny);
  },
  // after every tool: the diff of files the model has seen that changed since, appended to the result
  "tool.execute.after": async (input, output) => {
    const a = input.args || {};
    const ev = { event: "after", root: directory, session: input.sessionID };
    if (input.tool === "bash") Object.assign(ev, { command: a.command, output: output.output });
    if (input.tool === "read") Object.assign(ev, { read: a.filePath, lines: a.offset ? [a.offset, a.offset + (a.limit || 2000) - 1] : undefined });
    if (["write", "edit", "patch", "apply_patch"].includes(input.tool)) ev.authored = a.filePath;
    const r = wt(ev);
    if (r.context) output.output = `${output.output}\n\n${r.context}`;
  },
});
