// OpenCode plugin (V1 API, OpenCode 1.x): write-through through the align-agent-wt command.
// Copy to .opencode/plugins/ or ~/.config/opencode/plugins/. Needs `pip install align-agent`.
import { spawnSync } from "node:child_process";

// ALIGN_WT may hold arguments too, e.g. "uvx --from align-agent align-agent-wt"
const [WT, ...WT_ARGS] = (process.env.ALIGN_WT || "align-agent-wt").trim().split(/\s+/);
const PATCH_FILE = /^\*\*\* (?:(?:Update|Add|Delete) File|Move to): (.+?)\r?$/gm;

function wt(event) {
  // fail open: a missing command, a crash or a hang (held lock) must never stall the tool call
  const r = spawnSync(WT, WT_ARGS, { input: JSON.stringify(event), encoding: "utf8", timeout: 30000 });
  if (r.error || r.status !== 0) return {};
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
    if (["write", "edit", "patch"].includes(input.tool)) ev.authored = a.filePath;
    // apply_patch (given to gpt-* models instead of edit/write) names its files inside patchText
    if (input.tool === "apply_patch") ev.authored = [...(a.patchText || "").matchAll(PATCH_FILE)].map((m) => m[1].trim());
    const r = wt(ev);
    if (r.context) output.output = `${output.output}\n\n${r.context}`;
  },
});
