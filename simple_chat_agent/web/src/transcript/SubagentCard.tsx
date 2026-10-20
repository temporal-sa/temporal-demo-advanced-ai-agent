// A subagent as a nested row (PRD U2): its task, its own tool calls, and its report (a
// TaskResult). A reviewer's report also carries a Review, drawn as a card with a verdict and
// issues (severity, file:line). Shapes: simple_chat_agent/agent/contracts.py.

import { useState } from "react";
import type { Message, SubagentPart, ToolPart } from "../session";
import { agentLabel } from "../workbench/tree";
import { useTranscript } from "./context";
import { argPreview, oneLine } from "./toolInfo";
import { Markdown } from "./Markdown";
import { ErrorNote } from "./ErrorNote";

interface TaskFileChange {
  path: string;
  status: string;
}
interface TaskResult {
  text: string;
  review?: Review | null;
  changed_files?: TaskFileChange[];
}
interface Issue {
  severity: string;
  path: string;
  line?: number | null;
  comment: string;
}
interface Review {
  verdict: string;
  summary: string;
  issues?: Issue[];
}

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null;
export const asTaskResult = (v: unknown): TaskResult | null => (isObj(v) && typeof v.text === "string" ? (v as unknown as TaskResult) : null);
export const asReview = (v: unknown): Review | null =>
  isObj(v) && typeof v.verdict === "string" && typeof v.summary === "string" ? (v as unknown as Review) : null;

function title(part: SubagentPart, message: Message | undefined): string {
  const input = (message?.input ?? {}) as Record<string, unknown>;
  if (typeof input.description === "string" && input.description) return input.description;
  if (typeof input.prompt === "string" && input.prompt) return oneLine(input.prompt, 120);
  return part.handler;
}

function duration(message: Message | undefined): string {
  if (!message?.startedAt || !message.endedAt) return "";
  const s = Math.max(0, Math.round(message.endedAt - message.startedAt));
  return ` · ${s}s`;
}

const where = (path: string, line?: number | null) => (line ? `${path}:${line}` : path);

export function ReviewCard({ review }: { review: Review }) {
  const issues = review.issues ?? [];
  const blocking = issues.filter((i) => i.severity === "blocker").length;
  const approve = review.verdict === "approve";
  return (
    <div className="review">
      <div className="rh">
        <span className="badge agent">Review</span>
        <span className={`verdict${approve ? "" : " changes"}`}>{approve ? "Approve" : "Request changes"}</span>
        <span className="hint">
          {issues.length} {issues.length === 1 ? "issue" : "issues"} · {blocking} blocking
        </span>
      </div>
      {review.summary && <div className="issue summary">{review.summary}</div>}
      {issues.map((i, n) => (
        <div key={n} className="issue">
          <span className={`sev ${i.severity}`}>{i.severity}</span>
          <span>{i.comment}</span>
          <span className="where">{where(i.path, i.line)}</span>
        </div>
      ))}
    </div>
  );
}

export function SubagentCard({ part }: { part: SubagentPart }) {
  const { session } = useTranscript();
  const [open, setOpen] = useState(false);
  const node = session.agent(part.subagentId);
  const message = part.messageId ? node?.messages.find((m) => m.id === part.messageId) : undefined;
  const tools = (message?.parts ?? []).filter((p): p is ToolPart => p.type === "tool");
  const result = asTaskResult(message?.output);
  const review = asReview(result?.review);
  const changed = result?.changed_files ?? [];
  const status =
    part.status === "running" ? (
      <span className="badge prog">Running</span>
    ) : part.status === "ok" ? (
      <span className="badge good">Done{duration(message)}</span>
    ) : part.status === "unavailable" ? (
      <span className="badge idle">Not readable</span>
    ) : (
      <span className="badge error">Failed</span>
    );
  return (
    <div className="subagent-wrap">
      <div className="subagent">
        <button type="button" className="head" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
          <span className="badge agent">{agentLabel(part.agentKey || null)}</span>
          <span className="q">{title(part, message)}</span>
          {status}
        </button>
        {tools.length > 0 && (
          <div className="inner">
            {(open ? tools : tools.slice(-3)).map((t) => (
              <span key={t.toolId}>
                {t.toolName}  {argPreview(t)}
              </span>
            ))}
            {!open && tools.length > 3 && <span>… {tools.length - 3} earlier calls</span>}
          </div>
        )}
        {result && !review && (
          <div className="find">
            <b>{part.agentKey === "explore" ? "Findings" : "Report"}</b> <Markdown text={result.text} />
            {changed.length > 0 && (
              <div className="evidence">
                Changed {changed.map((f) => f.path).join(", ")}
              </div>
            )}
          </div>
        )}
        {message?.error && <ErrorNote text="The subagent stopped with an error. The main agent carries on without its answer." detail={message.error} />}
        {node?.unavailable && <div className="live">{node.unavailable}</div>}
      </div>
      {review && <ReviewCard review={review} />}
    </div>
  );
}
