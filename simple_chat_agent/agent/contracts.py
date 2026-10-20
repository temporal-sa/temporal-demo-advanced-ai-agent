"""Wire contracts for the coding agent.

Imported by the worker, the API and tests. Keep this module free of provider SDKs and of
anything that needs a worker (no Pydantic AI, no model registry), so the API can import it
without LLM keys.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from temporalio.contrib.workflow_streams import WorkflowStreamState

AGENT_WORKFLOW_TYPE = "CodingAgent"
# The subagent workflow the ``task`` tool starts (agent/subagents.py).
TASK_AGENT_WORKFLOW_TYPE = "TaskAgent"

Provider = Literal["anthropic", "openai", "google"]
Posture = Literal["ask_everything", "trust_safe", "auto", "never_ask"]

# Signal names of the front door (PRD S8). Signals land in history with no worker running;
# the agent admits them into the harness runner when it next runs.
ENQUEUE_MESSAGE_SIGNAL = "enqueue_message"
ENQUEUE_APPROVAL_SIGNAL = "enqueue_approval"
# The answer to a browser callback tool (ask_user, pick_files; PRD A12).
ENQUEUE_CALLBACK_SIGNAL = "enqueue_callback_result"
# A subagent reports the human decisions it collected this turn, so they join the session's
# approval history (the auto-mode judge reads that history on the parent and on later children).
SUBAGENT_APPROVAL_HISTORY_SIGNAL = "subagent_approval_history"
# The Redis bridge reports the highest stream offset it has stored, once per turn end. The
# agent then trims only what the bridge holds when it continues as new (P2 decision 4).
STREAM_ACKED_SIGNAL = "stream_acked"

# Bumped whenever CarriedState changes shape. Every field keeps a default so a newer
# version can read input written by an older one (PINNED + continue-as-new, PRD §6).
CARRIED_STATE_VERSION = 1


class ModelChoice(BaseModel):
    """A provider plus a model id from that provider's allowlist."""

    provider: Provider = "anthropic"
    model: str = "claude-sonnet-5-5"
    effort: Literal["low", "medium", "high"] | None = None

    @property
    def registry_key(self) -> str:
        return f"{self.provider}:{self.model}"


class FrontDoorMessage(BaseModel):
    """Payload of the enqueue_message signal."""

    request_id: str
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ApprovalHistoryEntry(BaseModel):
    """One call a human decided after it was escalated (asked), kept so the auto-mode judge
    can recognize a similar future call instead of asking again.

    ``args_summary`` is a compact, bounded rendering of the call's arguments (see
    ``approvals.summarize_tool_args``), never the raw dict: it rides along in CarriedState and
    in every judge prompt for the tool, so it has to stay small regardless of how large a call's
    real arguments were. ``reason`` is the human's stated reason, if they gave one, also bounded.
    """

    tool_name: str
    args_summary: str
    approved: bool
    reason: str = ""


class CarriedState(BaseModel):
    """Everything a run hands to the next one when it continues as new.

    The conversation is opaque JSON here so this module stays free of Pydantic AI; the
    workflow validates it on the way in.
    """

    state_version: int = CARRIED_STATE_VERSION
    conversation: list[dict[str, Any]] = Field(default_factory=list)
    stream_state: WorkflowStreamState | None = None
    turn_counter: int = 0
    stream_acked: int = 0
    recent_request_ids: list[str] = Field(default_factory=list)
    model: ModelChoice = Field(default_factory=ModelChoice)
    pending_model: ModelChoice | None = None
    # Observable state (plan, changes, workspace) as JSON, restored into the new run.
    observable: dict[str, dict[str, Any]] = Field(default_factory=dict)
    steer_queue: list[str] = Field(default_factory=list)
    # Front-door messages that arrived during the handoff; the next run admits them first.
    pending_messages: list[FrontDoorMessage] = Field(default_factory=list)
    # The MCP tool list as last listed (``mcp_client.McpServerTools`` as JSON; P5), so the
    # next run reuses fresh entries instead of listing every server again. Empty for runs
    # that predate it: the next run then lists as before.
    mcp_tools: list[dict[str, Any]] = Field(default_factory=list)
    # The remembered allow-list ("always allow") survives through AgentConfig.approval_policy,
    # which the workflow rebuilds from the live runner policy before continuing as new.
    # Human decisions on escalated calls, for the auto-mode judge to recognize a similar call
    # (approvals.record_decision / JudgeRequest.history). Empty for runs that predate it: the
    # judge then sees no history, exactly as before.
    approval_history: list[ApprovalHistoryEntry] = Field(default_factory=list)


class CodingAgentInit(BaseModel):
    """The agent's init data: what a session starts with, plus anything carried across runs."""

    owner_id: str = ""
    session_id: str = ""
    model: ModelChoice = Field(default_factory=ModelChoice)
    posture: Posture = "trust_safe"
    # Where the session runs (catalog rules are stricter) and which auto-mode judge it uses.
    # Decided by the API at session start and carried as data, never read from the
    # environment in workflow code.
    environment: Literal["catalog", "local"] = "catalog"
    judge: Literal["llm", "jev"] = "llm"
    # Research tools this session may offer, decided once by whoever starts the session
    # (from its configured backends) and carried as data, so replay never depends on a
    # worker's environment.
    research_tools: list[str] = Field(default_factory=lambda: ["fetch_url"])
    workspace_id: str = ""
    depth: int = 0
    # MCP tools from the owner's registered servers (PRD A8). Set by the API at session start,
    # only for local sessions (D7); off for sessions that predate it.
    mcp: bool = False
    system_prompt: str | None = None
    carried: CarriedState | None = None
    # Provider web search (P4 4.5): whether this session may turn it on (the API decides from
    # its environment and config; off in the catalog unless configured), and whether it is on.
    # Off by default. set_web_search changes `web_search`; it is carried with this data.
    web_search_allowed: bool = False
    web_search: bool = False
    # Test hooks: force continue-as-new at the next idle point (P0 step 0.5), and offer
    # test-only tools such as wait_seconds.
    force_continue_as_new_after_turns: int | None = None
    test_tools: bool = False
    # Test hook: a low history-event cap to exercise the checkpoint (P1 step 1.8a).
    test_history_event_cap: int | None = None


MAX_CHAT_ATTACHMENTS = 20


class ChatAttachment(BaseModel):
    """A file the user attached to a chat message (uploaded first with
    ``POST /api/agent/sessions/{id}/attachments``, which returns its id)."""

    attachment_id: str = Field(min_length=1, max_length=200)
    name: str = Field(default="", max_length=500)
    size: int | None = Field(default=None, ge=0)
    content_type: str | None = Field(default=None, max_length=200)


class ChatMessage(BaseModel):
    """A user message to the coding agent."""

    text: str
    # Defaulted: messages from before attachments existed (and carried, deferred ones) still
    # validate.
    attachments: list[ChatAttachment] = Field(default_factory=list, max_length=MAX_CHAT_ATTACHMENTS)


MAX_WEB_SOURCES = 20


class WebSource(BaseModel):
    """One page the provider's web search used for a reply."""

    title: str = ""
    url: str


class ChatReply(BaseModel):
    """The agent's final answer for a turn."""

    text: str
    # Pages the provider's built-in web search returned during this turn (P4 4.5), first seen
    # first, deduplicated by URL, at most MAX_WEB_SOURCES. Empty when web search is off or
    # wasn't used. The UI lists them under the reply.
    sources: list[WebSource] = Field(default_factory=list)
    interrupted: bool = False
    # Set when the turn stopped early to keep the workflow inside Temporal's history limits
    # (PRD A20). The agent continues as new; the user says "continue" to carry on.
    checkpoint: str | None = None
    # Set when the turn stopped on a known, explainable failure (spend cap, provider
    # rejection, context overflow) rather than an unexpected error.
    error: str | None = None


class SetModel(BaseModel):
    """Switch provider and/or model. Applies from the next turn."""

    model: ModelChoice


class SteerMessage(BaseModel):
    """Guidance for the turn in progress, injected before the agent's next model call."""

    text: str


class SetPosture(BaseModel):
    """Change how tool calls are approved. Applies to new calls; pending ones stay with you."""

    posture: Posture


class InterruptRequest(BaseModel):
    """Stop the current turn. Pending approvals in the turn are declined."""

    reason: str = "Stopped by the user."


class Ack(BaseModel):
    """Acknowledgement for control operations."""

    ok: bool = True
    detail: str = ""


class SettingsAck(Ack):
    """Acknowledgement of ``set_model`` / ``set_posture`` with the value the agent now holds:
    the new one when ``ok``, the unchanged one when it refused. The bridge copies it into the
    session list, so the list never shows a setting the agent refused."""

    setting: Literal["model", "posture", "web_search"]
    # set_model: the model for the next turn (the current one if the change was refused).
    model: ModelChoice | None = None
    posture: Posture | None = None
    # set_web_search: whether web search is now on.
    web_search: bool | None = None


class SetWebSearch(BaseModel):
    """Turn the model provider's built-in web search on or off for this session. Applies from
    the next turn. Its searches run inside the provider and are not approved one by one."""

    enabled: bool


class ForgetAlwaysAllowed(BaseModel):
    """Forget the "always allow" approvals given in this session (no fields)."""


class AlwaysAllowedForgotten(Ack):
    """The tools that were always allowed and now ask again (as the posture says)."""

    forgotten: list[str] = Field(default_factory=list)


class StreamAck(BaseModel):
    """Payload of the stream_acked signal."""

    offset: int


class FrontDoorCallbackResult(BaseModel):
    """Payload of the enqueue_callback_result signal: the browser's answer to a pending
    callback tool call. Exactly one of ``result`` / ``error`` is meaningful."""

    request_id: str
    tool_id: str
    result: Any = None
    error: str | None = None


# -- Browser callback tools (PRD A12) --------------------------------------------------------

# pick_files limits, enforced by the upload route (size, count, names) and again by the agent
# (count and path shape) when the answer arrives.
PICK_MAX_FILES = 10
PICK_MAX_FILE_BYTES = 5 * 1024 * 1024  # the executor's per-write cap
PICK_MAX_NAME_CHARS = 100
PICK_UPLOAD_DIR = "uploads"
ASK_MAX_OPTIONS = 6


class ChoiceOption(BaseModel):
    """One choice card."""

    id: str = Field(description="Short stable id for this option, e.g. 'a' or 'keep'.")
    label: str = Field(description="What the card says.")
    description: str = Field("", description="One optional line under the label.")


class AskUserAnswer(BaseModel):
    """The option the user chose (one of the offered ids)."""

    option_id: str


class PickedFiles(BaseModel):
    """Workspace paths of the files the user chose, uploaded under ``uploads/``."""

    paths: list[str] = Field(default_factory=list)


class FrontDoorApproval(BaseModel):
    """Payload of the enqueue_approval signal."""

    request_id: str
    tool_id: str
    approved: bool
    reason: str | None = None
    remember: bool = False


# -- Subagents (key-free, so the API can describe subagent interfaces) -----------------

SubagentType = Literal["general", "explore", "review"]
SubagentModel = Literal["inherit", "fast"]

SUBAGENT_TYPE_HELP = (
    "general: does a multi-step task on its own with the full workspace toolset (reads, edits, "
    "shell); its risky calls ask the user like yours do. explore: read-only, fast; answers a "
    "question about the code with file:line evidence. review: read-only; reviews the "
    "uncommitted diff (else the last commit) and returns a verdict with issues."
)


class TaskCall(BaseModel):
    """The ``task`` tool's input: one self-contained job for a subagent."""

    description: str = Field(
        max_length=120, description="A short (3-6 word) label for the task, shown to the user."
    )
    prompt: str = Field(
        description=(
            "Everything the subagent needs: the goal, the files or symbols involved, constraints, "
            "and what to report back. It sees none of your conversation."
        )
    )
    subagent_type: SubagentType = Field("general", description=SUBAGENT_TYPE_HELP)
    model: SubagentModel | None = Field(
        None,
        description=(
            "inherit: your model; fast: the provider's fast model. Omit for the type's default "
            "(explore runs fast, the others inherit)."
        ),
    )
    resume: str | None = Field(
        None,
        description=(
            "The subagent id of a task you started earlier in this turn, to send it this prompt "
            "as a follow-up. It keeps its context; subagent_type and model are ignored."
        ),
    )


class TaskRequest(BaseModel):
    """What the parent sends a subagent: one task to work on until done."""

    description: str = ""
    prompt: str
    # This session's approval decisions so far. The parent fills it; the subagent's judge reads
    # it, and decisions the subagent collects are signaled back. Empty for a parent that predates
    # it. Not part of the task the subagent was given.
    approval_history: list[ApprovalHistoryEntry] = Field(default_factory=list)


class SubagentApprovalHistory(BaseModel):
    """Payload of ``subagent_approval_history``: decisions a subagent's human gate collected
    during one task, for the parent to merge into the session history."""

    entries: list[ApprovalHistoryEntry] = Field(default_factory=list)


class TaskFileChange(BaseModel):
    """A file a subagent changed while it worked."""

    path: str
    status: str = "modified"
    added: int = 0
    removed: int = 0


class TaskCommit(BaseModel):
    """A local commit a subagent made."""

    hash: str
    message: str = ""


class Issue(BaseModel):
    """One problem the reviewer found."""

    severity: str = Field(description="blocker, major or minor")
    path: str
    line: int | None = None
    comment: str


class Review(BaseModel):
    """The reviewer's verdict on the diff."""

    verdict: str = Field(description="approve or request_changes")
    summary: str
    issues: list[Issue] = Field(default_factory=list)

    def as_text(self) -> str:
        lines = [f"Verdict: {self.verdict}. {self.summary}"]
        for i in self.issues:
            where = f"{i.path}:{i.line}" if i.line else i.path
            lines.append(f"- [{i.severity}] {where}: {i.comment}")
        return "\n".join(lines)


class TaskResult(BaseModel):
    """A subagent's report on its task. ``text`` is its final message; the parent's Changes
    panel folds in ``changed_files`` and ``commits``."""

    subagent_id: str = ""
    subagent_type: SubagentType = "general"
    text: str
    # Set for the review type: the structured verdict the UI draws as a card.
    review: Review | None = None
    changed_files: list[TaskFileChange] = Field(default_factory=list)
    commits: list[TaskCommit] = Field(default_factory=list)

    def __str__(self) -> str:
        parts = [self.text.strip() or "(The subagent finished without a final message.)"]
        if self.changed_files:
            parts.append("Files it changed: " + ", ".join(f"{f.path} ({f.status})" for f in self.changed_files))
        if self.commits:
            parts.append("Commits it made: " + ", ".join(f"{c.hash[:10]} {c.message}" for c in self.commits))
        parts.append(f'(subagent id: {self.subagent_id}; pass resume="{self.subagent_id}" to follow up this turn.)')
        return "\n\n".join(parts)
