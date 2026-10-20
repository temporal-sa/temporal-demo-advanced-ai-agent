"""Subagents the coding agent delegates to through one ``task`` tool (PRD A10), in the shape of
Claude Code's and Cursor's Task tool.

``task`` starts a ``TaskAgent`` child workflow of the requested type, or resumes one started
earlier in the same turn, waits for its final report and returns it. Types live in
``SUBAGENT_TYPES``: ``general`` gets the full workspace toolset, ``explore`` and ``review`` only
read-only tools. A child runs under its parent's approval policy, including auto mode: the
same rulebook, the same on/off switch, and this session's approval history, so the judge
treats a child's call the way it would treat the parent's. Decisions a child escalates to a
person are reported back and join that history. Limits: depth 1 (children get no ``task``
tool), caps on tasks running at once and on subagents per turn (enforced by the parent), and
every child stops when the parent's turn ends. Children share the parent's workspace with no
locking. Usage is charged to the parent session.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from temporalio import workflow
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel, Field
    from pydantic_ai.messages import ModelMessage

    from temporal_agent_harness.harness import agent
    from temporal_agent_harness.harness.agent_protocol import (
        AgentConfig,
        ToolApprovalDecision,
        ToolApprovalPolicy,
    )
    from temporal_agent_harness.harness.agent_workflow import AgentWorkflowRunner, _current_runner

    from simple_chat_agent.agent import approvals, context, harness_ext
    from simple_chat_agent.agent.contracts import (  # message models live in contracts (key-free)
        SUBAGENT_APPROVAL_HISTORY_SIGNAL,
        TASK_AGENT_WORKFLOW_TYPE,
        ApprovalHistoryEntry,
        ModelChoice,
        Review,
        SubagentApprovalHistory,
        SubagentModel,
        SubagentType,
        TaskCall,
        TaskCommit,
        TaskFileChange,
        TaskRequest,
        TaskResult,
    )
    from simple_chat_agent.agent.runtime import (
        CODING_AGENT,
        RESEARCH_TOOLS,
        CodingDeps,
        fast_model_for,
        model_settings,
    )
    from simple_chat_agent.agent.state import ChangesState, merge_changes, new_commits, touched_files
    from simple_chat_agent.agent.tools.workspace import TEST_TOOLS, WORKSPACE_TOOLS

READ_ONLY_TOOL_NAMES = frozenset({"read_file", "list_files", "search_code", "git_status", "git_diff"})
# Tasks running at once (a resumed task counts again while it runs).
MAX_CONCURRENT_SUBAGENTS = 4
# Subagents alive in one turn, running or idle (idle ones wait to be resumed).
MAX_SUBAGENTS_PER_TURN = 8


class SubagentInit(BaseModel):
    """What a subagent starts with: its type and model, and the parent's workspace, identity
    and session settings."""

    subagent_type: SubagentType = "general"
    workspace_id: str
    owner_id: str = ""
    session_id: str = ""
    model: ModelChoice = Field(default_factory=ModelChoice)
    environment: Literal["catalog", "local"] = "catalog"
    judge: Literal["llm", "jev"] = "llm"
    research_tools: list[str] = Field(default_factory=list)
    test_tools: bool = False


GENERAL_INSTRUCTIONS = """\
You are a subagent: another coding agent gave you one task in a shared, sandboxed workspace. \
Work on it with your tools until it is done. Nobody will answer questions, so make reasonable \
assumptions and say what they were. Paths are relative to the workspace root.

Other subagents may be working in the same workspace at the same time: touch only the files \
your task needs, read a file again right before you edit it, never revert or reformat changes \
you didn't make, and don't commit unless the task says to. If a tool call is declined, don't \
retry it the same way; say what you would have done.

Your final message is all the other agent sees: say what you did, which files you changed, how \
you checked it, and anything left open. Be brief."""

EXPLORE_INSTRUCTIONS = """\
You are a code explorer helping another engineer. Answer the question by reading the \
workspace with your read-only tools. Cite the files and lines that support your answer \
(path:line). Be brief and factual; don't propose changes. Your final message is your answer."""

REVIEW_INSTRUCTIONS = """\
You are a careful code reviewer. Inspect the uncommitted diff (git_diff); if there is none, \
review the last commit instead (git_diff with commit="HEAD"). Read the surrounding code, \
then judge whether the change is correct and complete. Report concrete issues with \
file and line; approve only if you would merge it."""


@dataclass(frozen=True)
class SubagentSpec:
    instructions: str
    read_only: bool
    model: SubagentModel
    output_type: type = str


SUBAGENT_TYPES: dict[str, SubagentSpec] = {
    "general": SubagentSpec(GENERAL_INSTRUCTIONS, read_only=False, model="inherit"),
    "explore": SubagentSpec(EXPLORE_INSTRUCTIONS, read_only=True, model="fast"),
    "review": SubagentSpec(REVIEW_INSTRUCTIONS, read_only=True, model="inherit", output_type=Review),
}


def child_run_config(config: AgentConfig, environment: Literal["catalog", "local"]) -> AgentConfig:
    """The ``AgentConfig`` a subagent starts with: the parent's policy and criteria, kept.

    Auto mode stays on when the parent is in auto mode, and "never ask" stays on in the
    catalog too. The criteria the parent is judging with are the criteria the child judges
    with; only an allow-list name the catalog forbids is removed (see
    :func:`approvals.inherit_approval_policy`)."""
    return config.model_copy(
        update={"approval_policy": approvals.inherit_approval_policy(config.approval_policy, environment)}
    )


def _child_tools(data: SubagentInit) -> list[Any]:
    if SUBAGENT_TYPES[data.subagent_type].read_only:
        return [t for t in WORKSPACE_TOOLS if getattr(t, "__name__", "") in READ_ONLY_TOOL_NAMES]
    tools: list[Any] = [*WORKSPACE_TOOLS, *(t for t in RESEARCH_TOOLS if t.__name__ in set(data.research_tools))]
    if data.test_tools and data.environment == "local":
        tools += TEST_TOOLS
    return tools


@agent.defn(name=TASK_AGENT_WORKFLOW_TYPE)
class TaskAgent:
    """A subagent: works on tasks from its parent in the parent's workspace."""

    @agent.init
    def __init__(self, config: AgentConfig, data: SubagentInit) -> None:
        self._data = data
        # Decisions this task escalated to a person, signaled to the parent when the task
        # returns. The session history itself is adopted from each TaskRequest.
        self._approval_history: list[ApprovalHistoryEntry] = []
        self._turn_decisions: list[ApprovalHistoryEntry] = []
        # The harness requires the stream to be built directly in @workflow.init.
        self._stream = harness_ext.DrainableStream()
        self._runner = AgentWorkflowRunner(
            child_run_config(config, data.environment),
            stream=self._stream,
            # Used only when the parent passed no policy. The parent always passes its live one.
            approval_policy_default=ToolApprovalPolicy.allow_inherently_safe(),
            auto_approval_criteria_default=approvals.default_criteria(data.environment),
            auto_mode_evaluator=approvals.build_evaluator(
                data.environment,
                data.judge,
                session_id=data.session_id,
                owner_id=data.owner_id,
                history_provider=self._recent_approval_history,
            ),
        )
        harness_ext.guard_tool_approval(self._runner, self._limit_remember)
        self._history: list[ModelMessage] = []

    def _limit_remember(self, decision: ToolApprovalDecision) -> ToolApprovalDecision:
        """Same "always allow" limit as the parent, and the same approval history: a decision
        that reached a person here is precedent for the next call, on this child and (once
        signaled) on the parent."""
        tool = harness_ext.pending_tool_name(self._runner, decision.tool_id) or ""
        tool_input = harness_ext.pending_tool_input(self._runner, decision.tool_id)
        if tool and tool_input is not None:
            self._turn_decisions.append(
                approvals.record_decision(
                    self._approval_history, tool, tool_input, approved=decision.approved, reason=decision.reason
                )
            )
        return approvals.limit_remember(decision, tool, self._runner.auto_approval_criteria)

    def _recent_approval_history(self, tool_name: str) -> list[ApprovalHistoryEntry]:
        return approvals.recent_decisions_for(self._approval_history, tool_name)

    def _adopt_parent_history(self, entries: list[ApprovalHistoryEntry]) -> None:
        """Start this task from the session history the parent sent. Decisions collected
        during the task are tracked separately so only those are signaled back."""
        self._approval_history = list(entries)
        self._turn_decisions = []

    async def _report_approval_decisions(self) -> None:
        """Tell the parent about decisions this task escalated, so its judge sees them too."""
        if not self._turn_decisions:
            return
        parent = workflow.info().parent
        if parent is None:
            return
        await workflow.get_external_workflow_handle(parent.workflow_id).signal(
            SUBAGENT_APPROVAL_HISTORY_SIGNAL,
            SubagentApprovalHistory(entries=list(self._turn_decisions)),
        )

    @agent.teardown
    async def _drain_stream(self) -> None:
        """Let the bridge read this subagent's last events before the workflow completes: a
        completed workflow's stream can't be read, and the parent stops a subagent when its
        turn ends (see harness_ext.DrainableStream)."""
        await harness_ext.drain_stream(self._stream)

    @agent.accepts
    async def task(self, msg: TaskRequest) -> TaskResult:
        """Work on one task in the workspace until it is done, then report what was done."""
        self._adopt_parent_history(msg.approval_history)
        spec = SUBAGENT_TYPES[self._data.subagent_type]
        changes = ChangesState()

        def on_result(name: str, tool_id: str, args: dict[str, Any], result: Any) -> None:
            merge_changes(changes, touched_files(result), new_commits(result, args), tool_id=tool_id)

        toolset = harness_ext.per_run_toolset(
            _child_tools(self._data),
            id="subagent",
            injections={
                "workspace_id": self._data.workspace_id,
                "owner_id": self._data.owner_id,
                "session_id": self._data.session_id,
            },
            on_result=on_result,
        )
        model = self._data.model
        result = await CODING_AGENT.run(
            msg.prompt,
            output_type=spec.output_type,
            instructions=spec.instructions,
            deps=CodingDeps(runner=self._runner, session_id=self._data.session_id, owner_id=self._data.owner_id),
            message_history=self._history,
            model=model.registry_key,
            model_settings=model_settings(model, session_id=self._data.session_id),
            toolsets=[toolset],
            capabilities=[context.ContextFit(context.budget_for(model.registry_key))],
        )
        self._history = result.all_messages()
        review = result.output if isinstance(result.output, Review) else None
        await self._report_approval_decisions()
        return TaskResult(
            subagent_id=harness_ext.agent_id(self._runner),
            subagent_type=self._data.subagent_type,
            text=review.as_text() if review else str(result.output),
            review=review,
            changed_files=[
                TaskFileChange(path=f.path, status=f.status, added=f.added, removed=f.removed) for f in changes.files
            ],
            commits=[TaskCommit(hash=c.hash, message=c.message) for c in changes.commits],
        )


SUBAGENT_WORKFLOWS = (TaskAgent,)


def _model_for(call: TaskCall, parent: ModelChoice) -> ModelChoice:
    choice = call.model or SUBAGENT_TYPES[call.subagent_type].model
    return fast_model_for(parent) if choice == "fast" else parent


def subagent_tools(
    *,
    task_queue: str,
    init: SubagentInit,
    config: Callable[[], AgentConfig],
    history: Callable[[], list[ApprovalHistoryEntry]],
) -> list[Any]:
    """The parent's ``task`` tool. ``init`` is the base init data for every child (its type
    and model are filled in per call); ``config`` gives the child's AgentConfig, called at
    each start so the child gets the parent's current approval policy and criteria.
    ``history`` is this session's approval decisions, sent with every task so the child's
    judge sees the same precedent the parent's does."""

    @agent.tool_defn()
    async def task(call: TaskCall) -> TaskResult:
        """Hand a self-contained task to a subagent and wait for its report.

        The subagent starts with a fresh context: put everything it needs in `prompt`. Use it
        for work that would fill your context (broad searches, reading many files) or that
        splits into independent parts. To run tasks in parallel, make several task calls in
        one response. Subagents share your workspace: give parallel general subagents
        disjoint files. Its report is not shown to the user; summarize what matters. Pass
        `resume` with a subagent id from an earlier report in this turn to follow up with
        the same subagent. A subagent uses your approval rules, including auto mode: a call
        it makes is judged or asked the same way the call would be if you made it."""
        runner = _current_runner()
        active = harness_ext.active_subagent_ids(runner)
        if call.resume:
            if call.resume not in active:
                raise ApplicationError(
                    f"No subagent {call.resume!r} is open in this turn; start a new task instead.",
                    non_retryable=True,
                )
            handle = call.resume
        else:
            if len(active) >= MAX_SUBAGENTS_PER_TURN:
                raise ApplicationError(
                    f"This turn already has {MAX_SUBAGENTS_PER_TURN} subagents. Resume one of "
                    f"them ({', '.join(active)}) instead of starting another.",
                    non_retryable=True,
                )
            data = init.model_copy(update={"subagent_type": call.subagent_type, "model": _model_for(call, init.model)})
            handle = await runner.start_subagent(
                call.subagent_type, TASK_AGENT_WORKFLOW_TYPE, task_queue, config=config(), data=data
            )
        output = await runner.run_subagent_turn(
            handle,
            "task",
            TaskRequest(
                description=call.description, prompt=call.prompt, approval_history=list(history())
            ).model_dump(mode="json"),
        )
        return TaskResult.model_validate(output)

    return [task]
