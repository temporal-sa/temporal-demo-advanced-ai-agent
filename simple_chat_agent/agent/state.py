"""Observable state for the coding agent's product view (PRD A13).

Declared on the workflow class with ``agent.state(...)``; the harness publishes a snapshot and
then one JSON patch per change on the agent's event stream, so the UI's Plan and Changes
panels update live with no Queries.

* ``plan``: the agent's own checklist, written by the inline ``update_plan`` tool.
* ``changes``: files the agent touched and commits it made, folded from tool results.
* ``workspace``: the repo's current branch, the last command and its exit code.

No ``from __future__ import annotations``: tool annotations are read concretely by the harness.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from temporal_agent_harness.harness import agent
from temporal_agent_harness.harness.state import HarnessState, StateRef

PlanStatus = Literal["pending", "in_progress", "done", "skipped"]
FileStatus = Literal["added", "modified", "deleted", "renamed"]

MAX_PLAN_ITEMS = 30
MAX_PLAN_TEXT = 300
MAX_CHANGED_FILES = 200
MAX_COMMITS = 50
MAX_BRANCH_CHARS = 250
BRANCH_TOOLS = frozenset({"git_status", "git_commit", "run_command"})


class PlanItem(HarnessState):
    """One step of the agent's plan."""

    text: str
    status: PlanStatus = "pending"


class PlanState(HarnessState):
    """The agent's checklist for the current task."""

    goal: str = ""
    items: list[PlanItem] = Field(default_factory=list)


class ChangedFile(HarnessState):
    """A file the agent changed in its workspace this session."""

    path: str
    status: FileStatus = "modified"
    added: int = 0
    removed: int = 0
    tool_id: str = ""


class Commit(HarnessState):
    """A local commit the agent made."""

    hash: str
    message: str


class ChangesState(HarnessState):
    """Files changed and commits made, newest last."""

    files: list[ChangedFile] = Field(default_factory=list)
    commits: list[Commit] = Field(default_factory=list)


class WorkspaceState(HarnessState):
    """What the workspace's terminal did last, and the branch it is on.

    Every field has a default: a run restores this from CarriedState JSON written by an
    older version.
    """

    # The workspace repo's checked-out branch ("HEAD (no branch)" when detached), from the
    # latest git_status, git_commit or run_command that reported one; "" until then.
    branch: str = ""
    last_command: str = ""
    last_exit_code: int | None = None
    commands_run: int = 0


# -- update_plan ------------------------------------------------------------------------


class PlanUpdate(BaseModel):
    """The full plan, replacing the previous one."""

    goal: str = Field("", description="One line saying what you are trying to achieve.")
    items: list[PlanItem] = Field(
        description="Every step, in order, with its current status. Send the whole list each time."
    )


class PlanReply(BaseModel):
    """Acknowledgement of a plan update."""

    summary: str

    def __str__(self) -> str:
        return self.summary


@agent.tool_defn(inherently_safe=True)
async def update_plan(update: PlanUpdate, plan: agent.Injected[StateRef[PlanState]]) -> PlanReply:
    """Write your plan for the current task as a checklist the user can watch.

    Call it at the start of a multi-step task, and again every time a step finishes (mark it
    `done` and the next one `in_progress`); a plan left unticked tells the user nothing was
    done. Send the whole list each time; mark at most one step `in_progress`.
    """
    # Bounded: every plan change becomes a state patch in workflow history.
    items = [
        PlanItem(text=i.text.strip()[:MAX_PLAN_TEXT], status=i.status) for i in update.items[:MAX_PLAN_ITEMS]
    ]
    plan.set(PlanState(goal=update.goal.strip()[:MAX_PLAN_TEXT], items=items))
    done = sum(1 for i in items if i.status == "done")
    summary = f"Plan updated: {done}/{len(items)} steps done."
    if done < len(items):
        # Models tend to write a plan and never tick it; the reply is where they look next.
        summary += " Call update_plan again as each step finishes, marking it done."
    return PlanReply(summary=summary)


STATE_TOOLS = (update_plan,)


# -- folding tool results into state ----------------------------------------------------


def fold_tool_result(
    changes: StateRef[ChangesState],
    workspace: StateRef[WorkspaceState],
    *,
    tool_name: str,
    tool_id: str,
    args: dict[str, Any],
    result: Any,
) -> None:
    """Update the Changes and Workspace panels from one tool result.

    Duck-typed over the tool result models (see tools/workspace.py): anything with
    ``changed_files`` (run_command, or a subagent's ``task`` with line counts), ``path`` with
    ``added_lines``/``removed_lines`` (write/edit), ``commit_hash`` (git_commit), ``commits``
    (``task``), or ``exit_code``.
    """
    touched = touched_files(result)
    if touched:
        with changes.mutate() as draft:
            merge_changes(draft, touched, [], tool_id=tool_id)
    commits = new_commits(result, args)
    if commits:
        with changes.mutate() as draft:
            merge_changes(draft, [], commits, tool_id=tool_id)

    # git_status, git_commit and run_command report the branch; None means unknown, so the
    # last known branch stays.
    branch = getattr(result, "branch", None) if tool_name in BRANCH_TOOLS else None
    branch = branch[:MAX_BRANCH_CHARS] if isinstance(branch, str) and branch else None
    if tool_name == "run_command" and hasattr(result, "exit_code"):
        with workspace.mutate() as draft:
            draft.last_command = str(args.get("command", ""))[:500]
            draft.last_exit_code = result.exit_code
            draft.commands_run += 1
            if branch is not None:
                draft.branch = branch
    elif branch is not None and branch != workspace.current.branch:
        with workspace.mutate() as draft:
            draft.branch = branch


def touched_files(result: Any) -> list[tuple[str, str, int, int]]:
    """(path, status, added, removed) for each file a tool result says it changed."""
    touched: list[tuple[str, str, int, int]] = []
    if getattr(result, "path", None) and hasattr(result, "added_lines"):
        status = "added" if getattr(result, "created", False) else "modified"
        touched.append((result.path, status, result.added_lines or 0, result.removed_lines or 0))
    for item in getattr(result, "changed_files", None) or []:
        path = _field(item, "path")
        if path:
            status = _normalize_status(_field(item, "status", "modified"))
            touched.append((path, status, _field(item, "added", 0) or 0, _field(item, "removed", 0) or 0))
    return touched


def new_commits(result: Any, args: dict[str, Any]) -> list[tuple[str, str]]:
    """(hash, message) for each commit a tool result reports."""
    # GitCommitResult reports `committed` and the new hash as `commit`.
    commit_hash = getattr(result, "commit_hash", None) or (
        getattr(result, "commit", None) if getattr(result, "committed", False) else None
    )
    commits: list[tuple[str, str]] = []
    if commit_hash:
        message = getattr(result, "message", "") or args.get("message", "") or getattr(result, "summary", "")
        commits.append((str(commit_hash), str(message)))
    for c in getattr(result, "commits", None) or []:
        if getattr(c, "hash", None):
            commits.append((str(c.hash), str(getattr(c, "message", ""))))
    return commits


def merge_changes(
    draft: ChangesState,
    touched: list[tuple[str, str, int, int]],
    commits: list[tuple[str, str]],
    *,
    tool_id: str = "",
) -> None:
    """Fold touched files and commits into ``draft`` (in place), bounded."""
    for path, status, added, removed in touched:
        existing = next((f for f in draft.files if f.path == path), None)
        if existing is None:
            draft.files.append(ChangedFile(path=path, status=status, added=added, removed=removed, tool_id=tool_id))
        else:
            existing.added += added
            existing.removed += removed
            existing.tool_id = tool_id
            if status == "deleted" or existing.status != "added":
                existing.status = status
    if len(draft.files) > MAX_CHANGED_FILES:
        del draft.files[: len(draft.files) - MAX_CHANGED_FILES]
    for hash_, message in commits:
        draft.commits.append(Commit(hash=hash_, message=message[:300]))
    if len(draft.commits) > MAX_COMMITS:
        del draft.commits[: len(draft.commits) - MAX_COMMITS]


def _field(item: Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def _normalize_status(status: str) -> str:
    s = (status or "").lower()
    if s.startswith("a") or s in ("new", "created", "untracked", "?"):
        return "added"
    if s.startswith("d"):
        return "deleted"
    if s.startswith("r"):
        return "renamed"
    return "modified"
