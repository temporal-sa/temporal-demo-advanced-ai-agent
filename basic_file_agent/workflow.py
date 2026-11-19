"""The agent Workflow: Agent Kit owns turns, tool approvals and the event stream;
Pydantic AI owns the model/tool loop inside each turn."""

from __future__ import annotations

import os

from pydantic import BaseModel
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelMessage
    from temporal_agent_kit.ai_sdks.pydantic_ai import (
        KitDeps,
        TemporalAgentKitDurability,
        build_kit_toolset,
    )
    from temporal_agent_kit import AgentWorkflowRunner, agent
    from temporal_agent_kit.agent_protocol import AgentConfig, MidTurn, ToolApprovalPolicy
    from temporalio.contrib.workflow_streams import WorkflowStream

    from basic_file_agent import WORKFLOW_TYPE
    from basic_file_agent.tools import TOOLS

MODEL = os.environ.get("BASIC_FILE_AGENT_MODEL", "anthropic:claude-haiku-5-5")

# Built once at import, outside any Workflow, as Pydantic AI's Temporal integration requires.
# The kit tools run in-workflow (build_kit_toolset marks them so), so the kit can gate and
# stream them. TemporalAgentKitDurability runs the model in activities and streams its output
# onto the turn.
FILE_AGENT = Agent(
    MODEL,
    name="basic-file-agent",
    instructions="You work with files in a workspace. Use relative paths. Be brief.",
    toolsets=[build_kit_toolset(TOOLS, id="files")],
    deps_type=KitDeps,
    model_settings={"anthropic_effort": "low"} if MODEL.startswith("anthropic:") else None,
    capabilities=[TemporalAgentKitDurability()],
)


class Chat(BaseModel):
    """A message from the user."""

    text: str


class Reply(BaseModel):
    """The agent's reply."""

    text: str


@agent.defn(name=WORKFLOW_TYPE)
class BasicFileAgent:
    @agent.init
    def __init__(self, config: AgentConfig) -> None:
        # read_file is inherently safe; write_file only touches the workspace, so allow it too.
        policy = ToolApprovalPolicy.allow_tools(["write_file"], also_inherently_safe=True)
        self._runner = AgentWorkflowRunner(config, stream=WorkflowStream(), approval_policy_default=policy)
        self._history: list[ModelMessage] = []

    @agent.accepts(mid_turn=MidTurn.ENQUEUE)
    async def chat(self, msg: Chat) -> Reply:
        """Send the agent a message and get its reply."""
        result = await FILE_AGENT.run(
            msg.text, deps=KitDeps(runner=self._runner), message_history=self._history
        )
        self._history = result.all_messages()
        return Reply(text=result.output)
