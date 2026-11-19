"""An idle CodingAgent session closes itself (Agent Kit ``close_after_idle``).

Runs the real workflow on a local dev server (``WorkflowEnvironment.start_local``). No model
call is made: the sessions stay idle.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from datetime import timedelta

os.environ.setdefault("ANTHROPIC_API_KEY", "idle-close-not-used")

from temporalio.client import Client, WorkflowExecutionStatus
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.testing import WorkflowEnvironment

from temporal_agent_kit.agent_protocol import TURN_EVENTS_TOPIC, AgentConfig, AgentEvent
from temporal_agent_kit.agent_protocol.events import AgentClosed


class IdleCloseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from pydantic_ai.durable_exec.temporal import PydanticAIPlugin
        from temporal_agent_kit.plugin import AgentKitPlugin

        from simple_chat_agent.agent.kit_ext import payload_storage
        from simple_chat_agent.agent.tools.workspace import TEST_TOOLS, WORKSPACE_TOOLS
        from tests.integration import support

        self.env = await WorkflowEnvironment.start_local()
        self.client = await Client.connect(
            self.env.client.service_client.config.target_host,
            namespace=self.env.client.namespace,
            plugins=[PydanticAIPlugin(), AgentKitPlugin(
                tools=[*WORKSPACE_TOOLS, *TEST_TOOLS], large_payload_offload=payload_storage()
            )],
        )
        self.queue = f"idle-{uuid.uuid4().hex[:6]}"
        self.worker = support.running_worker(self.client, self.queue)
        await self.worker.__aenter__()

    async def asyncTearDown(self) -> None:
        await self.worker.__aexit__(None, None, None)
        await self.env.shutdown()

    async def start(self, config: AgentConfig) -> object:
        from simple_chat_agent.agent.contracts import AGENT_WORKFLOW_TYPE, CodingAgentInit

        workflow_id = f"idle-{uuid.uuid4().hex[:10]}"
        data = CodingAgentInit(
            session_id=workflow_id, workspace_id=workflow_id, environment="local", owner_id="test-user"
        )
        return await self.client.start_workflow(
            AGENT_WORKFLOW_TYPE, args=[config, data], id=workflow_id, task_queue=self.queue
        )

    async def closed_event(self, workflow_id: str) -> AgentClosed:
        sub = WorkflowStreamClient.create(self.client, workflow_id).subscribe(
            topics=[TURN_EVENTS_TOPIC], result_type=AgentEvent
        )
        async for item in sub:
            if isinstance(item.data.event, AgentClosed):
                return item.data.event
        self.fail("no agent_closed event")

    async def test_an_idle_session_closes_itself(self) -> None:
        handle = await self.start(AgentConfig(close_after_idle=timedelta(seconds=2)))
        await asyncio.wait_for(handle.result(), timeout=30)
        self.assertEqual((await handle.describe()).status, WorkflowExecutionStatus.COMPLETED)
        self.assertEqual((await self.closed_event(handle.id)).reason, "idle_timeout_expired")

    async def test_zero_never_closes_on_idle(self) -> None:
        handle = await self.start(AgentConfig(close_after_idle=timedelta(0)))
        await asyncio.sleep(4)
        self.assertEqual((await handle.describe()).status, WorkflowExecutionStatus.RUNNING)
        await handle.signal("close")
        await asyncio.wait_for(handle.result(), timeout=30)
        self.assertEqual((await self.closed_event(handle.id)).reason, "requested")


if __name__ == "__main__":
    unittest.main()
