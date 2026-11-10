"""The harness event-stream backend this branch runs on: Redis instead of WorkflowStream.

Workers and the API read and write through ``EVENTS``, which holds the connection. Workflow
code builds ``RedisEventStream()`` itself: it needs no URL, since its events go through the
writer the worker's AgentHarnessPlugin runs.
"""

import os

from temporal_agent_harness.harness.redis_event_stream import RedisEventStream

EVENTS = RedisEventStream(os.environ.get("AGENT_EVENT_STREAM_URL", "redis://127.0.0.1:6390/1"))
