"""What every agent shares.

`Context` holds what every agent needs: the database, the log recorder, the LLM
provider, where the knowledge files live, the conversation mode, the id of the
current conversation and the Critic that validates what the agents produce.

`Agent` gives every agent two shortcuts:
- send(...)     creates a JSON Message and records it in the log
- call_llm(...) calls the LLM, records how long it took and checks the shape of the answer (answers.py)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from ..answers import shape
from ..database import Database
from ..log import Recorder
from ..messages import SYSTEM, Message
from ..providers import InvalidAnswer

MODES = {
    "in_meeting": "during the meeting: fast, only uses experts that already exist",
    "after_meeting": "after the meeting: may create new experts",
}


@dataclass
class Context:
    db: Database
    recorder: Recorder
    provider: object
    knowledge_folder: str
    mode: str = "after_meeting"
    conversation_id: str | None = None
    with_quality: bool = True  # consult the quality expert (FR/NFR class and subtype)
    critic: object | None = None  # the Critic agent (validation); None switches validation off
    hub: object | None = None  # the connection to the memory and traceability service (hub/); None = not connected


def requirement_summary(req: dict) -> dict:
    """Compact form of a requirement to send to the agents."""
    c = req["content"]
    return {
        "id": req["id"],
        "version": req["current_version"],
        "status": req["status"],
        "title": c.get("title"),
        "statement": c.get("statement"),
        "story": c.get("story"),
        "topics": req["topics"],
    }


def as_json(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1)


class Agent:
    name = "agent"

    def __init__(self, ctx: Context):
        self.ctx = ctx

    def send(self, recipient: str, kind: str, content: dict) -> Message:
        msg = Message(sender=self.name, recipient=recipient, kind=kind, content=content, conversation_id=self.ctx.conversation_id)
        return self.ctx.recorder.record(msg)

    def call_llm(self, role: str, system: str, messages: list[dict], **options) -> dict:
        """`options` (e.g. temperature) go to the provider only when given."""
        start = time.perf_counter()
        hub, invalid = self.ctx.hub, None
        try:
            answer = self.ctx.provider.complete_json(role, system, messages, **options)
        except InvalidAnswer as exc:  # the model answered, but not in JSON: as if it had said nothing
            self.send(SYSTEM, "error", {"role": role, "error": str(exc)})
            answer, invalid = {}, exc
        except Exception as exc:
            self.send(SYSTEM, "error", {"role": role, "error": str(exc)})
            if hub:
                hub.on_llm_call(self.name, role, system, messages, None, None, round((time.perf_counter() - start) * 1000), exc)
            raise
        duration_ms = round((time.perf_counter() - start) * 1000)
        usage = getattr(self.ctx.provider, "last_usage", {})
        self.send("llm", "llm_call", {
            "role": role,
            "provider": getattr(self.ctx.provider, "name", "?"),
            "duration_ms": duration_ms,
            **usage,
        })
        if hub:
            hub.on_llm_call(self.name, role, system, messages, answer, usage, duration_ms, invalid)
        return shape(role, answer)
