"""The connection to the memory and traceability service (hub/), against a fake service in memory.

No network and no real model: the service is an httpx transport that stores what it receives, and the
LLM is the simulated provider. What is checked is what GigaBrain sends and what it does with the answers.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import unittest
import uuid
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from gigabrain.agents import expert as expert_module
from gigabrain.agents import twin as twin_module
from gigabrain.api import create_app
from gigabrain.council import Council, open_context
from gigabrain.hub import HubConfigError, HubRuntime, HubSettings, connect, decisions
from gigabrain.providers import SimulatedProvider
from gigabrain.validation import host


class FakeService:
    """Just enough of the service: it stores what it is sent and answers like the real one."""

    def __init__(self, recent=None, hits=None, down=None, facts=None):
        self.recent, self.hits, self.down = recent or [], hits or [], down
        self.facts = facts or []  # what Mem0 answers when searched
        self.hold: threading.Event | None = None  # while set up and not set, a Mem0 write waits (a slow one)
        self.mem_searches: list[dict] = []
        self.mem_writes: list[dict] = []
        self.calls: list[tuple[str, str, dict | None]] = []
        self.projects, self.models, self.agents, self.prompts = {}, {}, {}, {}
        self.sessions, self.executions, self.steps, self.documents = {}, {}, {}, []
        self.searches: list[dict] = []
        self.agent_by_name: dict[str, str] = {}
        self.prompt_by_text: dict[tuple[str, str], str] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        method, path = request.method, request.url.path
        self.calls.append((method, path, body))
        if self.down and self.down(method, path):
            raise httpx.ConnectError("the service is down", request=request)
        parts = path.strip("/").split("/")
        if (method, path) == ("POST", "/projetos"):
            self.projects[body["id"]] = body
            return httpx.Response(200, json={"id": body["id"], "nome": body["nome"]})
        if (method, path) == ("POST", "/modelos"):
            self.models[(body["provedor"], body["nome"])] = len(self.models) + 1
            return httpx.Response(200, json={"id": self.models[(body["provedor"], body["nome"])]})
        if (method, path) == ("POST", "/agentes"):  # an agent is found by name, as in the service
            agent_id = self.agent_by_name.setdefault(body["nome"], body["id"])
            self.agents[agent_id] = body
            return httpx.Response(200, json={"id": agent_id})
        if method == "POST" and parts[0] == "agentes" and parts[-1] == "versoes":
            if parts[1] not in self.agents:
                return httpx.Response(404, json={"detail": "Agente não encontrado"})
            return httpx.Response(200, json={"id": body["id"]})
        if (method, path) == ("POST", "/prompts"):  # a prompt is found by its text, as in the service
            prompt_id = self.prompt_by_text.setdefault((body["chave"], body["template"]), body["id"])
            self.prompts[prompt_id] = body
            return httpx.Response(200, json={"id": prompt_id})
        if (method, path) == ("POST", "/sessoes"):
            self.sessions[body["id"]] = {**body, "closed": False}
            return httpx.Response(200, json={"id": body["id"]})
        if method == "PATCH" and parts[0] == "sessoes":
            self.sessions[parts[1]]["closed"] = True
            return httpx.Response(200, json={})
        if (method, path) == ("POST", "/execucoes"):
            self.executions[body["id"]] = {**body, "status": "running"}
            return httpx.Response(200, json={"id": body["id"]})
        if method == "PATCH" and parts[0] == "execucoes":
            self.executions[parts[1]].update(body)
            return httpx.Response(200, json={})
        if method == "POST" and parts[0] == "execucoes" and parts[-1] == "passos":
            if parts[1] not in self.executions:
                return httpx.Response(404, json={"detail": "Execução não encontrada"})
            if body.get("prompt_id") and body["prompt_id"] not in self.prompts:  # a foreign key, as in the service
                return httpx.Response(500, json={"detail": "prompt_id violates a foreign key"})
            self.steps[body["id"]] = {**body, "execucao_id": parts[1]}
            return httpx.Response(200, json={"id": body["id"]})
        if (method, path) == ("POST", "/documentos"):
            self.documents.append(body)
            return httpx.Response(200, json={"id": str(uuid.uuid4()), "status": "indexado"})
        if (method, path) == ("POST", "/documentos/buscar"):
            self.searches.append(body)
            return httpx.Response(200, json=self.hits)
        if (method, path) == ("POST", "/documentos/recentes"):
            return httpx.Response(200, json=self.recent)
        if (method, path) == ("POST", "/memoria/buscar"):
            self.mem_searches.append(body)
            return httpx.Response(200, json=self.facts)
        if (method, path) == ("POST", "/memoria/conversas"):
            if self.hold is not None:
                self.hold.wait(10)
            self.mem_writes.append(body)
            return httpx.Response(200, json={"resultados": [{"id": "m", "event": "ADD"}]})
        return httpx.Response(404, json={"detail": f"{method} {path}"})

    # ------------------------------------------------------------------ what a test asks

    def steps_of(self, tipo: str) -> list[dict]:
        return [s for s in self.steps.values() if s["tipo"] == tipo]

    def text_of(self, record_key: str) -> str:
        doc = next(d for d in self.documents if d["chave"] == record_key)
        return base64.b64decode(doc["conteudo_base64"]).decode("utf-8")

    def keys(self) -> list[str]:
        return [d["chave"] for d in self.documents]

    def doc(self, record_key: str) -> dict:
        return next(d for d in self.documents if d["chave"] == record_key)

    def already_has_agent(self, name: str, agent_id: str) -> None:
        self.agent_by_name[name] = agent_id
        self.agents[agent_id] = {"nome": name}

    def already_has_prompt(self, key: str, template: str, prompt_id: str) -> None:
        self.prompt_by_text[(key, template)] = prompt_id
        self.prompts[prompt_id] = {"chave": key, "template": template}


class Spy(SimulatedProvider):
    """The simulated provider that remembers what it was asked."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, str, list[dict]]] = []

    def complete_json(self, role, system, messages, **options):
        self.calls.append((role, system, list(messages)))  # a copy: the Twin keeps appending to its own list
        return super().complete_json(role, system, messages, **options)


class Billed(SimulatedProvider):
    def complete_json(self, role, system, messages, **options):
        answer = super().complete_json(role, system, messages, **options)
        self.last_usage = {"tokens_in": 120, "tokens_out": 30}
        return answer


class Failing(SimulatedProvider):
    def complete_json(self, role, system, messages, **options):
        raise RuntimeError("401 your API key has expired")


FIRST = "I want to notify the customer when the payment is confirmed"
SECOND = "The customer gets an email within 1 minute"
MEETING = """\
Analyst: Okay, next topic: login. How should that work?
Client: Users must log in with their email and a password of at least 8 characters.
Analyst: So, let's talk about reports. What do you need there?
Client: The system shall export monthly sales reports to PDF for the finance team.
"""
CONTINUATION = """\
Analyst: Back to login for a moment.
Client: Users must log in with their email and a password of at least 12 characters.
"""


class HubCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = os.path.join(self.tmp.name, "projects", "demo")
        patcher = mock.patch("gigabrain.hub.runtime.RETRIES", ())  # a failing service must not make a test wait
        patcher.start()
        self.addCleanup(patcher.stop)

    def runtime(self, service: FakeService, **settings) -> HubRuntime:
        return HubRuntime(HubSettings(url="http://hub.test", **settings), transport=httpx.MockTransport(service))

    def open(self, service, provider=None, **settings):
        rt = self.runtime(service, **settings)
        ctx = open_context(self.folder, provider or SimulatedProvider(), validation=False, hub=rt)
        self.addCleanup(ctx.db.close)
        return ctx, rt

    def council(self, service, provider=None, **settings):
        ctx, rt = self.open(service, provider, **settings)
        council = Council(ctx)
        council.start()
        return council, rt

    def flush(self, rt):
        self.assertTrue(rt.flush(10), "the queue did not empty")


class TestSettings(unittest.TestCase):
    def test_url_is_required(self):
        with self.assertRaises(HubConfigError):
            HubSettings.from_env({})

    def test_flags_and_defaults(self):
        s = HubSettings.from_env({"GIGABRAIN_HUB_URL": "http://x:8002", "GIGABRAIN_HUB_MEMORY": "off", "GIGABRAIN_HUB_MIN_SCORE": "0.2"})
        self.assertEqual((s.url, s.trace, s.memory, s.min_score, s.user, s.token), ("http://x:8002", True, False, 0.2, "po", None))

    def test_the_memory_of_the_po_has_its_own_switch_and_floor(self):
        s = HubSettings.from_env({"GIGABRAIN_HUB_URL": "http://x", "GIGABRAIN_HUB_PERSON_MEMORY": "0", "GIGABRAIN_HUB_PERSON_MIN_SCORE": "0.4"})
        self.assertEqual((s.person_memory, s.person_min_score, s.memory), (False, 0.4, True))
        d = HubSettings.from_env({"GIGABRAIN_HUB_URL": "http://x"})
        self.assertEqual((d.person_memory, d.person_min_score), (True, 0.2))

    def test_an_invalid_number_is_reported(self):
        with self.assertRaises(HubConfigError):
            HubSettings.from_env({"GIGABRAIN_HUB_URL": "http://x", "GIGABRAIN_HUB_MIN_SCORE": "high"})


class TestOffByDefault(HubCase):
    def test_nothing_is_connected_unless_asked(self):
        ctx = open_context(self.folder, SimulatedProvider(), validation=False)
        self.addCleanup(ctx.db.close)
        self.assertIsNone(ctx.hub)
        self.assertIsNone(ctx.recorder.hub)
        council = Council(ctx)
        council.start()
        self.assertEqual(council.talk(FIRST)["action"], "ask")  # the conversation runs as ever

    def test_both_flags_off_connects_nothing(self):
        service = FakeService()
        ctx, _ = self.open(service, trace=False, memory=False)
        self.assertIsNone(ctx.hub)
        self.assertEqual(service.calls, [])


class TestTraceability(HubCase):
    def test_a_conversation_is_a_session_and_each_thing_the_po_does_an_execution(self):
        service = FakeService()
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.approve()
        self.flush(rt)

        self.assertEqual([p["nome"] for p in service.projects.values()], ["gigabrain:demo"])
        self.assertIn(("simulated", "simulated"), service.models)
        (session,) = service.sessions.values()
        self.assertEqual((session["usuario_externo_id"], session["closed"]), ("po", False))  # approving does not end it
        roots = [e for e in service.executions.values() if "execucao_pai_id" not in e]
        self.assertEqual([e["entrada"]["kind"] for e in roots], ["open", "po_message", "po_message", "approval"])
        children = [e for e in service.executions.values() if "execucao_pai_id" in e]
        self.assertTrue({"leader"} <= {e["entrada"]["agent"] for e in children})
        self.assertTrue(any(e["entrada"]["agent"].startswith("expert:") for e in children))
        self.assertTrue(all(e["execucao_pai_id"] in {r["id"] for r in roots} for e in children))
        self.assertTrue(all(e["status"] == "success" for e in service.executions.values()))
        by_execution: dict[str, list[int]] = {}
        for step in service.steps.values():
            by_execution.setdefault(step["execucao_id"], []).append(step["indice_passo"])
        self.assertTrue(set(by_execution) <= set(service.executions))
        for indexes in by_execution.values():
            self.assertEqual(sorted(indexes), list(range(len(indexes))))

        council.end()
        self.flush(rt)
        self.assertTrue(session and next(iter(service.sessions.values()))["closed"])
        self.assertIn("end", [e["entrada"].get("kind") for e in service.executions.values()])

    def test_every_llm_call_of_the_log_is_a_step_with_its_prompt(self):
        service = FakeService()
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.approve()
        self.flush(rt)

        in_log = council.ctx.db.con.execute("SELECT COUNT(*) FROM event WHERE kind = 'llm_call'").fetchone()[0]
        calls = service.steps_of("llm_call")
        self.assertGreater(in_log, 0)
        self.assertEqual(len(calls), in_log)
        self.assertTrue(all(c["prompt_id"] in service.prompts for c in calls))  # the prompt was registered first
        self.assertIn("gigabrain.twin", {p["chave"] for p in service.prompts.values()})
        self.assertTrue(all({"role", "messages"} <= set(c["entrada"]) and "answer" in c["saida"] for c in calls))

    def test_tokens_travel_with_the_step(self):
        service = FakeService()
        council, rt = self.council(service, Billed())
        council.talk(FIRST)
        self.flush(rt)

        model_id = service.models[("simulated", "simulated")]
        for call in service.steps_of("llm_call"):
            self.assertEqual({(t["tipo_token"], t["quantidade"]) for t in call["tokens"]}, {("entrada", 120), ("saida", 30)})
            self.assertTrue(all(t["modelo_id"] == model_id for t in call["tokens"]))

    def test_messages_between_agents_are_steps_and_the_llm_is_not_counted_twice(self):
        service = FakeService()
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.approve()
        self.flush(rt)

        types = {s["entrada"]["kind"]: s["tipo"] for s in service.steps.values() if "kind" in s["entrada"]}
        self.assertNotIn("llm_call", types)  # that one is the "llm_call" step, with the prompt and the tokens
        self.assertEqual((types["po_request"], types["query"], types["requirement_saved"], types["draft_board"]),
                         ("message", "handoff", "repository", "message"))

    def test_a_meeting_is_one_execution_with_every_agent_it_set_off(self):
        service = FakeService()
        council, rt = self.council(service)
        council.hear_meeting(MEETING, "Kick-off")
        self.flush(rt)

        meeting = next(e for e in service.executions.values() if e["entrada"].get("kind") == "meeting")
        self.assertEqual((meeting["entrada"]["title"], meeting["status"]), ("Kick-off", "success"))
        self.assertEqual(meeting["saida"]["new"], 2)
        agents = {e["entrada"]["agent"] for e in service.executions.values() if e.get("execucao_pai_id") == meeting["id"]}
        self.assertTrue({"leader"} <= agents and any(a.startswith("expert:") for a in agents))
        in_log = council.ctx.db.con.execute("SELECT COUNT(*) FROM event WHERE kind = 'llm_call'").fetchone()[0]
        self.assertEqual(len(service.steps_of("llm_call")), in_log)

    def test_a_failed_llm_call_is_an_error_step_and_the_turn_is_an_error(self):
        service = FakeService()
        council, rt = self.council(service, Failing())
        with self.assertRaises(RuntimeError):
            council.talk(FIRST)
        self.flush(rt)

        (call,) = service.steps_of("llm_call")
        self.assertEqual(call["status"], "error")
        self.assertEqual((call["eventos"][0]["tipo"], call["eventos"][0]["categoria"]), ("erro", "falha_api"))
        self.assertIn("401", call["eventos"][0]["mensagem"])
        turn = next(e for e in service.executions.values() if e["entrada"].get("kind") == "po_message")
        self.assertEqual(turn["status"], "error")

    def test_trace_off_sends_no_steps_but_the_memory_still_works(self):
        service = FakeService()
        council, rt = self.council(service, trace=False)
        council.talk(FIRST)
        council.talk(SECOND)
        council.approve()
        self.flush(rt)

        self.assertEqual((service.sessions, service.executions, service.steps), ({}, {}, {}))
        self.assertIn("decision:R1:v1", service.keys())

    def test_a_conversation_brought_back_from_the_log_is_the_same_session(self):
        service = FakeService()
        council, rt = self.council(service)
        council.talk(FIRST)
        cid = council.ctx.conversation_id
        ctx2 = open_context(self.folder, SimulatedProvider(), validation=False, hub=rt)
        self.addCleanup(ctx2.db.close)
        Council.resume(ctx2, cid).talk(SECOND)
        self.flush(rt)

        self.assertEqual(len(service.sessions), 1)  # not a second one
        self.assertEqual([e["entrada"].get("kind") for e in service.executions.values() if "execucao_pai_id" not in e][-1], "po_message")


class TestIdsOfTheService(HubCase):
    """The id of an agent or a prompt is the one the service answers with: it may already have it under another."""

    def test_an_agent_and_a_prompt_the_service_already_had_are_used_and_no_step_is_refused(self):
        service = FakeService()
        service.already_has_agent("gigabrain.twin", "00000000-0000-0000-0000-00000000a001")
        service.already_has_prompt("gigabrain.twin", twin_module.SYSTEM_PROMPT + twin_module.MEMORY_NOTE, "00000000-0000-0000-0000-00000000b001")
        service.already_has_prompt("gigabrain.leader_decompose", __import__("gigabrain.agents.leader", fromlist=["x"]).DECOMPOSE_PROMPT,
                                   "00000000-0000-0000-0000-00000000b002")
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.approve()
        self.flush(rt)

        self.assertEqual(rt.dropped, 0)  # nothing was refused, so nothing was lost
        in_log = council.ctx.db.con.execute("SELECT COUNT(*) FROM event WHERE kind = 'llm_call'").fetchone()[0]
        calls = service.steps_of("llm_call")
        self.assertEqual(len(calls), in_log)
        self.assertTrue(all(c["prompt_id"] in service.prompts for c in calls))
        self.assertIn("00000000-0000-0000-0000-00000000b001", {c["prompt_id"] for c in calls})  # the twin's, as the service had it
        twin_executions = [e for e in service.executions.values() if e["entrada"].get("kind") == "po_message"]
        self.assertTrue(all(e["agente_id"] == "00000000-0000-0000-0000-00000000a001" for e in twin_executions))

    def test_a_service_that_cannot_register_does_not_make_the_conversation_wait_for_it(self):
        service = FakeService(down=lambda m, p: p.startswith(("/agentes", "/prompts")))
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        self.flush(rt)

        asked = [p for _, p, _ in service.calls if p.startswith(("/agentes", "/prompts"))]
        self.assertEqual(len(asked), 1)  # it asked once, failed, and stopped asking for a while
        self.assertTrue(service.steps_of("llm_call"))  # the steps went on, just without a prompt
        self.assertTrue(all("prompt_id" not in c for c in service.steps_of("llm_call")))


class TestTheServiceIsDown(HubCase):
    def test_unreachable_at_the_start_the_conversation_runs_without_the_hub(self):
        service = FakeService(down=lambda m, p: True)
        ctx, rt = self.open(service)
        self.assertIsNone(ctx.hub)
        self.assertIsNone(ctx.recorder.hub)
        council = Council(ctx)
        council.start()
        self.assertEqual(council.talk(FIRST)["action"], "ask")

    def test_going_down_in_the_middle_changes_nothing_for_the_conversation(self):
        service = FakeService(down=lambda m, p: p.startswith(("/execucoes", "/sessoes", "/documentos", "/prompts", "/agentes")))
        council, rt = self.council(service)
        actions = [council.talk(FIRST)["action"], council.talk(SECOND)["action"], council.approve()["action"]]
        self.flush(rt)

        self.assertEqual(actions, ["ask", "propose", "approved"])
        self.assertEqual(council.ctx.db.get_requirement("R1")["status"], "active")  # GigaBrain's own repository is untouched
        self.assertGreater(rt.dropped, 0)  # and the loss is counted, not hidden

    def test_connect_returns_none_when_the_service_does_not_answer(self):
        service = FakeService(down=lambda m, p: True)
        self.assertIsNone(connect(self.runtime(service), "demo", SimulatedProvider()))


class TestDecisionMemory(HubCase):
    def test_an_approved_draft_is_written_with_its_reason_and_who_decided(self):
        service = FakeService()
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.approve()
        council.end()
        self.flush(rt)

        cid = council.ctx.conversation_id
        self.assertEqual(sorted(service.keys()), sorted(["decision:R1:v1", f"decision:{cid}:summary"]))
        text = service.text_of("decision:R1:v1")
        self.assertIn("requirement R1 v1 APPROVED", text)
        self.assertIn("Draft: D1 (chat)", text)
        self.assertIn("Decided by: PO (explicit approval)", text)
        doc = service.doc("decision:R1:v1")
        self.assertEqual((doc["tipo"], doc["contextualizar"], doc["origem"]), ("decisao", False, "gerado_por_agente"))
        meta = doc["metadados"]
        self.assertEqual((meta["tipo_registro"], meta["requisito"], meta["status_decisao"], meta["rascunho"], meta["origem"]),
                         ("requirement_approved", ["R1"], "approved", "D1", "chat"))

    def test_each_record_is_linked_to_the_step_that_wrote_it(self):
        service = FakeService()
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.approve()
        council.end()
        self.flush(rt)

        writes = {s["id"] for s in service.steps_of("memory_write") if "chave" in s["entrada"]}  # the others are turns given to Mem0
        self.assertEqual(len(writes), len(service.documents))
        self.assertTrue(all(d["passo_origem_id"] in writes for d in service.documents))

    def test_a_draft_the_po_changed_by_talking_keeps_what_changed_and_why(self):
        service = FakeService()
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.talk("In D1 the email must go out within 30 seconds, one minute is too slow")
        council.approve()
        self.flush(rt)

        cid = council.ctx.conversation_id
        revised = service.text_of(f"decision:{cid}:D1:rev2:revised")
        self.assertIn("draft D1 CHANGED (now revision 2)", revised)
        self.assertIn("What changed: statement:", revised)
        self.assertIn("one minute is too slow", revised)  # the PO's own words are the reason
        approved = service.text_of("decision:R1:v1")
        self.assertIn("changed 1 time(s)", approved)
        self.assertIn("because the PO said", approved)

    def test_asking_the_twin_to_drop_a_draft_records_a_rejection_with_the_po_words(self):
        service = FakeService()
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.talk("Reject D1, the customers do not want emails")
        self.flush(rt)

        cid = council.ctx.conversation_id
        text = service.text_of(f"decision:{cid}:D1:rev1:rejected-1")
        self.assertIn("draft D1 REJECTED by the PO", text)
        self.assertIn("the customers do not want emails", text)
        self.assertEqual(service.doc(f"decision:{cid}:D1:rev1:rejected-1")["metadados"]["status_decisao"], "rejected")

    def test_rejecting_with_a_button_keeps_the_reason_given_and_a_restore_takes_it_back(self):
        service = FakeService()
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.reject("D1", "not for this release")
        council.restore("D1")
        council.reject("D1")  # and this time without saying why
        self.flush(rt)

        cid = council.ctx.conversation_id
        self.assertIn("not for this release", service.text_of(f"decision:{cid}:D1:rev1:rejected-1"))
        self.assertIn("the PO took back the rejection", service.text_of(f"decision:{cid}:D1:rev1:restored-1"))
        self.assertIn(f"decision:{cid}:D1:rev1:rejected-2", service.keys())  # rejected again: its own record, not a new version of the first
        self.assertEqual(len(service.keys()), len(set(service.keys())))

    def test_a_correction_in_a_later_part_of_a_meeting_is_recorded_with_the_utterance(self):
        service = FakeService()
        council, rt = self.council(service)
        council.hear_meeting(MEETING)
        council.hear_meeting(CONTINUATION)
        self.flush(rt)

        cid = council.ctx.conversation_id
        text = service.text_of(f"decision:{cid}:D1:rev2:revised")
        self.assertIn("a later part of the meeting corrected it (T6)", text)
        self.assertIn("12 characters", text)
        self.assertEqual(service.doc(f"decision:{cid}:D1:rev2:revised")["metadados"]["decidido_por"], "twin")

    def test_a_new_version_says_what_changed_from_the_one_before(self):
        service = FakeService()
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.approve()
        council.end()
        ctx2 = open_context(self.folder, SimulatedProvider(), validation=False, hub=rt)
        self.addCleanup(ctx2.db.close)
        second = Council(ctx2)
        second.start()
        second.talk("update R1 to the customer gets an email within 2 minutes")
        second.approve()
        self.flush(rt)

        self.assertEqual(ctx2.db.get_requirement("R1")["current_version"], 2)
        text = service.text_of("decision:R1:v2")
        self.assertIn("requirement R1 v2 APPROVED", text)
        self.assertIn("Changes since v1:", text)
        self.assertIn("statement:", text)
        self.assertIn("decision:R1:v1", service.keys())  # the old record stays: the memory is a history

    def test_leaving_with_drafts_not_approved_keeps_them_and_the_reason_given(self):
        service = FakeService()
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.end("closed", "too expensive for this release")
        self.flush(rt)

        cid = council.ctx.conversation_id
        closed = service.text_of(f"decision:{cid}:D1:rev1:closed")
        self.assertIn("ended without approval", closed)
        self.assertIn("too expensive for this release", closed)
        summary = service.text_of(f"decision:{cid}:summary")
        self.assertIn("ended: closed", summary)
        self.assertIn("Saved: nothing", summary)
        self.assertIn("1 abandoned", summary)

    def test_leaving_without_saying_why_is_recorded_as_such(self):
        service = FakeService()
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.end()
        self.flush(rt)

        self.assertIn("the PO gave no reason", service.text_of(f"decision:{council.ctx.conversation_id}:D1:rev1:closed"))

    def test_ending_twice_writes_once(self):
        service = FakeService()
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.end()
        council.end()
        self.flush(rt)

        self.assertEqual(len(service.keys()), len(set(service.keys())))

    def test_memory_off_writes_nothing(self):
        service = FakeService()
        council, rt = self.council(service, memory=False)
        council.talk(FIRST)
        council.talk(SECOND)
        council.talk("Reject D1")
        council.end("closed", "no")
        self.flush(rt)

        self.assertEqual((service.documents, service.searches), ([], []))
        self.assertNotIn("/documentos/recentes", [p for _, p, _ in service.calls])
        self.assertTrue(service.steps)  # tracing is independent

    def test_the_log_of_gigabrain_says_what_was_written(self):
        service = FakeService()
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.approve()
        council.end()
        self.flush(rt)

        kinds = [r[0] for r in council.ctx.db.con.execute("SELECT kind FROM event")]
        self.assertEqual(kinds.count("memory_saved"), len(service.documents))


class TestAttachments(HubCase):
    def test_a_file_attached_to_a_message_is_kept_in_the_memory(self):
        service = FakeService()
        council, rt = self.council(service)
        council.talk("Here are the rules\n\n--- Attached document: rules.md ---\n# Rules\n- refunds in 7 days",
                     [("rules.md", "# Rules\n- refunds in 7 days")])
        self.flush(rt)

        doc = service.doc("attachment:rules.md")
        self.assertEqual((doc["tipo"], doc["formato"], doc["origem"], doc["contextualizar"]), ("outro", "md", "upload_humano", True))
        self.assertEqual(doc["metadados"]["conversa"], council.ctx.conversation_id)
        self.assertIn("refunds in 7 days", service.text_of("attachment:rules.md"))
        self.assertIn(doc["passo_origem_id"], {s["id"] for s in service.steps_of("memory_write")})

    def test_through_the_api_the_files_the_reject_reason_and_the_close_reason_all_arrive(self):
        service = FakeService()
        rt = self.runtime(service)
        client = TestClient(create_app(self.tmp.name, SimulatedProvider, validation=False, hub=rt))
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        client.post("/api/v1/projects", json={"name": "demo"})
        cid = client.post("/api/v1/projects/demo/conversations", json={}).json()["id"]
        base = f"/api/v1/projects/demo/conversations/{cid}"

        client.post(f"{base}/messages", data={"text": FIRST}, files=[("files", ("rules.txt", b"refunds are paid in 7 days", "text/plain"))])
        client.post(f"{base}/messages", json={"text": SECOND})
        self.assertEqual(client.request("DELETE", f"{base}/drafts/D1", json={"reason": "not for now"}).status_code, 200)
        client.post(f"{base}/close", json={"reason": "we will come back to it"})
        self.flush(rt)

        self.assertIn("refunds are paid in 7 days", service.text_of("attachment:rules.txt"))
        self.assertIn("not for now", service.text_of(f"decision:{cid}:D1:rev1:rejected-1"))
        self.assertIn("ended: closed", service.text_of(f"decision:{cid}:summary"))


class TestRecords(unittest.TestCase):
    DRAFT = {"draft_id": "D1", "origin": "chat", "revision": 1}

    def test_a_record_is_short_and_always_the_same(self):
        long = "word " * 400
        content = {"title": long, "statement": long, "class": "NFR", "subtype": "PE", "topics": ["payments"],
                   "constraint": {"quality": long, "scope": long, "measure": long}}
        args = dict(conversation_id="c-1", requirement_id="R9", version=2, draft=self.DRAFT, content=content, reason=long,
                    links=[], decided_by="PO", changes=[long] * 5, revisions=[{"by": "the PO said", "reason": long}] * 6,
                    date="2026-11-18")
        a, b = decisions.requirement_approved(**args), decisions.requirement_approved(**args)
        self.assertEqual(a, b)
        self.assertLessEqual(len(a.text), decisions.MAX_CHARS)

    def test_an_nfr_is_recorded_with_its_constraint_card(self):
        content = {"title": "Search time", "statement": "The search shall answer within 2 seconds.", "class": "NFR", "subtype": "PE",
                   "topics": ["search"], "constraint": {"quality": "response time", "scope": "every search", "measure": "95% under 2 s"}}
        text = decisions.requirement_approved(conversation_id="c-1", requirement_id="R2", version=1, draft=self.DRAFT, content=content,
                                              reason="r", links=[], decided_by="PO", changes=None, revisions=[]).text
        self.assertIn("Constraint: quality response time · scope every search · measure 95% under 2 s", text)

    def test_diff_between_versions(self):
        old = {"title": "T", "statement": "The system shall send an email within 1 minute.", "class": "FR", "subtype": None,
               "story": {"role": "customer", "goal": "know", "benefit": "peace"}, "topics": ["payments"],
               "acceptance_criteria": [{"scenario": "Main"}]}
        new = {**old, "statement": "The system shall send an email within 2 minutes.", "class": "NFR", "subtype": "PE",
               "story": {"role": "customer", "goal": "know", "benefit": "trust"}, "topics": ["payments", "emails"],
               "constraint": {"quality": "speed", "scope": "emails", "measure": "2 min"},
               "acceptance_criteria": [{"scenario": "Main"}, {"scenario": "Late"}]}
        changes = "\n".join(decisions.diff_versions(old, new))
        for expected in ("statement:", "within 1 minute", "within 2 minutes", "class:", "story benefit", "constraint quality",
                         "added Late", "topics:"):
            self.assertIn(expected, changes)
        self.assertEqual(decisions.diff_versions(old, dict(old)), [])


class TestMemoryForTheAgents(HubCase):
    FACTS = [{"id": "m1", "memory": "The PO prefers short answers", "score": 0.71, "updated_at": "2026-11-10T09:00:00"},
             {"id": "m2", "memory": "Something that hardly fits", "score": 0.05}]
    RECENT = [
        {"documento_id": "1", "chave": "decision:R1:v1", "versao": 1, "criado_em": "2026-11-10T10:00:00+00:00",
         "metadados": {"tipo_registro": "requirement_approved"},
         "texto": "Decision record: requirement R1 v1 APPROVED. Statement: the customer shall get an email."},
        {"documento_id": "2", "chave": "decision:c-9:D2:rev1:rejected", "versao": 1, "criado_em": "2026-11-09T10:00:00+00:00",
         "metadados": {"tipo_registro": "draft_rejected"}, "texto": "draft D2 REJECTED by the PO. Why: too expensive."},
    ]
    HITS = [{"id": "p-1", "score": 0.61, "payload": {
        "chave_documento": "manual-pagamentos", "versao_documento": 3, "tipo_documento": "especificacao",
        "criado_em": "2026-10-01T09:00:00+00:00", "texto": "Payments are confirmed by the gateway within 30 seconds."}}]

    def test_the_twin_gets_where_the_last_conversations_stopped(self):
        service, spy = FakeService(recent=self.RECENT), Spy()
        council, rt = self.council(service, spy)
        council.talk(FIRST)
        self.flush(rt)

        self.assertEqual([r["ref"] for r in council.twin.briefing], ["decision:R1:v1@v1", "decision:c-9:D2:rev1:rejected@v1"])
        role, system, messages = spy.calls[0]
        self.assertEqual(role, "twin")
        self.assertTrue(system.endswith(twin_module.MEMORY_NOTE))
        self.assertEqual([r["ref"] for r in json.loads(messages[0]["content"])["memory"]], [r["ref"] for r in council.twin.briefing])
        db = council.ctx.db
        self.assertEqual(db.con.execute("SELECT COUNT(*) FROM event WHERE kind = 'memory_briefing'").fetchone()[0], 1)
        self.assertEqual(len(service.steps_of("memory_read")), 1)

    def test_without_a_past_there_is_no_briefing_but_the_prompt_already_tells_what_may_come(self):
        service, spy = FakeService(), Spy()
        council, rt = self.council(service, spy)
        council.talk(FIRST)
        council.talk(SECOND)

        self.assertIsNone(council.twin.briefing)
        twin_calls = [(s, m) for r, s, m in spy.calls if r == "twin"]
        # the history of a conversation is sent again at each message: its prompt must not change in the middle
        self.assertTrue(all(s == twin_module.SYSTEM_PROMPT + twin_module.MEMORY_NOTE for s, _ in twin_calls))
        self.assertTrue(all("memory" not in json.loads(m[-1]["content"]) for _, m in twin_calls))

    def test_with_the_memory_off_the_prompt_is_the_one_of_always(self):
        service, spy = FakeService(recent=self.RECENT, facts=self.FACTS), Spy()
        council, rt = self.council(service, spy, memory=False)
        council.talk(FIRST)

        self.assertTrue(all(s == twin_module.SYSTEM_PROMPT for r, s, _ in spy.calls if r == "twin"))

    def test_a_briefing_is_in_the_first_message_only(self):
        service, spy = FakeService(recent=self.RECENT), Spy()
        council, rt = self.council(service, spy)
        council.talk(FIRST)
        council.talk(SECOND)

        twin_messages = [m for r, _, m in spy.calls if r == "twin"]
        self.assertIn("memory", json.loads(twin_messages[0][0]["content"]))
        for later in twin_messages[1:]:
            self.assertTrue(all("memory" not in json.loads(m["content"]) for m in later[1:] if m["role"] == "user"))

    def test_an_expert_searches_the_memory_when_it_is_consulted(self):
        service, spy = FakeService(hits=self.HITS), Spy()
        council, rt = self.council(service, spy)
        council.talk(FIRST)
        self.flush(rt)

        expert_calls = [(s, m) for r, s, m in spy.calls if r == "expert_answer"]
        self.assertTrue(expert_calls)
        system, messages = expert_calls[0]
        data = json.loads(messages[0]["content"])
        self.assertTrue(system.endswith(expert_module.MEMORY_NOTE))
        self.assertEqual(data["memory"][0]["ref"], "manual-pagamentos@v3")
        self.assertIn("30 seconds", data["memory"][0]["text"])
        self.assertEqual(len(service.searches), len(expert_calls))  # one for each expert consulted
        search = service.searches[0]
        self.assertEqual((search["filtros_metadados"], search["score_minimo"]), ({}, 0.1))
        retrievals = {s["id"]: s for s in service.steps_of("retrieval")}
        self.assertIn(search["passo_execucao_id"], retrievals)  # the service links the search to its step
        self.assertTrue(retrievals[search["passo_execucao_id"]]["execucao_id"] in service.executions)

    def test_what_the_memory_had_is_in_the_log_and_the_saved_requirement_cites_it(self):
        service = FakeService(hits=self.HITS)
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.approve()

        db = council.ctx.db
        recalled = [json.loads(r[0]) for r in db.con.execute("SELECT content FROM event WHERE kind = 'memory_recall'")]
        self.assertEqual(recalled[0]["records"][0]["ref"], "manual-pagamentos@v3")
        self.assertIn({"kind": "memory", "ref": "manual-pagamentos@v3"}, db.get_requirement("R1")["sources"])

    def test_the_critic_can_read_a_memory_source_a_requirement_cites(self):
        service = FakeService(hits=self.HITS)
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.approve()

        db = council.ctx.db
        conversation_id = db.requirement_versions("R1")[0]["conversation_id"]
        evidence, sources = host.RepositoryReader(db).declared(db.get_requirement("R1"), conversation_id)
        self.assertIn("memory:manual-pagamentos@v3", [e.source_id for e in evidence])
        memory = next(s for s in sources if s.id == "memory:manual-pagamentos@v3")
        self.assertEqual(memory.kind, "memory")
        self.assertIn("30 seconds", memory.text)
        self.assertIsNotNone(host.validation_of(db, "R1"))  # scoring a saved requirement that cites it does not break

    def test_while_a_proposal_is_made_the_critic_gets_the_memory_as_a_source(self):
        service = FakeService(hits=self.HITS)
        council, rt = self.council(service)
        council.talk(FIRST)
        evidence, sources = host.conversation_evidence(council.ctx.db, council.ctx.conversation_id)
        self.assertIn("memory:manual-pagamentos@v3", [e.source_id for e in evidence])
        self.assertIn("memory", {s.kind for s in sources})

    def test_a_search_that_fails_leaves_the_expert_with_what_it_always_had(self):
        service = FakeService(hits=self.HITS, down=lambda m, p: p == "/documentos/buscar")
        spy = Spy()
        council, rt = self.council(service, spy)
        self.assertEqual(council.talk(FIRST)["action"], "ask")
        for role, system, messages in spy.calls:
            if role == "expert_answer":
                self.assertNotIn("memory", json.loads(messages[0]["content"]))

    def test_memory_off_gives_the_agents_nothing(self):
        service, spy = FakeService(recent=self.RECENT, hits=self.HITS), Spy()
        council, rt = self.council(service, spy, memory=False)
        council.talk(FIRST)
        council.talk(SECOND)

        self.assertEqual(service.searches, [])
        for _, system, messages in spy.calls:
            self.assertNotIn("memory service", system)  # neither note was added to a prompt
            for message in messages:
                if message["role"] == "user":
                    self.assertNotIn("memory", json.loads(message["content"]))


class TestMemoryOfThePo(HubCase):
    """Mem0 as the Twin's own memory of the PO: it gets the talk, and hands back short facts."""

    FACTS = TestMemoryForTheAgents.FACTS

    def test_the_twin_gets_the_facts_that_fit_with_the_message_and_the_prompt_says_how_to_use_them(self):
        service, spy = FakeService(facts=self.FACTS), Spy()
        council, rt = self.council(service, spy)
        council.talk(FIRST)
        self.flush(rt)

        role, system, messages = spy.calls[0]
        data = json.loads(messages[-1]["content"])
        self.assertEqual([f["ref"] for f in data["po_memory"]], ["mem0:m1"])  # m2 is below the floor
        self.assertEqual(data["po_memory"][0]["text"], "The PO prefers short answers")
        self.assertTrue(system.endswith(twin_module.MEMORY_NOTE))
        self.assertIn("trust the\n  record", twin_module.MEMORY_NOTE)  # the records win over the facts
        search = service.mem_searches[0]
        self.assertEqual((search["usuario_externo_id"], search["query"], search["projeto_agente_id"]),
                         ("po", FIRST, str(council.ctx.hub.person_agent_id)))
        retrievals = {s["id"]: s for s in service.steps_of("retrieval")}
        self.assertEqual(retrievals[search["passo_execucao_id"]]["entrada"]["store"], "mem0")
        logged = [json.loads(r[0]) for r in council.ctx.db.con.execute("SELECT content FROM event WHERE kind = 'memory_facts'")]
        self.assertEqual([f["ref"] for f in logged[0]["facts"]], ["mem0:m1"])

    def test_the_facts_are_not_a_source_for_the_critic(self):
        service = FakeService(facts=self.FACTS)
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)

        evidence, sources = host.conversation_evidence(council.ctx.db, council.ctx.conversation_id)
        self.assertNotIn("memory", {s.kind for s in sources})  # memory is context, not a source of requirements

    def test_each_turn_goes_to_mem0_with_the_previous_message_of_the_twin_as_context(self):
        service = FakeService()
        council, rt = self.council(service)
        first = council.talk(FIRST)
        council.talk(SECOND)
        self.flush(rt)

        a, b = service.mem_writes
        self.assertEqual([m["role"] for m in a["mensagens"]], ["user", "assistant"])
        self.assertEqual(a["mensagens"][0]["content"], FIRST)
        self.assertEqual(a["mensagens"][1]["content"], first["message_to_po"])
        self.assertEqual([m["role"] for m in b["mensagens"]], ["assistant", "user", "assistant"])
        self.assertEqual(b["mensagens"][0]["content"], first["message_to_po"])  # the question the PO is answering
        self.assertEqual(b["mensagens"][1]["content"], SECOND)
        self.assertEqual((a["usuario_externo_id"], a["projeto_agente_id"]), (b["usuario_externo_id"], b["projeto_agente_id"]))
        writes = {s["id"]: s for s in service.steps_of("memory_write")}
        self.assertEqual({a["passo_execucao_id"], b["passo_execucao_id"]} <= set(writes), True)
        self.assertEqual(writes[a["passo_execucao_id"]]["entrada"]["store"], "mem0")

    def test_the_memory_of_the_twin_is_one_for_each_project_and_the_same_in_all_its_conversations(self):
        service = FakeService()
        first, rt = self.council(service)
        first.talk(FIRST)
        second_ctx = open_context(self.folder, SimulatedProvider(), validation=False, hub=rt)
        self.addCleanup(second_ctx.db.close)
        second = Council(second_ctx)
        second.start()
        second.talk(FIRST)
        other_ctx = open_context(os.path.join(self.tmp.name, "projects", "other"), SimulatedProvider(), validation=False, hub=rt)
        self.addCleanup(other_ctx.db.close)
        other = Council(other_ctx)
        other.start()
        other.talk(FIRST)
        self.flush(rt)

        ids = [w["projeto_agente_id"] for w in service.mem_writes]
        self.assertEqual(ids[0], ids[1])  # two conversations of the same project
        self.assertNotEqual(ids[0], ids[2])  # another project does not share it

    def test_approvals_meetings_and_the_files_are_not_given_to_mem0(self):
        service = FakeService()
        council, rt = self.council(service)
        council.talk(FIRST)
        council.talk(SECOND)
        council.talk("approved")  # an approval, not a turn of the talk
        council.hear_meeting(MEETING)  # a meeting is a document, kept exactly in the other memory
        council.talk("Here is the policy", [("policy.md", "# Policy\n- a very long attached text")])
        self.flush(rt)

        said = [m["content"] for w in service.mem_writes for m in w["mensagens"] if m["role"] == "user"]
        self.assertEqual(said, [FIRST, SECOND, "Here is the policy"])  # only what the PO typed, no files, no meeting

    def test_person_memory_off_leaves_the_decisions_and_documents_as_they_are(self):
        service = FakeService(facts=self.FACTS)
        council, rt = self.council(service, person_memory=False)
        council.talk(FIRST)
        council.talk(SECOND)
        council.approve()
        self.flush(rt)

        self.assertEqual((service.mem_searches, service.mem_writes), ([], []))
        self.assertIn("decision:R1:v1", service.keys())

    def test_a_memory_that_fails_changes_nothing_for_the_conversation(self):
        service = FakeService(facts=self.FACTS, down=lambda m, p: p.startswith("/memoria"))
        council, rt = self.council(service)
        actions = [council.talk(FIRST)["action"], council.talk(SECOND)["action"]]
        self.flush(rt)

        self.assertEqual(actions, ["ask", "propose"])

    def test_a_slow_write_to_mem0_does_not_hold_back_the_trace(self):
        service = FakeService()
        service.hold = threading.Event()
        self.addCleanup(service.hold.set)
        council, rt = self.council(service)
        council.talk(FIRST)

        self.assertTrue(rt.flush(3, lane="fast"))  # the trace is already there...
        self.assertTrue(service.steps_of("llm_call"))
        self.assertFalse(rt.flush(0.3))  # ...while Mem0 is still thinking
        self.assertEqual(service.mem_writes, [])
        service.hold.set()
        self.assertTrue(rt.flush(5))
        self.assertEqual(len(service.mem_writes), 1)

    def test_a_search_is_traced_even_while_a_write_is_slow(self):
        service = FakeService(facts=self.FACTS)
        service.hold = threading.Event()
        self.addCleanup(service.hold.set)
        council, rt = self.council(service)
        council.talk(FIRST)  # leaves a slow write behind
        council.talk(SECOND)  # and searches again meanwhile
        rt.flush(3, lane="fast")

        searched = [s for s in service.mem_searches if s["passo_execucao_id"]]
        self.assertEqual(len(searched), 2)  # both searches were linked to their step


if __name__ == "__main__":
    unittest.main()
