"""The hub of one conversation: what its agents do, sent to the memory and traceability service,
and what the service remembers, handed back to them.

Two things it never does: stop a conversation, and decide anything. If the service is down, slow or
refuses a request, the conversation goes on as if there were no hub (the reason goes to the log).

How what GigaBrain does maps to the service:

    conversation                 -> session
    a thing the PO does (a message, a meeting, an approval, a rejection...)
                                 -> an execution of the Twin ("turn"); the other agents it calls get
                                    their own execution, children of the Twin's
    a call to the LLM            -> a step "llm_call": prompt (versioned), messages, answer, tokens
    a message between agents     -> a step: "handoff" (agent to agent), "reasoning" (what an agent
                                    produced), "message" (to or from the PO), "repository" (saved)
    a memory search              -> a step "retrieval" (the service links it to what it returned)
    a decision or a document written to memory -> a step "memory_write" (and the document keeps it as its origin)
    what the PO says to the Twin -> Mem0, as the Twin's own memory of the PO: a "retrieval" step before the
                                    answer and a "memory_write" step after it
"""

from __future__ import annotations

import base64
import contextlib
import datetime
import json
import uuid
from urllib.parse import urlparse

from .client import HubError
from .decisions import TYPE, Record, clip
from .runtime import AGENT_ROLES, HubRuntime, log, stable_id

#: What kind of step each message kind is. Anything not listed is "reasoning".
STEP_TYPE = {
    "po_request": "message", "question_to_po": "message", "proposal": "message", "approval": "message",
    "confirmation": "message", "meeting": "message", "draft_board": "message",
    "query": "handoff", "consolidated_answer": "handoff", "expert_answer": "handoff", "validation_return": "handoff",
    "requirement_saved": "repository", "requirement_under_review": "repository", "knowledge_updated": "repository",
    "expert_created": "repository", "pending": "repository", "epics_organized": "repository",
    "memory_briefing": "memory_read",
}
#: Messages the hub records in another way (the LLM call, the search and the write have their own step).
SKIP = {"llm_call", "memory_recall", "memory_saved", "memory_facts"}


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _provider_identity(provider) -> tuple[str, str]:
    name = str(getattr(provider, "name", None) or "unknown")
    base = getattr(getattr(provider, "client", None), "base_url", None)
    host = urlparse(str(base)).hostname if base else None
    return (host or ("simulated" if name == "simulated" else "local")), name


class TurnHandle:
    """What the Council can tell about how a turn ended."""

    def __init__(self):
        self.result: dict | None = None

    def finish(self, result: dict | None) -> None:
        self.result = result


class _Execution:
    def __init__(self, id_: uuid.UUID, label: str):
        self.id, self.label, self._steps = id_, label, 0

    def next_index(self) -> int:
        self._steps += 1
        return self._steps - 1


class _Turn:
    def __init__(self, root: _Execution):
        self.root = root
        self.children: dict[str, _Execution] = {}


class Hub:
    def __init__(self, runtime: HubRuntime, project_key: str, project_id: uuid.UUID, model_id: int, provider_name: str):
        self.runtime = runtime
        self.settings = runtime.settings
        self.project_key, self.project_id, self.model_id, self.provider_name = project_key, project_id, model_id, provider_name
        self.conversation_id: str | None = None
        self._session_id: uuid.UUID | None = None
        self._turn: _Turn | None = None
        self._depth = 0

    @classmethod
    def create(cls, runtime: HubRuntime, project_key: str, provider) -> Hub | None:
        """None if the service cannot be reached: the conversation then runs without it."""
        try:
            project_id = runtime.ensure_project(project_key)
            model_id = runtime.ensure_model(*_provider_identity(provider))
        except (HubError, KeyError, ValueError) as exc:
            log.warning("hub: not connected, this conversation runs without it (%s)", exc)
            return None
        return cls(runtime, project_key, project_id, model_id, str(getattr(provider, "name", "unknown")))

    @property
    def trace_on(self) -> bool:
        return self.settings.trace

    @property
    def memory_on(self) -> bool:
        return self.settings.memory

    @property
    def person_memory_on(self) -> bool:
        return self.settings.memory and self.settings.person_memory

    @property
    def person_agent_id(self) -> uuid.UUID:
        """The id Mem0 isolates the Twin's memory by: one for each project, the same in every conversation of it."""
        return stable_id("mem0-agent", self.project_key, "twin")

    # ------------------------------------------------------------------ session and turns

    def open_session(self, conversation_id: str, mode: str) -> None:
        """Opens (or, for a conversation brought back from the log, finds again) the session of a conversation."""
        self.conversation_id = conversation_id
        if not self.trace_on:
            return
        self._session_id = stable_id("session", self.project_key, conversation_id)
        payload = {"id": self._session_id, "projeto_id": self.project_id, "usuario_externo_id": self.settings.user}
        self.runtime.submit("session", lambda: self.runtime.client.post("/sessoes", payload))

    def close_session(self) -> None:
        session_id, self._session_id = self._session_id, None
        if session_id:
            self.runtime.submit("session close", lambda: self.runtime.client.patch(f"/sessoes/{session_id}", {"finalizar": True}))

    @contextlib.contextmanager
    def turn(self, kind: str, entrada: dict | None = None):
        """One thing the PO did (a message, a meeting, an approval, ending the conversation) and everything it set off.

        Inside another turn it joins that one: approving by writing "approved" is a message and an approval.
        """
        handle = TurnHandle()
        if not self.trace_on or self._session_id is None:
            yield handle
            return
        if self._turn is not None:
            self._depth += 1
            try:
                yield handle
            finally:
                self._depth -= 1
            return
        try:
            self._turn = _Turn(self._open("twin", None, {"kind": kind, **(entrada or {})}))
        except Exception as exc:  # tracing must never be what stops a conversation
            log.warning("hub: could not open a turn (%s)", exc)
            yield handle
            return
        status = "success"
        try:
            yield handle
        except BaseException as exc:
            status, handle.result = "error", {"error": f"{type(exc).__name__}: {exc}"[:500]}
            raise
        finally:
            self._close_turn(status, handle.result)

    def _open(self, label: str, parent: uuid.UUID | None, entrada: dict) -> _Execution:
        head = label.split(":")[0]
        agent_id = self.runtime.register_agent(head, self.model_id)
        execution = _Execution(uuid.uuid4(), label)
        payload = {"id": execution.id, "sessao_id": self._session_id, "projeto_id": self.project_id, "agente_id": agent_id,
                   "versao_agente": 1, "entrada": self._cap(entrada), "iniciado_em": _now().isoformat()}
        if parent:
            payload["execucao_pai_id"] = parent
        self.runtime.submit(f"execution {label}", lambda: self.runtime.client.post("/execucoes", payload))
        return execution

    def _close_turn(self, status: str, result: dict | None) -> None:
        turn, self._turn = self._turn, None
        if turn is None:
            return
        for execution in (*turn.children.values(), turn.root):
            body = {"status": status, **({"saida": self._cap(result)} if result and execution is turn.root else {})}
            exec_id = execution.id
            self.runtime.submit("execution close", lambda i=exec_id, b=body: self.runtime.client.patch(f"/execucoes/{i}", b))

    def _execution_for(self, label: str) -> _Execution | None:
        turn = self._turn
        if turn is None:
            return None
        if label.split(":")[0] not in AGENT_ROLES or label == turn.root.label:
            return turn.root
        if label not in turn.children:
            turn.children[label] = self._open(label, turn.root.id, {"agent": label})
        return turn.children[label]

    def _cap(self, value) -> dict | None:
        """The payload as a dict, and never larger than the limit."""
        if value is None:
            return None
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return {"unserializable": str(value)[:500]}
        if len(text) > self.settings.max_chars:
            return {"truncated": True, "chars": len(text), "preview": text[: self.settings.max_chars // 10]}
        return value if isinstance(value, dict) else {"value": value}

    def _step(self, execution: _Execution, tipo: str, entrada, saida, *, status: str = "success", prompt_id=None,
              tokens=None, eventos=None, started: datetime.datetime | None = None) -> uuid.UUID:
        step_id = uuid.uuid4()
        payload = {"id": step_id, "indice_passo": execution.next_index(), "tipo": tipo, "status": status,
                   "entrada": self._cap(entrada), "saida": self._cap(saida), "iniciado_em": (started or _now()).isoformat()}
        if prompt_id:
            payload["prompt_id"] = prompt_id
        if tokens:
            payload["tokens"] = tokens
        if eventos:
            payload["eventos"] = eventos
        exec_id = execution.id
        self.runtime.submit(f"step {tipo}", lambda: self.runtime.client.post(f"/execucoes/{exec_id}/passos", payload))
        return step_id

    # ------------------------------------------------------------------ tracing

    def on_llm_call(self, agent: str, role: str, system: str, messages: list[dict], answer, usage: dict | None,
                    duration_ms: int, error: Exception | None = None) -> None:
        if not self.trace_on:
            return
        try:
            execution = self._execution_for(agent)
            if execution is None:
                return
            tokens = [{"modelo_id": self.model_id, "tipo_token": kind, "quantidade": int(n)}
                      for kind, key in (("entrada", "tokens_in"), ("saida", "tokens_out"))
                      if isinstance((n := (usage or {}).get(key)), int)]
            events = ([{"tipo": "erro", "categoria": "schema_invalido" if type(error).__name__ == "InvalidAnswer" else "falha_api",
                        "mensagem": clip(error, 500)}] if error else None)
            self._step(execution, "llm_call", {"role": role, "provider": self.provider_name, "messages": messages},
                       {"answer": answer, "duration_ms": duration_ms}, status="error" if error else "success",
                       prompt_id=self.runtime.register_prompt(role, system), tokens=tokens, eventos=events,
                       started=_now() - datetime.timedelta(milliseconds=duration_ms))
        except Exception as exc:
            log.warning("hub: could not trace an LLM call (%s)", exc)

    def on_message(self, msg) -> None:
        if not self.trace_on or msg.kind in SKIP:
            return
        try:
            content = msg.content or {}
            if msg.kind == "error" and "stage" not in content:
                return  # a failed LLM call: its step already carries the error
            execution = self._execution_for(msg.sender)
            if execution is None:
                return
            failed = msg.kind == "error"
            self._step(execution, STEP_TYPE.get(msg.kind, "reasoning"),
                       {"sender": msg.sender, "recipient": msg.recipient, "kind": msg.kind, "message_id": msg.id}, content,
                       status="error" if failed else "success",
                       eventos=[{"tipo": "erro", "categoria": str(content.get("stage") or "erro"),
                                 "mensagem": clip(content.get("error"), 500)}] if failed else None)
        except Exception as exc:
            log.warning("hub: could not trace a message (%s)", exc)

    # ------------------------------------------------------------------ memory

    def briefing(self) -> list[dict]:
        """The latest decisions of the project, newest first: where the last conversations stopped."""
        if not self.memory_on:
            return []
        try:
            documents = self.runtime.client.post(
                "/documentos/recentes",
                {"projeto_id": self.project_id, "tipo": TYPE, "limite": self.settings.briefing_limit})
        except HubError as exc:
            log.warning("hub: no briefing (%s)", exc)
            return []
        return [{"ref": f"{d['chave']}@v{d['versao']}", "kind": (d.get("metadados") or {}).get("tipo_registro"),
                 "date": (d.get("criado_em") or "")[:10], "text": clip(d.get("texto"), 350)} for d in documents]

    def recall(self, query: str, agent: str) -> list[dict]:
        """What the memory has about a question: documents and past decisions, best first.

        The agent is waiting for this, so it is the one call made on the spot. The step is registered
        first, so that the service can link the search to what it returned (which excerpts, scoring what).
        """
        if not self.memory_on or not query.strip():
            return []
        step_id = self._retrieval_step(agent, {"query": query, "store": "documents", "limit": self.settings.recall_limit})
        try:
            found = self.runtime.client.post("/documentos/buscar", {
                "projeto_id": self.project_id, "query": query, "limite": self.settings.recall_limit,
                "passo_execucao_id": step_id, "filtros_metadados": {}, "score_minimo": self.settings.min_score})
        except HubError as exc:
            log.warning("hub: memory search failed (%s)", exc)
            return []
        excerpts = []
        for hit in found:
            p = hit.get("payload") or {}
            excerpts.append({"ref": f"{p.get('chave_documento')}@v{p.get('versao_documento')}", "chunk": hit.get("id"),
                             "kind": p.get("tipo_documento"), "date": (p.get("criado_em") or "")[:10],
                             "score": round(float(hit.get("score") or 0), 3), "text": clip(p.get("texto"), 600)})
        return excerpts

    def _retrieval_step(self, agent: str, entrada: dict) -> uuid.UUID | None:
        """Registers a search as a step before it is made, so that the service can link it to what it returns.

        Waits for the fast lane only: a slow write to the memory must not keep a search from being traced.
        """
        try:
            execution = self._execution_for(agent) if self.trace_on else None
            if execution is None or not self.runtime.flush(self.settings.timeout, lane="fast"):
                return None
            step_id = uuid.uuid4()
            self.runtime.client.post(f"/execucoes/{execution.id}/passos", {
                "id": step_id, "indice_passo": execution.next_index(), "tipo": "retrieval", "entrada": self._cap(entrada),
                "iniciado_em": _now().isoformat()})
            return step_id
        except HubError as exc:
            log.warning("hub: could not register the search step (%s)", exc)
            return None

    def recall_person(self, query: str) -> list[dict]:
        """What the Twin remembers about the PO and the project from earlier conversations (Mem0), best first.

        Short facts, looser than the decision records: preferences, context, how the PO works. A fact scoring
        less than the settings say is left out: a fact that does not fit the message only gets in the way.
        """
        if not self.person_memory_on or not query.strip():
            return []
        step_id = self._retrieval_step("twin", {"query": query, "store": "mem0", "limit": self.settings.recall_limit})
        try:
            found = self.runtime.client.post("/memoria/buscar", {
                "projeto_agente_id": self.person_agent_id, "usuario_externo_id": self.settings.user, "query": query,
                "limite": self.settings.recall_limit, "passo_execucao_id": step_id})
        except HubError as exc:
            log.warning("hub: search in the memory of the PO failed (%s)", exc)
            return []
        facts = []
        for hit in found:
            score = float(hit.get("score") or 0)
            if score >= self.settings.person_min_score and (hit.get("memory") or "").strip():
                facts.append({"ref": f"mem0:{hit.get('id')}", "score": round(score, 3),
                              "date": str(hit.get("updated_at") or hit.get("created_at") or "")[:10],
                              "text": clip(hit.get("memory"), 400)})
        return facts

    def remember_turn(self, messages: list[dict]) -> None:
        """Give Mem0 a turn of the conversation (what the PO said and what the Twin answered), in the background.

        Mem0 reads it with an LLM and decides what is worth keeping, updating or forgetting. Only the talk goes
        here: the decisions and the documents are kept exactly, in the other memory (see decisions.py).
        """
        if not self.person_memory_on or not any(m["role"] == "user" and m["content"].strip() for m in messages):
            return
        try:
            body = {"projeto_agente_id": self.person_agent_id, "usuario_externo_id": self.settings.user, "mensagens": messages}
            execution = self._execution_for("twin")
            if execution is not None:
                body["passo_execucao_id"] = self._step(execution, "memory_write", {"store": "mem0", "turns": len(messages)},
                                                       {"messages": messages})
            self.runtime.submit("memory of the PO", lambda: self.runtime.client.post("/memoria/conversas", body, timeout=60), slow=True)
        except Exception as exc:
            log.warning("hub: could not write to the memory of the PO (%s)", exc)

    def remember(self, record: Record) -> None:
        """Write one decision in the memory. Sent in the background; the same record twice changes nothing."""
        if not self.memory_on:
            return
        self._write({"chave": record.key, "nome": record.name, "tipo": TYPE, "formato": "txt", "origem": "gerado_por_agente",
                     "enviado_por": "gigabrain", "metadados": record.metadata, "contextualizar": False},
                    record.text, {"chave": record.key, "tipo_registro": record.metadata.get("tipo_registro")}, 30)

    def remember_document(self, name: str, text: str) -> None:
        """A file the PO attached, kept so that the experts can find it later (in this conversation and the next ones).

        Unlike a decision it is long and free: the service reads it as a whole to give each piece its context
        and its attributes, which takes a while, so it is sent in the background.
        """
        if not self.memory_on or not text.strip():
            return
        extension = name.rsplit(".", 1)[-1].lower() if "." in name else "txt"
        self._write({"chave": f"attachment:{name}", "nome": name, "tipo": "outro", "formato": "md" if extension == "md" else "txt",
                     "origem": "upload_humano", "enviado_por": self.settings.user,
                     "metadados": {"origem": "chat", "conversa": self.conversation_id}, "contextualizar": True},
                    text, {"documento": name, "chars": len(text)}, 180)

    def _write(self, body: dict, text: str, summary: dict, timeout: float) -> None:
        body = {**body, "projeto_id": self.project_id, "conteudo_base64": base64.b64encode(text.encode("utf-8")).decode("ascii")}
        try:
            execution = self._execution_for("twin")
            if execution is not None:
                body["passo_origem_id"] = self._step(execution, "memory_write", summary, {"texto": clip(text, 2000)})
            self.runtime.submit(f"memory {body['chave']}", lambda: self.runtime.client.post("/documentos", body, timeout=timeout),
                                slow=True)
        except Exception as exc:
            log.warning("hub: could not write to the memory (%s)", exc)

    def flush(self, timeout: float = 10.0) -> bool:
        return self.runtime.flush(timeout)
