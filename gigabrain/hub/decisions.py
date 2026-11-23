"""What GigaBrain writes in the memory: one short record for each decision.

A record says what was decided, who decided, why, and what changed from the version before. They
are built here, from data GigaBrain already has, with no LLM: the same decision is always the
same text, so sending it twice is harmless and the memory can be trusted as a record, not a summary.

Records are only ever added, never edited. A requirement that changes gets a new record next to
the old one, and a draft the PO rejected and then restored has both records: what is true now is
in the Requirements Repository and in the draft list, what happened is in the memory.

Each record is kept under ~1,500 characters so it is one piece of text for the search.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

MAX_CHARS = 1450
TYPE = "decisao"  # `tipo` of the document in the service


@dataclass
class Record:
    key: str                 # unique within the project: sending the same key and text again changes nothing
    name: str
    text: str
    metadata: dict = field(default_factory=dict)


def clip(text, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _fit(lines: list[str]) -> str:
    text = "\n".join(lines)
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 1].rstrip() + "…"


def _today() -> str:
    return datetime.date.today().isoformat()


def _class(content: dict) -> str:
    cls = content.get("class") or "?"
    return f"{cls}/{content['subtype']}" if content.get("subtype") else cls


def _constraint(card: dict | None) -> str:
    card = card or {}
    return " · ".join(f"{k} {clip(card[k], 80)}" for k in ("quality", "scope", "measure") if card.get(k))


def _metadata(kind: str, status: str, conversation_id: str, *, topics=None, requirements=None, draft: dict | None = None,
              decided_by: str = "po") -> dict:
    meta = {"tipo_registro": kind, "status_decisao": status, "conversa": conversation_id, "decidido_por": decided_by}
    if topics:
        meta["topicos"] = list(topics)
    if requirements:
        meta["requisito"] = list(requirements)
    if draft:
        meta["rascunho"] = draft.get("draft_id")
        meta["origem"] = draft.get("origin")
    return meta


# ---------------------------------------------------------------------------- what changed

def diff_versions(old: dict, new: dict) -> list[str]:
    """What changed between two versions of a requirement (or a draft before and after), field by field."""
    changes = []
    for key in ("title", "statement", "class", "subtype"):
        if (old.get(key) or None) != (new.get(key) or None):
            changes.append(f'{key}: "{clip(old.get(key), 160)}" -> "{clip(new.get(key), 160)}"')
    old_story, new_story = old.get("story") or {}, new.get("story") or {}
    for part in ("role", "goal", "benefit"):
        if (old_story.get(part) or "") != (new_story.get(part) or ""):
            changes.append(f'story {part}: "{clip(old_story.get(part), 100)}" -> "{clip(new_story.get(part), 100)}"')
    old_card, new_card = old.get("constraint") or {}, new.get("constraint") or {}
    for part in ("quality", "scope", "measure"):
        if (old_card.get(part) or "") != (new_card.get(part) or ""):
            changes.append(f'constraint {part}: "{clip(old_card.get(part), 100)}" -> "{clip(new_card.get(part), 100)}"')
    old_c, new_c = old.get("acceptance_criteria") or [], new.get("acceptance_criteria") or []
    if old_c != new_c:
        added = [c.get("scenario") for c in new_c if c not in old_c]
        removed = [c.get("scenario") for c in old_c if c not in new_c]
        parts = ([f"added {', '.join(clip(a, 50) for a in added)}"] if added else []) + \
                ([f"removed {', '.join(clip(a, 50) for a in removed)}"] if removed else [])
        changes.append("acceptance criteria: " + "; ".join(parts) if parts else "acceptance criteria changed")
    if sorted(old.get("topics") or []) != sorted(new.get("topics") or []):
        changes.append(f"topics: {', '.join(old.get('topics') or []) or '-'} -> {', '.join(new.get('topics') or []) or '-'}")
    return changes


# ---------------------------------------------------------------------------- the records

def requirement_approved(*, conversation_id: str, requirement_id: str, version: int, draft: dict, content: dict,
                         reason: str | None, links: list[dict], decided_by: str, changes: list[str] | None,
                         revisions: list[dict], date: str | None = None) -> Record:
    """A draft was approved: a requirement (or a new version of one) was saved."""
    lines = [
        f"Decision record: requirement {requirement_id} v{version} APPROVED",
        f"Conversation: {conversation_id} · Draft: {draft.get('draft_id')} ({draft.get('origin')}) · Date: {date or _today()}",
        f"Title: {clip(content.get('title'), 120)}",
        f"Statement: {clip(content.get('statement'), 300)}",
        f"Class: {_class(content)} · Topics: {', '.join(content.get('topics') or []) or '-'}",
    ]
    if content.get("constraint"):
        lines.append(f"Constraint: {_constraint(content['constraint'])}")
    lines += [f"Reason: {clip(reason, 200) or '-'}", f"Decided by: {decided_by}"]
    if changes is not None:
        lines.append(f"Changes since v{version - 1}: " + ("; ".join(changes) if changes else "none in the content"))
    if links:
        lines.append("Links: " + "; ".join(f"{requirement_id} {l['kind'].replace('_', ' ')} {l['target']}" for l in links))
    if revisions:
        lines.append(f"Before it was approved this draft was changed {len(revisions)} time(s):")
        lines += [f'- because {r["by"]}: "{clip(r.get("reason"), 140)}"' for r in revisions[-3:]]
    return Record(
        key=f"decision:{requirement_id}:v{version}",
        name=f"{requirement_id} v{version} approved",
        text=_fit(lines),
        metadata={**_metadata("requirement_approved", "approved", conversation_id, topics=content.get("topics"),
                              requirements=[requirement_id], draft=draft,
                              decided_by="twin" if decided_by.lower().startswith("twin") else "po"),
                  "versao_requisito": version},
    )


def _what(draft: dict) -> list[str]:
    lines = [f'Draft {draft.get("draft_id")} ({draft.get("origin")}): {clip(draft.get("title"), 80)}',
             f'Statement: {clip(draft.get("statement"), 250)}']
    if draft.get("constraint"):
        lines.append(f"Constraint: {_constraint(draft['constraint'])}")
    return lines


def draft_rejected(*, conversation_id: str, draft: dict, reason: str | None, number: int = 1, date: str | None = None) -> Record:
    """The PO rejected a draft, with a button or by asking the Twin to drop it. `number`: the first time, the second..."""
    lines = [f"Decision record: draft {draft.get('draft_id')} REJECTED by the PO",
             f"Conversation: {conversation_id} · Date: {date or _today()}", *_what(draft),
             f'Why (the PO said): "{clip(reason, 400)}"' if reason else "Why: the PO gave no reason."]
    return Record(
        key=f"decision:{conversation_id}:{draft.get('draft_id')}:rev{draft.get('revision', 1)}:rejected-{number}",
        name=f"draft {draft.get('draft_id')} rejected", text=_fit(lines),
        metadata=_metadata("draft_rejected", "rejected", conversation_id, topics=draft.get("topics"), draft=draft))


def draft_restored(*, conversation_id: str, draft: dict, number: int = 1, date: str | None = None) -> Record:
    """The PO undid a rejection: the earlier 'rejected' record no longer holds."""
    lines = [f"Decision record: draft {draft.get('draft_id')} RESTORED — the PO took back the rejection",
             f"Conversation: {conversation_id} · Date: {date or _today()}", *_what(draft)]
    return Record(
        key=f"decision:{conversation_id}:{draft.get('draft_id')}:rev{draft.get('revision', 1)}:restored-{number}",
        name=f"draft {draft.get('draft_id')} restored", text=_fit(lines),
        metadata=_metadata("draft_restored", "restored", conversation_id, topics=draft.get("topics"), draft=draft))


def draft_revised(*, conversation_id: str, draft: dict, before: dict, reason: str, by: str,
                  date: str | None = None) -> Record:
    """A draft was changed before being approved. `by` says what asked for it (the PO's message, a later part of a meeting)."""
    changes = diff_versions(before, draft)
    lines = [f"Decision record: draft {draft.get('draft_id')} CHANGED (now revision {draft.get('revision')})",
             f"Conversation: {conversation_id} · Date: {date or _today()}", *_what(draft),
             "What changed: " + ("; ".join(changes) if changes else "nothing in the content"),
             f'Why ({by}): "{clip(reason, 400)}"']
    return Record(
        key=f"decision:{conversation_id}:{draft.get('draft_id')}:rev{draft.get('revision')}:revised",
        name=f"draft {draft.get('draft_id')} changed", text=_fit(lines),
        metadata=_metadata("draft_revised", "revised", conversation_id, topics=draft.get("topics"), draft=draft,
                           decided_by="po" if by.startswith("the PO") else "twin"))


def draft_abandoned(*, conversation_id: str, draft: dict, reason: str | None, date: str | None = None) -> Record:
    """The conversation ended with this draft never approved."""
    lines = [f"Decision record: draft {draft.get('draft_id')} NOT approved — the conversation ended without approval",
             f"Conversation: {conversation_id} · Date: {date or _today()}", *_what(draft),
             f'Why (the PO said): "{clip(reason, 400)}"' if reason else "Why: the PO gave no reason."]
    return Record(
        key=f"decision:{conversation_id}:{draft.get('draft_id')}:rev{draft.get('revision', 1)}:closed",
        name=f"draft {draft.get('draft_id')} abandoned", text=_fit(lines),
        metadata=_metadata("draft_abandoned", "abandoned", conversation_id, topics=draft.get("topics"), draft=draft))


def session_summary(*, conversation_id: str, status: str, po_messages: list[str], meeting_utterances: int,
                    drafts: list[dict], saved: list[str], open_question: str | None, date: str | None = None) -> Record:
    """What one conversation came to, so the next one can start from it."""
    outcomes: dict[str, int] = {}
    for d in drafts:
        outcomes[d["status"]] = outcomes.get(d["status"], 0) + 1
    lines = [f"Session summary · conversation {conversation_id} · ended: {status} · Date: {date or _today()}"]
    if po_messages:
        lines += ["The PO asked for:", *[f'- "{clip(m, 150)}"' for m in po_messages[:5]]]
    if meeting_utterances:
        lines.append(f"Meetings read: {meeting_utterances} utterances.")
    lines.append(f"Drafts: {len(drafts)}" + (" (" + ", ".join(f"{n} {k}" for k, n in outcomes.items()) + ")" if drafts else ""))
    lines.append("Saved: " + (", ".join(saved) if saved else "nothing"))
    if open_question:
        lines.append(f'Open: the Twin asked and the PO did not answer: "{clip(open_question, 200)}"')
    return Record(
        key=f"decision:{conversation_id}:summary",
        name="session summary", text=_fit(lines),
        metadata=_metadata("session_summary", status, conversation_id, requirements=saved))
