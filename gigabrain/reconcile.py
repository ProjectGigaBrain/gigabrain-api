"""Chronological reconciliation: what a meeting made obsolete in the Requirements Repository.

A meeting is read one utterance at a time, so each draft only sees what was said up to it. A
decision taken later (the LCD panel cut for budget, rubber replaced by plastic, an idea dropped)
saves a new requirement, but the earlier ones it contradicts stay "active". At the end of each
meeting, the Twin reads the meeting as a whole against the Repository and says what happens to
each requirement the meeting changed:

    supersede   another requirement states the decision that takes its place ("R50 supersedes R32")
    revise      it still holds, but part of it changed: a new version
    withdraw    it no longer holds and nothing replaces it (status "withdrawn")

The later decision prevails. Every change must cite utterances of the meeting; a change that
names an unknown requirement, cites nothing, or lets an older requirement replace a newer one is
dropped before anything changes (check).

In a meeting processed in batch, the Twin has the final vote and the changes are applied at
once (meeting.py). In a conversation, they become drafts for the PO to approve (council.py).
"""

from __future__ import annotations

from . import requirement_format
from .database import LIVE_STATUSES
from .messages import REPOSITORY, TWIN, Message

ACTIONS = ("supersede", "revise", "withdraw")


def _order(rid: str) -> tuple[int, int]:
    """Chronological order of an id: saved requirements (R) first, then pending drafts (D)."""
    try:
        return (1 if rid.startswith("D") else 0, int(rid[1:]))
    except (ValueError, AttributeError):
        return (2, 0)


def repository_view(db, conversation_id: str | None, drafts: list[dict] | None = None) -> list[dict]:
    """What the Twin reconciles: every requirement that still holds, oldest first (+ pending drafts)."""
    view = []
    for req in db.list_requirements(only_active=True):
        first = db.requirement_versions(req["id"])[0]
        here = first["conversation_id"] == conversation_id
        item = {"id": req["id"], "statement": req["content"].get("statement") or req["title"], "topics": req["topics"],
                "from_this_meeting": here}
        if here and req["content"].get("turns"):
            item["turns"] = req["content"]["turns"]
        view.append(item)
    for d in drafts or []:
        r = d["requirement"]
        view.append({"id": d["draft_id"], "statement": r.get("statement"), "topics": r.get("topics") or [],
                     "from_this_meeting": True, "turns": r.get("turns") or [], "pending_draft": True})
    return sorted(view, key=lambda item: _order(item["id"]))


def open_conflicts(db) -> list[dict]:
    """conflicts_with links between two requirements that both still hold."""
    live = {r["id"] for r in db.list_requirements(only_active=True)}
    return [{"newer": l["source_id"], "older": l["target_id"], "reason": l["reason"]}
            for l in db.list_links() if l["kind"] == "conflicts_with" and l["source_id"] in live and l["target_id"] in live]


def check(changes: list[dict], view: list[dict], turn_ids: set[str]) -> list[dict]:
    """Keep only the changes that can be applied. Nothing in the Repository changes here."""
    known = {item["id"]: item for item in view}
    candidates = []
    for c in changes:
        action, target, by = c.get("action"), c.get("target"), c.get("by") or None
        turns = [t for t in (c.get("turns") or []) if t in turn_ids]
        if action not in ACTIONS or target not in known or not turns:
            continue
        if action == "supersede" and (by not in known or by == target or _order(by) <= _order(target)):
            continue  # only a newer requirement takes the place of an older one
        if action == "revise":
            statement = (c.get("statement") or "").strip()
            if not statement or statement == (known[target]["statement"] or "").strip():
                continue
            c = {**c, "statement": statement}
        candidates.append({**c, "action": action, "target": target, "by": by if action == "supersede" else None,
                           "turns": turns, "reason": c.get("reason") or ""})
    # At most one change per requirement, and a requirement that goes away cannot replace another.
    kept, targets = [], set()
    for c in candidates:
        if c["target"] not in targets:
            kept.append(c)
            targets.add(c["target"])
    gone = {c["target"] for c in kept if c["action"] != "revise"}
    return [c for c in kept if not (c["action"] == "supersede" and c["by"] in gone)]


def run(ctx, twin, turns: list[dict], language: str, drafts: list[dict] | None = None) -> list[dict]:
    """Ask the Twin what the meeting made obsolete, and keep what can be applied."""
    view = repository_view(ctx.db, ctx.conversation_id, drafts)
    if not view or not turns:
        return []
    changes = twin.reconcile(turns, view, open_conflicts(ctx.db), language)
    return check(changes, view, {t["id"] for t in turns})


def revised_content(ctx, twin, target: str, change: dict, utterances: dict[str, str], language: str) -> dict:
    """The new version of a revised requirement: the Twin rewrites the whole card around the new statement."""
    req = ctx.db.get_requirement(target)
    old = req["content"]
    draft = {**requirement_format.normalize(old), "topic": (req["topics"] or ["general"])[0], "turns": old.get("turns") or []}
    feedback = (f"A later decision of the meeting changed this requirement: {change['reason']}. "
                f"The statement is now: \"{change['statement']}\". Make the title, the card and the "
                f"acceptance criteria agree with it.")
    cited = {t: utterances[t] for t in change["turns"] if t in utterances}
    revised = twin.revise(draft, feedback, cited, language) or draft
    content = requirement_format.normalize({**revised, "statement": change["statement"]})
    turns = list(dict.fromkeys([*(old.get("turns") or []), *change["turns"]]))
    return {**content, "topics": req["topics"], "turns": turns}


def apply(ctx, twin, change: dict, utterances: dict[str, str], language: str, decided_by: str,
          extra_sources: list[dict] | None = None) -> dict | None:
    """Apply one checked change to the Repository. Returns what changed, or None if nothing could."""
    db, conv = ctx.db, ctx.conversation_id
    target = db.get_requirement(change["target"])
    if not target or target["status"] not in LIVE_STATUSES:
        return None
    sources = [{"kind": "turn", "ref": t} for t in change["turns"]] + list(extra_sources or [])
    reason = change.get("reason") or "changed by a later decision"
    if change["action"] == "withdraw":
        result = db.withdraw(change["target"], reason, conv, sources)
        if not result:
            return None
        outcome = {"action": "withdraw", "target": change["target"], "under_review": result["under_review"]}
    elif change["action"] == "supersede":
        by = db.get_requirement(change["by"] or "")
        if not by or by["status"] not in LIVE_STATUSES:
            return None
        link = db.link(change["by"], "supersedes", change["target"], reason, conv)
        if not link:
            return None
        outcome = {"action": "supersede", "target": change["target"], "by": change["by"], "under_review": link["under_review"]}
    else:
        content = revised_content(ctx, twin, change["target"], change, utterances, language)
        result = db.save_requirement(content, conv, base_id=change["target"], reason=reason,
                                     sources=[{"kind": "conversation", "ref": conv}, *sources])
        outcome = {"action": "revise", "target": change["target"], "version": result["version"],
                   "statement": content["statement"], "under_review": result["under_review"]}
    outcome.update({"reason": reason, "turns": change["turns"], "decided_by": decided_by})
    ctx.recorder.record(Message(sender=TWIN, recipient=REPOSITORY, kind="requirement_reconciled", content=outcome,
                                conversation_id=conv))
    for rid in outcome["under_review"]:
        ctx.recorder.record(Message(sender=REPOSITORY, recipient=TWIN, kind="requirement_under_review",
                                    content={"requirement_id": rid, "because_of": change["target"]}, conversation_id=conv))
    _tell_experts(ctx, outcome, target["topics"])
    return outcome


def _tell_experts(ctx, outcome: dict, topics: list[str]) -> None:
    from .agents import Expert  # agents import this module's neighbours; import here to avoid a cycle
    for topic in topics:
        exp = Expert.load(ctx, topic)
        if exp:
            exp.note_change(outcome)
