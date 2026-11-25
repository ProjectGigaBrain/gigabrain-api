"""Chronological reconciliation: a later decision supersedes, revises or withdraws what was saved before."""

from __future__ import annotations

import os
import tempfile
import unittest

from gigabrain import dataset, meeting, reconcile
from gigabrain.council import Council, open_context
from gigabrain.providers import SimulatedProvider
from tests.test_meeting import make_dataset

# A meeting that changes its mind: the LCD screen of T2 is removed in T4.
UTTERANCES = [
    ("Analyst", "Okay, next topic: display. What should the remote show?"),
    ("Client", "The remote control shall have a small LCD screen to confirm each command."),
    ("Analyst", "And the case?"),
    ("Client", "Because of the budget the LCD screen was removed from the remote control, replies go to the TV."),
]
GOLD = [("p:R1", "The remote shall show replies on the TV screen.", "FR", None, "T4")]


def content(statement: str) -> dict:
    return {"title": statement[:30], "statement": statement, "class": "FR", "topics": ["display"]}


class TempFolder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def context(self, name: str):
        ctx = open_context(os.path.join(self.tmp.name, name), SimulatedProvider())
        self.addCleanup(ctx.db.close)
        return ctx


class TestWithdraw(TempFolder):
    def test_withdraw_keeps_history_and_reviews_dependents(self):
        db = self.context("db").db
        db.save_requirement(content("The remote shall find itself when the user whistles."), None)
        db.save_requirement(content("The remote shall beep while it is being found."), None,
                            links=[{"kind": "depends_on", "target": "R1"}])
        result = db.withdraw("R1", "idea dropped for cost", sources=[{"kind": "turn", "ref": "T9"}])
        self.assertEqual(result, {"requirement_id": "R1", "status": "withdrawn", "under_review": ["R2"]})
        self.assertEqual(db.get_requirement("R1")["status"], "withdrawn")
        self.assertEqual([r["id"] for r in db.list_requirements(only_active=True)], ["R2"])
        self.assertEqual(db.status_changes("R1")[0]["sources"], [{"kind": "turn", "ref": "T9"}])
        self.assertEqual([s["event"] for s in db.trace("R1")], ["version", "status"])
        self.assertIsNone(db.withdraw("R1", "again"))  # only what still holds can be withdrawn


class TestCheck(unittest.TestCase):
    VIEW = [{"id": "R1", "statement": "LCD screen"}, {"id": "R2", "statement": "rubber buttons"},
            {"id": "R3", "statement": "LCD removed"}, {"id": "D1", "statement": "plastic only"}]

    def test_drops_what_cannot_be_applied(self):
        changes = [
            {"action": "supersede", "target": "R1", "by": "R3", "turns": ["T4"]},  # ok
            {"action": "withdraw", "target": "R9", "turns": ["T4"]},  # unknown requirement
            {"action": "withdraw", "target": "R2", "turns": ["T99"]},  # cites nothing of this meeting
            {"action": "supersede", "target": "R3", "by": "R2", "turns": ["T4"]},  # older replacing newer
            {"action": "withdraw", "target": "R1", "turns": ["T4"]},  # second change of R1
            {"action": "revise", "target": "R2", "statement": "rubber buttons", "turns": ["T4"]},  # nothing changes
            {"action": "supersede", "target": "R2", "by": "D1", "turns": ["T4"]},  # a newer draft: ok
        ]
        kept = reconcile.check(changes, self.VIEW, {"T4"})
        self.assertEqual([(c["action"], c["target"], c["by"]) for c in kept],
                         [("supersede", "R1", "R3"), ("supersede", "R2", "D1")])

    def test_what_goes_away_cannot_replace(self):
        changes = [{"action": "supersede", "target": "R1", "by": "R3", "turns": ["T4"]},
                   {"action": "withdraw", "target": "R3", "turns": ["T4"]}]
        self.assertEqual([c["target"] for c in reconcile.check(changes, self.VIEW, {"T4"})], ["R3"])


class TestMeeting(TempFolder):
    def setUp(self):
        super().setUp()
        make_dataset(self.tmp.name, UTTERANCES, GOLD)
        self.project = dataset.load_project("test-project", self.tmp.name)

    def test_council_withdraws_what_a_later_utterance_removed(self):
        ctx = self.context("council")
        r = meeting.process_and_evaluate(ctx, self.project, os.path.join(self.tmp.name, "council"), "council")
        self.assertEqual(ctx.db.get_requirement("R1")["status"], "withdrawn")
        self.assertEqual([p["id"] for p in r["retired"]], ["R1"])
        self.assertNotIn("R1", [p["id"] for p in r["predicted"]])  # the final specification only
        self.assertEqual(r["reconciled"][0]["action"], "withdraw")
        reasons = [v["reason"] for e in ctx.db.list_experts() for v in ctx.db.knowledge_versions(e["id"])]
        self.assertIn("R1 withdraw", reasons)  # the expert of the topic writes it down

    def test_baselines_are_not_reconciled(self):
        ctx = self.context("single")
        r = meeting.process_and_evaluate(ctx, self.project, os.path.join(self.tmp.name, "single"), "single_agent")
        self.assertEqual(r["retired"], [])
        self.assertEqual(ctx.db.get_requirement("R1")["status"], "active")


class TestConversation(TempFolder):
    def test_change_of_a_saved_requirement_waits_for_the_po(self):
        ctx = self.context("chat")
        council = Council(ctx)
        council.start()
        council.hear_meeting("Client: The remote control shall have a small LCD screen to confirm each command.")
        council.approve()
        self.assertEqual(ctx.db.get_requirement("R1")["status"], "active")

        answer = council.hear_meeting("Client: Because of the budget the LCD screen was removed from the remote control, "
                                      "replies go to the TV.")
        changes = [d for d in council.drafts if d["requirement"].get("change")]
        self.assertEqual([d["requirement"]["change"]["target"] for d in changes], ["R1"])
        self.assertEqual(ctx.db.get_requirement("R1")["status"], "active")  # nothing changes before the PO approves
        self.assertIn(changes[0]["draft_id"], answer["new"])

        council.approve([changes[0]["draft_id"]])
        self.assertEqual(ctx.db.get_requirement("R1")["status"], "withdrawn")
        self.assertEqual(changes[0]["status"], "approved")


if __name__ == "__main__":
    unittest.main()
