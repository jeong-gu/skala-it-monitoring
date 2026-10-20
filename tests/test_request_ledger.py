import unittest
from types import SimpleNamespace

from agent.request_ledger import RequestLedger, RequirementSpec


class RequestLedgerTests(unittest.TestCase):
    def test_obtaining_information_does_not_complete_requested_calculation(self):
        ledger = RequestLedger()
        ledger.add([RequirementSpec(description="계산된 결과도 제공", server_ids=["node-x"],
                                    information_ids=["cpu"], kind="calculation")], ["node-x"], "plan")
        ledger.sync([], [], "first_round")
        ledger.sync([], [{"server_id": "node-x", "facts": [{"information_id": "cpu"}]}], "next_round")
        self.assertEqual(ledger.requirements[0]["status"], "information_ready")
        self.assertEqual(ledger.requirements[0]["history"][1]["status"], "needs_more")
        self.assertFalse(ledger.issues[0]["resolved"])
        ledger.apply_checks([SimpleNamespace(requirement_id="r1", status="satisfied", reason="계산과 출처 포함")], "final_validation")
        self.assertTrue(ledger.issues[0]["resolved"])
        self.assertEqual(ledger.requirements[0]["status"], "satisfied")

    def test_unsupported_requirement_is_not_dropped_and_snapshot_is_detached(self):
        ledger = RequestLedger()
        spec = RequirementSpec(description="과거 이력", server_ids=["node-x"], information_ids=["history"])
        ledger.add([spec], ["node-x"], "plan")
        ledger.add([spec], ["node-x"], "review")
        ledger.sync([{"server_id": "node-x", "information_ids": ["history"],
                      "information_status": {"history": "unsupported"}}], [], "collection")
        payload = ledger.payload()
        payload["requirements"].clear()
        self.assertEqual(len(ledger.requirements), 1)
        self.assertEqual(ledger.requirements[0]["status"], "unsupported")
        self.assertTrue(ledger.issues)

    def test_model_cannot_add_another_target(self):
        ledger = RequestLedger()
        with self.assertRaises(ValueError):
            ledger.add([RequirementSpec(description="다른 대상", server_ids=["node-y"])], ["node-x"], "review")
