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


class CalculationRejectionLedgerTests(unittest.TestCase):
    def test_terminal_rejection_survives_sync_judge_and_alias_registration(self):
        ledger=RequestLedger()
        spec=RequirementSpec(description="CPU 비율 계산",server_ids=["node-x"],information_ids=["cpu","process"],kind="calculation")
        ledger.add([spec],["node-x"],"plan")
        record=dict(code="incompatible_cpu_basis",requirement_ids=["r1"],operand_refs=[],operation="ratio",fingerprint="fp",repairable=False,reason="집계 기준이 다릅니다")
        ledger.record_calculation_rejection(record)
        ledger.sync([],[{"server_id":"node-x","facts":[{"information_id":"cpu"},{"information_id":"process"}]}],"repair")
        ledger.apply_checks([SimpleNamespace(requirement_id="r1",status="satisfied",reason="모델의 잘못된 통과")],"review")
        self.assertEqual(ledger.requirements[0]["status"],"unsupported")
        self.assertTrue(ledger.is_calculation_blocked("r1","fp"))
        ledger.add([spec.model_copy(update={"description":"같은 지표 비율 비교","kind":"comparison","calculation_fingerprint":"fp"})],["node-x"],"final_validation")
        self.assertEqual(len(ledger.requirements),1)
        # Independent requirements are never suppressed by the terminal record.
        ledger.add([spec.model_copy(update={"description":"RSS 환산","information_ids":["process"]})],["node-x"],"final_validation")
        self.assertEqual(len(ledger.requirements),2)

    def test_same_information_with_different_operation_is_not_suppressed(self):
        ledger=RequestLedger()
        spec=RequirementSpec(description="CPU 비율",server_ids=["node-x"],information_ids=["cpu","process"],kind="calculation")
        ledger.add([spec],["node-x"],"plan")
        ledger.record_calculation_rejection(dict(code="incompatible_cpu_basis",requirement_ids=["r1"],operand_refs=[],operation="ratio",fingerprint="fp",repairable=False,reason="집계 기준"))
        ledger.add([spec.model_copy(update={"description":"다른 피연산자와 연산"})],["node-x"],"final_validation")
        self.assertEqual(len(ledger.requirements),2)
        self.assertFalse(ledger.is_calculation_blocked("r2"))

    def test_same_fingerprint_is_not_revived_by_broader_or_narrower_information_ids(self):
        for ids in (["cpu"],["cpu","process","memory.summary"]):
            with self.subTest(ids=ids):
                ledger=RequestLedger()
                spec=RequirementSpec(description="CPU 비율",server_ids=["node-x"],information_ids=["cpu","process"],kind="calculation")
                ledger.add([spec],["node-x"],"plan")
                ledger.record_calculation_rejection(dict(code="incompatible_cpu_basis",requirement_ids=["r1"],operand_refs=[],operation="ratio",fingerprint="fp",repairable=False,reason="집계 기준"))
                ledger.add([spec.model_copy(update={"description":"다른 표기","kind":"comparison","information_ids":ids,"calculation_fingerprint":"fp"})],["node-x"],"final_validation")
                self.assertEqual(len(ledger.requirements),1)
