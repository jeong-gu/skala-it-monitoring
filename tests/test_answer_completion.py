"""Completion protocol tests: scripted omissions and new numeric inputs."""
import copy
import json
import unittest
from types import SimpleNamespace
from agent.answer_calculations import calculate
from agent.answer_contracts import Calculation
from agent.fake_llm import FastLLM
from agent.investigation import investigate
from test_information_investigation import FixtureMCP, task


def req(*ids,kind="lookup"):
    return dict(description="요청을 빠짐없이 설명",server_ids=["server-a"],information_ids=list(ids),kind=kind)


def chk(rid,status="satisfied",reason="답변과 출처 확인"):
    return dict(requirement_id=rid,status=status,reason=reason)


def review(*checks,tasks=None,missing=None):
    return dict(checks=list(checks),repair_tasks=tasks or [],missing_requirements=missing or [],rewrite_instructions="미충족 요구 보완")


class Model(FastLLM):
    def __init__(self,**scripts):
        super().__init__("scripted",0);self.scripts=scripts;self.inputs=[]
    def monitoring_json(self,purpose,payload,schema):
        self.inputs.append((purpose,copy.deepcopy(payload)))
        values=self.scripts.get(purpose,[])
        if values:
            result=values.pop(0)
            return result if isinstance(result,str) else json.dumps(result,ensure_ascii=False),{"usd":0}
        return super().monitoring_json(purpose,payload,schema)


class CompletionTests(unittest.IsolatedAsyncioTestCase):
    async def run_flow(self,requirements,reviewer,*,tasks=None,writer=None,reviews=None):
        self.mcp=FixtureMCP();self.writer=writer or Model()
        self.writer.scripts.update(reviewer.scripts)
        self.writer.name="standard-writer-and-validator"
        self.fast=Model();self.fast.name="unused-fast-model"
        self.planner=Model(plan=[dict(tasks=tasks or [task("server-a","cpu")],requirements=requirements)],
                           review=reviews or [dict(tasks=[])])
        return await investigate(dict(query="서버 A 요청 항목을 빠짐없이 설명해줘",history=[],events=[],cost_usd=0),
            SimpleNamespace(mcp=self.mcp,fast_llm=self.fast,monitoring_llm=self.planner,quality_llm=self.writer))

    async def test_missing_information_loops_back_and_gap_reaches_writer(self):
        fast=Model(final_review=[review(chk("r1"),chk("r2","unmet","프로세스 근거가 없음"),tasks=[task("server-a","process")]),
                                 review(chk("r1"),chk("r2"))])
        r=await self.run_flow([req("cpu"),req("process")],fast)
        self.assertEqual(self.mcp.calls,[("server-a","top"),("server-a","ps")])
        self.assertEqual(r["final_validation"]["status"],"passed")
        self.assertEqual([p for p,_ in self.writer.inputs],["report","final_review","report","final_review"])
        self.assertEqual(self.fast.inputs,[])
        self.assertEqual(r["final_validation"]["model"],self.writer.name)
        self.assertTrue(all(v["model"]==self.writer.name for v in r["final_reviews"]))
        second=[p for name,p in self.writer.inputs if name=="report"][1]
        self.assertEqual(second["query"],"서버 A 요청 항목을 빠짐없이 설명해줘")
        self.assertTrue(any("프로세스 근거가 없음" in g["reason"] for g in second["requirement_issues"]))
        self.assertTrue(any(g["resolved"] for g in r["requirement_issues"]))

    async def test_rewrite_without_requerying_obtained_information(self):
        text=dict(text="제공된 서버 A CPU 정보를 확인했습니다",requirement_ids=["r1"],evidence_ids=["e1"],quotes=["16.0 us"])
        fast=Model(final_review=[review(chk("r1","unmet")),review(chk("r1"))])
        writer=Model(report=[dict(hypotheses=[]),dict(hypotheses=[],answers=[text])])
        r=await self.run_flow([req("cpu",kind="comparison")],fast,writer=writer)
        self.assertEqual(len(self.mcp.calls),1);self.assertIn(text["text"],r["final"])
        self.assertEqual(r["final_validation"]["repair_rounds"],1)

    async def test_program_rejects_false_calculation_pass_then_computes_from_source(self):
        calc=dict(label="원본 필드 계산",requirement_ids=["r1"],operation="sum",
                  operands=[dict(evidence_id="e1",information_id="cpu",field="user_percent")],output_unit="native")
        # Both report attempts omit the calculation; a permissive final judge
        # must still fail the requirement before the later repair succeeds.
        writer=Model(report=[dict(hypotheses=[]),dict(hypotheses=[]),dict(hypotheses=[],calculations=[calc])])
        r=await self.run_flow([req("cpu",kind="calculation")],
            Model(final_review=[review(chk("r1")),review(chk("r1"))]),writer=writer)
        self.assertEqual(r["final_reviews"][0]["checks"][0]["status"],"unmet")
        self.assertEqual(r["final_validation"]["status"],"passed")
        self.assertEqual(float(r["calculation_results"][0]["value"]),16)
        self.assertEqual(len(self.mcp.calls),1)

    async def test_incompatible_cpu_calculation_cannot_pass_a_permissive_judge(self):
        calc=dict(label="시스템 CPU 중 프로세스 비중",requirement_ids=["r1"],operation="percent",operands=[
            dict(evidence_id="e2",information_id="process",field="entries.0.cpu_percent"),
            dict(evidence_id="e1",information_id="cpu",field="usage_percent")],output_unit="native")
        writer=Model(report=[dict(hypotheses=[],calculations=[calc])]*6)
        r=await self.run_flow([req("cpu","process",kind="calculation")],
            Model(final_review=[review(chk("r1"))]*3),tasks=[task("server-a","cpu","process")],writer=writer)
        self.assertEqual(r["calculation_results"],[])
        self.assertFalse(r["final_validation"]["request_fully_satisfied"])
        self.assertTrue(any("기준" in issue["reason"] for issue in r["requirement_issues"]))

    async def test_missing_diagnostic_candidate_cannot_pass_even_if_model_says_yes(self):
        r=await self.run_flow([req("cpu",kind="diagnosis")],
            Model(final_review=[review(chk("r1"))]*3))
        self.assertEqual(r["final_validation"]["status"],"partial")
        self.assertEqual(len(r["final_reviews"]),3)
        self.assertTrue(all(c["checks"][0]["status"]=="unmet" for c in r["final_reviews"]))

    async def test_direct_answer_may_use_only_its_own_source_verified_calculation(self):
        calc=dict(label="비율 차이",requirement_ids=["r1"],operation="difference",
                  operands=[dict(evidence_id="e1",information_id="cpu",field=f) for f in ["user_percent","system_percent"]],output_unit="native")
        answer=dict(text="비율 차이는 13퍼센트포인트입니다",requirement_ids=["r1"],evidence_ids=["e1"],quotes=["16.0 us"])
        r=await self.run_flow([req("cpu",kind="calculation")],Model(final_review=[review(chk("r1"))]),
            writer=Model(report=[dict(hypotheses=[],calculations=[calc],answers=[answer])]))
        self.assertEqual(r["final_validation"]["status"],"passed")
        self.assertEqual(float(r["calculation_results"][0]["value"]),13)
        self.assertIn(answer["text"],r["final"])

    async def test_midchain_support_tasks_do_not_become_new_user_requirements(self):
        r=await self.run_flow([req("cpu")],Model(final_review=[review(chk("r1"))]),
            reviews=[dict(tasks=[task("server-a","process")],requirements=[req("process",kind="diagnosis")]),dict(tasks=[])])
        self.assertEqual(len(r["requirements"]),1)
        self.assertEqual(r["requirements"][0]["information_ids"],["cpu"])
        self.assertEqual(r["final_validation"]["status"],"passed")

    async def test_gate_detects_requirement_omitted_by_planner(self):
        fast=Model(final_review=[review(chk("r1"),missing=[req("process")],tasks=[task("server-a","process")]),
                                 review(chk("r1"),chk("r2"))])
        r=await self.run_flow([req("cpu")],fast)
        self.assertEqual(len(r["requirements"]),2)
        self.assertEqual(self.mcp.calls[-1],("server-a","ps"))
        self.assertTrue(r["final_validation"]["request_fully_satisfied"])

    async def test_unsupported_is_preserved_as_partial_not_success(self):
        r=await self.run_flow([req("history")],Model(final_review=[review(chk("r1","unavailable"))]),tasks=[task("server-a","history")])
        self.assertEqual(self.mcp.calls,[]);self.assertEqual(r["requirements"][0]["status"],"unsupported")
        self.assertFalse(r["final_validation"]["request_fully_satisfied"])
        self.assertIn("history (unsupported)",r["final"])

    async def test_gate_failure_is_closed_not_silent_success(self):
        cases=[([req("process")],[review(chk("r1"))]*2),
               ([req("cpu")],[review()]*2),([req("cpu")],["null","bad JSON"]),
               ([req("cpu")],[review(chk("r1","unmet"),tasks=[task("server-b","process")])]*2),
               ([req("cpu")],[review(chk("r1","unavailable"))]*2)]
        for requirements,responses in cases:
            with self.subTest(responses=responses):
                r=await self.run_flow(requirements,Model(final_review=responses))
                self.assertEqual(r["final_validation"]["status"],"unverified")
                self.assertFalse(r["final_validation"]["request_fully_satisfied"])
                self.assertEqual(self.mcp.calls,[("server-a","top")])
                self.assertIn("최종 답변 검증을 완료하지 못"," ".join(r["limitations"]))

    async def test_persistent_gap_stops_after_two_answer_repairs(self):
        fast=Model(final_review=[review(chk("r1","unmet"),tasks=[task("server-a","cpu")])]*3)
        r=await self.run_flow([req("cpu",kind="explanation")],fast)
        self.assertEqual(len(self.mcp.calls),1);self.assertEqual(len(r["final_reviews"]),3)
        self.assertEqual(r["final_validation"]["repair_rounds"],2)
        self.assertFalse(r["final_validation"]["request_fully_satisfied"])
        self.assertIn("최종 답변 보완은 최대 2회"," ".join(r["limitations"]))

    async def test_initial_and_final_repairs_share_query_round_limit(self):
        fast=Model(final_review=[review(chk("r1","unmet"),tasks=[task("server-a","disk")])]*3)
        r=await self.run_flow([req("cpu",kind="explanation")],fast,
            reviews=[dict(tasks=[task("server-a","process")]),dict(tasks=[task("server-a","log.content")])])
        self.assertEqual({c for _,c in self.mcp.calls},{"top","ps","tail"})
        self.assertEqual(r["replan_rounds"],2)
        self.assertTrue(any("한도" in g["reason"] for g in r["requirement_issues"]))


class CalculationTests(unittest.TestCase):
    def test_noncanonical_process_indices_cannot_bypass_cpu_basis_check(self):
        evidence=[dict(evidence_id="e1",server_id="server-a",facts=[dict(information_id="process",
            fields={"entries":[{"cpu_percent":231.5,"memory_percent":7.0}]})])]
        for index in ["-1", "+0", " 0"]:
            spec=Calculation(label="잘못된 경로",requirement_ids=["r1"],operation="ratio",operands=[
                dict(evidence_id="e1",information_id="process",field=f"entries.{index}.cpu_percent"),
                dict(evidence_id="e1",information_id="process",field="entries.0.memory_percent")])
            with self.subTest(index=index), self.assertRaises(ValueError):calculate(spec,evidence,["r1"])

    def setUp(self):
        self.evidence=[dict(evidence_id="x",server_id="node-x",facts=[dict(information_id="memory.detail",fields={"page_size_bytes":24576,"free_pages":37})]),
                       dict(evidence_id="y",server_id="node-y",facts=[dict(information_id="cpu",fields={"usage_percent":42.5})]),
                       dict(evidence_id="z",server_id="node-z",facts=[dict(information_id="cpu",fields={"usage_percent":17.25})])]
    def calc(self,operation,refs,unit="native"):
        return calculate(Calculation(label="새 입력 계산",requirement_ids=["r9"],operation=operation,
            operands=[dict(evidence_id=e,information_id=i,field=f) for e,i,f in refs],output_unit=unit),self.evidence,["r9"])
    def test_page_conversion_uses_observed_page_size_and_changed_inputs(self):
        refs=[("x","memory.detail","free_pages"),("x","memory.detail","page_size_bytes")]
        r=self.calc("product",refs,"MiB")
        self.assertEqual(float(r["value"]),37*24576/1024**2);self.assertIn("1048576",r["equation"])
        self.evidence[0]["facts"][0]["fields"]["free_pages"]=41
        self.assertEqual(float(self.calc("product",refs,"MiB")["value"]),41*24576/1024**2)
    def test_system_and_process_cpu_percentages_are_not_comparable(self):
        self.evidence.append(dict(evidence_id="p",server_id="synthetic",facts=[dict(information_id="process",
            fields=dict(entries=[dict(cpu_percent=231.5,memory_percent=7.0)]))]))
        refs=[("p","process","entries.0.cpu_percent"),("y","cpu","usage_percent")]
        for op in ["sum","difference","ratio","percent"]:
            with self.subTest(operation=op):
                with self.assertRaisesRegex(ValueError,"집계|기준"):self.calc(op,refs)
        with self.assertRaisesRegex(ValueError,"집계|기준"):
            self.calc("ratio",[("p","process","entries.0.cpu_percent"),("p","process","entries.0.memory_percent")])
        # A process may legitimately exceed 100%; do not clamp or reject it alone.
        self.assertEqual(float(self.calc("sum",refs[:1])["value"]),231.5)

    def test_difference_is_percentage_points_with_both_sources(self):
        r=self.calc("difference",[("y","cpu","usage_percent"),("z","cpu","usage_percent")])
        self.assertEqual(float(r["value"]),25.25);self.assertEqual(r["unit"],"percentage_points")
        self.assertEqual(r["evidence_ids"],["y","z"])
    def test_missing_field_invalid_unit_and_zero_division_are_rejected(self):
        for refs,unit in [([("missing","cpu","usage_percent")],"native"),([("x","memory.detail","__class__")],"native"),
                          ([("y","cpu","usage_percent")],"MiB")]:
            with self.assertRaises(ValueError):self.calc("sum",refs,unit)
        self.evidence[2]["facts"][0]["fields"]["usage_percent"]=0
        with self.assertRaises(ValueError):self.calc("ratio",[("y","cpu","usage_percent"),("z","cpu","usage_percent")])


class ResultPreservationTests(unittest.IsolatedAsyncioTestCase):
    run_flow=CompletionTests.run_flow
    async def test_repairing_explanation_preserves_verified_calculation(self):
        calc=dict(label="CPU 사용자 비율",requirement_ids=["r1"],operation="sum",operands=[dict(evidence_id="e1",information_id="cpu",field="user_percent")],output_unit="native")
        answer=dict(text="제공된 서버 A CPU 정보를 확인했습니다",requirement_ids=["r2"],evidence_ids=["e1"],quotes=["16.0 us"])
        writer=Model(report=[dict(hypotheses=[],calculations=[calc]),dict(hypotheses=[],answers=[answer])])
        r=await self.run_flow([req("cpu",kind="calculation"),req("cpu",kind="explanation")],Model(final_review=[review(chk("r1"),chk("r2","unmet")),review(chk("r1"),chk("r2"))]),writer=writer)
        self.assertEqual(r["final_validation"]["status"],"passed");self.assertEqual(r["calculation_results"][0]["value"],"16")
        self.assertEqual(len(self.mcp.calls),1);self.assertIn("e1",r["calculation_results"][0]["evidence_ids"])
        second=[p for name,p in self.writer.inputs if name=="report"][1]
        self.assertEqual(second["verified_results"]["calculations"][0]["value"],"16");self.assertEqual(second["repair_requirement_ids"],["r2"])
    async def test_invalid_repair_keeps_previous_verified_results(self):
        calc=dict(label="CPU 사용자 비율",requirement_ids=["r1"],operation="sum",operands=[dict(evidence_id="e1",information_id="cpu",field="user_percent")],output_unit="native")
        writer=Model(report=[dict(hypotheses=[],calculations=[calc]),"invalid JSON","invalid JSON","invalid JSON","invalid JSON"])
        r=await self.run_flow([req("cpu",kind="calculation"),req("cpu",kind="explanation")],Model(final_review=[review(chk("r1"),chk("r2","unmet"))]*3),writer=writer)
        self.assertEqual(r["calculation_results"][0]["value"],"16");self.assertEqual(r["final_validation"]["status"],"partial")


class CalculationRejectionFlowTests(unittest.IsolatedAsyncioTestCase):
    run_flow = CompletionTests.run_flow
    async def test_incompatible_cpu_is_terminal_but_independent_rss_calculation_survives(self):
        bad=dict(label="CPU 비율",requirement_ids=["r1"],operation="ratio",operands=[dict(evidence_id="e1",information_id="cpu",field="usage_percent"),dict(evidence_id="e2",information_id="process",field="entries.0.cpu_percent")])
        good=dict(label="RSS 환산",requirement_ids=["r2"],operation="sum",operands=[dict(evidence_id="e2",information_id="process",field="entries.0.rss_kib")],output_unit="MiB")
        writer=Model(report=[dict(hypotheses=[],calculations=[bad,good])])
        alias=req("cpu","process",kind="comparison")
        class AliasReviewer(Model):
            def monitoring_json(self,purpose,payload,schema):
                if purpose=="final_review":
                    self.scripts["final_review"][0]["missing_requirements"][0]["calculation_fingerprint"]=payload["calculation_rejections"][0]["fingerprint"]
                return super().monitoring_json(purpose,payload,schema)
        r=await self.run_flow([req("cpu","process",kind="calculation"),req("process",kind="calculation")],
            Model(final_review=[review(chk("r1","unmet","비율이 없음"),chk("r2"),missing=[alias])]),
            tasks=[task("server-a","cpu","process")],writer=AliasReviewer(report=[dict(hypotheses=[],calculations=[bad,good])]))
        self.assertEqual([p for p,_ in self.writer.inputs].count("report"),1)
        self.assertEqual(len(r["final_reviews"]),1)
        self.assertEqual(r["final_validation"]["status"],"partial")
        self.assertFalse(r["final_validation"]["request_fully_satisfied"])
        self.assertEqual(len(r["calculation_results"]),1)
        self.assertEqual(r["calculation_results"][0]["label"],"RSS 환산")
        self.assertIn("집계 기준",r["final"])
        self.assertEqual(r["requirements"][0]["status"],"unsupported")


class FinalBodyImmutabilityTests(unittest.IsolatedAsyncioTestCase):
    run_flow=CompletionTests.run_flow
    async def test_failed_review_and_exhausted_repairs_do_not_append_unreviewed_body(self):
        for responses in (["invalid","invalid"],[review(chk("r1","unmet"))]*3):
            with self.subTest(responses=responses):
                result=await self.run_flow([req("cpu",kind="explanation")],Model(final_review=responses))
                reviewed=[p["draft_answer"] for name,p in self.writer.inputs if name=="final_review"]
                self.assertEqual(result["final"],reviewed[-1])
                self.assertFalse(result["final_validation"]["request_fully_satisfied"])


class MixedAnswerBindingTests(unittest.IsolatedAsyncioTestCase):
    run_flow=CompletionTests.run_flow
    async def test_rejected_calculation_removes_only_its_answer_binding(self):
        bad=dict(label="CPU 비율",requirement_ids=["r1"],operation="ratio",operands=[dict(evidence_id="e1",information_id="cpu",field="usage_percent"),dict(evidence_id="e2",information_id="process",field="entries.0.cpu_percent")])
        explanation=dict(text="제공된 서버 A CPU 정보를 확인했습니다",requirement_ids=["r1","r2"],evidence_ids=["e1"],quotes=["16.0 us"])
        result=await self.run_flow([req("cpu","process",kind="calculation"),req("cpu",kind="explanation")],Model(final_review=[review(chk("r1"),chk("r2"))]),
            tasks=[task("server-a","cpu","process")],writer=Model(report=[dict(hypotheses=[],calculations=[bad],answers=[explanation])]))
        self.assertIn(explanation["text"],result["final"])
        self.assertEqual(result["answers"][0]["requirement_ids"],["r2"])
        self.assertEqual(result["final_validation"]["status"],"partial")


class CalculationScopeRepairTests(unittest.IsolatedAsyncioTestCase):
    run_flow=CompletionTests.run_flow
    def specifications(self):
        bad=dict(label="잘못 고른 CPU 비율",requirement_ids=["r1"],operation="ratio",operands=[dict(evidence_id="e1",information_id="cpu",field="user_percent"),dict(evidence_id="e2",information_id="process",field="entries.0.cpu_percent")])
        good=dict(label="사용자/시스템 비율",requirement_ids=["r1"],operation="ratio",operands=[dict(evidence_id="e1",information_id="cpu",field="user_percent"),dict(evidence_id="e1",information_id="cpu",field="system_percent")])
        return bad,good

    async def test_wrong_operand_information_is_repairable_not_terminal(self):
        bad,good=self.specifications()
        result=await self.run_flow([req("cpu",kind="calculation")],Model(final_review=[review(chk("r1"))]),
            tasks=[task("server-a","cpu","process")],writer=Model(report=[dict(hypotheses=[],calculations=[bad]),dict(hypotheses=[],calculations=[good])]))
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual(result["calculation_results"][0]["label"],good["label"])
        self.assertFalse(any(h["status"]=="unsupported" for h in result["requirements"][0]["history"]))

    async def test_validator_can_repair_wrong_binding_without_reviving_rejected_operation(self):
        bad,good=self.specifications()
        mismatch={**chk("r1","unmet","요청은 사용자/시스템 비율이며 프로세스 비율이 아닙니다"),"rejection_matches_request":False}
        result=await self.run_flow([req("cpu","process",kind="calculation")],Model(final_review=[review(mismatch),review(chk("r1"))]),
            tasks=[task("server-a","cpu","process")],writer=Model(report=[dict(hypotheses=[],calculations=[bad]),dict(hypotheses=[],calculations=[good])]))
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual([c["label"] for c in result["calculation_results"]],[good["label"]])
        self.assertNotIn("직접 계산할 수 없습니다",result["final"])
        self.assertTrue(any(h["status"]=="unsupported" for h in result["requirements"][0]["history"]))


    async def test_reopened_comparison_without_replacement_cannot_pass(self):
        bad,_=self.specifications()
        mismatch={**chk("r1","unmet"),"rejection_matches_request":False}
        result=await self.run_flow([req("cpu","process",kind="comparison")],
            Model(final_review=[review(mismatch),review(chk("r1")),review(chk("r1"))]),
            tasks=[task("server-a","cpu","process")],
            writer=Model(report=[dict(hypotheses=[],calculations=[bad]),dict(hypotheses=[],calculations=[],answers=[]),dict(hypotheses=[],calculations=[],answers=[])]))
        self.assertEqual(result["final_validation"]["status"],"partial")
        self.assertFalse(result["final_validation"]["request_fully_satisfied"])
        self.assertEqual(result["calculation_results"],[])

    async def test_reopened_qualitative_comparison_accepts_grounded_visible_answer(self):
        bad,_=self.specifications()
        mismatch={**chk("r1","unmet"),"rejection_matches_request":False}
        answer=dict(text="서버 A CPU 사용자 비율이 시스템 비율보다 큽니다",requirement_ids=["r1"],evidence_ids=["e1"],quotes=["16.0 us"])
        result=await self.run_flow([req("cpu","process",kind="comparison")],
            Model(final_review=[review(mismatch),review(chk("r1"))]),tasks=[task("server-a","cpu","process")],
            writer=Model(report=[dict(hypotheses=[],calculations=[bad]),dict(hypotheses=[],answers=[answer])]))
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertIn(answer["text"],result["final"])
        self.assertEqual(result["calculation_results"],[])


class GroundedOutputRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_calculation_intent_cannot_be_downgraded_to_lookup_by_plan(self):
        bad=dict(tasks=[task("server-a","memory.summary")], requirements=[req("memory.summary")])
        good=dict(tasks=[task("server-a","memory.summary")], requirements=[req("memory.summary",kind="calculation")])
        planner=Model(plan=[bad,good],review=[dict(tasks=[])])
        writer=Model(final_review=[review(chk("r1"))]*3)
        result=await investigate(dict(query="전체 대비 사용 중인 RAM의 비율을 %로 알려줘.",scenario_id="server-a",
            single_server_mode=True,history=[],events=[],request_decision=dict(requested_information_ids=["memory.summary"],
            needs_calculation=True,needs_diagnosis=False,needs_explanation=False)),
            SimpleNamespace(mcp=FixtureMCP(),fast_llm=planner,monitoring_llm=planner,quality_llm=writer))
        self.assertEqual(len([p for p,_ in planner.inputs if p=="plan"]),2)
        self.assertEqual(result["requirements"][0]["kind"],"calculation")
        self.assertFalse(result["final_validation"]["request_fully_satisfied"])

    def test_quote_ids_bind_exact_whitespace_without_model_copying(self):
        from agent.investigation import bind_report_sources
        from agent.investigation_models import InvestigationReport
        evidence=[dict(evidence_id="e1",stdout="header\n  2140  batch-worker    372.0\n",facts=[])]
        report=InvestigationReport.model_validate(dict(hypotheses=[dict(text="점유 후보입니다",
            evidence_ids=["e1"],quote_ids=["e1:l2"])],answers=[]))
        bind_report_sources(report,evidence)
        self.assertEqual(report.hypotheses[0].quotes,["  2140  batch-worker    372.0"])

    def test_quote_id_cannot_borrow_a_different_evidence_source(self):
        from agent.investigation import bind_report_sources
        from agent.investigation_models import InvestigationReport
        evidence=[dict(evidence_id="e1",stdout="cpu"),dict(evidence_id="e2",stdout="process")]
        for quote in ["e2:l1","e1:l9"]:
            report=InvestigationReport.model_validate(dict(hypotheses=[dict(text="후보",
                evidence_ids=["e1"],quote_ids=[quote])]))
            with self.assertRaises(ValueError):bind_report_sources(report,evidence)


class StrictContractRegressionTests(unittest.TestCase):
    def test_openai_schema_closes_nested_objects_without_mutating_original(self):
        from agent.openai_llm import strict_json_schema
        from agent.investigation_models import TaskPlan, InvestigationReport
        for contract in [TaskPlan,InvestigationReport]:
            original=contract.model_json_schema();before=copy.deepcopy(original)
            converted=strict_json_schema(original)
            def inspect(value):
                if isinstance(value,dict):
                    if value.get("type")=="object":
                        self.assertFalse(value["additionalProperties"])
                        self.assertEqual(set(value["required"]),set(value["properties"]))
                    self.assertNotIn("default",value)
                    for child in value.values():inspect(child)
                elif isinstance(value,list):
                    for child in value:inspect(child)
            inspect(converted)
            self.assertEqual(before,original)

    def test_qualified_numeric_field_is_canonicalized_before_basis_checks(self):
        from agent.answer_contracts import SourceField
        field=SourceField(evidence_id="e1",information_id="memory.summary",field="fields.used_mib")
        self.assertEqual(field.field,"used_mib")
        evidence=[dict(evidence_id="e1",server_id="server-a",facts=[dict(information_id="memory.summary",
            fields=dict(used_mib=1792,total_mib=7168))])]
        spec=Calculation(label="사용 비율",requirement_ids=["r1"],operation="percent",output_unit="%",operands=[
            dict(evidence_id="e1",information_id="memory.summary",field="fields.used_mib"),
            dict(evidence_id="e1",information_id="memory.summary",field="fields.total_mib")])
        result=calculate(spec,evidence,["r1"])
        self.assertEqual(float(result["value"]),25)
        self.assertEqual(result["inputs"][0]["field"],"used_mib")


class ConditionalDiagnosisRegressionTests(unittest.IsolatedAsyncioTestCase):
    run_flow = CompletionTests.run_flow
    async def test_conditional_diagnosis_can_explain_absent_evidence_without_inventing_cause(self):
        explanation=dict(text="현재 시스템 CPU 사용률은 19%로, 이 표본에서 높은 CPU 부하를 전제로 한 원인 후보를 제시할 근거는 부족합니다",
            requirement_ids=["r1"],evidence_ids=["e1"],quote_ids=["e1:l3"])
        result=await self.run_flow([req("cpu",kind="diagnosis")],Model(final_review=[review(chk("r1"))]),
            writer=Model(report=[dict(hypotheses=[],diagnoses=[{**explanation,"symptom_observed":False}])]))
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual(result["hypotheses"],[])


class IntentRequirementHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_typed_memory_percent_survives_without_a_second_semantic_plan(self):
        original=dict(description="전체 대비 사용 물리 메모리 비율 %",kind="calculation",information_ids=["memory.summary"])
        calc=dict(label="메모리 사용 비율",requirement_ids=["r1"],operation="percent",output_unit="%",operands=[
            dict(evidence_id="e1",information_id="memory.summary",field="used_mib"),
            dict(evidence_id="e1",information_id="memory.summary",field="total_mib")])
        writer=Model(report=[dict(hypotheses=[],calculations=[calc])],final_review=[review(chk("r1"))])
        planner=Model();mcp=FixtureMCP()
        result=await investigate(dict(query="물리 메모리의 사용 비율을 알려줘",scenario_id="server-a",single_server_mode=True,
            history=[],events=[],request_decision=dict(requested_information_ids=["memory.summary"],requested_results=[original],
            needs_calculation=True,needs_diagnosis=False,needs_explanation=False)),
            SimpleNamespace(mcp=mcp,fast_llm=planner,monitoring_llm=planner,quality_llm=writer))
        self.assertFalse(any(p=="plan" for p,_ in planner.inputs))
        self.assertEqual(result["requirements"][0]["description"],original["description"])
        self.assertEqual(result["requirements"][0]["kind"],"calculation")
        self.assertEqual(float(result["calculation_results"][0]["value"]),37.5)
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual(mcp.calls,[("server-a","free")])

    async def test_supporting_process_task_does_not_become_a_new_user_requirement(self):
        original=dict(description="CPU가 높은 경우 원인 후보와 한계",kind="diagnosis",information_ids=["cpu"])
        planner=Model(review=[dict(tasks=[task("server-a","process")],requirements=[req("cpu",kind="calculation")]),dict(tasks=[])])
        writer=Model(final_review=[review(chk("r1","unmet"))]*3)
        result=await investigate(dict(query="CPU 병목의 주범을 짚어줘",scenario_id="server-a",single_server_mode=True,
            history=[],events=[],request_decision=dict(requested_information_ids=["cpu"],requested_results=[original],
            needs_calculation=False,needs_diagnosis=True,needs_explanation=False)),
            SimpleNamespace(mcp=FixtureMCP(),fast_llm=planner,monitoring_llm=planner,quality_llm=writer))
        self.assertEqual([r["kind"] for r in result["requirements"]],["diagnosis"])
        self.assertTrue(any("process" in t["information_ids"] for t in result["tasks"]))
        self.assertFalse(any(p=="plan" for p,_ in planner.inputs))


class TypedContractBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def run_contract(self,query,results,ids,writer=None):
        planner=Model(review=[dict(tasks=[])])
        mcp=FixtureMCP()
        result=await investigate(dict(query=query,scenario_id="server-a",single_server_mode=True,history=[],events=[],
            request_decision=dict(requested_information_ids=ids,requested_results=results,needs_calculation=True,
                                  needs_diagnosis=False,needs_explanation=False)),
            SimpleNamespace(mcp=mcp,fast_llm=planner,monitoring_llm=planner,quality_llm=writer or Model()))
        return result,mcp

    async def test_explicit_command_preserves_typed_calculation_requirement(self):
        result,mcp=await self.run_contract("free 결과로 전체 대비 사용 메모리 비율을 %로 알려줘",
            [dict(description="사용/전체 %",kind="calculation",information_ids=["memory.summary"])],["memory.summary"],
            Model(final_review=[review(chk("r1"))]*3))
        self.assertEqual(result["requirements"][0]["kind"],"calculation")
        self.assertFalse(result["final_validation"]["request_fully_satisfied"])
        self.assertEqual(mcp.calls,[("server-a","free")])

    async def test_result_scope_is_collected_even_if_top_level_ids_omit_it(self):
        result,mcp=await self.run_contract("CPU와 메모리 비율을 확인해줘",[
            dict(description="CPU %",kind="lookup",information_ids=["cpu"]),
            dict(description="메모리 사용/전체 %",kind="calculation",information_ids=["memory.summary"])],["cpu"])
        self.assertTrue(any("memory.summary" in t["information_ids"] for t in result["tasks"]))
        self.assertTrue(any(f["information_id"]=="memory.summary" for e in result["evidence"] for f in e["facts"]))

    async def test_many_valid_information_ids_do_not_overflow_single_task(self):
        ids=["system","cpu","process","memory.summary","memory.detail","disk","network","log.files","log.usage","log.content"]
        result,mcp=await self.run_contract("가능한 정보와 배포 이력을 확인해줘",[
            dict(description="기본 정보",kind="lookup",information_ids=ids),
            dict(description="배포 이력",kind="lookup",information_ids=["deployment"])],ids)
        self.assertTrue(any("deployment" in t["information_ids"] for t in result["tasks"]))
        self.assertTrue(all(len(t["information_ids"])<=10 for t in result["tasks"]))
        self.assertFalse(result["final_validation"]["request_fully_satisfied"])

    async def test_calculation_text_is_rendered_once_from_verified_operands(self):
        spec=dict(label="사용 비율",requirement_ids=["r1"],operation="percent",output_unit="%",operands=[
            dict(evidence_id="e1",information_id="memory.summary",field="used_mib"),
            dict(evidence_id="e1",information_id="memory.summary",field="total_mib")])
        bad_text=dict(text="사용 비율은 37.6%",requirement_ids=["r1"],evidence_ids=["e1"],quote_ids=["e1:l2"])
        writer=Model(report=[dict(hypotheses=[],calculations=[spec],answers=[bad_text])],final_review=[review(chk("r1"))])
        result,_=await self.run_contract("사용 메모리의 비율을 계산해줘",
            [dict(description="사용/전체 %",kind="calculation",information_ids=["memory.summary"])],["memory.summary"],writer)
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertIn("37.5",result["final"]);self.assertNotIn("37.6",result["final"])
        self.assertEqual(result["answers"],[])
        self.assertEqual(len([p for p,_ in writer.inputs if p=="report"]),1)

    def test_diagnosis_block_is_bound_to_its_requirement_and_evidence(self):
        from agent.response_policy import bind_answer_blocks
        blocks=[dict(block_id="b1",text="batch-worker가 기여할 가능성",section="원인")]
        state=dict(hypotheses=[dict(text=blocks[0]["text"],requirement_ids=["r1"],evidence_ids=["e2"])])
        result=bind_answer_blocks(blocks,state)
        self.assertEqual(result[0]["requirement_ids"],["r1"])
        self.assertEqual(result[0]["evidence_ids"],["e2"])


class DiagnosisAssessmentTests(unittest.IsolatedAsyncioTestCase):
    run_flow = CompletionTests.run_flow
    def assessment(self,observed=False,rid="r1"):
        return dict(text="현재 CPU 사용률은 19%이며 이 표본만으로 높은 시스템 부하의 원인을 지목할 근거는 부족합니다",
            requirement_ids=[rid],evidence_ids=["e1"],quote_ids=["e1:l3"],symptom_observed=observed)

    async def test_absent_symptom_cannot_coexist_with_a_cause_candidate(self):
        bad=dict(diagnoses=[self.assessment()],hypotheses=[dict(text="CPU 과다 점유 후보입니다",
            requirement_ids=["r1"],evidence_ids=["e1"],quote_ids=["e1:l3"])])
        good=dict(diagnoses=[self.assessment()],hypotheses=[])
        result=await self.run_flow([req("cpu",kind="diagnosis")],Model(final_review=[review(chk("r1"))]),
            writer=Model(report=[bad,good]))
        self.assertEqual(result["hypotheses"],[])
        self.assertIn("19%",result["final"])
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual(len([p for p,_ in self.writer.inputs if p=="report"]),2)

    async def test_observed_symptom_alone_is_not_a_cause_answer(self):
        statement=self.assessment(observed=True)
        statement["text"]="현재 시스템 CPU 사용률은 19%입니다"
        result=await self.run_flow([req("cpu",kind="diagnosis")],Model(final_review=[review(chk("r1"))]*3),
            writer=Model(report=[dict(diagnoses=[statement],hypotheses=[])]*3))
        self.assertNotEqual(result["final_validation"]["status"],"passed")

    async def test_final_review_receives_current_contract_without_old_rejection_prose(self):
        source=dict(text="현재 CPU 19%를 관측했습니다",requirement_ids=["r1"],evidence_ids=["e1"],quote_ids=["e1:l3"])
        result=await self.run_flow([req("cpu",kind="explanation")],
            Model(final_review=[review(chk("r1","unmet","OLD_REJECTION")),review(chk("r1"))]),
            writer=Model(report=[dict(hypotheses=[]),dict(hypotheses=[],answers=[source])]))
        payload=[x for purpose,x in self.writer.inputs if purpose=="final_review"][-1]
        self.assertNotIn("requirement_issues",payload)
        self.assertNotIn("history",payload["requirements"][0])
        self.assertNotIn("review_reason",payload["requirements"][0])
        self.assertTrue(any("OLD_REJECTION" in i["reason"] for i in result["requirement_issues"]))


class SpecialistFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_specialist_result_is_used_without_cause_writer(self):
        class Expert(Model):
            expert_calls=0
            def diagnostic_json(self,payload,schema):
                self.expert_calls+=1
                assert payload["role"]=="metrics"
                assert "history" not in payload
                return json.dumps(dict(diagnoses=[dict(
                    text="현재 CPU 사용률은 19%로, 높은 CPU 부하의 전제가 관측되지 않았습니다",
                    requirement_ids=["r1"],evidence_ids=["e1"],quote_ids=["e1:l3"],symptom_observed=False)],tasks=[])),{"usd":0}
        model=Expert(final_review=[review(chk("r1"))]);mcp=FixtureMCP()
        result=await investigate(dict(query="CPU가 높은 경우 원인 후보를 설명해줘",scenario_id="server-a",single_server_mode=True,
            events=[],history=[],request_decision=dict(requested_information_ids=["cpu"],needs_diagnosis=True,needs_explanation=False,
                requested_results=[dict(description="높은 CPU의 원인 후보와 한계",kind="diagnosis",information_ids=["cpu"])])),
            SimpleNamespace(mcp=mcp,quality_llm=model,fast_llm=Model(),monitoring_llm=Model()))
        self.assertEqual(model.expert_calls,1)
        self.assertFalse(any(p=="report" for p,_ in model.inputs))
        self.assertEqual(mcp.calls,[("server-a","top")])
        self.assertEqual(result["hypotheses"],[])
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertTrue(any(e["type"]=="SpecialistReturned" for e in result["events"]))


class SpecialistBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def run_specialist(self,model,information_ids):
        mcp=FixtureMCP()
        result=await investigate(dict(query="관측한 상태와 로그의 원인 후보를 확인해줘",scenario_id="server-a",single_server_mode=True,
            events=[],history=[],request_decision=dict(requested_information_ids=information_ids,needs_diagnosis=True,
            needs_explanation=False,requested_results=[dict(description="배정된 관측에서 증상과 원인 후보 확인",kind="diagnosis",information_ids=information_ids)])),
            SimpleNamespace(mcp=mcp,quality_llm=model,fast_llm=Model(),monitoring_llm=Model()))
        return result,mcp

    async def test_mixed_cpu_log_requirement_reaches_both_roles(self):
        class Expert(Model):
            roles=[]
            def diagnostic_json(self,payload,schema):
                self.roles.append((payload["role"],{f["information_id"] for e in payload["observations"] for f in e["facts"]}))
                eid=payload["observations"][0]["evidence_id"]
                source=next(q for q in payload["quote_catalog"] if q["quote_id"]==eid+(":l3" if payload["role"]=="metrics" else ":l1"))
                return json.dumps(dict(diagnoses=[dict(text="배정된 관측에서 원인을 지목할 근거는 부족합니다",
                    requirement_ids=["r1"],evidence_ids=[eid],quote_ids=[source["quote_id"]],symptom_observed=False)],tasks=[])),{"usd":0}
        model=Expert(final_review=[review(chk("r1"))]);model.roles=[]
        result,_=await self.run_specialist(model,["cpu","log.content"])
        self.assertEqual([r for r,_ in model.roles],["metrics","logs"])
        self.assertIn("log.content",model.roles[1][1]);self.assertNotIn("log.content",model.roles[0][1])
        self.assertEqual(len(result["answers"]),2)
        self.assertEqual(result["final_validation"]["status"],"passed")

    async def test_empty_assessment_with_duplicate_task_is_retried_by_specialist(self):
        class Expert(Model):
            expert_calls=0
            def diagnostic_json(self,payload,schema):
                self.expert_calls+=1
                if self.expert_calls==1:return json.dumps(dict(diagnoses=[],tasks=[task("server-a","cpu")])),{"usd":0}
                assert "validation_feedback" in payload
                return json.dumps(dict(diagnoses=[dict(text="현재 CPU 사용률은 19%로 높은 부하 전제가 없습니다",
                    requirement_ids=["r1"],evidence_ids=["e1"],quote_ids=["e1:l3"],symptom_observed=False)],tasks=[])),{"usd":0}
        model=Expert(final_review=[review(chk("r1"))])
        result,mcp=await self.run_specialist(model,["cpu"])
        self.assertEqual(model.expert_calls,2)
        self.assertEqual(mcp.calls,[("server-a","top")])
        self.assertEqual(result["final_validation"]["status"],"passed")


class WithdrawnCauseTests(unittest.IsolatedAsyncioTestCase):
    run_flow=CompletionTests.run_flow
    async def test_corrected_absent_symptom_withdraws_previous_candidate(self):
        assessment=dict(text="시스템 CPU 사용률은 19%입니다",requirement_ids=["r1"],evidence_ids=["e1"],quote_ids=["e1:l3"],symptom_observed=True)
        candidate=dict(text="CPU 사용량 기여 후보입니다",requirement_ids=["r1"],evidence_ids=["e1"],quote_ids=["e1:l3"])
        corrected={**assessment,"symptom_observed":False,"text":"시스템 CPU는 19%로 높은 부하 전제가 관측되지 않았습니다"}
        model=Model(report=[dict(diagnoses=[assessment],hypotheses=[],causes=[dict(requirement_id="r1",factor_id="e2:process:0:cpu")]),dict(diagnoses=[corrected],hypotheses=[])])
        result=await self.run_flow([req("cpu",kind="diagnosis")],
            Model(final_review=[review(chk("r1","unmet"),tasks=[task("server-a","disk")]),review(chk("r1"))]),writer=model,tasks=[task("server-a","cpu","process")])
        second=[payload for purpose,payload in model.inputs if purpose=="report"][1]
        self.assertTrue(second["verified_results"]["hypotheses"])
        self.assertEqual(result["hypotheses"],[])
        self.assertNotIn(candidate["text"],result["final"])
        self.assertIn(corrected["text"],result["final"])


class SpecialistIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_either_role_failure_keeps_other_role_and_never_false_passes(self):
        for failed_role in ("metrics", "logs"):
            with self.subTest(failed_role=failed_role):
                class Expert(Model):
                    def __init__(self):
                        super().__init__(); self.roles=[]
                    def diagnostic_json(self, payload, schema):
                        role = payload["role"]; self.roles.append(role)
                        if role == failed_role: return "not-json", {"usd": 0}
                        eid = payload["observations"][0]["evidence_id"]
                        qid = eid + (":l3" if role == "metrics" else ":l1")
                        return json.dumps(dict(diagnoses=[dict(text=role+" 영역에서 원인을 특정할 근거가 부족합니다",
                            requirement_ids=["r1"], evidence_ids=[eid], quote_ids=[qid], symptom_observed=False)], tasks=[])), {"usd": 0}
                model = Expert()
                model.scripts["final_review"] = [review(chk("r1"))]*3
                result, mcp = await SpecialistBoundaryTests.run_specialist(self, model, ["cpu", "log.content"])
                successful_role = "logs" if failed_role == "metrics" else "metrics"
                self.assertIn(successful_role, model.roles)
                self.assertEqual(model.roles.count(successful_role), 1)
                self.assertEqual(model.roles.count(failed_role), 2)
                self.assertTrue(any(successful_role in a["text"] for a in result["answers"]))
                self.assertIn(successful_role+" 영역", result["final"])
                self.assertEqual(result["final_validation"]["status"], "partial")
                self.assertTrue(any(e["type"] == "SpecialistFailed" and e["role"] == failed_role for e in result["events"]))
                self.assertEqual(len(mcp.calls), 2)
                self.assertEqual(len(result["final_reviews"]), 1)

    async def test_new_role_evidence_retries_failed_role_and_reuses_successful_role(self):
        class Expert(Model):
            def __init__(self):
                super().__init__(); self.roles=[]
            def diagnostic_json(self, payload, schema):
                role=payload["role"];self.roles.append(role)
                if role=="metrics" and self.roles.count(role)<=2:return "not-json", {"usd":0}
                eid=payload["observations"][0]["evidence_id"]
                return json.dumps(dict(diagnoses=[dict(text=role+" 영역에서 현재 원인을 지목할 근거가 부족합니다",
                    requirement_ids=["r1"],evidence_ids=[eid],quote_ids=[eid+(":l3" if role=="metrics" else ":l1")],
                    symptom_observed=False)],tasks=[])), {"usd":0}
        model=Expert()
        model.scripts["final_review"]=[review(chk("r1","unmet"),tasks=[task("server-a","process")]),review(chk("r1"))]
        result,mcp=await SpecialistBoundaryTests.run_specialist(self,model,["cpu","log.content"])
        self.assertEqual(model.roles.count("metrics"),3)
        self.assertEqual(model.roles.count("logs"),1)
        self.assertEqual(len(mcp.calls),3)
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertTrue(any(e["type"]=="SpecialistReused" and e["role"]=="logs" for e in result["events"]))
        self.assertFalse(any("전문 판단을 완료하지 못했습니다" in note for note in result["limitations"]))


class EarlyCalculationCompletenessTests(unittest.IsolatedAsyncioTestCase):
    run_flow=CompletionTests.run_flow

    async def test_mib_paraphrase_cannot_replace_percent_before_final_review(self):
        calculation=dict(label="물리 메모리 사용 비율",requirement_ids=["r1"],operation="percent",operands=[
            dict(evidence_id="e1",information_id="memory.summary",field="used_mib"),
            dict(evidence_id="e1",information_id="memory.summary",field="total_mib")],output_unit="native")
        paraphrase=dict(text="전체 메모리 8192 MiB 중 3072 MiB를 사용 중입니다",requirement_ids=["r1"],
            evidence_ids=["e1"],quote_ids=["e1:l2"])
        writer=Model(report=[dict(hypotheses=[],answers=[paraphrase],calculations=[]),
                             dict(hypotheses=[],calculations=[calculation])])
        result=await self.run_flow([req("memory.summary",kind="calculation")],Model(final_review=[review(chk("r1"))]),
            tasks=[task("server-a","memory.summary")],writer=writer)
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual(len(result["final_reviews"]),1)
        self.assertEqual(result["final_validation"]["repair_rounds"],0)
        self.assertEqual([p for p,_ in writer.inputs], ["report","report","final_review"])
        self.assertEqual(float(result["calculation_results"][0]["value"]),37.5)
        self.assertIn("37.5",result["final"])


    async def test_each_ready_calculation_must_be_covered_before_final_review(self):
        def calculation(rid, operation, fields):
            return dict(label="원본 메모리 계산",requirement_ids=[rid],operation=operation,
                operands=[dict(evidence_id="e1",information_id="memory.summary",field=f) for f in fields])
        percent=calculation("r1","percent",["used_mib","total_mib"])
        remaining=calculation("r2","difference",["total_mib","used_mib"])
        writer=Model(report=[dict(hypotheses=[],calculations=[percent]),
                             dict(hypotheses=[],calculations=[percent,remaining])])
        ratio=req("memory.summary",kind="calculation");ratio["description"]="사용/전체 메모리 비율"
        difference=req("memory.summary",kind="calculation");difference["description"]="전체-사용 메모리 용량"
        result=await self.run_flow([ratio,difference],Model(final_review=[review(chk("r1"),chk("r2"))]),
            tasks=[task("server-a","memory.summary")],writer=writer)
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual(len(result["final_reviews"]),1)
        self.assertEqual([p for p,_ in writer.inputs],["report","report","final_review"])
        self.assertEqual({rid for c in result["calculation_results"] for rid in c["requirement_ids"]},{"r1","r2"})


    def two_memory_calculations(self):
        def calculation(rid, operation, fields):
            return dict(label="원본 메모리 계산",requirement_ids=[rid],operation=operation,output_unit="%" if operation=="percent" else "MiB",
                operands=[dict(evidence_id="e1",information_id="memory.summary",field=f) for f in fields])
        requirements=[req("memory.summary",kind="calculation"),req("memory.summary",kind="calculation")]
        requirements[0]["description"]="사용/전체 메모리 비율"
        requirements[1]["description"]="전체-사용 메모리 용량"
        return requirements, calculation("r1","percent",["used_mib","total_mib"]), calculation("r2","difference",["total_mib","used_mib"])

    async def test_valid_sibling_calculation_survives_exhausted_report_retries(self):
        requirements,percent,_=self.two_memory_calculations()
        writer=Model(report=[dict(hypotheses=[],calculations=[percent]) for _ in range(6)])
        result=await self.run_flow(requirements,Model(final_review=[review(chk("r1"),chk("r2"))]*3),
            tasks=[task("server-a","memory.summary")],writer=writer)
        self.assertEqual(result["final_validation"]["status"],"partial")
        self.assertEqual(float(result["calculation_results"][0]["value"]),37.5)
        self.assertIn("37.5",result["final"])
        self.assertEqual(result["requirements"][0]["status"],"satisfied")
        self.assertNotEqual(result["requirements"][1]["status"],"satisfied")

    async def test_retry_only_supplies_missing_calculation_and_retains_first(self):
        requirements,percent,remaining=self.two_memory_calculations()
        writer=Model(report=[dict(hypotheses=[],calculations=[percent]),dict(hypotheses=[],calculations=[remaining])])
        result=await self.run_flow(requirements,Model(final_review=[review(chk("r1"),chk("r2"))]),
            tasks=[task("server-a","memory.summary")],writer=writer)
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual([p for p,_ in writer.inputs],["report","report","final_review"])
        self.assertEqual(writer.inputs[1][1]["required_calculation_ids"],["r2"])
        self.assertEqual(writer.inputs[1][1]["verified_results"]["calculations"][0]["value"],"37.5")
        self.assertEqual({(float(c["value"]),c["unit"]) for c in result["calculation_results"]},{(37.5,"%"),(5120.0,"MiB")})


class ActionableReviewTests(unittest.IsolatedAsyncioTestCase):
    run_flow=CompletionTests.run_flow

    async def test_unmet_without_any_repair_instruction_retries_judge_only(self):
        first=dict(checks=[chk("r1","unmet","답변이 부족합니다")],repair_tasks=[],missing_requirements=[],rewrite_instructions="")
        writer=Model(final_review=[first,review(chk("r1"))])
        result=await self.run_flow([req("cpu")],writer)
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual(result["final_validation"]["repair_rounds"],0)
        self.assertEqual([p for p,_ in self.writer.inputs],["report","final_review","final_review"])
        self.assertIn("구체적인",self.writer.inputs[2][1]["validation_feedback"])
        self.assertEqual(len(self.mcp.calls),1)


    async def test_terminal_basis_rejection_needs_no_rewrite_instruction(self):
        bad=dict(label="시스템 CPU 중 프로세스 비율",requirement_ids=["r1"],operation="percent",operands=[
            dict(evidence_id="e2",information_id="process",field="entries.0.cpu_percent"),
            dict(evidence_id="e1",information_id="cpu",field="usage_percent")])
        verdict=dict(checks=[chk("r1","unmet","집계 기준이 달라 요청한 비율을 계산할 수 없습니다")],
                     repair_tasks=[],missing_requirements=[],rewrite_instructions="")
        writer=Model(report=[dict(hypotheses=[],calculations=[bad])],final_review=[verdict,verdict])
        result=await self.run_flow([req("cpu","process",kind="calculation")],writer,
            tasks=[task("server-a","cpu","process")])
        self.assertEqual(result["final_validation"]["status"],"partial")
        self.assertEqual(result["final_reviews"][0]["checks"][0]["status"],"unavailable")
        self.assertEqual([p for p,_ in self.writer.inputs],["report","final_review"])


class SpecialistContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, tasks, bad_quote=False):
        class Expert(Model):
            expert_calls=0
            def diagnostic_json(self, payload, schema):
                self.expert_calls+=1
                has_process=any(f["information_id"]=="process" for e in payload["observations"] for f in e["facts"])
                statement=dict(text="현재 CPU는 19%로 높은 부하 전제가 관측되지 않았습니다" if has_process else "잘못된 진단 수치 999%",
                    requirement_ids=["r1"],evidence_ids=["e1"],quote_ids=["e1:l3"],symptom_observed=False)
                if not has_process and bad_quote:
                    statement.update(text="확인되지 않은 진단 문장",quote_ids=["e1:l999"])
                return json.dumps(dict(diagnoses=[statement],tasks=[] if has_process else tasks)),{"usd":0}
        model=Expert(final_review=[review(chk("r1"))]*3)
        result,mcp=await SpecialistBoundaryTests.run_specialist(self,model,["cpu"])
        return result,mcp,model

    async def test_invalid_prose_does_not_discard_valid_new_observation_task(self):
        for bad_quote in (False,True):
            with self.subTest(bad_quote=bad_quote):
                result,mcp,model=await self.run_case([task("server-a","process")],bad_quote)
                self.assertEqual(mcp.calls,[("server-a","top"),("server-a","ps")])
                self.assertEqual(model.expert_calls,2)
                self.assertEqual(result["final_validation"]["status"],"passed")
                self.assertNotIn("999",result["final"])
                self.assertNotIn("확인되지 않은 진단 문장",result["final"])
                self.assertTrue(any(i["stage"]=="diagnostic_metrics" for i in result["requirement_issues"]))

    async def test_invalid_task_scope_cannot_rescue_invalid_prose(self):
        for invalid in (task("server-b","process"),task("server-a","log.content")):
            with self.subTest(invalid=invalid):
                result,mcp,_=await self.run_case([task("server-a","process"),invalid])
                self.assertEqual(mcp.calls,[("server-a","top")])
                self.assertNotEqual(result["final_validation"]["status"],"passed")
                self.assertNotIn("999",result["final"])

    async def test_no_task_or_already_observed_task_does_not_accept_invalid_prose(self):
        for tasks in ([],[task("server-a","cpu")]):
            with self.subTest(tasks=tasks):
                result,mcp,model=await self.run_case(tasks)
                self.assertEqual(mcp.calls,[("server-a","top")])
                self.assertEqual(model.expert_calls,2)
                self.assertNotEqual(result["final_validation"]["status"],"passed")
                self.assertNotIn("999",result["final"])


class ReportItemPreservationTests(unittest.IsolatedAsyncioTestCase):
    run_flow=CompletionTests.run_flow

    def two_explanations(self):
        total=req("memory.summary",kind="explanation");total["description"]="전체 물리 메모리를 설명"
        used=req("memory.summary",kind="explanation");used["description"]="사용 물리 메모리를 설명"
        return [total,used]

    def memory_calculation(self):
        return dict(label="물리 메모리 사용 비율",requirement_ids=["r1"],operation="percent",operands=[
            dict(evidence_id="e1",information_id="memory.summary",field="used_mib"),
            dict(evidence_id="e1",information_id="memory.summary",field="total_mib")],output_unit="native")

    def answer(self, rid, text, quote="e1:l2"):
        return dict(text=text,requirement_ids=[rid],evidence_ids=["e1"],quote_ids=[quote])

    async def test_valid_answer_survives_sibling_invalid_source(self):
        good=self.answer("r1","전체 메모리는 8192 MiB입니다")
        bad=self.answer("r2","사용 메모리는 99999 MiB입니다","e1:l999")
        writer=Model(report=[dict(hypotheses=[],answers=[good,bad]) for _ in range(6)])
        result=await self.run_flow(self.two_explanations(),
            Model(final_review=[review(chk("r1"),chk("r2","unmet","잘못된 인용"))]*3),
            tasks=[task("server-a","memory.summary")],writer=writer)
        self.assertIn(good["text"],result["final"])
        self.assertNotIn(bad["text"],result["final"])
        self.assertEqual({rid for a in result["answers"] for rid in a["requirement_ids"]},{"r1"})
        self.assertEqual(result["final_validation"]["status"],"partial")

    async def test_valid_calculation_survives_sibling_invalid_source(self):
        calc=self.memory_calculation();bad=self.answer("r2","사용 메모리는 99999 MiB입니다","e1:l999")
        writer=Model(report=[dict(hypotheses=[],calculations=[calc],answers=[bad]) for _ in range(6)])
        result=await self.run_flow([req("memory.summary",kind="calculation"),req("memory.summary",kind="explanation")],
            Model(final_review=[review(chk("r1"),chk("r2","unmet","잘못된 인용"))]*3),
            tasks=[task("server-a","memory.summary")],writer=writer)
        self.assertEqual([float(c["value"]) for c in result["calculation_results"]],[37.5])
        self.assertIn("37.5",result["final"])
        self.assertNotIn("99999",result["final"])
        self.assertEqual(result["final_validation"]["status"],"partial")

    async def test_retry_only_returns_missing_answer_without_losing_verified_sibling(self):
        first=self.answer("r1","전체 메모리는 8192 MiB입니다")
        bad=self.answer("r2","사용 메모리는 99999 MiB입니다","e1:l999")
        repaired=self.answer("r2","사용 메모리는 3072 MiB입니다")
        writer=Model(report=[dict(hypotheses=[],answers=[first,bad]),dict(hypotheses=[],answers=[repaired])])
        result=await self.run_flow(self.two_explanations(),
            Model(final_review=[review(chk("r1"),chk("r2"))]),tasks=[task("server-a","memory.summary")],writer=writer)
        self.assertIn(first["text"],result["final"])
        self.assertIn(repaired["text"],result["final"])
        self.assertNotIn(bad["text"],result["final"])
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual([p for p,_ in writer.inputs],["report","report","final_review"])
        self.assertEqual(writer.inputs[1][1]["verified_results"]["answers"][0]["text"],first["text"])

    async def test_bad_calculation_does_not_discard_independent_explanation(self):
        calc=self.memory_calculation();calc["operands"][0]["evidence_id"]="e999"
        good=self.answer("r2","전체 메모리는 8192 MiB입니다")
        writer=Model(report=[dict(hypotheses=[],calculations=[calc],answers=[good]) for _ in range(6)])
        result=await self.run_flow([req("memory.summary",kind="calculation"),req("memory.summary",kind="explanation")],
            Model(final_review=[review(chk("r1","unmet","계산 근거 없음"),chk("r2"))]*3),
            tasks=[task("server-a","memory.summary")],writer=writer)
        self.assertEqual(result["calculation_results"],[])
        self.assertIn(good["text"],result["final"])
        self.assertEqual(result["final_validation"]["status"],"partial")


class PartialReportWithdrawalTests(unittest.IsolatedAsyncioTestCase):
    run_flow=CompletionTests.run_flow

    async def test_verified_absence_withdraws_old_candidate_despite_sibling_error(self):
        assessment=dict(text="시스템 CPU 사용률은 19%입니다",requirement_ids=["r1"],evidence_ids=["e1"],quote_ids=["e1:l3"],symptom_observed=True)
        candidate=dict(text="CPU 사용량 기여 후보입니다",requirement_ids=["r1"],evidence_ids=["e1"],quote_ids=["e1:l3"])
        corrected={**assessment,"symptom_observed":False,"text":"시스템 CPU는 19%로 높은 부하 전제가 관측되지 않았습니다"}
        bad=dict(text="검증되지 않은 메모리 설명",requirement_ids=["r2"],evidence_ids=["e1"],quote_ids=["e1:l999"])
        writer=Model(report=[dict(diagnoses=[assessment],hypotheses=[],causes=[dict(requirement_id="r1",factor_id="e2:process:0:cpu")],answers=[bad])]+
                           [dict(diagnoses=[corrected],hypotheses=[],answers=[bad]) for _ in range(5)])
        result=await self.run_flow([req("cpu",kind="diagnosis"),req("memory.summary",kind="explanation")],
            Model(final_review=[review(chk("r1"),chk("r2","unmet","인용 오류"))]*3),writer=writer,tasks=[task("server-a","cpu","process")])
        second=[payload for purpose,payload in writer.inputs if purpose=="report"][1]
        self.assertTrue(second["verified_results"]["hypotheses"])
        self.assertIn(corrected["text"],result["final"])
        self.assertNotIn(candidate["text"],result["final"])
        self.assertEqual(result["hypotheses"],[])
        self.assertEqual(result["final_validation"]["status"],"partial")

    def test_withdrawal_removes_only_absent_requirement_binding(self):
        from agent.investigation import withdraw_absent_hypotheses
        h=dict(text="두 요구의 후보",requirement_ids=["r1","r2"],evidence_ids=["e1"])
        result=withdraw_absent_hypotheses([h],[dict(requirement_ids=["r1"],symptom_observed=False),dict(requirement_ids=["r2"],symptom_observed=True)])
        self.assertEqual(result,[{**h,"requirement_ids":["r2"]}])
        self.assertEqual(h["requirement_ids"],["r1","r2"])

    def test_observed_log_role_is_not_withdrawn_by_normal_metrics_role(self):
        from agent.investigation import withdraw_absent_hypotheses
        h=dict(text="로그에서 관측한 후보",requirement_ids=["r1"],evidence_ids=["e2"])
        result=withdraw_absent_hypotheses([h],[dict(requirement_ids=["r1"],symptom_observed=False),dict(requirement_ids=["r1"],symptom_observed=True)])
        self.assertEqual(result,[h])


class CausalSelectionTests(unittest.IsolatedAsyncioTestCase):
    run_flow=CompletionTests.run_flow

    async def test_impact_in_hypothesis_cannot_pass_permissive_final_judge(self):
        observed=dict(text="현재 시스템 CPU 사용률은 19%입니다", requirement_ids=["r1"],
            evidence_ids=["e1"],quote_ids=["e1:l3"],symptom_observed=True)
        impact=dict(text="CPU 사용으로 시스템 성능 저하가 발생할 가능성이 있습니다",
            requirement_ids=["r1"],evidence_ids=["e1"],quote_ids=["e1:l3"])
        result=await self.run_flow([req("cpu",kind="diagnosis")],
            Model(final_review=[review(chk("r1"))]*3),
            writer=Model(report=[dict(diagnoses=[observed],hypotheses=[impact]) for _ in range(6)]))
        self.assertNotEqual(result["final_validation"]["status"],"passed")
        self.assertNotIn(impact["text"],result["final"])
        self.assertEqual(result["hypotheses"],[])

    def observed(self,info="memory.summary",sid="server-x"):
        from monitoring.evidence import extract_facts
        rows=[("e1","free","              total used free shared buff/cache available\nMem: 10000 9400 200 0 400 600\nSwap: 2000 1000 1000\n"),
              ("e2","ps","PID PPID COMMAND %CPU %MEM RSS\n777 1 unfamiliar-worker 145.5 61.2 6266880\n"),
              ("e3","top","top - load average: 3.0, 2.0, 1.0\n%Cpu(s): 91.0 us, 5.0 sy, 4.0 id\n")]
        evidence=[]
        for eid,cmd,stdout in rows:
            e=dict(evidence_id=eid,server_id=sid,command_id=cmd,os="linux",stdout=stdout,exit_code=0,error=None)
            e["facts"]=extract_facts(e);evidence.append(e)
        requirements=[dict(requirement_id="r1",kind="diagnosis",description="현재 증상의 기여 후보",server_ids=[sid],information_ids=[info])]
        source="e3" if info=="cpu" else "e1"
        quote=next(f["quote"] for e in evidence if e["evidence_id"]==source for f in e["facts"] if f["information_id"]==info)
        assessment=[dict(requirement_ids=["r1"],evidence_ids=[source],quotes=quote.splitlines(),symptom_observed=True)]
        return evidence,requirements,assessment

    def test_unknown_process_name_uses_its_own_memory_observation(self):
        from agent.investigation import cause_catalog,bind_cause
        from agent.investigation_models import CauseSelection
        evidence,requirements,assessments=self.observed()
        choices=cause_catalog(evidence,requirements)
        item=next(c for c in choices if c["kind"]=="process_memory")
        result=bind_cause(CauseSelection(requirement_id="r1",factor_id=item["factor_id"]),evidence,requirements,assessments)
        self.assertIn("unfamiliar-worker",result.text)
        self.assertIn("6266880 KiB",result.text)
        self.assertIn("61.2%",result.text)
        self.assertNotIn("145.5",result.text)
        self.assertEqual(set(result.evidence_ids),{"e1","e2"})
        self.assertEqual(result.requirement_ids,["r1"])
        self.assertTrue(any("777" in q for q in result.quotes))

    def test_cpu_basis_is_explicit_and_cannot_be_selected_for_memory(self):
        from agent.investigation import cause_catalog,bind_cause
        from agent.investigation_models import CauseSelection
        evidence,cpu,assessments=self.observed("cpu")
        choice=next(c for c in cause_catalog(evidence,cpu) if c["kind"]=="process_cpu")
        selection=CauseSelection(requirement_id="r1",factor_id=choice["factor_id"])
        result=bind_cause(selection,evidence,cpu,assessments)
        self.assertIn("145.5%",result.text)
        self.assertIn("코어 합산",result.text)
        self.assertEqual(set(result.evidence_ids),{"e2","e3"})
        _,memory,ma=self.observed()
        with self.assertRaises(ValueError):bind_cause(selection,evidence,memory,ma)

    def test_missing_or_other_target_factor_cannot_be_selected(self):
        from agent.investigation import cause_catalog,bind_cause
        from agent.investigation_models import CauseSelection
        evidence,requirements,assessments=self.observed()
        choice=cause_catalog(evidence,requirements)[0]
        selection=CauseSelection(requirement_id="r1",factor_id=choice["factor_id"])
        for changed in [evidence[:1], [dict(e,server_id="server-y") if e["evidence_id"]=="e2" else e for e in evidence]]:
            with self.assertRaises(ValueError):bind_cause(selection,changed,requirements,assessments)
        with self.assertRaises(ValueError):bind_cause(CauseSelection(requirement_id="r1",factor_id="made-up"),evidence,requirements,assessments)

    def test_normal_or_unverified_symptom_does_not_allow_factor(self):
        from agent.investigation import cause_catalog,bind_cause
        from agent.investigation_models import CauseSelection
        evidence,requirements,assessments=self.observed()
        selection=CauseSelection(requirement_id="r1",factor_id=cause_catalog(evidence,requirements)[0]["factor_id"])
        for invalid in [[],[dict(assessments[0],symptom_observed=False)], [dict(assessments[0],quotes=["not in source"])]]:
            with self.assertRaises(ValueError):bind_cause(selection,evidence,requirements,invalid)

    def test_log_storage_factor_belongs_to_longest_matching_mount(self):
        from agent.investigation import cause_catalog
        from monitoring.evidence import extract_facts
        evidence=[]
        for eid,cmd,stdout in [("d","df","Filesystem Size Used Avail Use% Mounted on\n/dev/root 100G 95G 5G 95% /\n/dev/log 20G 2G 18G 10% /var/log\n"),
                               ("l","du","1048576 /var/log/\n")]:
            e=dict(evidence_id=eid,server_id="server-x",command_id=cmd,os="linux",stdout=stdout,exit_code=0,error=None)
            e["facts"]=extract_facts(e);evidence.append(e)
        requirements=[dict(requirement_id="r1",kind="diagnosis",description="공간 점유 후보",server_ids=["server-x"],information_ids=["disk"])]
        choices=cause_catalog(evidence,requirements)
        storage=[c for c in choices if c["kind"]=="log_storage"]
        self.assertEqual(storage,[])  # Healthy separate log mount cannot explain full root.
        evidence[0]["stdout"]=evidence[0]["stdout"].replace("2G 18G 10%","19G 1G 95%")
        evidence[0]["facts"]=extract_facts(evidence[0])
        storage=[c for c in cause_catalog(evidence,requirements) if c["kind"]=="log_storage"]
        self.assertEqual(len(storage),1)
        self.assertEqual(storage[0]["mount"],"/var/log")
        self.assertFalse(any("/dev/root" in q for q in storage[0]["quotes"]))

    def test_only_explicit_log_reason_becomes_a_factor(self):
        from monitoring.evidence import extract_facts,causal_factors
        e=dict(evidence_id="l",server_id="server-x",command_id="tail",os="linux",exit_code=0,error=None,
            stdout="2026-10-22T01:00:00Z ERROR memory pressure\n2026-10-22T01:01:00Z ERROR request failed reason=connection refused\n")
        e["facts"]=extract_facts(e)
        factors=causal_factors([e])
        self.assertEqual(len(factors),1)
        self.assertEqual(factors[0]["kind"],"log_reason")
        self.assertIn("connection refused",factors[0]["text"])
        self.assertNotIn("memory pressure",factors[0]["text"])


    async def test_omitted_assessment_cannot_pass_with_impact_in_answers(self):
        impact=dict(text="CPU 사용으로 시스템 성능 저하가 발생할 가능성이 있습니다",
            requirement_ids=["r1"],evidence_ids=["e1"],quote_ids=["e1:l3"])
        result=await self.run_flow([req("cpu",kind="diagnosis")],Model(final_review=[review(chk("r1"))]*3),
            writer=Model(report=[dict(hypotheses=[],answers=[impact]) for _ in range(6)]))
        self.assertFalse(result["final_validation"]["request_fully_satisfied"])
        self.assertNotEqual(result["final_validation"]["status"],"passed")

    def test_memory_quote_in_shared_top_does_not_authorize_cpu_factor(self):
        from agent.investigation import cause_catalog,bind_cause
        from agent.investigation_models import CauseSelection
        from monitoring.evidence import extract_facts
        evidence,requirements,_=self.observed()
        e=dict(evidence_id="e3",server_id="server-x",command_id="top",os="macos",exit_code=0,error=None,
            stdout="Load Avg: 1.0, 1.0, 1.0\nCPU usage: 10.0% user, 5.0% sys, 85.0% idle\nPhysMem: 7900M used, 100M unused\n")
        e["facts"]=extract_facts(e);evidence=[evidence[1],e]
        requirements[0]["information_ids"]=["cpu","memory.summary"]
        assessment=[dict(requirement_ids=["r1"],evidence_ids=["e3"],quotes=["PhysMem: 7900M used, 100M unused"],symptom_observed=True)]
        choices=cause_catalog(evidence,requirements)
        cpu=next(c for c in choices if c["kind"]=="process_cpu")
        memory=next(c for c in choices if c["kind"]=="process_memory")
        with self.assertRaises(ValueError):
            bind_cause(CauseSelection(requirement_id="r1",factor_id=cpu["factor_id"]),evidence,requirements,assessment)
        valid=bind_cause(CauseSelection(requirement_id="r1",factor_id=memory["factor_id"]),evidence,requirements,assessment)
        self.assertIn("RSS",valid.text)
        self.assertNotIn("메모리 압박",valid.text)


    async def test_each_diagnosis_requires_its_own_candidate_before_final_review(self):
        assessments=[dict(text="현재 시스템 CPU 사용률은 19%입니다",requirement_ids=[rid],
                          evidence_ids=["e1"],quote_ids=["e1:l3"],symptom_observed=True) for rid in ("r1","r2")]
        candidate=lambda rid:dict(requirement_id=rid,factor_id="e2:process:0:cpu")
        first=dict(diagnoses=assessments,hypotheses=[],causes=[candidate("r1")])
        second=dict(diagnoses=assessments,hypotheses=[],causes=[candidate("r2")])
        result=await self.run_flow([req("cpu",kind="diagnosis"),dict(req("cpu",kind="diagnosis"),description="관측된 CPU 점유 프로세스의 기여 후보")],
            Model(final_review=[review(chk("r1"),chk("r2"))]),writer=Model(report=[first,second]),
            tasks=[task("server-a","cpu","process")])
        reports=[p for name,p in self.writer.inputs if name=="report"]
        self.assertEqual(len(reports),2)
        self.assertIn('"requirement_id": "r2"',reports[1]["validation_feedback"])
        self.assertNotIn('"requirement_id": "r1"',reports[1]["validation_feedback"])
        self.assertEqual({rid for h in result["hypotheses"] for rid in h["requirement_ids"]},{"r1","r2"})
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual(result["final_validation"]["repair_rounds"],0)

    def test_candidate_choices_exclude_absent_unverified_and_other_target_symptoms(self):
        from agent.investigation import eligible_cause_choices
        evidence,requirements,assessment=self.observed()
        self.assertTrue(eligible_cause_choices(evidence,requirements,assessment))
        for invalid in ([],[dict(assessment[0],symptom_observed=False)],
                        [dict(assessment[0],quotes=["unobserved line"])],
                        [dict(assessment[0],evidence_ids=["missing"])]):
            self.assertEqual(eligible_cause_choices(evidence,requirements,invalid),[])
        changed=[dict(e,server_id="other-server") if e["command_id"]=="ps" else e for e in evidence]
        self.assertEqual(eligible_cause_choices(changed,requirements,assessment),[])

    async def test_redundant_process_candidates_retry_to_one_primary_factor(self):
        assessment=dict(text="현재 시스템 CPU 사용률은 19%입니다",requirement_ids=["r1"],evidence_ids=["e1"],quote_ids=["e1:l3"],symptom_observed=True)
        first=dict(diagnoses=[assessment],hypotheses=[],causes=[dict(requirement_id="r1",factor_id=f"e2:process:{i}:cpu") for i in (0,1)])
        second=dict(diagnoses=[assessment],hypotheses=[],causes=[dict(requirement_id="r1",factor_id="e2:process:0:cpu")])
        result=await self.run_flow([req("cpu",kind="diagnosis")],Model(final_review=[review(chk("r1"))]),
            writer=Model(report=[first,second]),tasks=[task("server-a","cpu","process")])
        self.assertEqual(len([p for p,_ in self.writer.inputs if p=="report"]),2)
        self.assertEqual(len(result["hypotheses"]),1)
        self.assertEqual(result["final_validation"]["status"],"passed")


    def test_later_disk_and_log_rows_bind_to_the_selected_factor(self):
        from agent.investigation import cause_catalog,bind_cause
        from agent.investigation_models import CauseSelection
        from monitoring.evidence import extract_facts
        evidence=[]
        for eid,cmd,stdout in [
            ("d","df","Filesystem Size Used Avail Use% Mounted on\n/dev/root 100G 20G 80G 20% /\n/dev/log 20G 19G 1G 95% /var/log\n"),
            ("u","du","1048576 /var/log/\n"),
            ("l","tail","2026-10-22T01:00:00Z INFO ready\n2026-10-22T01:01:00Z ERROR failed reason=connection_refused\n")]:
            e=dict(evidence_id=eid,server_id="server-x",command_id=cmd,os="linux",exit_code=0,error=None,stdout=stdout)
            e["facts"]=extract_facts(e);evidence.append(e)
        for info,kind,eid,quote in [("disk","log_storage","d","/dev/log 20G 19G 1G 95% /var/log"),
                                    ("log.content","log_reason","l","2026-10-22T01:01:00Z ERROR failed reason=connection_refused")]:
            with self.subTest(info=info):
                requirements=[dict(requirement_id="r1",kind="diagnosis",description="원인 후보",server_ids=["server-x"],information_ids=[info])]
                factor=next(c for c in cause_catalog(evidence,requirements) if c["kind"]==kind)
                assessment=[dict(requirement_ids=["r1"],evidence_ids=[eid],quotes=[quote],symptom_observed=True)]
                candidate=bind_cause(CauseSelection(requirement_id="r1",factor_id=factor["factor_id"]),evidence,requirements,assessment)
                self.assertIn(quote,candidate.quotes)
                wrong=[dict(assessment[0],quotes=[evidence[0]["stdout"].splitlines()[1]] if info=="disk" else [evidence[2]["stdout"].splitlines()[0]])]
                with self.assertRaises(ValueError):bind_cause(CauseSelection(requirement_id="r1",factor_id=factor["factor_id"]),evidence,requirements,wrong)

    def test_cpu_load_only_quote_does_not_authorize_utilization_cause(self):
        from agent.investigation import cause_catalog,bind_cause
        from agent.investigation_models import CauseSelection
        evidence,requirements,assessment=self.observed("cpu")
        factor=next(c for c in cause_catalog(evidence,requirements) if c["kind"]=="process_cpu")
        assessment[0]["quotes"]=["load average: 3.0, 2.0, 1.0"]
        with self.assertRaises(ValueError):bind_cause(CauseSelection(requirement_id="r1",factor_id=factor["factor_id"]),evidence,requirements,assessment)


class DiskSupportHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def flow(self, *, stdout=None, failed=False, repair=False, mixed=False, absence_text=None, timeout_repair=False, force_disk_absent=False, logs_absent=False, support_text=None, omit_first_cause=False, omit_all_causes=False, explanation=False):
        class Expert(Model):
            def __init__(self):
                super().__init__();self.expert_inputs=[];self.reviews=0
            def diagnostic_json(self,payload,schema):
                self.expert_inputs.append(copy.deepcopy(payload))
                if timeout_repair and payload["role"]=="metrics" and sum(p["role"]=="metrics" for p in self.expert_inputs)==2:
                    import time
                    time.sleep(.35)
                facts=[(e,f) for e in payload["observations"] for f in e["facts"]]
                if payload["role"]=="logs":
                    assert "disk" not in payload["allowed_information_ids"]
                    assert "disk" in payload["read_only_information_ids"]
                    assert any(f["information_id"]=="disk" for e,f in facts)
                    if not any(f["information_id"]=="log.usage" for e,f in facts):
                        return json.dumps(dict(diagnoses=[],tasks=[task("server-d","log.usage")])),{"usd":0}
                diagnoses=[]
                for r in payload["requirements"]:
                    iid="log.usage" if payload["role"]=="logs" else r["information_ids"][0]
                    e,f=next((e,f) for e,f in facts if f["information_id"]==iid)
                    line=f["quote"].splitlines()[0]
                    quote=next(q for q in payload["quote_catalog"] if q["evidence_id"]==e["evidence_id"] and q["text"]==line)
                    observed=iid=="log.usage" or (iid=="disk" and f["fields"]["entries"][0]["usage_percent"]>=90)
                    if force_disk_absent and iid=="disk":observed=False
                    if payload["role"]=="logs" and logs_absent:observed=False
                    text=absence_text if iid=="disk" and not observed and absence_text else f["summary"]
                    if payload["role"]=="logs" and support_text:text=support_text
                    source_ids=[e["evidence_id"]];quote_ids=[quote["quote_id"]]
                    if payload["role"]=="logs" and support_text:
                        disk=next(e for e,f in facts if f["information_id"]=="disk")
                        source_ids.append(disk["evidence_id"]);quote_ids.append(disk["evidence_id"]+":l2")
                    diagnoses.append(dict(text=text,requirement_ids=[r["requirement_id"]],
                        evidence_ids=source_ids,quote_ids=quote_ids,symptom_observed=observed))
                return json.dumps(dict(diagnoses=diagnoses,tasks=[])),{"usd":0}
            def monitoring_json(self,purpose,payload,schema):
                if purpose=="report":
                    self.inputs.append((purpose,copy.deepcopy(payload)))
                    if omit_all_causes or (omit_first_cause and len([p for kind,p in self.inputs if kind=="report"])==1):
                        answers=[]
                        if explanation:
                            answers=[dict(text="현재 파일시스템 사용률은 95%입니다",requirement_ids=["r2"],evidence_ids=["e1"],quote_ids=["e1:l2"])]
                        return json.dumps(dict(hypotheses=[],causes=[],answers=answers)),{"usd":0}
                    if omit_first_cause:
                        assert "원인 후보" in payload.get("validation_feedback", "")
                    return json.dumps(dict(hypotheses=[],causes=[dict(requirement_id=c["requirement_ids"][0],factor_id=c["factor_id"])
                        for c in payload["cause_catalog"] if c["kind"]=="log_storage"])),{"usd":0}
                if purpose=="final_review":
                    self.reviews+=1
                    first=(repair or timeout_repair) and self.reviews==1
                    return json.dumps(review(*[chk(r["requirement_id"],"unmet" if first else "satisfied") for r in payload["requirements"]],
                        tasks=[task("server-d","process" if timeout_repair else "log.usage")] if first else [])),{"usd":0}
                return super().monitoring_json(purpose,payload,schema)
        self.model=Expert();self.mcp=FixtureMCP()
        if stdout is not None:self.mcp.overrides[("server-d","df")]={"stdout":stdout}
        if failed:self.mcp.overrides[("server-d","du")]={"stdout":"","stderr":"saved lookup failure","exit_code":1}
        ids=["cpu","disk"] if mixed else ["disk"]
        state=dict(query="현재 디스크 사용률과 공간 부족의 원인 후보를 확인해줘",scenario_id="server-d",single_server_mode=True,
            events=[],history=[],request_decision=dict(requested_information_ids=ids,needs_diagnosis=True,needs_explanation=False,
                requested_results=[dict(description=i+" 증상이 있을 경우 원인 후보 확인",kind="diagnosis",information_ids=[i]) for i in ids]))
        if explanation:
            state["request_decision"]["requested_results"].append(dict(description="확인한 사실 설명",kind="explanation",information_ids=["disk"]))
        deps=SimpleNamespace(mcp=self.mcp,quality_llm=self.model,fast_llm=Model(),monitoring_llm=Model())
        if timeout_repair:
            from agent.deps import run_graph
            class Graph:
                async def astream(self,state,stream_mode):
                    yield "updates",{"investigate":await investigate(state,deps)}
            return await run_graph(Graph(),state,timeout_seconds=.15)
        return await investigate(state,deps)

    async def test_missing_candidate_retries_before_final_review_and_preserves_explanation(self):
        result=await self.flow(omit_first_cause=True,explanation=True)
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual(result["final_validation"]["repair_rounds"],0)
        reports=[p for kind,p in self.model.inputs if kind=="report"]
        self.assertEqual(len(reports),2)
        self.assertIn("r1",reports[1]["validation_feedback"])
        self.assertIn("e2:storage",reports[1]["validation_feedback"])
        self.assertTrue(result["hypotheses"])
        self.assertTrue(any(a["requirement_ids"]==["r2"] for a in result["answers"]))
        self.assertEqual(self.mcp.calls,[("server-d","df"),("server-d","du")])

    async def test_persistent_candidate_omission_keeps_facts_and_stays_partial(self):
        result=await self.flow(omit_all_causes=True,explanation=True)
        self.assertEqual(result["final_validation"]["status"],"partial")
        self.assertEqual(result["hypotheses"],[])
        self.assertTrue(any(a["requirement_ids"]==["r2"] for a in result["answers"]))
        self.assertEqual(len(result["evidence"]),2)
        self.assertEqual(len([p for kind,p in self.model.inputs if kind=="report"]),6)
        self.assertEqual(result["final_validation"]["repair_rounds"],2)

    async def test_disk_pressure_delegates_storage_without_creating_user_requirement(self):
        result=await self.flow()
        self.assertEqual(self.mcp.calls,[("server-d","df"),("server-d","du")])
        self.assertEqual([p["role"] for p in self.model.expert_inputs],["metrics","logs","logs"])
        self.assertEqual(len(result["requirements"]),1)
        self.assertEqual(result["requirements"][0]["information_ids"],["disk"])
        self.assertEqual(result["hypotheses"][0]["requirement_ids"],["r1"])
        self.assertEqual(set(result["hypotheses"][0]["evidence_ids"]),{"e1","e2"})
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual(result["workflow_budget"]["query_rounds"],2)

    async def test_normal_disk_has_no_support_queries(self):
        result=await self.flow(stdout="Filesystem Size Used Avail Use% Mounted on\n/dev/root 100G 30G 70G 30% /\n")
        self.assertEqual(self.mcp.calls,[("server-d","df")])
        self.assertEqual([p["role"] for p in self.model.expert_inputs],["metrics"])
        self.assertEqual(result["hypotheses"],[])
        self.assertEqual(result["final_validation"]["status"],"passed")

    async def test_full_root_with_healthy_separate_log_mount_does_not_query_logs(self):
        result=await self.flow(stdout="Filesystem Size Used Avail Use% Mounted on\n/dev/root 100G 95G 5G 95% /\n/dev/log 100G 10G 90G 10% /var/log\n")
        self.assertEqual(self.mcp.calls,[("server-d","df")])
        self.assertEqual(result["hypotheses"],[])
        self.assertFalse(result["final_validation"]["request_fully_satisfied"])

    async def test_failed_support_retains_disk_and_does_not_pass_or_repeat(self):
        result=await self.flow(failed=True)
        self.assertEqual(self.mcp.calls,[("server-d","df"),("server-d","du")])
        self.assertIn("95",result["final"])
        self.assertTrue(any(e["command_id"]=="df" and e["facts"] for e in result["evidence"]))
        self.assertEqual(result["hypotheses"],[])
        self.assertFalse(result["final_validation"]["request_fully_satisfied"])

    async def test_repair_reuses_support_and_preserves_query_budget(self):
        result=await self.flow(repair=True)
        self.assertEqual(self.mcp.calls,[("server-d","df"),("server-d","du")])
        self.assertEqual(result["workflow_budget"]["query_rounds"],2)
        self.assertEqual(result["final_validation"]["status"],"passed")

    async def test_mixed_normal_cpu_and_pressure_disk_keeps_separate_requirements(self):
        result=await self.flow(mixed=True)
        self.assertEqual(set(self.mcp.calls),{("server-d","top"),("server-d","df"),("server-d","du")})
        self.assertEqual(len(self.mcp.calls),3)
        self.assertEqual(len(result["requirements"]),2)
        self.assertEqual([h["requirement_ids"] for h in result["hypotheses"]],[["r2"]])
        self.assertEqual(result["final_validation"]["status"],"passed")


    async def test_support_cannot_exceed_shared_collection_limits(self):
        from unittest.mock import patch
        for setting in ("MAX_ROUNDS","MAX_COMMAND_CALLS"):
            with self.subTest(setting=setting),patch("agent.investigation."+setting,1):
                result=await self.flow()
                self.assertEqual(self.mcp.calls,[("server-d","df")])
                self.assertEqual(result["workflow_budget"]["query_rounds"],1)
                self.assertFalse(result["final_validation"]["request_fully_satisfied"])


    async def test_repair_timeout_keeps_verified_support_role_answer(self):
        result=await self.flow(timeout_repair=True)
        self.assertEqual(self.mcp.calls,[("server-d","df"),("server-d","du"),("server-d","ps")])
        self.assertEqual(result["final_validation"]["status"],"unverified")
        self.assertTrue(any("/var/log/" in a["text"] for a in result["answers"]))

    async def test_negative_disk_pressure_statement_is_grounded_not_rejected(self):
        result=await self.flow(stdout="Filesystem Size Used Avail Use% Mounted on\n/dev/root 100G 30G 70G 30% /\n",
            absence_text="디스크 공간이 부족하지 않습니다. 사용률은 30%이며 여유 공간은 70G입니다.")
        self.assertEqual(self.mcp.calls,[("server-d","df")])
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual(result["hypotheses"],[])

    async def test_absence_flag_cannot_hide_observed_disk_pressure(self):
        result=await self.flow(force_disk_absent=True)
        self.assertFalse(result["final_validation"]["request_fully_satisfied"])


    async def test_log_support_absence_is_not_reinterpreted_as_disk_absence(self):
        result=await self.flow(logs_absent=True)
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual(sum(p=="report" for p,_ in self.model.inputs),1)
        self.assertTrue(any(a["evidence_ids"]==["e2"] for a in result["answers"]))
        self.assertTrue(result["hypotheses"])


    async def test_support_text_cannot_turn_disk_symptom_into_a_cause(self):
        wrong="디스크 사용량이 95%로 높으며 이는 공간 부족의 원인 후보입니다."
        result=await self.flow(support_text=wrong)
        self.assertNotIn(wrong,result["final"])
        self.assertFalse(any(a["text"]==wrong for a in result["answers"]))
        self.assertTrue(any("83886080 KiB" in a["text"] for a in result["answers"]))
        self.assertEqual(result["final_validation"]["status"],"passed")


class MetricValueBindingTests(unittest.TestCase):
    """A number in the source is not evidence for every metric in that source."""
    def sources(self):
        from monitoring.evidence import extract_facts
        raw='top - 12:00:00 load average: 0.40, 0.30, 0.20\n%Cpu(s): 16.0 us, 2.0 sy, 82.0 id\nMiB Mem : 8192 total, 1024 free, 4096 used\n2048 avail Mem\n'
        e=dict(evidence_id='e1',server_id='server-a',os='linux',command_id='top',exit_code=0,stdout=raw)
        p=dict(evidence_id='e2',server_id='server-a',os='linux',command_id='ps',exit_code=0,
               stdout='PID PPID COMM %CPU %MEM RSS\n101 1 alpha 40.0 5.0 2048\n102 1 beta 18.0 8.0 4096\n')
        for item in (e,p):item['facts']=extract_facts(item)
        return [e,p]

    def verify(self,text,*,sources=None,calculated=()):
        from agent.investigation import verify_report
        from agent.investigation_models import InvestigationReport
        sources=sources or self.sources()
        report=InvestigationReport(hypotheses=[dict(text=text,evidence_ids=[e['evidence_id'] for e in sources],
            quotes=[line for e in sources for line in e['stdout'].splitlines() if line])])
        verify_report(report,sources,calculated)

    def test_cross_metric_numbers_and_cpu_subfields_are_not_interchangeable(self):
        for text in ['시스템 CPU 사용률은 40%입니다.','CPU 사용률은 16%입니다.',
                     'CPU 유휴율은 18%입니다.','CPU 사용자 비율은 2%입니다.',
                     'CPU 시스템 비율은 16%입니다.']:
            with self.subTest(text=text),self.assertRaises(ValueError):self.verify(text,calculated=[40])

    def test_process_identity_and_metric_are_bound_together(self):
        for text in ['alpha 프로세스 CPU 사용률은 18%입니다.',
                     'alpha 메모리 점유율은 8%입니다.', 'alpha가 메모리 8%를 점유합니다.',
                     'PID 101의 RSS는 4096 KiB입니다.']:
            with self.subTest(text=text),self.assertRaises(ValueError):self.verify(text)

    def test_memory_subfields_are_not_interchangeable(self):
        for text in ['메모리 전체 용량은 4096 MiB입니다.',
                     '사용 메모리는 8192 MiB입니다.',
                     '메모리 available은 1024 MiB입니다.', '메모리 total 값은 4096 MiB입니다.']:
            with self.subTest(text=text),self.assertRaises(ValueError):self.verify(text)

    def test_correct_bindings_and_raw_output_override_cached_fields(self):
        for text in ['시스템 CPU 사용률은 18%입니다.', 'CPU 유휴율은 82%입니다.',
                     'alpha 프로세스 CPU 사용률은 40%입니다.', 'alpha 메모리 점유율은 5%입니다.', 'alpha가 메모리 5%를 점유합니다.',
                     'PID 101의 RSS는 2048 KiB입니다.', '메모리 전체 용량은 8192 MiB입니다.',
                     '사용 메모리는 4096 MiB입니다.', '메모리 available은 2048 MiB입니다.', '메모리 total 값은 8192 MiB입니다.',
                     'CPU 사용률은 18%이며 user 16%, system 2%, idle 82%입니다.',
                     'alpha 프로세스 CPU 사용률은 40%이며 CPU 전체 사용률은 18%입니다.']:
            with self.subTest(text=text):self.verify(text)
        sources=self.sources();sources[0]['facts'][0]['fields']['usage_percent']=40
        with self.assertRaises(ValueError):self.verify('시스템 CPU 사용률은 40%입니다.',sources=sources)
