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
    async def run_flow(self,requirements,fast,*,tasks=None,writer=None,reviews=None):
        self.mcp=FixtureMCP();self.writer=writer or Model()
        self.planner=Model(plan=[dict(tasks=tasks or [task("server-a","cpu")],requirements=requirements)],
                           review=reviews or [dict(tasks=[])])
        return await investigate(dict(query="서버 A 요청 항목을 빠짐없이 설명해줘",history=[],events=[],cost_usd=0),
            SimpleNamespace(mcp=self.mcp,fast_llm=fast,monitoring_llm=self.planner,quality_llm=self.writer))

    async def test_missing_information_loops_back_and_gap_reaches_writer(self):
        fast=Model(final_review=[review(chk("r1"),chk("r2","unmet","프로세스 근거가 없음"),tasks=[task("server-a","process")]),
                                 review(chk("r1"),chk("r2"))])
        r=await self.run_flow([req("cpu"),req("process")],fast)
        self.assertEqual(self.mcp.calls,[("server-a","top"),("server-a","ps")])
        self.assertEqual(r["final_validation"]["status"],"passed")
        self.assertEqual([p for p,_ in fast.inputs],["final_review","final_review"])
        self.assertFalse(any(p=="final_review" for p,_ in self.writer.inputs))
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
        writer=Model(report=[dict(hypotheses=[]),dict(hypotheses=[],calculations=[calc])])
        r=await self.run_flow([req("cpu",kind="calculation")],
            Model(final_review=[review(chk("r1")),review(chk("r1"))]),writer=writer)
        self.assertEqual(r["final_reviews"][0]["checks"][0]["status"],"unmet")
        self.assertEqual(r["final_validation"]["status"],"passed")
        self.assertEqual(float(r["calculation_results"][0]["value"]),16)
        self.assertEqual(len(self.mcp.calls),1)

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
                self.assertIn("최종 답변 검증을 완료하지 못",r["final"])

    async def test_persistent_gap_stops_after_two_answer_repairs(self):
        fast=Model(final_review=[review(chk("r1","unmet"),tasks=[task("server-a","cpu")])]*3)
        r=await self.run_flow([req("cpu",kind="explanation")],fast)
        self.assertEqual(len(self.mcp.calls),1);self.assertEqual(len(r["final_reviews"]),3)
        self.assertEqual(r["final_validation"]["repair_rounds"],2)
        self.assertFalse(r["final_validation"]["request_fully_satisfied"])
        self.assertIn("최종 답변 보완은 최대 2회",r["final"])

    async def test_initial_and_final_repairs_share_query_round_limit(self):
        fast=Model(final_review=[review(chk("r1","unmet"),tasks=[task("server-a","disk")])]*3)
        r=await self.run_flow([req("cpu",kind="explanation")],fast,
            reviews=[dict(tasks=[task("server-a","process")]),dict(tasks=[task("server-a","log.content")])])
        self.assertEqual({c for _,c in self.mcp.calls},{"top","ps","tail"})
        self.assertEqual(r["replan_rounds"],2)
        self.assertTrue(any("한도" in g["reason"] for g in r["requirement_issues"]))


class CalculationTests(unittest.TestCase):
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
