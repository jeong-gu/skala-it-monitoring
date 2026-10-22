import json
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from agent.fake_llm import FastLLM, QualityLLM
from agent.graph import build_graph
from agent.response_policy import present_response
from test_information_investigation import FixtureMCP
from test_remediation import match_for, intent_payload


class FailedResponseCostTests(unittest.IsolatedAsyncioTestCase):
    class Paid:
        name = "paid-test"
        def __init__(self, outputs): self.outputs = iter(outputs); self.calls = 0
        def response_json(self, *args):
            self.calls += 1
            item = next(self.outputs)
            if isinstance(item, Exception): raise item
            return item, {"usd": .125}

    async def test_invalid_intent_retains_both_paid_attempts(self):
        from agent.response_policy import request_route
        m = self.Paid(['{}', '{}'])
        result = await request_route(dict(query="CPU 확인", scenario_id="server-b", cost_usd=1), deps(RejectMCP(), m), set())
        self.assertEqual(m.calls, 2)
        self.assertEqual(result['route'], 'chat')
        self.assertEqual(result['cost_usd'], 1.25)

    async def test_transport_error_after_paid_invalid_result_retains_known_cost(self):
        from agent.response_policy import request_route
        m = self.Paid(['{}', RuntimeError('provider unavailable')])
        result = await request_route(dict(query="CPU 확인", scenario_id="server-b", cost_usd=1), deps(RejectMCP(), m), set())
        self.assertEqual(result['cost_usd'], 1.125)

    async def test_direct_and_layout_fallback_retain_cost_and_attempts(self):
        for route in ('chat', 'commands'):
            with self.subTest(route=route):
                m = self.Paid(['{}', '{}'])
                result = await present_response(dict(query="설명", route=route, final="기존 검증 본문", cost_usd=1), deps(RejectMCP(), m))
                self.assertEqual(result['cost_usd'], 1.25)
                self.assertEqual(result['presentation']['model_calls'], 2)
                self.assertEqual(result['final'], '기존 검증 본문')

    async def test_successful_retry_counts_each_call_once(self):
        answer = json.dumps(dict(style='brief', brief='간단한 설명입니다.', sections=[], summary=[]))
        result = await present_response(dict(query="설명", route="chat", cost_usd=1), deps(RejectMCP(), self.Paid(['{}', answer])))
        self.assertEqual(result['cost_usd'], 1.25)
        self.assertEqual(result['presentation']['model_calls'], 2)


class RejectMCP:
    def __init__(self): self.calls=[]
    async def call_tool(self, name, args):
        self.calls.append(name)
        raise AssertionError("This request must not use MCP: "+name)


def deps(mcp, quality=None):
    return SimpleNamespace(mcp=mcp, fast_llm=FastLLM("unused-fast",0),
                           quality_llm=quality or QualityLLM("standard",0))


class ResponseFlowTests(unittest.IsolatedAsyncioTestCase):
    async def run_graph(self, query, mcp, quality=None, history=None):
        return await build_graph(deps(mcp,quality)).ainvoke({"query":query,"history":history or [],"events":[],"cost_usd":0})

    async def test_clarify_and_invalid_intents_never_query_tools(self):
        class Uncertain(QualityLLM):
            def response_json(self, purpose, payload, schema):
                if purpose == "intent":
                    self.asserted_schema = schema
                    return json.dumps(intent_payload(route="commands", style="brief", reason="요청 시점 불명확",
                        observation_policy="clarify")), {"usd": 0}
                raise AssertionError("Clarification must not generate observations")
        model=Uncertain("test",0);mcp=RejectMCP()
        r=await build_graph(deps(mcp,model)).ainvoke(dict(query="그 값을 확인할까 말까?",scenario_id="server-b",history=[],events=[]))
        self.assertIn("clarify",model.asserted_schema["properties"]["observation_policy"]["enum"])
        self.assertEqual(r["request_decision"]["observation_policy"],"clarify")
        self.assertEqual(r["route"],"chat");self.assertEqual(mcp.calls,[])
        self.assertIn("확인",r["final"])

    async def test_missing_required_intent_fields_fail_closed_after_one_retry(self):
        from agent.response_policy import Intent
        complete = intent_payload(route="commands", style="brief", reason="incomplete",
            requested_information_ids=["cpu"])
        for field in complete:
            if field in {"needs_diagnosis", "needs_calculation", "needs_explanation"}:continue
            with self.subTest(field=field):
                class Missing(QualityLLM):
                    calls = 0
                    def response_json(self, purpose, payload, schema):
                        self.calls += 1
                        return json.dumps({k:v for k,v in complete.items() if k != field}), {"usd":0}
                model=Missing("test",0);mcp=RejectMCP()
                r=await self.run_graph("서버 A CPU는 조회하지 마",mcp,model)
                self.assertEqual(mcp.calls,[])
                self.assertEqual(r["route"],"chat")
                self.assertEqual(model.calls,2)
                self.assertIn("요청 의도",r["final"])

    async def test_model_policy_is_not_overridden_by_query_regex(self):
        class Chosen(QualityLLM):
            def response_json(self,purpose,payload,schema):
                if purpose=="intent":return json.dumps(intent_payload(route="commands",style="brief",reason="controlled decision",
                    requested_information_ids=["cpu"],observation_policy="refresh")),{"usd":0}
                return super().response_json(purpose,payload,schema)
        # Deliberately wrong semantic decision: this test verifies the NEW contract
        # that code does not reclassify language. Real-model semantic tests are separate.
        mcp=FixtureMCP()
        r=await self.run_graph("서버 A CPU는 조회하지 말고 기존 내용으로 설명해줘",mcp,Chosen("test",0))
        self.assertEqual(r["request_decision"]["observation_policy"],"refresh")
        self.assertEqual(mcp.calls,[("server-a","top")])

    async def test_explicit_metric_with_basic_flag_retries_then_stops_without_fetch(self):
        class OverBroad(QualityLLM):
            calls=0
            def response_json(self,purpose,payload,schema):
                self.calls+=1
                return json.dumps(intent_payload(route="commands",style="brief",reason="wrong basic flag",
                    basic_status=True,requested_information_ids=["cpu","memory.summary","disk"])),{"usd":0}
        mcp=RejectMCP();model=OverBroad("test",0)
        r=await self.run_graph("서버 B CPU 상태를 새로 측정해줘",mcp,model)
        self.assertEqual(r["route"],"chat");self.assertEqual(mcp.calls,[])
        self.assertEqual(model.calls,2)

    async def test_contradictory_continuation_retries_then_clarifies_without_tools(self):
        from agent.response_policy import request_route
        for route,policy in [("commands","refresh"),("chat","reuse"),("incident","reuse"),("remediation","reuse")]:
            with self.subTest(route=route,policy=policy):
                class Contradictory(QualityLLM):
                    calls=0
                    def response_json(self,purpose,payload,schema):
                        self.calls+=1
                        return json.dumps(intent_payload(route=route,style="brief",reason="inconsistent",
                            requested_information_ids=["cpu"],observation_policy=policy,continue_previous_task=True)),{"usd":0}
                model=Contradictory("test",0);mcp=RejectMCP()
                r=await request_route(dict(query="이 결과를 쉽게 풀어줘",scenario_id="server-b",history=[],
                    investigation_context=dict(query="CPU 상태 확인",evidence=[dict(server_id="server-b")],tasks=[])),deps(mcp,model),set())
                self.assertEqual(r["route"],"chat");self.assertEqual(model.calls,2)
                self.assertEqual(mcp.calls,[]);self.assertIn("요청 의도",r["direct_response_override"])

    async def test_prior_context_is_supplied_for_an_unrecognized_followup_phrase(self):
        from agent.response_policy import request_route
        class Continue(QualityLLM):
            def response_json(self,purpose,payload,schema):
                self.payload=payload
                return json.dumps(intent_payload(route="commands",style="brief",reason="앞선 결과 재설명",
                    requested_information_ids=["cpu"],observation_policy="reuse",continue_previous_task=True)),{"usd":0}
        mcp=RejectMCP();model=Continue("test",0)
        context=dict(query="CPU 상태 확인",subject_query="CPU 상태 확인",tasks=[dict(information_ids=["cpu"])],
                     evidence=[dict(server_id="server-b",facts=[dict(information_id="cpu")])],manual_requirements=[])
        r=await request_route(dict(query="이 결과를 쉽게 풀어줘",history=[dict(role="user",content="CPU 상태 확인")],
            scenario_id="server-b",investigation_context=context),deps(mcp,model),set())
        self.assertEqual(model.payload["previous_request"],"CPU 상태 확인")
        self.assertTrue(model.payload["previous_is_latest_topic"])
        self.assertEqual(r["request_subject"],"CPU 상태 확인")
        self.assertEqual(r["request_decision"]["observation_policy"],"reuse")
        self.assertEqual(mcp.calls,[])

    async def test_decided_reuse_policy_reaches_writer_and_final_validator(self):
        class Capture(QualityLLM):
            def __init__(self):super().__init__("test",0);self.payloads=[]
            def monitoring_json(self,purpose,payload,schema):
                self.payloads.append((purpose,payload))
                return super().monitoring_json(purpose,payload,schema)
        model=Capture();mcp=FixtureMCP()
        first=await self.run_graph("서버 A CPU 상태 확인",mcp,model)
        mcp.calls.clear();model.payloads.clear()
        result=await build_graph(deps(mcp,model)).ainvoke(dict(query="서버 A CPU는 조회하지 말고 기존 결과로 설명해줘",
            history=[dict(role="user",content="서버 A CPU 상태 확인")],events=[],cost_usd=0,
            investigation_context={**first,"query":"서버 A CPU 상태 확인"}))
        selected=[v for purpose,v in model.payloads if purpose in {"report","final_review"}]
        self.assertTrue(any(p=="report" for p,_ in model.payloads))
        self.assertTrue(any(p=="final_review" for p,_ in model.payloads))
        self.assertTrue(all(p["request_decision"]["observation_policy"]=="reuse" for p in selected))
        self.assertEqual(mcp.calls,[]);self.assertEqual(result["final_validation"]["status"],"passed")

    async def test_concept_with_monitoring_words_never_queries_tools(self):
        for query in ["CPU 사용률이 뭐야?", "top과 ps의 차이가 뭐야?", "서버 B라는 이름에서 서버의 뜻이 뭐야?"]:
            mcp=RejectMCP();r=await self.run_graph(query,mcp)
            self.assertEqual(r["route"],"chat")
            self.assertEqual(mcp.calls,[])
            self.assertEqual(r["presentation"]["status"],"formatted")
            self.assertLessEqual(len(r["final"].splitlines()),2)

    async def test_previous_server_does_not_turn_a_new_concept_into_an_observation(self):
        mcp=RejectMCP()
        r=await self.run_graph("메모리와 디스크의 차이가 뭐야?",mcp,
                               history=[{"role":"user","content":"서버 B CPU 상태 확인"}])
        self.assertEqual(r["route"],"chat");self.assertEqual(mcp.calls,[])

    async def test_simple_actual_status_still_observes_and_has_at_most_two_lines(self):
        mcp=FixtureMCP();r=await self.run_graph("서버 A CPU 상태 확인",mcp)
        self.assertEqual(mcp.calls,[("server-a","top")])
        self.assertIn("19.0%",r["final"])
        self.assertIn("모의 시나리오",r["final"])
        self.assertNotIn("메모리 전체",r["final"])
        self.assertLessEqual(len(r["final"].splitlines()),2)

    async def test_manual_request_without_history_searches_and_formats_with_last_summary(self):
        class Manuals:
            def __init__(self): self.calls=[]
            async def call_tool(self,name,args):
                self.calls.append((name,args))
                if name!="search_remediation_manuals": raise AssertionError(name)
                return {"matches":[match_for("kernel-cpu-control")],"timings_ms":{}}
        mcp=Manuals();r=await self.run_graph("Linux CPU가 높을 때 대처방안을 알려줘",mcp)
        self.assertEqual([n for n,_ in mcp.calls],["search_remediation_manuals"])
        self.assertEqual(mcp.calls[0][1]["evidence_summary"],"")
        self.assertEqual(r["remediation"]["status"],"provided")
        self.assertEqual(r["presentation"]["style"],"detailed")
        self.assertIn("**조치 전 확인**",r["final"])
        self.assertIn("**출처**",r["final"])
        summary=r["final"].split("**요약**\n")[-1].splitlines()
        self.assertTrue(3<=len(summary)<=4)

    async def test_raw_json_bypasses_intent_and_final_formatter(self):
        class NeverFormat(QualityLLM):
            def response_json(self,*args): raise AssertionError("raw request must not generate")
        r=await self.run_graph("서버 A top JSON만",FixtureMCP(),NeverFormat("standard",0))
        self.assertEqual(json.loads(r["final"])["command_id"],"top")
        self.assertEqual(r["presentation"],{"status":"raw","model_calls":0})

    async def test_complex_general_explanation_is_sectioned_without_tools(self):
        class Explanation(QualityLLM):
            def response_json(self,purpose,payload,schema):
                if purpose=="direct":
                    return json.dumps({"style":"detailed","brief":"","sections":[{"title":"배경","body":"컴퓨터 구성 요소의 역할을 비교합니다."},{"title":"원리","body":"각 구성 요소가 맡는 작업을 구분합니다."}],"summary":["첫째 요점","둘째 요점","셋째 요점"]}),{"usd":0}
                return super().response_json(purpose,payload,schema)
        mcp=RejectMCP();r=await self.run_graph("컴퓨터 구성 요소를 자세하게 설명해줘",mcp,Explanation("standard",0))
        self.assertIn("**배경**",r["final"])
        self.assertEqual(len(r["final"].split("**요약**\n")[-1].splitlines()),3)
        self.assertEqual(mcp.calls,[])

    async def test_failed_intent_does_not_accidentally_query_a_server(self):
        class Broken(QualityLLM):
            def response_json(self,*args): raise RuntimeError("model offline")
        mcp=RejectMCP();r=await self.run_graph("메모리 상태를 확인하고 싶어",mcp,Broken("standard",0))
        self.assertEqual(mcp.calls,[])
        self.assertIn("요청 의도를 확인하지 못했습니다",r["final"])


class SourceProtectionTests(unittest.IsolatedAsyncioTestCase):
    def state(self):
        return {"query":"서버 상태와 원인 설명", "route":"commands","response_style":"detailed", "events":[],
                "final":"확인한 사실\n- CPU 37% [e1]\n원인 후보 — 확정되지 않음\n- 작업 점유 가능성 [e1]\n확인 한계·미확인 항목\n- 이전 배포 이력은 확인하지 못했습니다."}

    async def test_invented_prose_or_values_are_rejected_and_existing_text_is_kept(self):
        class Invented(QualityLLM):
            calls=0
            def response_json(self,*args):
                self.calls+=1
                return '{"brief_ids":[],"sections":[],"summary_ids":[],"text":"CPU 99%, 재시작 완료"}',{"usd":0}
        model=Invented("standard",0);state=self.state()
        r=await present_response(state,deps(RejectMCP(),model))
        self.assertEqual(model.calls,2)
        self.assertEqual(r["final"],state["final"])
        self.assertEqual(r["presentation"]["status"],"fallback")

    async def test_unconfirmed_items_cannot_be_dropped(self):
        class Misleading(QualityLLM):
            calls=0
            def response_json(self,purpose,payload,schema):
                self.calls+=1
                raw,usage=super().response_json(purpose,payload,schema)
                value=json.loads(raw)
                if self.calls==1:
                    value["sections"]=[s for s in value["sections"] if s["title"]!="미확인 사항"]
                return json.dumps(value),usage
        model=Misleading("standard",0)
        r=await present_response(self.state(),deps(RejectMCP(),model))
        self.assertEqual(model.calls,2)
        self.assertEqual(r["presentation"]["status"],"formatted")
        self.assertIn("이전 배포 이력은 확인하지 못했습니다",r["final"])
        self.assertIn("**원인**",r["final"])

    async def test_unconfirmed_information_cannot_be_formatted_as_a_cause(self):
        class Relabel(QualityLLM):
            def response_json(self,purpose,payload,schema):
                raw,usage=super().response_json(purpose,payload,schema)
                value=json.loads(raw)
                for section in value["sections"]:
                    if section["title"]=="미확인 사항":section["title"]="원인"
                return json.dumps(value),usage
        state=self.state();r=await present_response(state,deps(RejectMCP(),Relabel("standard",0)))
        self.assertEqual(r["presentation"]["status"],"fallback")
        self.assertEqual(r["final"],state["final"])


class OutputContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_and_sse_only_publish_the_completed_final_presentation(self):
        import httpx
        import server
        from agent.session_store import SqliteSessionStore
        with tempfile.TemporaryDirectory() as tmp:
            store=SqliteSessionStore(Path(tmp)/"sessions.sqlite");mcp=RejectMCP();d=deps(mcp)
            d.session_store=store;d.version="v2"
            saved={k:getattr(server.app.state,k,None) for k in ("deps","graph","llm_backend","instance_id")}
            server.app.state.deps=d;server.app.state.graph=build_graph(d)
            server.app.state.llm_backend="fake";server.app.state.instance_id=1
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url="http://test") as c:
                    body=(await c.post('/chat',json={"message":"CPU가 뭐야?","thread_id":"plain"})).json()
                    self.assertEqual(body["presentation"]["status"],"formatted")
                    response=await c.get('/chat/stream',params={"message":"CPU가 뭐야?","thread_id":"stream"})
                    events=[json.loads(s[6:]) for s in response.text.splitlines() if s.startswith('data: ') and s!='data: [DONE]']
                    final=next(e for e in events if e['type']=='FinalAnswer')
                    self.assertEqual(final['presentation']['style'],'brief')
                    self.assertEqual(final['reply'],body['reply'])
                    self.assertLess(next(i for i,e in enumerate(events) if e['type']=='FinalResponseFormatted'),next(i for i,e in enumerate(events) if e['type']=='FinalAnswer'))
                    self.assertFalse(any(e['type']=='AnswerToken' for e in events))
                    self.assertEqual(mcp.calls,[])
            finally:
                store.close()
                for k,v in saved.items():setattr(server.app.state,k,v)


if __name__=="__main__": unittest.main()


class LLMTraceTests(unittest.TestCase):
    def test_structured_call_has_one_span_and_provider_token_counts(self):
        from unittest.mock import MagicMock, patch
        from agent.ollama_llm import OllamaLLM
        model=OllamaLLM("qwen3:8b",0)
        response=MagicMock()
        response.json.return_value={"message":{"content":"{}"},"prompt_eval_count":7,"eval_count":3,"total_duration":1000000}
        manager=MagicMock(); span=manager.__enter__.return_value
        with patch("agent.ollama_llm.trace",return_value=manager) as traced, patch("agent.ollama_llm.httpx.post",return_value=response) as posted:
            text,usage=model.response_json("direct",{"query":"hello"},{"type":"object"})
        self.assertEqual(text,"{}");self.assertEqual(posted.call_count,1)
        self.assertEqual(traced.call_count,1);self.assertEqual(traced.call_args.args[0],"llm.direct")
        self.assertEqual(usage["input_tokens"],7);self.assertEqual(usage["output_tokens"],3)
        self.assertEqual(span.end.call_args.kwargs["outputs"]["usage_metadata"],{"input_tokens":7,"output_tokens":3,"total_tokens":10})

    def test_tracing_entry_failure_does_not_retry_or_change_model_input(self):
        from unittest.mock import MagicMock, patch
        from agent.ollama_llm import OllamaLLM
        response=MagicMock();response.json.return_value={"message":{"content":"{}"}}
        model=OllamaLLM("qwen3:8b",0)
        with patch("agent.ollama_llm.trace",side_effect=RuntimeError("tracing unavailable")),patch("agent.ollama_llm.httpx.post",return_value=response) as posted:
            text,usage=model.response_json("direct",{"query":"hello"},{"type":"object"})
        self.assertEqual(text,"{}");self.assertEqual(posted.call_count,1)
        self.assertIsNone(usage["input_tokens"])
        self.assertEqual(posted.call_args.kwargs["json"]["messages"][1]["content"],'{"query": "hello"}')

    def test_tracing_exit_failure_does_not_repeat_provider_call(self):
        from unittest.mock import MagicMock, patch
        from agent.ollama_llm import trace_json_call
        manager=MagicMock();manager.__exit__.side_effect=RuntimeError("flush")
        call=MagicMock(return_value=("{}",{"model":"ollama:test","input_tokens":2,"output_tokens":1,"tokens":3}))
        with patch("agent.ollama_llm.trace",return_value=manager):
            self.assertEqual(trace_json_call("test","intent",{},call)[0],"{}")
        self.assertEqual(call.call_count,1)

    def test_provider_failure_propagates_once_and_retry_span_is_separate(self):
        from unittest.mock import MagicMock, patch
        from agent.ollama_llm import trace_json_call
        call=MagicMock(side_effect=ValueError("provider failed"))
        with patch("agent.ollama_llm.trace",return_value=MagicMock()) as traced:
            for payload in [{},{"validation_feedback":"bad json"}]:
                with self.assertRaises(ValueError):trace_json_call("test","report",payload,call)
        self.assertEqual(call.call_count,2)
        self.assertEqual([c.kwargs["metadata"]["attempt"] for c in traced.call_args_list],[1,2])


class LookupReportGateTests(unittest.TestCase):
    def test_report_gate_is_semantic_conservative_and_preserves_interpretation(self):
        from agent.investigation import should_generate_report
        lookup=[{"kind":"lookup"}]
        base=dict(needs_explanation=False,needs_diagnosis=False,needs_calculation=False)
        self.assertFalse(should_generate_report(base,lookup,None))
        self.assertTrue(should_generate_report({},lookup,None))
        for flag in ("needs_explanation","needs_diagnosis","needs_calculation"):
            self.assertTrue(should_generate_report({**base,flag:True},lookup,None))
        for kind in ("explanation","calculation","comparison","diagnosis"):
            self.assertTrue(should_generate_report(base,[{"kind":kind}],None))
        self.assertTrue(should_generate_report(base,lookup,{"rewrite_instructions":"누락된 설명 보완"}))

    def test_requirement_coverage_uses_only_rendered_blocks_and_preserves_ids(self):
        from agent.response_policy import build_requirement_coverage
        requirements=[{"requirement_id":"r1"},{"requirement_id":"r2"}]
        blocks=[dict(block_id="b1",text="실제 CPU 조치",section="대처방안",requirement_ids=["r1"],evidence_ids=["e1"],step_ids=["a1"],chunk_ids=["c1"])]
        coverage=build_requirement_coverage(requirements,blocks)
        self.assertEqual(coverage["r1"]["block_ids"],["b1"])
        self.assertEqual(coverage["r1"]["chunk_ids"],["c1"])
        self.assertEqual(coverage["r2"]["block_ids"],[])

class LookupReportFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_plain_status_skips_report_but_keeps_format_and_final_review(self):
        class Count(QualityLLM):
            def __init__(self):super().__init__("test",0);self.stages=[]
            def monitoring_json(self,purpose,payload,schema):
                self.stages.append(purpose);return super().monitoring_json(purpose,payload,schema)
            def response_json(self,purpose,payload,schema):
                self.stages.append(purpose)
                if purpose=="intent":return json.dumps(intent_payload(route="commands",style="brief",reason="기본 상태",basic_status=True,requested_information_ids=["cpu","memory.summary","disk"],needs_explanation=False)),{"usd":0}
                return super().response_json(purpose,payload,schema)
        mcp=FixtureMCP();model=Count()
        result=await build_graph(deps(mcp,model)).ainvoke(dict(query="상태 확인해줘",scenario_id="server-b",history=[],events=[]))
        self.assertNotIn("report",model.stages)
        self.assertEqual(model.stages.count("format"),1);self.assertEqual(model.stages.count("final_review"),1)
        self.assertEqual(mcp.calls,[("server-b","top"),("server-b","df")])
        self.assertIn("다른 지표가 필요하면",result["final"])
        self.assertEqual(result["final_validation"]["status"],"passed")


class OpenAIConfigurationTests(unittest.TestCase):
    def test_openai_formatter_enforces_all_source_ids_in_provider_schema(self):
        import os, asyncio
        from unittest.mock import MagicMock, patch
        from agent.openai_llm import OpenAILLM
        from agent.response_policy import model_json, Layout, validate_layout
        blocks=[dict(block_id="b0",section="배경",text="모의 데이터"),
                dict(block_id="b1",section="근거",text="CPU 96%")]
        response=MagicMock();response.json.return_value={"choices":[{"message":{"content":'{"brief_ids":["b0","b1"]}'}}]}
        with patch.dict(os.environ,{"OPENAI_API_KEY":"test-key","LANGCHAIN_TRACING_V2":"false","LANGSMITH_TRACING":"false"}),patch("agent.openai_llm.httpx.post",return_value=response) as posted:
            model=OpenAILLM("gpt-4o-mini",0)
            layout,_,_=asyncio.run(model_json(model,"format",{"query":"상태 확인","style":"brief","blocks":blocks},Layout,
                lambda layout:validate_layout(layout,blocks,"brief")))
        self.assertEqual(layout.brief_ids,["b0","b1"])
        response_format=posted.call_args.kwargs["json"]["response_format"]
        self.assertEqual(response_format["type"],"json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        ids=response_format["json_schema"]["schema"]["properties"]["brief_ids"]
        self.assertEqual((ids["minItems"],ids["maxItems"]),(2,2))
        self.assertEqual(ids["items"]["enum"],["b0","b1"])

    def test_all_structured_openai_paths_include_the_requested_schema(self):
        import os
        from unittest.mock import MagicMock, patch
        from agent.openai_llm import OpenAILLM, OPENAI_URL
        schema={"type":"object","properties":{"sentinel":{"type":"string"}},"required":["sentinel"]}
        response=MagicMock();response.json.return_value={"choices":[{"message":{"content":"{}"}}],
            "usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}
        with patch.dict(os.environ,{"OPENAI_API_KEY":"test-key","LANGCHAIN_TRACING_V2":"false","LANGSMITH_TRACING":"false"}):
            model=OpenAILLM("gpt-4o",0)
            for method,args in [(model.response_json,("intent",{},schema)),
                                (model.monitoring_json,("plan",{},schema)),
                                (model.remediation_json,({},schema))]:
                with self.subTest(method=method.__name__),patch("agent.openai_llm.httpx.post",return_value=response) as posted:
                    text,usage=method(*args)
                    self.assertEqual(posted.call_count,1)
                    self.assertEqual(posted.call_args.args[0],OPENAI_URL)
                    payload=posted.call_args.kwargs["json"]
                    self.assertIn(json.dumps(schema,ensure_ascii=False),payload["messages"][0]["content"])
                    self.assertEqual(payload["messages"][1]["content"],"{}")
                    contract=payload["response_format"]
                    self.assertEqual(contract["type"],"json_schema")
                    self.assertTrue(contract["json_schema"]["strict"])
                    self.assertEqual(contract["json_schema"]["schema"]["properties"],schema["properties"])
                    self.assertEqual(contract["json_schema"]["schema"]["required"],["sentinel"])
                    self.assertFalse(contract["json_schema"]["schema"]["additionalProperties"])
                    self.assertEqual((usage["input_tokens"],usage["output_tokens"]),(7,3))

    def test_empty_openai_key_stops_before_http(self):
        import os
        from unittest.mock import patch
        from agent.openai_llm import OpenAILLM
        for key in ("","  "):
            with self.subTest(key=repr(key)),patch.dict(os.environ,{"OPENAI_API_KEY":key}),patch("agent.openai_llm.httpx.post") as posted:
                model=OpenAILLM("gpt-4o",0)
                with self.assertRaisesRegex(RuntimeError,"OPENAI_API_KEY"):
                    model._chat("JSON","{}",json_mode=True)
                posted.assert_not_called()

    def test_openai_main_selection_keeps_local_manual_embedder(self):
        import os
        from unittest.mock import patch
        import server
        from agent.openai_llm import OpenAIFastLLM, OpenAIQualityLLM
        from remediation.index import ManualIndex, OllamaEmbedder
        with patch.dict(os.environ,{"LLM_BACKEND":"openai","OPENAI_API_KEY":"",
                "OPENAI_FAST_MODEL":"gpt-4o-mini","OPENAI_QUALITY_MODEL":"gpt-4o",
                "MANUAL_EMBED_MODEL":"bge-m3","MANUAL_OLLAMA_URL":"http://127.0.0.1:11434"}),patch("httpx.post") as posted:
            backend,fast,quality=server._build_llms()
            self.assertEqual(backend,"openai")
            self.assertIsInstance(fast,OpenAIFastLLM);self.assertIsInstance(quality,OpenAIQualityLLM)
            self.assertEqual((fast.model,quality.model),("gpt-4o-mini","gpt-4o"))
            self.assertIsInstance(ManualIndex().embedder,OllamaEmbedder)
            self.assertEqual(ManualIndex().embedder.model,"bge-m3")
            posted.assert_not_called()


class SemanticIntentRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_diagnosis_synonym_preserves_model_intent_and_supporting_information(self):
        from agent.response_policy import request_route
        class Chosen(QualityLLM):
            def response_json(self, purpose, payload, schema):
                return json.dumps(intent_payload(route="commands", style="detailed", reason="병목 원인 조사",
                    requested_information_ids=["cpu", "process"], needs_diagnosis=True)), {"usd":0}
        result=await request_route(dict(query="CPU 병목의 주범을 짚어줘.", scenario_id="server-b",
            single_server_mode=True, history=[], events=[]), deps(RejectMCP(),Chosen("test",0)),set())
        self.assertTrue(result["request_decision"]["needs_diagnosis"])
        self.assertEqual(result["request_decision"]["requested_information_ids"],["cpu","process"])


class TypedResultFlagTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_predicate_quote_retry_preserves_lookup_and_calculation(self):
        from agent.response_policy import request_route
        query = "현재 시스템 CPU 사용률과 전체 대비 사용 메모리 비율을 각각 %로 알려줘."
        invented = "현재 시스템 CPU 사용률을 각각 %로 알려줘."
        class Chosen(QualityLLM):
            def __init__(self):
                super().__init__("test", 0)
                self.payloads = []
            def response_json(self, purpose, payload, schema):
                self.payloads.append(payload)
                if len(self.payloads) == 1:
                    self.first_quote_unconstrained = 'enum' not in schema['$defs']['RequestedResult']['anyOf'][0]['properties']['request_quote']
                if len(self.payloads) == 2:
                    self.retry_quotes = schema['$defs']['RequestedResult']['anyOf'][0]['properties']['request_quote']['enum']
                quote = invented if len(self.payloads) == 1 else "현재 시스템 CPU 사용률"
                value = intent_payload(route="commands", style="detailed", reason="CPU 조회와 메모리 비율 계산",
                    requested_information_ids=["cpu", "memory.summary"], needs_calculation=False,
                    needs_diagnosis=False, needs_explanation=False)
                value["requested_results"] = [
                    dict(request_quote=quote, description="CPU", kind="lookup", information_ids=["cpu"]),
                    dict(request_quote="전체 대비 사용 메모리 비율", description="메모리", kind="calculation", information_ids=["memory.summary"])]
                return json.dumps(value), {"usd": .125}
        model = Chosen()
        result = await request_route(dict(query=query, scenario_id="server-b", single_server_mode=True,
            history=[], events=[]), deps(RejectMCP(), model), set())
        self.assertEqual(len(model.payloads), 2)
        self.assertTrue(model.first_quote_unconstrained)
        feedback = model.payloads[1]["validation_feedback"]
        self.assertIn(invented, feedback)
        self.assertIn(query, feedback)
        self.assertIn("부분 구간", feedback)
        self.assertIn("현재 시스템 CPU 사용률", model.retry_quotes)
        self.assertNotIn(invented, model.retry_quotes)
        self.assertTrue(all(q in query for q in model.retry_quotes))
        self.assertEqual(result["route"], "commands")
        self.assertEqual([r["kind"] for r in result["request_decision"]["requested_results"]], ["lookup", "calculation"])
        self.assertEqual(result["cost_usd"], .25)

    def test_source_quote_choices_do_not_import_unrelated_history(self):
        from agent.response_policy import source_quote_choices
        decision = dict(requested_results=[dict(request_quote="메모리 상태를 확인해줘")])
        payload = dict(query="CPU 상태와 사용률 확인", previous_request="메모리 상태를 확인해줘",
            has_previous_observations=True, previous_is_latest_topic=True)
        self.assertTrue(all(q in payload['query'] for q in source_quote_choices(decision, payload)))
        decision.update(continue_previous_task=True, observation_policy='reuse')
        self.assertIn(payload['previous_request'], source_quote_choices(decision, payload))

    def test_long_source_quote_choices_respect_contract_limit(self):
        from agent.response_policy import source_quote_choices
        query = "긴 배경 설명 " * 70 + "현재 CPU 사용률과 메모리 사용률을 확인해줘"
        choices = source_quote_choices(dict(requested_results=[dict(request_quote="현재 CPU 사용률을 확인해줘")]), dict(query=query))
        self.assertIn("현재 CPU 사용률", choices)
        self.assertNotIn(query, choices)
        self.assertTrue(all(4 <= len(q) <= 400 and q in query for q in choices))

    async def test_result_kinds_drive_flags_without_reinterpreting_language(self):
        from agent.response_policy import request_route
        class Chosen(QualityLLM):
            def response_json(self,purpose,payload,schema):
                value=intent_payload(route="commands",style="brief",reason="복합 의미 분류",
                    requested_information_ids=["memory.summary"],needs_calculation=False,needs_diagnosis=False,
                    needs_explanation=False)
                value["requested_results"]=[
                    dict(description="사용/전체 메모리의 비율 %",kind="calculation",information_ids=["memory.summary"]),
                    dict(description="부족한 경우 원인 후보와 한계",kind="diagnosis",information_ids=["memory.summary"])]
                return json.dumps(value),{"usd":0}
        r=await request_route(dict(query="현재 메모리 사용 비율과 부족한 경우 원인 후보, 확인 한계를 설명해줘.",
            scenario_id="server-c",single_server_mode=True,events=[],history=[]),deps(RejectMCP(),Chosen("test",0)),set())
        self.assertTrue(r["request_decision"]["needs_calculation"])
        self.assertTrue(r["request_decision"]["needs_diagnosis"])
        self.assertEqual(r["response_style"],"detailed")


class IntentCoverageConsistencyTests(unittest.IsolatedAsyncioTestCase):
    def payload(self,**overrides):
        value=intent_payload(route="commands",style="brief",reason="메모리 계산과 조건부 원인",
            requested_information_ids=["memory.summary"],needs_calculation=True,needs_diagnosis=True,needs_explanation=False)
        value["requested_results"]=[dict(description="사용/전체 비율과 원인 후보",kind="calculation",information_ids=["memory.summary"])]
        return {**value,**overrides}

    async def run_route(self,values):
        from agent.response_policy import request_route
        class Chosen(QualityLLM):
            def __init__(self):super().__init__("test",0);self.inputs=[]
            def response_json(self,purpose,payload,schema):
                self.inputs.append(payload)
                return json.dumps(values[min(len(self.inputs)-1,len(values)-1)]),{"usd":0}
        model=Chosen()
        result=await request_route(dict(query="사용 메모리 비율과 부족한 경우 원인 후보를 설명해줘.",
            scenario_id="server-c",single_server_mode=True,events=[],history=[]),deps(RejectMCP(),model),set())
        return result,model.inputs

    async def test_missing_diagnosis_contract_is_retried_not_dropped(self):
        first=self.payload()
        corrected=self.payload(requested_results=[*first["requested_results"],
            dict(description="부족한 경우 원인 후보",kind="diagnosis",information_ids=["memory.summary"])])
        result,inputs=await self.run_route([first,corrected])
        self.assertEqual(len(inputs),2)
        self.assertIn("diagnosis",inputs[1]["validation_feedback"])
        self.assertTrue(result["request_decision"]["needs_diagnosis"])
        self.assertEqual(len(result["request_decision"]["requested_results"]),2)
        self.assertEqual(result["response_style"],"detailed")

    async def test_repeated_missing_contract_clarifies_without_tool_calls(self):
        result,inputs=await self.run_route([self.payload()])
        self.assertEqual(len(inputs),2)
        self.assertEqual(result["route"],"chat")
        self.assertIn("확인하지 못",result["direct_response_override"])
        self.assertNotIn("request_decision",result)

    async def test_typed_observation_only_request_does_not_enter_manual_route(self):
        value=self.payload(route="investigate_remediate",needs_calculation=False,
            requested_information_ids=["disk"],requested_results=[
                dict(description="디스크 사용률",kind="lookup",information_ids=["disk"]),
                dict(description="공간 부족의 원인 후보",kind="diagnosis",information_ids=["disk"])])
        result,inputs=await self.run_route([value])
        self.assertEqual(result["route"],"commands")
        self.assertEqual(result["request_decision"]["manual_information_ids"],[])
        self.assertEqual(len(inputs),1)


class RawOutputBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_command_result_sentence_keeps_raw_fast_path(self):
        from agent.response_policy import raw_fast_path
        for query in ["top 명령 결과를 원문 JSON만 보여줘.","top 원문만 보여주세요.","top 결과 JSON만"]:
            with self.subTest(query=query):
                self.assertTrue(raw_fast_path(query,"server-b"))
        for query in ["top 조회하지 말고 원문 JSON만 설명해줘", "top 출력 구조를 JSON만으로 설명해줘."]:
            self.assertFalse(raw_fast_path(query,"server-b"))
        class NoModel(QualityLLM):
            def response_json(self,*args):raise AssertionError("Raw explicit command must not use a model")
        mcp=FixtureMCP();d=deps(mcp);d.quality_llm=NoModel("test",0)
        result=await build_graph(d).ainvoke(dict(query="top 명령 결과를 원문 JSON만 보여줘.",scenario_id="server-b",single_server_mode=True,history=[],events=[],cost_usd=0))
        body=json.loads(result["final"])
        self.assertTrue(body)
        self.assertEqual(mcp.calls,[("server-b","top")])
        self.assertEqual(result["presentation"]["status"],"raw")

    async def test_concept_json_wording_does_not_suppress_chat_answer(self):
        from agent.response_policy import Intent
        class Concept(QualityLLM):
            def response_json(self,purpose,payload,schema):
                if purpose=="intent":
                    return Intent(route="chat",style="brief",reason="명령 출력 구조 설명",needs_explanation=True).model_dump_json(),{"usd":0}
                return super().response_json(purpose,payload,schema)
        mcp=RejectMCP();d=deps(mcp);d.quality_llm=Concept("test",0)
        result=await build_graph(d).ainvoke(dict(query="top 출력 구조를 JSON만으로 설명해줘.",scenario_id="server-b",single_server_mode=True,history=[],events=[],cost_usd=0))
        self.assertTrue(result["final"].strip())
        self.assertNotEqual(result["presentation"]["status"],"raw")
        self.assertFalse(result["raw_output"])
        self.assertEqual(mcp.calls,[])


class InformationIntentBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def route(self, values, query="선택 상태 확인", **extra):
        from agent.response_policy import request_route
        from monitoring.api import ScenarioStore
        class Chosen(QualityLLM):
            def __init__(self):super().__init__("test",0);self.inputs=[]
            def response_json(self,purpose,payload,schema):
                self.inputs.append(payload)
                return json.dumps(values[min(len(self.inputs)-1,len(values)-1)]),{"usd":0}
        model=Chosen();mcp=RejectMCP();d=deps(mcp,model)
        d.information_catalog=[i.model_dump() for i in ScenarioStore().information]
        result=await request_route(dict(query=query,scenario_id="server-g",single_server_mode=True,
            history=[],events=[],**extra),d,set())
        return result,model.inputs,mcp

    async def test_intent_receives_canonical_information_semantics_without_tools(self):
        value=intent_payload(route="chat",style="brief",reason="개념")
        result,inputs,mcp=await self.route([value],query="로그 파일과 로그 내용의 차이가 뭐야?")
        catalog={i["information_id"]:i for i in inputs[0].get("information_catalog",[])}
        self.assertIn("log.content",catalog);self.assertIn("log.files",catalog)
        self.assertNotEqual(catalog["log.content"]["description"],catalog["log.files"]["description"])
        self.assertIn("limitations",catalog["network"])
        self.assertNotIn("commands_by_os",catalog["network"])
        self.assertEqual(result["route"],"chat");self.assertEqual(mcp.calls,[])

    async def test_chat_observation_conflict_requires_consistent_model_retry(self):
        value=intent_payload(route="chat",style="detailed",reason="조회와 설명",
            requested_information_ids=["network"],needs_explanation=True,
            requested_results=[dict(kind="lookup",description="소켓 연결 상태",information_ids=["network"]),
                               dict(kind="explanation",description="외부 응답 확인 한계",information_ids=["network"])])
        result,inputs,mcp=await self.route([value,{**value,"route":"commands"}])
        self.assertEqual(result["route"],"commands");self.assertEqual(len(inputs),2)
        self.assertEqual(mcp.calls,[])

    async def test_greeting_with_invented_basic_observations_requires_retry(self):
        bad=intent_payload(route='chat',style='brief',reason='인사에는 관측이 필요 없음',basic_status=True,
            requested_information_ids=['cpu','memory.summary','disk'],
            requested_results=[dict(description='안녕!',kind='lookup',information_ids=['cpu','memory.summary','disk'])])
        good=intent_payload(route='chat',style='brief',reason='인사',basic_status=False)
        result,inputs,mcp=await self.route([bad,good],query='안녕!')
        self.assertEqual(result['route'],'chat')
        self.assertNotIn('direct_response_override',result)
        self.assertEqual(len(inputs),2)
        self.assertEqual(mcp.calls,[])
        failed,_,_=await self.route([bad,bad],query='안녕!')
        self.assertEqual(failed['route'],'chat')
        self.assertIn('direct_response_override',failed)

    async def test_observation_plus_manual_keeps_both_contracts_and_selected_route(self):
        first=intent_payload(route="remediation",style="detailed",reason="메모리 조치",
            requested_information_ids=["memory.summary"],manual_information_ids=["memory.summary"],
            requested_results=[dict(kind="remediation",description="부족할 때 조치",information_ids=["memory.summary"])])
        result,inputs,_=await self.route([first])
        # The model already selected observation IDs: preserve that contract
        # directly instead of buying another classification of the same intent.
        self.assertEqual(len(inputs),1)
        self.assertEqual(result["route"],"investigate_remediate")
        self.assertEqual(len(result["request_decision"]["requested_results"]),2)

    async def test_os_specific_general_manual_does_not_acquire_observations(self):
        value=intent_payload(route="remediation",style="detailed",reason="일반 매뉴얼",
            manual_information_ids=["cpu"],requested_results=[dict(kind="remediation",description="Linux CPU 대응",information_ids=["cpu"])])
        result,inputs,mcp=await self.route([value],query="Linux CPU 과부하 대응 방법을 설명해줘")
        self.assertEqual(result["route"],"remediation");self.assertEqual(mcp.calls,[])

    async def test_unsupported_only_manual_returns_scope_before_os_question(self):
        class Unsupported(QualityLLM):
            def response_json(self,purpose,payload,schema):
                if purpose=="intent":return json.dumps(intent_payload(route="remediation",style="detailed",reason="미지원 매뉴얼",
                    observation_policy="reuse",unsupported_manual_topics=["네트워크 지연"],
                    requested_results=[dict(kind="remediation",description="네트워크 지연 매뉴얼",information_ids=["network"])])),{"usd":0}
                return super().response_json(purpose,payload,schema)
        mcp=RejectMCP()
        result=await build_graph(deps(mcp,Unsupported("test",0))).ainvoke(dict(
            query="네트워크 지연 문제의 공식 조치 매뉴얼을 찾아줘. 서버 상태 조회는 하지 마.",
            scenario_id="server-a",single_server_mode=True,history=[],events=[],cost_usd=0))
        self.assertIn("미지원",result["final"]);self.assertNotIn("어떤 환경",result["final"])
        self.assertEqual(mcp.calls,[])
        self.assertTrue(result["requirements"])
        self.assertTrue(all(r["status"]=="unsupported" for r in result["requirements"]))
        self.assertEqual(result["final_validation"]["status"],"partial")


class TypedIntentReconciliationTests(unittest.IsolatedAsyncioTestCase):
    async def test_followup_can_quote_only_the_bound_previous_request(self):
        from agent.response_policy import request_route
        d=intent_payload(route='commands',style='brief',reason='같은 관측 재설명',
            requested_information_ids=['cpu'],observation_policy='reuse',continue_previous_task=True,
            requested_results=[dict(request_quote='CPU 상태 확인',description='CPU 재설명',kind='explanation',information_ids=['cpu'])])
        class Model:
            def response_json(self,*args):return json.dumps(d),{'usd':.1}
        state=dict(query='방금 결과만 설명해줘',scenario_id='server-b',single_server_mode=True,
            history=[dict(role='user',content='CPU 상태 확인')],investigation_context={
                'query':'CPU 상태 확인','subject_query':'CPU 상태 확인','evidence':[{'server_id':'server-b'}],
                'tasks':[{'information_ids':['cpu']}]})
        result=await request_route(state,deps(RejectMCP(),Model()),set())
        self.assertEqual(result['route'],'commands')
        self.assertEqual(result['request_decision']['observation_policy'],'reuse')
        # Unrelated/old history is not a source of new requirements.
        state['history']=[dict(role='user',content='다른 개념 설명')]
        self.assertEqual((await request_route(state,deps(RejectMCP(),Model()),set()))['route'],'chat')

    async def test_chat_explanation_subject_is_not_a_request_for_observation(self):
        d=intent_payload(route="chat",style="brief",reason="개념 설명",requested_information_ids=[],
            requested_results=[dict(description="CPU 사용률 의미",kind="explanation",information_ids=["cpu"])])
        result=await self.route(d)
        self.assertEqual(result["route"],"chat")
        self.assertNotIn("direct_response_override",result)

    async def test_followup_action_can_reference_observations_without_new_lookup_result(self):
        from agent.response_policy import request_route
        d=intent_payload(route="investigate_remediate",style="detailed",reason="기존 관측 대응",observation_policy="reuse",continue_previous_task=True,
            requested_information_ids=["memory.summary"],manual_information_ids=["memory.summary"],
            requested_results=[dict(description="그 상태에서 대응",kind="remediation",information_ids=["memory.summary"],action_condition="if_needed")])
        class Model:
            def response_json(self,*args):return json.dumps(d),{"usd":0}
        state=dict(query="그 상태에서 어떻게 대응해야 해? 새로 조회하지는 마.",scenario_id="server-g",single_server_mode=True,
            history=[dict(role="user",content="메모리 상태 확인")],investigation_context=dict(query="메모리 상태 확인",subject_query="메모리 상태 확인",
                evidence=[dict(server_id="server-g")],tasks=[dict(information_ids=["memory.summary"])]))
        result=await request_route(state,deps(RejectMCP(),Model()),set())
        self.assertEqual(result["route"],"investigate_remediate")
        self.assertEqual(result["request_decision"]["observation_policy"],"reuse")

    async def test_explicit_typed_observation_is_kept_when_result_only_describes_action(self):
        d=intent_payload(route="remediation",style="detailed",reason="확인 후 조건부 조치",
            requested_information_ids=["memory.summary"],manual_information_ids=["memory.summary"],
            requested_results=[dict(description="메모리 확인 후 조치",kind="remediation",information_ids=["memory.summary"],action_condition="if_needed")])
        result=await self.route(d)
        self.assertEqual(result["route"],"investigate_remediate")
        self.assertTrue(any(r["kind"]=="lookup" and r["information_ids"]==["memory.summary"] for r in result["request_decision"]["requested_results"]))

    async def test_partial_manual_topic_contract_fails_closed(self):
        d=intent_payload(route="remediation",style="detailed",reason="두 대처 주제",manual_information_ids=["cpu","disk"],
            requested_results=[dict(description="CPU 절차",kind="remediation",information_ids=["cpu"],action_condition="always")])
        result=await self.route(d)
        self.assertEqual(result["route"],"chat")
        self.assertIn("요청 의도",result["direct_response_override"])

    async def test_provider_flags_are_derived_and_manual_gate_rejects_added_actions(self):
        d=intent_payload(route='investigate_remediate',style='detailed',reason='관측 후 대응',
            requested_information_ids=['cpu'],manual_information_ids=['cpu'])
        for key in ('needs_diagnosis','needs_calculation','needs_explanation'):d.pop(key)
        d.update(manual_requested=True,requested_results=[
            dict(description='CPU 확인',kind='lookup',information_ids=['cpu']),
            dict(description='CPU 대응',kind='remediation',information_ids=['cpu'],action_condition='if_needed')])
        r=await self.route(d)
        self.assertEqual(r['route'],'investigate_remediate')
        self.assertFalse(r['request_decision']['needs_diagnosis'])
        d['manual_requested']=False
        self.assertEqual((await self.route(d))['route'],'chat')

    async def test_conditional_topics_are_independently_actionable(self):
        d=intent_payload(route='investigate_remediate',style='detailed',reason='각 주제 조건부 조치',
            requested_information_ids=['cpu','disk'],manual_information_ids=['cpu','disk'],
            requested_results=[dict(description='CPU와 디스크 확인',kind='lookup',information_ids=['cpu','disk']),
                dict(description='각 항목 필요한 조치',kind='remediation',information_ids=['cpu','disk'],action_condition='if_needed')])
        r=await self.route(d)
        actions=[x for x in r['request_decision']['requested_results'] if x['kind']=='remediation']
        self.assertEqual([x['information_ids'] for x in actions],[['cpu'],['disk']])

    async def test_conflicting_manual_intent_does_not_retry_into_search_permission(self):
        from agent.response_policy import request_route
        class Conflict(QualityLLM):
            calls=0
            def response_json(self,purpose,payload,schema):
                self.calls+=1
                return json.dumps(intent_payload(route='investigate_remediate',style='detailed',reason='모순',
                    manual_requested=self.calls>1,requested_information_ids=['cpu'],manual_information_ids=['cpu'],
                    requested_results=[dict(description='CPU 확인',kind='lookup',information_ids=['cpu']),
                        dict(description='CPU 대처',kind='remediation',information_ids=['cpu'],action_condition='if_needed')])),{'usd':.001}
        model=Conflict('test',0)
        result=await request_route(dict(query='CPU 원인과 확인 한계를 설명해줘',scenario_id='server-b',
            single_server_mode=True,history=[],events=[]),deps(RejectMCP(),model),[])
        self.assertEqual(model.calls,1)
        self.assertEqual(result['route'],'chat')
        self.assertIn('요청 의도',result['direct_response_override'])
        self.assertEqual(result['cost_usd'],.001)

    async def route(self,decision):
        from agent.response_policy import request_route
        class Chosen(QualityLLM):
            def response_json(self,purpose,payload,schema):
                assert purpose=='intent'
                return json.dumps(decision,ensure_ascii=False),{'usd':0}
        return await request_route(dict(query='구조화된 모델 계약 재현',scenario_id='server-g',scenario_os='macos',
            single_server_mode=True,history=[],events=[]),deps(RejectMCP(),Chosen('test',0)),[])

    async def test_basic_scope_is_consistent_in_both_information_lists(self):
        d=intent_payload(route='commands',style='brief',reason='기본 상태',basic_status=True,
            requested_information_ids=['cpu','memory.summary','disk','system','network'])
        d['requested_results']=[dict(description='상태',kind='lookup',information_ids=d['requested_information_ids'])]
        r=await self.route(d);decision=r['request_decision']
        self.assertEqual(decision['requested_information_ids'],['cpu','memory.summary','disk'])
        self.assertEqual(decision['requested_results'][0]['information_ids'],['cpu','memory.summary','disk'])

    async def test_ambiguous_conditional_lookup_fails_closed_without_inventing_advice(self):
        d=intent_payload(route='investigate_remediate',style='detailed',reason='모순된 계약',requested_information_ids=['cpu','disk'])
        d['requested_results']=[dict(description=i,kind='lookup',information_ids=[i],action_condition='if_needed') for i in ['cpu','disk']]
        r=await self.route(d)
        self.assertEqual(r['route'],'chat')
        self.assertIn('요청 의도',r['direct_response_override'])

    async def test_integrated_route_materializes_its_missing_observation(self):
        d=intent_payload(route='investigate_remediate',style='detailed',reason='조회 후 조치',manual_information_ids=['memory.summary'])
        d['requested_results']=[dict(description='메모리 대처',kind='remediation',information_ids=['memory.summary'],action_condition='if_needed')]
        r=await self.route(d);decision=r['request_decision']
        self.assertEqual(decision['requested_information_ids'],['memory.summary'])
        self.assertTrue(any(x['kind']=='lookup' and x['information_ids']==['memory.summary'] for x in decision['requested_results']))

    async def test_single_known_manual_subject_binds_all_empty_manual_results(self):
        d=intent_payload(route='investigate_remediate',style='detailed',reason='디스크 조회 후 조치',
            requested_information_ids=['disk'],manual_information_ids=['disk'])
        d['requested_results']=[dict(description='디스크 조회',kind='lookup',information_ids=['disk'])] + [dict(description=label,kind='remediation',information_ids=[]) for label in ['대처','조건부 대처']]
        r=await self.route(d)
        self.assertTrue(all(x['information_ids']==['disk'] for x in r['request_decision']['requested_results'] if x['kind']=='remediation'))

    async def test_partial_invalid_contract_is_rejected_instead_of_dropping_a_subject(self):
        d=intent_payload(route='investigate_remediate',style='detailed',reason='각 항목 조건부 조치',requested_information_ids=['cpu','disk'],manual_information_ids=['disk'])
        d['requested_results']=[dict(description='CPU',kind='lookup',information_ids=['cpu'],action_condition='if_needed'),
            dict(description='디스크',kind='lookup',information_ids=['disk']),
            dict(description='디스크 조치',kind='remediation',information_ids=['disk'],action_condition='if_needed')]
        r=await self.route(d)
        self.assertEqual(r['route'],'chat')

    async def test_original_request_quote_controls_description_and_rejects_fabrication(self):
        d=intent_payload(route='commands',style='brief',reason='원문 계약',requested_information_ids=['cpu'])
        d['requested_results']=[dict(request_quote='구조화된 모델 계약 재현',description='사용자에게 없는 임계값 100%',kind='lookup',information_ids=['cpu'])]
        r=await self.route(d)
        self.assertEqual(r['request_decision']['requested_results'][0]['description'],'구조화된 모델 계약 재현')
        self.assertNotIn('request_quote',r['request_decision']['requested_results'][0])
        d['requested_results'][0]['request_quote']='없는 요구'
        self.assertEqual((await self.route(d))['route'],'chat')

    async def test_observation_without_information_id_is_not_silently_unfulfillable(self):
        d=intent_payload(route='commands',style='detailed',reason='비어 있는 설명')
        d['requested_results']=[dict(description='상태 해석',kind='explanation',information_ids=[])]
        self.assertEqual((await self.route(d))['route'],'chat')
