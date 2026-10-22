import asyncio
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.fake_llm import FastLLM, QualityLLM
from agent.graph import build_graph
from agent.remediation import remediation_node, wants_remediation
from remediation.actions import ACTIONS, candidates
from remediation.collect import chunk_section, sections
from remediation.context import recall, remember
from remediation.index import ManualIndex, ManualUnavailable, ROOT
from remediation.model import summarize


def intent_payload(**overrides):
    """Explicit complete semantic decisions from controlled test models."""
    return dict(requested_information_ids=[], manual_information_ids=[], needs_diagnosis=False,
        needs_calculation=False, needs_explanation=False, observation_policy="auto", continue_previous_task=False,
        basic_status=False, unsupported_manual_topics=[]) | overrides


def match_for(doc, text=None, os_name="linux", section=None):
    original=next(c for c in json.loads((ROOT/"data/manuals/chunks.json").read_text()) if c["doc_id"]==doc and (section is None or c["section"]==section))
    if text is not None: original["text"]=text
    return {**original,"os":os_name,"score":.8}


class CountingEmbedder:
    model="test-v1"
    def __init__(self): self.calls=[]
    def embed(self,texts):
        self.calls.extend(texts)
        return [[1.,0.,0.] if "cpu" in t else [0.,1.,0.] for t in texts]


class StubMCP:
    def __init__(self,matches=None,fail=False): self.matches=matches or [];self.calls=[];self.fail=fail
    async def call_tool(self,name,args):
        self.calls.append((name,args))
        if name=="list_command_servers": return [{"server_id":"server-b","os":"linux"},{"server_id":"server-g","os":"macos"}]
        if name=="search_remediation_manuals":
            if self.fail: raise ValueError("인덱스 없음")
            return {"matches":self.matches,"timings_ms":{"embedding":1,"search":2}}
        raise AssertionError(name)


class ManualDataTests(unittest.TestCase):
    def test_real_sources_hashes_and_token_caps(self):
        manifest=json.loads((ROOT/"data/manuals/manifest.json").read_text())
        self.assertEqual(len(manifest["documents"]),6)
        for doc in manifest["documents"]:
            self.assertEqual(hashlib.sha256((ROOT/doc["original_path"]).read_bytes()).hexdigest(),doc["sha256"])
            self.assertTrue(doc["source_url"].startswith("https://"))
        chunks=json.loads((ROOT/"data/manuals/chunks.json").read_text())
        self.assertTrue(all(c["token_count"]<=600 for c in chunks))
        self.assertTrue(all("Copyright and trademarks" not in c["text"] for c in chunks))
        self.assertFalse(any("data/runbooks" in json.dumps(c) for c in chunks))

    def test_every_reviewed_action_is_anchored_in_a_source_chunk(self):
        chunks=json.loads((ROOT/"data/manuals/chunks.json").read_text())
        available={a["action_id"] for a in candidates(chunks)}
        self.assertEqual(available,{a[0] for a in ACTIONS})

    def test_only_remedies_are_indexed_and_removed_sources_cannot_return(self):
        from remediation.sources import SOURCES
        removed={"rhel-monitoring","rhel-memory","gnu-df","gnu-du","apple-cpu","apple-memory"}
        chunks=json.loads((ROOT/"data/manuals/chunks.json").read_text())
        self.assertFalse(removed & {d["doc_id"] for d in SOURCES})
        self.assertTrue(all(candidates([c]) for c in chunks))
        self.assertTrue(all(a[-1]!="read_only" for a in ACTIONS))
        for directory in ["originals","text"]:
            self.assertFalse(removed & {p.stem for p in (ROOT/"data/manuals"/directory).iterdir()})

    def test_resource_control_sections_keep_cautions_but_exclude_lookup_methods(self):
        from remediation.sources import SOURCES
        spec=next(s for s in SOURCES if s["doc_id"]=="kernel-memory-control")
        raw=(ROOT/"data/manuals/originals/kernel-memory-control.html").read_text()
        parsed=sections(raw,spec)
        self.assertEqual([h for h,t in parsed],["memory.high"])
        self.assertIn("under extreme conditions the limit may be breached",parsed[0][1])
        self.assertIn("external process monitors",parsed[0][1])
        with self.assertRaisesRegex(ValueError,"Missing configuration"):
            sections("<main>removed upstream</main>",spec)

    def test_collection_prunes_only_retired_manifest_files(self):
        from remediation.collect import collect
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/"data/manuals"
            shutil.copytree(ROOT/"data/manuals",base)
            (root/".manual_cache").mkdir()
            (root/".manual_cache/bge-m3-tokenizer.json").write_text("test-tokenizer-placeholder")
            manifest=json.loads((base/"manifest.json").read_text())
            manifest["documents"].append(dict(doc_id="retired",original_path="data/manuals/originals/retired.html"))
            (base/"manifest.json").write_text(json.dumps(manifest))
            for rel in ["originals/retired.html","text/retired.txt","text/user-notes.txt"]:(base/rel).write_text("keep unrelated")
            with patch("remediation.collect.Tokenizer.from_file"), patch("remediation.collect.chunk_section",side_effect=lambda text,tokenizer:[(text,1)]):
                collect(root)
            self.assertFalse((base/"originals/retired.html").exists())
            self.assertFalse((base/"text/retired.txt").exists())
            self.assertTrue((base/"text/user-notes.txt").exists())

    def test_navigation_is_not_manual_text(self):
        raw='<main><nav><p>ignore</p></nav><h1>Activity Monitor User Guide</h1><p>Copyright and trademarks</p><h1>Actual procedure</h1><p>Inspect memory</p></main>'
        self.assertEqual(sections(raw,{"doc_id":"a","title":"x"}),[("Actual procedure","Inspect memory")])

    def test_evaluation_split_and_no_trigger_questions(self):
        cases=json.loads((ROOT/"eval/remediation_cases.json").read_text())["cases"]
        self.assertEqual(len(cases),24)
        self.assertEqual(sum(c["split"]=="holdout" for c in cases),6)
        self.assertEqual(sum(c["kind"]=="positive" for c in cases),12)
        for c in cases:
            self.assertEqual(wants_remediation(c["query"]),c["kind"]!="no_trigger",c["id"])

    def test_structured_context_survives_new_connection_and_is_session_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"session.db";store=SimpleNamespace(conn=sqlite3.connect(path))
            data={"evidence":[{"server_id":"server-b"}],"tasks":[]}
            remember(store,"one",data,"CPU 요청");store.conn.close()
            store=SimpleNamespace(conn=sqlite3.connect(path))
            self.assertEqual(recall(store,"one")["query"],"CPU 요청")
            self.assertIsNone(recall(store,"two"));store.conn.close()


class IndexLifecycleTests(unittest.TestCase):
    def test_persistence_incremental_embedding_and_os_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/"data/manuals";base.mkdir(parents=True)
            chunks=[]
            for sid,os_name in [("a","linux"),("b","macos")]:
                chunks.append({"chunk_id":sid,"doc_id":sid,"os":os_name,"portable":True,"product":"x","version":"1",
                               "title":"cpu","section":"diagnosis","text":"cpu report "+sid,"sha256":"source-one"})
            (base/"chunks.json").write_text(json.dumps(chunks));(base/"retrieval.json").write_text('{"minimum_score":0,"require_reviewed_actions":false}')
            embed=CountingEmbedder();index=ManualIndex(root,embed)
            self.assertEqual(index.build()["embedded"],2)
            self.assertEqual(index.build()["changed"],False)
            self.assertEqual(len(embed.calls),2)
            restarted=ManualIndex(root,embed)
            self.assertEqual([c["os"] for c in restarted.search("cpu","linux")["matches"]],["linux"])
            query_vector_input=embed.calls[-1]
            with_observations=restarted.search("cpu","linux","Memory: total 8192 MiB, free 3584 MiB; disk 95%")
            self.assertEqual(embed.calls[-1],query_vector_input)
            self.assertEqual([c["os"] for c in with_observations["matches"]],["linux"])
            # Reused vector first, newly embedded vector last (mixed Chroma/HTTP origins).
            chunks[1]["text"]+=" changed";chunks[1]["sha256"]="source-two"
            (base/"chunks.json").write_text(json.dumps(chunks))
            with self.assertRaisesRegex(ManualUnavailable,"본문"):
                index.search("cpu","linux")
            updated=index.build();self.assertEqual(updated["embedded"],1);self.assertEqual(updated["reused"],1)
            embed.model="test-v2"
            with self.assertRaises(ManualUnavailable): index.search("cpu","linux")
            self.assertEqual(index.build()["embedded"],2)

    def test_search_uses_five_candidates_and_returns_at_most_two_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manuals=root/"data/manuals";manuals.mkdir(parents=True)
            chunks=[dict(chunk_id=f"c{i}",doc_id=f"d{i}",os="linux",portable=True,product="x",version="1",
                title=f"cpu {i}",section="diagnosis",text=f"cpu report distinct {i}",sha256="source") for i in range(6)]
            (manuals/"chunks.json").write_text(json.dumps(chunks))
            (manuals/"retrieval.json").write_text(json.dumps(dict(minimum_score=0,require_reviewed_actions=False,
                candidate_count=5,result_chunk_limit=2)))
            index=ManualIndex(root,CountingEmbedder());index.build()
            collection=index.client().get_collection(json.loads((index.path/"active.json").read_text())["collection"])
            seen=[]
            def query(**args):seen.append(args);return collection.query(**args)
            index.client=lambda:SimpleNamespace(get_collection=lambda name, **kwargs:SimpleNamespace(query=query))
            result=index.search("cpu","linux")
            self.assertEqual(seen[0]["n_results"],5) # 5 candidates from 6 indexed chunks
            self.assertEqual(len(result["matches"]),2)
            self.assertEqual(len({m["chunk_id"] for m in result["matches"]}),2)

    def test_failed_rebuild_keeps_old_active_and_topic_metadata_is_not_source_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/"data/manuals";base.mkdir(parents=True)
            chunk=dict(chunk_id="one",doc_id="doc",os="linux",portable=True,product="x",version="1",
                incident_types=["disk-full"],title="Log rotation",section="DESCRIPTION",text="Rotation and compression.")
            (base/"chunks.json").write_text(json.dumps([chunk]))
            embed=CountingEmbedder();index=ManualIndex(root,embed);index.build()
            self.assertIn("disk full",embed.calls[0])
            active=(index.path/"active.json").read_text()
            collection=index.client().get_collection(json.loads(active)["collection"])
            stored=collection.get(include=["metadatas"])["metadatas"][0]
            self.assertEqual(json.loads(stored["chunk_json"])["text"],chunk["text"])
            chunk["text"]+=" updated";(base/"chunks.json").write_text(json.dumps([chunk]))
            with patch.object(embed,"embed",side_effect=ManualUnavailable("offline")):
                with self.assertRaises(ManualUnavailable):index.build()
            self.assertEqual((index.path/"active.json").read_text(),active)
            self.assertEqual(collection.count(),1)

    def test_deleted_vectors_are_removed_and_cleanup_recovers_after_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/"data/manuals";base.mkdir(parents=True)
            chunks=[dict(chunk_id=k,doc_id=k,os="linux",portable=True,product="x",version="1",title="cpu",section="action",text="cpu "+k) for k in ["keep","retire"]]
            (base/"chunks.json").write_text(json.dumps(chunks))
            (base/"retrieval.json").write_text('{"minimum_score":0,"require_reviewed_actions":false}')
            embed=CountingEmbedder();index=ManualIndex(root,embed);index.build()
            old=json.loads((index.path/"active.json").read_text())["collection"]
            index.client().create_collection("unrelated-user-data")
            (base/"chunks.json").write_text(json.dumps(chunks[:1]))
            with patch.object(index,"_prune_obsolete",side_effect=RuntimeError("interrupted cleanup")):
                with self.assertRaises(RuntimeError):index.build()
            self.assertEqual(index.client().get_collection(old).count(),2)
            self.assertFalse(index.build()["changed"])
            restarted=ManualIndex(root,embed)
            self.assertEqual([m["chunk_id"] for m in restarted.search("cpu","linux")["matches"]],["keep"])
            names={c.name for c in restarted.client().list_collections()}
            self.assertNotIn(old,names);self.assertIn("unrelated-user-data",names)

    def test_missing_index_never_creates_or_queries_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            embed=CountingEmbedder();index=ManualIndex(tmp,embed)
            with self.assertRaises(ManualUnavailable): index.search("cpu","linux")
            self.assertEqual(embed.calls,[])


class RouteAndSummaryTests(unittest.IsolatedAsyncioTestCase):
    def deps(self,mcp=None,quality=None):
        return SimpleNamespace(mcp=mcp or StubMCP(),fast_llm=FastLLM("test",0),quality_llm=quality or QualityLLM("test",0))

    async def test_generic_os_request_has_one_search_and_no_commands(self):
        matches=[match_for("kernel-cpu-control")];mcp=StubMCP(matches)
        result=await remediation_node({"query":"Linux CPU 대응 방법","events":[]},self.deps(mcp))
        self.assertEqual([n for n,_ in mcp.calls],["search_remediation_manuals"])
        self.assertEqual(result["remediation"]["status"],"provided")
        self.assertTrue(all(a["quote"] in matches[0]["text"] for a in result["remediation"]["steps"]))
        self.assertLessEqual(result["remediation"]["model_calls"],2)

    async def test_ambiguous_and_mismatched_os_do_not_search(self):
        for query in ["어떻게 조치해?","서버 G Linux CPU 대응 방법","서버 Z CPU 조치방법"]:
            mcp=StubMCP();r=await remediation_node({"query":query,"events":[]},self.deps(mcp))
            self.assertEqual(r["remediation"]["status"],"needs_clarification")
            self.assertNotIn("search_remediation_manuals",[n for n,_ in mcp.calls])

    async def test_target_without_prior_observation_is_conditional_and_does_not_query_commands(self):
        mcp=StubMCP([match_for("kernel-cpu-control")])
        r=await remediation_node({"query":"서버 B CPU 대처방안","events":[]},self.deps(mcp))
        self.assertEqual([n for n,_ in mcp.calls],["list_command_servers","search_remediation_manuals"])
        self.assertEqual(r["evidence"],[])
        self.assertIn("현재 서버 상태를 조회하지 않았습니다",r["final"])

    async def test_followup_reuses_previous_evidence_without_requerying(self):
        mcp=StubMCP([match_for("kernel-cpu-control")])
        context={"query":"서버 B CPU 상태 확인","evidence":[{"server_id":"server-b","observed_at":"2026-10-20T12:00:01+09:00","facts":[{"summary":"CPU 사용률 96%"}]}]}
        state={"query":"그 서버 대처방안 알려줘","history":[{"role":"user","content":"서버 B CPU 상태"}],"events":[],"investigation_context":context}
        result=await remediation_node(state,self.deps(mcp))
        self.assertEqual(result["command_call_count"],0)
        self.assertEqual(result["evidence"],context["evidence"])
        self.assertIn("96%",mcp.calls[-1][1]["evidence_summary"])
        self.assertEqual(result["remediation"]["status"],"provided")

    async def test_wrong_os_manual_is_never_accepted(self):
        r=await remediation_node({"query":"Linux CPU 대응 방법","events":[]},self.deps(StubMCP([match_for("apple-quit",os_name="macos")])))
        self.assertEqual(r["remediation"]["status"],"unavailable")
        self.assertEqual(r["remediation"]["steps"],[])

    async def test_search_failure_preserves_previous_evidence(self):
        evidence=[{"server_id":"server-b","facts":[{"summary":"CPU 96%"}]}]
        r=await remediation_node({"query":"서버 B CPU 조치방법","events":[],"investigation_context":{"evidence":evidence}},self.deps(StubMCP(fail=True)))
        self.assertEqual(r["evidence"],evidence)
        self.assertEqual(r["remediation"]["status"],"unavailable")

    async def test_model_cannot_add_invented_commands_and_has_one_repair(self):
        class Bad:
            calls=0
            def remediation_json(self,payload,schema):
                self.calls+=1;return '{"applicable":true,"action_ids":["rm-all-files"]}',{"usd":0}
        bad=Bad()
        with self.assertRaises(ValueError): await summarize(bad,"Linux 디스크 해결","",candidates([match_for("logrotate")]))
        self.assertEqual(bad.calls,2)

    async def test_summary_cannot_mix_disk_steps_into_memory_type(self):
        class Mixed:
            def remediation_json(self,payload,schema):
                return json.dumps({"applicable":True,"incident_types":["memory-pressure"],"action_ids":["linux-rotation"]}),{"usd":0}
        with self.assertRaisesRegex(ValueError,"두 번"):
            await summarize(Mixed(),"Linux 메모리 대응","",candidates([match_for("logrotate"),match_for("kernel-memory-control")]))

    async def test_one_applicable_remedy_is_valid_without_padding_or_retry(self):
        class OneAction:
            calls=0
            def remediation_json(self,payload,schema):
                self.calls+=1
                assert all(a.get("precondition") for a in payload["actions"])
                return json.dumps({"applicable":True,"incident_types":["cpu-spike"],"action_ids":["linux-cpu-limit"]}),{"usd":0}
        model=OneAction()
        steps,metrics=await summarize(model,"Linux CPU 대응","",candidates([match_for("kernel-cpu-control")]))
        self.assertEqual(len(steps),1);self.assertEqual(metrics["model_calls"],1)

    async def test_graph_keeps_ordinary_status_requests_out_of_rag(self):
        from test_information_investigation import FixtureMCP
        mcp=FixtureMCP();graph=build_graph(self.deps(mcp))
        r=await graph.ainvoke({"query":"서버 B CPU 상태 확인","history":[],"events":[],"cost_usd":0})
        self.assertEqual(r["route"],"commands")
        self.assertFalse(any(e["type"].startswith("Manual") for e in r["events"]))


class RemedyTopicRegressionTests(unittest.IsolatedAsyncioTestCase):
    def test_pronoun_clause_retains_typed_search_subject(self):
        from agent.remediation import manual_topic_query
        clause="그 상태에서 어떻게 대응해야 해? 새로 조회하지는 마."
        query=manual_topic_query([dict(information_ids=["memory.summary"],description=clause)])
        self.assertIn("memory pressure and RAM usage",query)
        self.assertIn(clause,query)
        self.assertNotIn("disk space",query)
        self.assertNotIn("memory.summary",query)

    async def test_missing_topic_still_gets_second_search_after_invalid_selection(self):
        from agent.remediation import build_manual_agent
        queries=[];selections=[]
        async def retrieve(query):
            queries.append(query)
            return {"matches":[match_for("kernel-cpu-control")] + ([match_for("logrotate")] if len(queries)>1 else [])}
        async def select(matches):
            selections.append(matches)
            if len(matches)==1:raise ValueError("No disk action to bind")
            return candidates(matches),{}
        requirements=[dict(requirement_id="r1",description="CPU 조치",information_ids=["cpu"]),
                      dict(requirement_id="r2",description="디스크 조치",information_ids=["disk"])]
        graph=build_manual_agent(retrieve,select,lambda state:None,lambda *a,**kw:None)
        result=await graph.ainvoke(dict(query="CPU and disk remedies",requirements=requirements,rounds=0,matches=[],selected=[]))
        self.assertEqual(len(queries),2)
        self.assertIn("disk space exhaustion",queries[1]);self.assertNotIn("high CPU",queries[1])
        self.assertEqual(result["status"],"assessed")
        self.assertTrue(any(a["doc_id"]=="logrotate" for a in result["selected"]))


class MCPContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_stdio_tool_reports_missing_index_as_error(self):
        from agent.mcp_client import MCPClient
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"MANUAL_INDEX_DIR":tmp,"LANGCHAIN_TRACING_V2":"false","LANGSMITH_TRACING":"false"}):
            client=MCPClient(flaky_first_tool_call=False,server_scripts={"ops":ROOT/"mcp_server.py"})
            await client.start()
            try:
                with self.assertRaisesRegex(ValueError,"인덱스"):
                    await client.call_tool("search_remediation_manuals",{"query":"Linux CPU 대응","os":"linux"})
            finally: await client.stop()


class CombinedWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from test_information_investigation import FixtureMCP
        from test_answer_completion import Model
        class Combined(FixtureMCP):
            def __init__(self):
                super().__init__();self.searches=[];self.matches=[[match_for("kernel-cpu-control"),match_for("kernel-memory-control")]];self.search_error=False
            async def call_tool(self,name,args):
                if name=="search_remediation_manuals":
                    self.searches.append(args)
                    if self.search_error:raise ValueError("offline index")
                    return {"matches":self.matches[min(len(self.searches)-1,len(self.matches)-1)],"timings_ms":{"embedding":1,"search":1}}
                return await super().call_tool(name,args)
        self.mcp=Combined();self.quality=Model();self.planner=Model()

    async def run_flow(self,query,context=None,history=None):
        deps=SimpleNamespace(mcp=self.mcp,fast_llm=FastLLM("unused",0),quality_llm=self.quality,monitoring_llm=self.planner)
        return await build_graph(deps).ainvoke({"query":query,"history":history or [],"events":[],"cost_usd":0,"investigation_context":context})

    def requirement(self,sid,kind,ids,description):
        return dict(description=description,server_ids=[sid],information_ids=ids,kind=kind)

    async def test_fresh_server_advice_investigates_then_formats_then_checks_exact_final_text(self):
        r=await self.run_flow("서버 B CPU 상태와 대처방안을 알려줘")
        self.assertEqual(r["route"],"investigate_remediate")
        self.assertEqual(self.mcp.calls,[("server-b","top")])
        self.assertEqual(len(self.mcp.searches),1)
        self.assertEqual(r["final_validation"]["status"],"passed")
        kinds=[e["type"] for e in r["events"]]
        self.assertLess(kinds.index("CommandCompleted"),kinds.index("ManualSearchStarted"))
        self.assertLess(kinds.index("ManualSearchCompleted"),kinds.index("FinalResponseFormatted"))
        self.assertLess(kinds.index("FinalResponseFormatted"),kinds.index("FinalAnswerValidation"))
        self.assertNotIn("linux-top",[a["action_id"] for a in r["remediation"]["steps"]])
        self.assertTrue(all(a["requirement_ids"] and a["evidence_ids"] for a in r["remediation"]["steps"]))
        reviewed=[p for purpose,p in self.quality.inputs if purpose=="final_review"]
        self.assertEqual(reviewed[-1]["draft_answer"],r["final"])

    async def test_manual_metadata_cannot_pass_when_final_body_drops_steps(self):
        from agent.response_policy import present_response as original
        async def omit_steps(state,deps):
            result=await original(state,deps)
            if state.get("remediation"):
                result["final"]="관측 사실만 제공합니다. 매뉴얼 단계·적용 조건은 누락되었습니다."
                result["answer_blocks"]=[];result["requirement_coverage"]={}
            return result
        with patch("agent.response_policy.present_response",side_effect=omit_steps):
            result=await self.run_flow("서버 B CPU 상태와 대처방안을 알려줘")
        self.assertEqual(result["remediation"]["status"],"provided")
        self.assertFalse(result["final_validation"]["request_fully_satisfied"])
        self.assertTrue(any(c["status"]=="unmet" for r in result["final_reviews"] for c in r.get("checks",[])))

    async def test_modified_manual_condition_is_not_grounded_despite_valid_source_hash(self):
        from agent.remediation import manual_requirement_covered
        result=await self.run_flow("서버 B CPU 상태와 대처방안을 알려줘")
        requirement=next(r for r in result["requirements"] if r["kind"]=="remediation")
        manual=json.loads(json.dumps(result["remediation"]))
        self.assertTrue(manual_requirement_covered(requirement,manual,result["evidence"]))
        manual["steps"][0]["precondition"]="모든 서버에서 즉시 재시작 완료"
        self.assertFalse(manual_requirement_covered(requirement,manual,result["evidence"]))

    async def test_final_review_receives_visible_manual_requirement_bindings(self):
        result=await self.run_flow("서버 B CPU 상태와 대처방안을 알려줘")
        payload=[p for purpose,p in self.quality.inputs if purpose=="final_review"][-1]
        requirement=next(r for r in result["requirements"] if r["kind"]=="remediation")
        coverage=payload["requirement_coverage"][requirement["requirement_id"]]
        self.assertTrue(coverage["step_ids"]);self.assertTrue(coverage["chunk_ids"])
        self.assertTrue(all(b["text"] in payload["draft_answer"] for b in payload["answer_blocks"]))
        for step in result["remediation"]["steps"]:
            self.assertIn(step["precondition"],result["final"])

    async def test_generic_manual_has_no_server_calls_and_gets_whole_request_validation(self):
        from test_answer_completion import Model
        class NoInventedTarget(Model):
            def monitoring_json(inner,purpose,payload,schema):
                if purpose=="final_review":
                    self.assertEqual(schema["properties"]["repair_tasks"]["maxItems"],0)
                    self.assertEqual(schema["$defs"]["RequirementSpec"]["properties"]["server_ids"]["maxItems"],0)
                return super().monitoring_json(purpose,payload,schema)
        self.quality=NoInventedTarget()
        r=await self.run_flow("Linux CPU 과부하 대처방안")
        self.assertEqual(self.mcp.calls,[]);self.assertEqual(len(self.mcp.searches),1)
        self.assertEqual(r["requirements"][0]["server_ids"],[])
        self.assertEqual(r["final_validation"]["status"],"passed")
        self.assertNotIn("모의 시나리오",r["final"])

    async def test_generic_manual_preserves_original_typed_requirement_for_search_and_validation(self):
        from test_answer_completion import Model
        description="Linux에서 메모리 부족을 조사하는 공식 매뉴얼 절차만 알려줘."
        class TypedManual(Model):
            def response_json(self,purpose,payload,schema):
                if purpose=="intent":return json.dumps(intent_payload(route="remediation",style="brief",reason="일반 절차",observation_policy="reuse",
                    manual_information_ids=["memory.summary"],requested_results=[dict(description=description,kind="remediation",information_ids=["memory.summary"],action_condition="always")])),{"usd":0}
                return super().response_json(purpose,payload,schema)
        self.quality=TypedManual()
        result=await self.run_flow("현재 서버는 조회하지 말고 "+description)
        self.assertEqual(len(result["requirements"]),1)
        self.assertEqual(result["requirements"][0]["description"],description)
        self.assertIn(description,self.mcp.searches[0]["query"])
        self.assertEqual(self.mcp.calls,[])
        self.assertEqual(result["final_validation"]["status"],"passed")

    async def cached_context(self):
        from agent.investigation import investigate
        from test_information_investigation import FixtureMCP
        mcp=FixtureMCP()
        result=await investigate({"query":"서버 A top JSON만","events":[],"history":[]},SimpleNamespace(mcp=mcp,fast_llm=self.planner,quality_llm=self.quality))
        return {**result,"query":"서버 A CPU 상태"}

    async def test_followup_reuses_observation_but_refresh_requests_requery(self):
        context=await self.cached_context();history=[{"role":"user","content":"서버 A CPU 상태"}]
        r=await self.run_flow("그 서버 CPU 대처방안",context,history)
        self.assertEqual(self.mcp.calls,[])
        self.assertEqual(r["evidence"],context["evidence"])
        self.assertEqual(r["workflow_budget"]["command_calls"],0)
        for query in ["그 서버 CPU 대처방안을 다시 설명해줘", "그 서버 CPU 대처방안. 새로 조회하지 말고 이전 관측을 사용해줘"]:
            self.mcp.calls.clear()
            repeated=await self.run_flow(query,context,history)
            self.assertEqual(self.mcp.calls,[])
            self.assertEqual(repeated["workflow_budget"]["command_calls"],0)
        self.mcp.calls.clear()
        fresh=await self.run_flow("서버 A CPU를 새로 확인하고 대처방안 알려줘",context,history)
        self.assertEqual(self.mcp.calls,[("server-a","top")])
        self.assertEqual(fresh["workflow_budget"]["command_calls"],1)

    async def test_multiple_servers_and_wrong_os_stop_before_results_or_manual_search(self):
        for query in ["서버 A와 B CPU 상태와 대처방안","서버 G Linux CPU 대처방안"]:
            r=await self.run_flow(query)
            self.assertEqual(self.mcp.calls,[]);self.assertEqual(self.mcp.searches,[])
            self.assertEqual(r["final_validation"]["status"],"not_run")

    async def test_failed_observation_keeps_conditional_manual_and_partial_status(self):
        self.mcp.overrides[("server-b","top")]={"stdout":"","exit_code":1,"stderr":"fixture denied"}
        r=await self.run_flow("서버 B CPU 상태와 대처방안")
        self.assertEqual(r["evidence"][0]["exit_code"],1)
        self.assertEqual(r["remediation"]["status"],"provided")
        self.assertTrue(all(a["conditional"] for a in r["remediation"]["steps"]))
        self.assertEqual(r["final_validation"]["status"],"partial")
        self.assertIn("조건부",r["final"])

    async def test_search_failure_keeps_observations_and_does_not_repeat_failed_search(self):
        self.mcp.search_error=True
        r=await self.run_flow("서버 A CPU 상태와 대처방안")
        self.assertEqual(len(self.mcp.searches),1)
        self.assertIn("19.0%",r["final"])
        self.assertEqual(r["final_validation"]["status"],"partial")
        self.assertEqual(r["remediation"]["status"],"unavailable")

    async def test_two_problems_have_separate_requirement_bindings(self):
        rs=[self.requirement("server-b","lookup",["cpu","memory.summary"],"두 지표 확인"),
            self.requirement("server-b","remediation",["cpu"],"CPU 대처"),
            self.requirement("server-b","remediation",["memory.summary"],"메모리 대처")]
        self.planner.scripts["plan"]=[{"tasks":[{"server_id":"server-b","goal":"상태 확인","information_ids":["cpu","memory.summary"]}],"requirements":rs}]
        r=await self.run_flow("서버 B CPU와 메모리 상태 및 각각의 대처방안")
        self.assertEqual(r["final_validation"]["status"],"passed")
        bindings={i for a in r["remediation"]["steps"] for i in a["requirement_ids"]}
        self.assertTrue({"r2","r3"}<=bindings)
        self.assertEqual(self.mcp.calls,[("server-b","top")])

    async def test_omitted_diagnosis_found_by_final_review_cannot_be_hidden_by_lookup_or_manual(self):
        from test_answer_completion import review,chk
        self.mcp.matches=[[match_for("logrotate")]]
        initial=[self.requirement("server-d","lookup",["disk"],"디스크 상태와 원인을 조사하고"),
                 self.requirement("server-d","remediation",["disk"],"해결 방법도 알려줘")]
        missing=[self.requirement("server-d","diagnosis",["disk"],"디스크 원인 조사")]
        self.planner.scripts["plan"]=[dict(tasks=[dict(server_id="server-d",goal="디스크 확인",information_ids=["disk"])],requirements=initial)]
        self.quality.scripts["final_review"]=[review(chk("r1"),chk("r2"),missing=missing,
            tasks=[dict(server_id="server-d",goal="로그 저장량 근거",information_ids=["log.usage"])]),
            review(chk("r1"),chk("r2"),chk("r3")),review(chk("r1"),chk("r2"),chk("r3"))]
        # Inject the deliberately incomplete plan directly. The graph's fake
        # intent adapter already identifies diagnosis and would mask this defect.
        from agent.investigation import investigate
        deps=SimpleNamespace(mcp=self.mcp,fast_llm=FastLLM("unused",0),quality_llm=self.quality,monitoring_llm=self.planner)
        result=await investigate(dict(query="서버 D 디스크 상태와 원인을 조사하고 해결 방법도 알려줘",
            history=[],events=[],cost_usd=0,route="investigate_remediate"),deps)
        self.assertTrue(any(r["kind"]=="diagnosis" for r in result["requirements"]))
        self.assertIn(("server-d","du"),self.mcp.calls)
        # Permissive subsequent judges cannot satisfy a new diagnosis with only
        # the old lookup plus manual. This fixture deliberately omits causes.
        self.assertNotEqual(result["final_validation"]["status"],"passed")
        self.assertEqual(result["remediation"]["status"],"provided")
        self.assertTrue(any(g["stage"]=="final_validation" and "누락" in g["reason"] for g in result["requirement_issues"]))

    async def test_new_missing_topic_uses_one_extra_search_without_resetting_commands_or_ids(self):
        from test_answer_completion import review,chk
        self.mcp.matches=[[match_for("kernel-cpu-control")],[match_for("logrotate")]]
        rs=[self.requirement("server-d","lookup",["cpu"],"CPU 상태"),self.requirement("server-d","remediation",["cpu"],"CPU 대처")]
        missing=[self.requirement("server-d","lookup",["disk"],"디스크 상태"),self.requirement("server-d","remediation",["disk"],"디스크 대처")]
        self.planner.scripts["plan"]=[{"tasks":[{"server_id":"server-d","goal":"CPU 확인","information_ids":["cpu"]}],"requirements":rs}]
        self.quality.scripts["final_review"]=[review(chk("r1"),chk("r2"),missing=missing,tasks=[{"server_id":"server-d","goal":"빠진 디스크 확인","information_ids":["disk"]}]),review(chk("r1"),chk("r2"),chk("r3"),chk("r4"))]
        r=await self.run_flow("서버 D CPU와 디스크 상태 및 각각의 대처방안")
        self.assertEqual(r["final_validation"]["status"],"passed")
        self.assertEqual(len(self.mcp.searches),2)
        self.assertEqual(self.mcp.calls,[("server-d","top"),("server-d","df")])
        self.assertEqual([q["requirement_id"] for q in r["requirements"]],["r1","r2","r3","r4"])
        self.assertEqual(r["workflow_budget"]["query_rounds"],2)
        self.assertEqual(r["workflow_budget"]["manual_searches"],2)
        self.assertTrue(any(x["stage"]=="final_validation" for x in r["requirement_issues"]))

    async def test_two_searches_retain_four_distinct_chunks_without_a_global_cap(self):
        from test_answer_completion import review,chk
        self.mcp.matches=[[match_for("kernel-cpu-control"),match_for("kernel-memory-control")],[match_for("logrotate"),match_for("kernel-cpu-control",section="cpu.idle")]]
        rs=[self.requirement("server-d","lookup",["cpu"],"CPU 상태"),self.requirement("server-d","remediation",["cpu"],"CPU 대처")]
        missing=[self.requirement("server-d","lookup",["disk"],"디스크 상태"),self.requirement("server-d","remediation",["disk"],"디스크 대처")]
        self.planner.scripts["plan"]=[{"tasks":[{"server_id":"server-d","goal":"CPU 확인","information_ids":["cpu"]}],"requirements":rs}]
        self.quality.scripts["final_review"]=[review(chk("r1"),chk("r2"),missing=missing,tasks=[{"server_id":"server-d","goal":"빠진 디스크 확인","information_ids":["disk"]}]),review(chk("r1"),chk("r2"),chk("r3"),chk("r4"))]
        r=await self.run_flow("서버 D CPU와 디스크 상태 및 각각의 대처방안")
        self.assertEqual(r["final_validation"]["status"],"passed")
        self.assertEqual(len(self.mcp.searches),2)
        self.assertEqual(self.mcp.calls,[("server-d","top"),("server-d","df")])
        self.assertEqual([q["requirement_id"] for q in r["requirements"]],["r1","r2","r3","r4"])
        self.assertEqual(r["workflow_budget"]["query_rounds"],2)
        self.assertEqual(r["workflow_budget"]["manual_searches"],2)
        self.assertTrue(any(x["stage"]=="final_validation" for x in r["requirement_issues"]))

        self.assertEqual(len(r["manual_state"]["matches"]),4)
        self.assertEqual(len({m["chunk_id"] for m in r["manual_state"]["matches"]}),4)

    async def test_bad_source_hash_never_publishes_steps_or_passes_even_if_judge_says_yes(self):
        bad=match_for("kernel-cpu-control");bad["sha256"]="wrong-hash";self.mcp.matches=[[bad]]
        r=await self.run_flow("서버 A CPU 상태와 대처방안")
        self.assertEqual(r["remediation"]["steps"],[])
        self.assertNotEqual(r["final_validation"]["status"],"passed")
        self.assertIn("19.0%",r["final"])

    async def test_repair_keeps_global_query_round_limit(self):
        from test_answer_completion import review,chk
        rs=[self.requirement("server-b","lookup",["cpu"],"CPU 상태"),self.requirement("server-b","remediation",["cpu"],"CPU 대처"),self.requirement("server-b","diagnosis",["cpu"],"CPU 원인 후보")]
        self.planner.scripts["plan"]=[{"tasks":[{"server_id":"server-b","goal":"CPU 확인","information_ids":["cpu"]}],"requirements":rs}]
        self.planner.scripts["review"]=[{"tasks":[{"server_id":"server-b","goal":"추가 프로세스","information_ids":["process"]}]},{"tasks":[{"server_id":"server-b","goal":"추가 로그","information_ids":["log.content"]}]}]
        self.quality.scripts["final_review"]=[review(chk("r1","unmet"),chk("r2"),chk("r3","unmet"),tasks=[{"server_id":"server-b","goal":"디스크 추가 확인","information_ids":["disk"]}])]*3
        r=await self.run_flow("서버 B CPU 문제 원인과 대처방안")
        self.assertEqual({c for _,c in self.mcp.calls},{"top","ps","tail"})
        self.assertEqual(r["workflow_budget"]["query_rounds"],3)
        self.assertEqual(r["workflow_budget"]["repair_rounds"],2)
        self.assertLessEqual(len(self.mcp.searches),2)
        self.assertEqual(r["final_validation"]["status"],"partial")

    async def test_invalid_final_review_returns_unverified_with_evidence_and_manual_intact(self):
        self.quality.scripts["final_review"]=["bad JSON","null"]
        r=await self.run_flow("서버 A CPU 상태와 대처방안")
        self.assertEqual(r["final_validation"]["status"],"unverified")
        self.assertEqual(r["remediation"]["status"],"provided")
        self.assertTrue(r["evidence"])
        self.assertIn("검증을 완료하지 못"," ".join(r["limitations"]))

    async def test_router_information_selection_is_reused_without_invented_plan_or_diagnosis(self):
        from test_answer_completion import Model
        class Scoped(Model):
            def response_json(self,purpose,payload,schema):
                if purpose=="intent":return json.dumps(intent_payload(route="investigate_remediate",style="detailed",reason="상태와 대처",requested_information_ids=["cpu"],manual_information_ids=["cpu"],needs_diagnosis=False,needs_calculation=False)),{"usd":0}
                return super().response_json(purpose,payload,schema)
        class NoExtraPlan(Model):
            def monitoring_json(self,*args):raise AssertionError("의도에서 선택한 정보로 충분합니다.")
        self.quality=Scoped();self.planner=NoExtraPlan()
        r=await self.run_flow("서버 B CPU 상태와 대처방안")
        self.assertEqual(r["route"],"investigate_remediate")
        self.assertEqual(self.mcp.calls,[("server-b","top")])
        self.assertFalse(any(q["kind"]=="diagnosis" for q in r["requirements"]))
        self.assertEqual([purpose for purpose,_ in self.quality.inputs],["final_review"])
        self.assertEqual(r["final_validation"]["status"],"passed")
        self.assertIn("final_review",r["workflow_timings_ms"])

    async def test_failed_final_format_cannot_be_reported_as_fully_satisfied(self):
        from test_answer_completion import Model
        class BadLayout(Model):
            def response_json(self,purpose,payload,schema):
                if purpose=="format":return '{"invalid":"invented layout"}',{"usd":0}
                return super().response_json(purpose,payload,schema)
        self.quality=BadLayout()
        r=await self.run_flow("서버 A CPU 상태와 대처방안")
        self.assertEqual(r["presentation"]["status"],"fallback")
        self.assertEqual(r["final_validation"]["status"],"partial")
        self.assertFalse(r["final_validation"]["request_fully_satisfied"])
        self.assertEqual(self.mcp.calls,[("server-a","top")])
        self.assertEqual(len(self.mcp.searches),1)

    async def test_shared_top_result_satisfies_cpu_and_memory_even_when_memory_is_planned_first(self):
        requirements=[self.requirement("server-b","lookup",["memory.summary","cpu"],"메모리와 CPU 확인"),self.requirement("server-b","remediation",["cpu"],"CPU 대처")]
        self.planner.scripts["plan"]=[{"tasks":[{"server_id":"server-b","goal":"두 지표 확인","information_ids":["memory.summary","cpu"]}],"requirements":requirements}]
        r=await self.run_flow("서버 B 메모리와 CPU를 확인하고 CPU 대처방안 알려줘")
        self.assertEqual(self.mcp.calls,[("server-b","top")])
        self.assertEqual(r["workflow_budget"]["command_calls"],1)
        self.assertEqual(r["final_validation"]["status"],"passed")

    async def test_typed_cpu_scope_is_preserved_without_additional_plan_or_observations(self):
        from test_answer_completion import Model
        class ScopedIntent(Model):
            def response_json(self,purpose,payload,schema):
                if purpose=="intent":return json.dumps(intent_payload(route="investigate_remediate",style="detailed",reason="CPU 상태와 CPU 대처만 요청",requested_information_ids=["cpu"],manual_information_ids=["cpu"],needs_diagnosis=False,needs_calculation=False)),{"usd":0}
                return super().response_json(purpose,payload,schema)
        self.quality=ScopedIntent()
        r=await self.run_flow("서버 B CPU 상태와 대처방안")
        self.assertEqual(self.mcp.calls,[("server-b","top")])
        self.assertTrue(all(q['information_ids']==['cpu'] for q in r['requirements']))
        self.assertEqual(r['final_validation']['status'],'passed')


    async def test_process_attribution_requires_process_measurement_not_start_log(self):
        from agent.investigation import verify_report
        from agent.investigation_models import InvestigationReport
        from monitoring.evidence import extract_facts
        from test_information_investigation import FixtureMCP
        mcp=FixtureMCP();evidence=[]
        for i,command in enumerate(["ps","tail"],1):
            item=await mcp.call_tool("get_command_result",{"server_id":"server-b","command":command})
            item.update(evidence_id=f"e{i}",facts=extract_facts(item),error=None);evidence.append(item)
        text="batch-worker 프로세스가 CPU를 과도하게 점유할 가능성이 있습니다."
        def report(eid,quote):return InvestigationReport(hypotheses=[dict(text=text,evidence_ids=[eid],quotes=[quote])])
        with self.assertRaises(ValueError):
            verify_report(report("e2",evidence[1]["stdout"].splitlines()[1]),evidence)
        verify_report(report("e1",evidence[0]["stdout"].splitlines()[1]),evidence)

    async def test_formatter_sampler_binds_heading_to_its_own_evidence_blocks(self):
        from agent.response_policy import model_json,Layout,fake_response,validate_layout
        blocks=[dict(block_id="a",section="근거",text="관측"),dict(block_id="b",section="원인",text="후보"),dict(block_id="c",section="미확인 사항",text="한계")]
        class Inspect:
            def response_json(self,purpose,payload,schema):
                groups=schema["properties"]["sections"]["items"]["anyOf"]
                self.groups={g["properties"]["title"]["enum"][0]:g["properties"]["block_ids"]["items"]["enum"] for g in groups}
                return json.dumps(fake_response(purpose,payload)),{"usd":0}
        model=Inspect()
        await model_json(model,"format",{"style":"detailed","blocks":blocks},Layout,lambda v:validate_layout(v,blocks,"detailed"))
        self.assertEqual(model.groups,{"근거":["a"],"원인":["b"],"미확인 사항":["c"]})


class APIResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_and_sse_expose_remediation_and_persist_followup_evidence(self):
        import httpx
        import server
        from agent.session_store import SqliteSessionStore
        from test_information_investigation import FixtureMCP
        class CombinedMCP(FixtureMCP):
            manual_calls=0
            async def call_tool(self,name,args):
                if name=="search_remediation_manuals":
                    self.manual_calls+=1
                    return {"matches":[match_for("kernel-cpu-control")],"timings_ms":{"embedding":1,"search":2}}
                return await super().call_tool(name,args)
        with tempfile.TemporaryDirectory() as tmp:
            store=SqliteSessionStore(Path(tmp)/"session.db");mcp=CombinedMCP();deps=SimpleNamespace(
                mcp=mcp,fast_llm=FastLLM("test",0),quality_llm=QualityLLM("test",0),session_store=store,version="v2")
            saved={k:getattr(server.app.state,k,None) for k in ["deps","graph","llm_backend","instance_id"]}
            server.app.state.deps=deps;server.app.state.graph=build_graph(deps);server.app.state.llm_backend="fake";server.app.state.instance_id=1
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url="http://test") as client:
                    a=await client.post("/chat",json={"message":"서버 B CPU 상태 확인","thread_id":"reuse"})
                    self.assertEqual(a.status_code,200);self.assertIsNone(a.json()["remediation"])
                    prior=len(mcp.calls)
                    b=await client.post("/chat",json={"message":"그 서버의 CPU 대처방안을 알려줘","thread_id":"reuse"})
                    self.assertEqual(b.json()["remediation"]["status"],"provided")
                    self.assertEqual(len(mcp.calls),prior);self.assertEqual(mcp.manual_calls,1)
                    c=await client.get("/chat/stream",params={"message":"Linux CPU 대응 방법","thread_id":"stream"})
                    events=[json.loads(line[6:]) for line in c.text.splitlines() if line.startswith('data: {')]
                    final=next(e for e in events if e["type"]=="FinalAnswer")
                    self.assertEqual(final["remediation"]["status"],"provided")
                    self.assertTrue(final["remediation"]["sources"])
            finally:
                store.close()
                for k,v in saved.items(): setattr(server.app.state,k,v)



class ScenarioSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_post_sse_share_private_numeric_request_trace(self):
        from unittest.mock import patch
        from langsmith.utils import tracing_is_enabled
        from agent.ollama_llm import trace_json_call
        class Recorder:
            def __init__(self):self.created=[];self.updated=[]
            def create_run(self, **data):self.created.append(data)
            def update_run(self, run_id, **data):self.updated.append((run_id,data))
        recorder=Recorder();original=self.server.app.state.graph
        class Graph:
            async def astream(self,state,stream_mode):
                assert not tracing_is_enabled(), 'Raw nested tracing must be disabled'
                trace_json_call('test', 'intent', {'private':'secret prompt'},
                    lambda:('secret model output',{'usd':.125,'input_tokens':10,'output_tokens':5}))
                yield 'updates', {'final':{'final':'secret answer','cost_usd':.125,'final_validation':{'status':'passed'}}}
        self.server.app.state.graph=Graph()
        try:
            with patch.object(self.server,'_metrics_client',return_value=recorder,create=True):
                post=(await self.post('secret question')).json()
                stream=await self.client.get('/chat/stream',params=dict(message='secret question',scenario_id='server-b',thread_id='trace'))
            final=[json.loads(x[6:]) for x in stream.text.splitlines() if x.startswith('data: {')][-1]
            roots=[r for r in recorder.created if r['name']=='chat_request']
            children=[r for r in recorder.created if r['name']!='chat_request']
            self.assertEqual(len(roots),2);self.assertEqual(len(children),2)
            ids={str(r['id']) for r in roots}
            self.assertEqual(ids,{post['metadata']['langsmith_run_id'],final['metadata']['langsmith_run_id']})
            self.assertEqual({str(r['parent_run_id']) for r in children},ids)
            self.assertEqual(len(recorder.updated),2)
            self.assertTrue(all(d['outputs']['metrics']['model_calls']==1 for _,d in recorder.updated))
            self.assertEqual(final['metadata']['cost_usd'],.125)
            self.assertEqual(final['metadata']['scenario_id'],'server-b')
            self.assertNotIn('secret',json.dumps([recorder.created,recorder.updated],default=str))
            self.assertTrue(all(r['inputs']=={} for r in recorder.created))
        finally:self.server.app.state.graph=original

    async def test_metrics_failure_never_breaks_answer_or_repeats_work(self):
        from unittest.mock import patch
        class Offline:
            def create_run(self, **kwargs):raise RuntimeError('telemetry offline')
        with patch.object(self.server,'_metrics_client',return_value=Offline(),create=True):
            response=(await self.post('top JSON만')).json()
        self.assertEqual(len(self.mcp.calls),1)
        self.assertIsNone(response['metadata']['langsmith_run_id'])
        self.assertEqual(json.loads(response['reply'])['command_id'],'top')

    async def test_request_trace_finalizes_cancellation_without_raw_error(self):
        import asyncio
        from unittest.mock import patch
        class Recorder:
            def __init__(self):self.updated=[]
            def create_run(self, **data):pass
            def update_run(self, run_id, **data):self.updated.append(data)
        recorder=Recorder()
        with patch.object(self.server,'_metrics_client',return_value=recorder,create=True):
            with self.assertRaises(asyncio.CancelledError):
                async with self.server._request_trace('sse'):
                    raise asyncio.CancelledError('secret cancellation content')
        self.assertEqual(len(recorder.updated),1)
        self.assertEqual(recorder.updated[0]['error'],'request_cancelled')
        self.assertNotIn('secret',json.dumps(recorder.updated,default=str))

    async def test_child_telemetry_failure_still_completes_root(self):
        from unittest.mock import patch
        from agent.ollama_llm import trace_json_call
        class Recorder:
            def __init__(self):self.updated=[]
            def create_run(self, **data):
                if data['name']!='chat_request':raise RuntimeError('offline child')
            def update_run(self, run_id, **data):self.updated.append(data)
        recorder=Recorder()
        with patch.object(self.server,'_metrics_client',return_value=recorder):
            async with self.server._request_trace('sse'):
                trace_json_call('test','intent',{},lambda:('{}',{'usd':.1}))
        self.assertEqual(len(recorder.updated),1)
        self.assertEqual(recorder.updated[0]['outputs']['metrics']['model_calls'],1)
        self.assertEqual(recorder.updated[0]['outputs']['metrics']['cost_usd'],.1)

    async def test_level_cancellation_still_finalizes_request_trace(self):
        import anyio
        from unittest.mock import patch
        from agent.ollama_llm import trace_json_call
        class Recorder:
            def __init__(self):self.updated=[]
            def create_run(self, **data):pass
            def update_run(self, run_id, **data):self.updated.append(data)
        recorder=Recorder()
        with patch.object(self.server,'_metrics_client',return_value=recorder):
            with anyio.CancelScope() as scope:
                async with self.server._request_trace('sse'):
                    trace_json_call('test','intent',{},lambda:('{}',{'usd':.1}))
                    scope.cancel()
                    await anyio.sleep(0)
        self.assertEqual(len(recorder.updated),1)
        self.assertEqual(recorder.updated[0]['error'],'request_cancelled')

    async def test_failed_post_keeps_completed_provider_cost_in_root(self):
        from unittest.mock import patch
        from agent.ollama_llm import trace_json_call
        class Recorder:
            def __init__(self):self.updated=[]
            def create_run(self, **data):pass
            def update_run(self, run_id, **data):self.updated.append(data)
        class Broken:
            async def astream(self,state,stream_mode):
                trace_json_call('test','intent',{},lambda:('{}',{'usd':.125}))
                raise RuntimeError('private provider detail')
                yield
        recorder=Recorder();prior=self.server.app.state.graph
        try:
            self.server.app.state.graph=Broken()
            with patch.object(self.server,'_metrics_client',return_value=recorder):
                with self.assertRaises(RuntimeError):await self.post('private request')
            self.assertEqual(recorder.updated[0]['outputs']['metrics']['cost_usd'],.125)
            self.assertEqual(recorder.updated[0]['error'],'request_failed')
            self.assertNotIn('private',json.dumps(recorder.updated,default=str))
        finally:self.server.app.state.graph=prior

    async def asyncSetUp(self):
        import httpx,server
        from agent.session_store import SqliteSessionStore
        from test_information_investigation import FixtureMCP
        class Manuals(FixtureMCP):
            def __init__(self):super().__init__();self.searches=0
            async def call_tool(self,name,args):
                if name=="search_remediation_manuals":
                    self.searches+=1
                    return {"matches":[match_for("kernel-cpu-control")],"timings_ms":{}}
                return await super().call_tool(name,args)
        self.server=server;self.tmp=tempfile.TemporaryDirectory();self.store=SqliteSessionStore(Path(self.tmp.name)/"session.db");self.mcp=Manuals()
        self.saved={k:getattr(server.app.state,k,None) for k in ["deps","graph","llm_backend","instance_id"]}
        self.deps=SimpleNamespace(mcp=self.mcp,fast_llm=FastLLM("test",0),quality_llm=QualityLLM("test",0),session_store=self.store,version="v2")
        server.app.state.deps=self.deps;server.app.state.graph=build_graph(self.deps);server.app.state.llm_backend="fake";server.app.state.instance_id=1
        self.client=httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url="http://test")
    async def asyncTearDown(self):
        await self.client.aclose();self.store.close();self.tmp.cleanup()
        for k,v in self.saved.items():setattr(self.server.app.state,k,v)
    async def post(self,message,scenario="server-b",thread="scenario-test"):
        return await self.client.post("/chat",json=dict(message=message,scenario_id=scenario,thread_id=thread))
    def status_quality(self, policy="auto"):
        class StatusQuality(QualityLLM):
            def __init__(self):
                super().__init__("status-regression", 0)
                self.inputs = []
            def response_json(self, purpose, payload, schema):
                self.inputs.append((purpose, payload))
                if purpose == "intent":
                    from agent.response_policy import explicit_information_scope
                    ids = sorted(explicit_information_scope(payload["query"]))
                    if "memory.summary" in ids:ids.remove("memory.detail")
                    ids = ids or ["cpu"]
                    chat = payload["query"] in {"안녕", "CPU 사용률이 뭐야?"}
                    return json.dumps(intent_payload(route="chat" if chat else "commands", style="brief",
                        reason="controlled intent", requested_information_ids=ids,
                        observation_policy="reuse" if payload["query"] == "새로 확인하지 말고 기존 결과로 다시 설명해줘" else policy,
                        continue_previous_task=payload["query"] == "새로 확인하지 말고 기존 결과로 다시 설명해줘",
                        basic_status=payload["query"] in {"상태 확인해줘", "사용량 확인해줘", "지금 어떤 상태야?"})), {"usd": 0}
                return super().response_json(purpose, payload, schema)
            def monitoring_json(self, purpose, payload, schema):
                self.inputs.append((purpose, payload))
                # The planner is deliberately not taught RAM aliases in this test.
                # A valid typed intent must reach initial lookup tasks unchanged.
                return super().monitoring_json(purpose, payload, schema)
        self.deps.quality_llm = StatusQuality()
        self.deps.monitoring_llm = self.deps.quality_llm
        return self.deps.quality_llm

    async def test_basic_status_requests_collect_three_metrics_without_rag(self):
        notice = "프로세스·네트워크·로그 등 다른 지표가 필요하면 따로 요청해주세요."
        for scenario in ["server-b", "server-g"]:
            for n, query in enumerate(["상태 확인해줘", "사용량 확인해줘", "지금 어떤 상태야?"]):
                with self.subTest(scenario=scenario, query=query):
                    model = self.status_quality()
                    self.mcp.calls.clear()
                    r = (await self.post(query, scenario, f"basic-{scenario}-{n}")).json()
                    self.assertTrue(any(p == "intent" for p, _ in model.inputs))
                    self.assertEqual(self.mcp.calls, [(scenario, "top"), (scenario, "df")])
                    self.assertEqual({i for t in r["tasks"] for i in t["information_ids"]},
                                     {"cpu", "memory.summary", "disk"})
                    self.assertEqual(r["final_validation"]["status"], "passed")
                    self.assertEqual(r["workflow_budget"]["command_calls"], 2)
                    self.assertEqual(self.mcp.searches, 0)
                    self.assertIn(notice, r["reply"])
                    reviews = [v for p, v in model.inputs if p == "final_review"]
                    self.assertTrue(reviews)
                    self.assertEqual(reviews[-1]["draft_answer"], r["reply"])
                    self.assertIn(notice, reviews[-1]["draft_answer"])

    async def test_selected_metric_aliases_do_not_expand_to_basic_status(self):
        for n, (query, expected, command) in enumerate([
            ("RAM 상태 확인해줘", {"memory.summary"}, "top"),
            ("RAM사용량 확인해줘", {"memory.summary"}, "top"),
            ("RAM은 사용량이 어떤 상태야?", {"memory.summary"}, "top"),
            ("swap 사용량 확인해줘", {"memory.detail"}, "vm_stat"),
            ("disk 상태 확인해줘", {"disk"}, "df")]):
            with self.subTest(query=query):
                model = self.status_quality()
                self.mcp.calls.clear()
                r = (await self.post(query, "server-g", f"alias-{n}")).json()
                self.assertTrue(any(p == "intent" for p, _ in model.inputs))
                self.assertEqual(self.mcp.calls, [("server-g", command)])
                self.assertEqual({i for t in r["tasks"] for i in t["information_ids"]}, expected)
                self.assertEqual(r["final_validation"]["status"], "passed")

    async def test_explicit_unsupported_periods_are_not_replaced_with_basic_snapshots(self):
        for n, (query, unsupported) in enumerate([
            ("지난 3시간 사용량 확인해줘", "history"),
            ("배포 이력 상태 확인해줘", "deployment")]):
            with self.subTest(query=query):
                self.status_quality()
                self.mcp.calls.clear()
                r = (await self.post(query, thread=f"unsupported-snapshot-{n}")).json()
                self.assertEqual(r["final_validation"]["status"], "partial")
                self.assertTrue(any(t["information_status"].get(unsupported) == "unsupported" for t in r["tasks"]))
                self.assertTrue(any(unsupported in q["information_ids"] and q["status"] == "unsupported" for q in r["requirements"]))
                self.assertNotIn(("server-b", "df"), self.mcp.calls)
                self.assertNotIn("다른 지표가 필요하면", r["reply"])

    async def test_basic_status_sse_uses_same_metrics_and_notice(self):
        self.status_quality()
        r = await self.client.get("/chat/stream", params=dict(message="상태 확인해줘",
            thread_id="basic-sse", scenario_id="server-g"))
        events = [json.loads(s[6:]) for s in r.text.splitlines() if s.startswith('data: {')]
        final = next(e for e in events if e["type"] == "FinalAnswer")
        self.assertEqual(self.mcp.calls, [("server-g", "top"), ("server-g", "df")])
        self.assertEqual(final["final_validation"]["status"], "passed")
        self.assertIn("다른 지표가 필요하면 따로 요청해주세요.", final["reply"])

    async def test_selected_scenario_still_classifies_concepts_and_greetings_without_tools(self):
        for query in ["안녕", "CPU 사용률이 뭐야?"]:
            model = self.status_quality()
            r = (await self.post(query, thread="concept-" + query)).json()
            self.assertTrue(any(p == "intent" for p, _ in model.inputs))
            self.assertEqual(self.mcp.calls, [])
            self.assertEqual(self.mcp.searches, 0)
            self.assertEqual(r["tasks"], [])

    async def test_past_no_lookup_declarations_do_not_block_explicit_refresh(self):
        queries = ["서버 B는 이전에는 조회하지 않았어. 지금 CPU 상태를 새로 측정해줘",
                   "서버 B 아직 조회 안 했으니 CPU 상태를 새로 측정해줘",
                   "서버 B는 조회하지 않은 상태야. CPU 상태를 새로 측정해줘"]
        for cached in [False, True]:
            for n, query in enumerate(queries):
                with self.subTest(cached=cached, query=query):
                    self.status_quality("refresh")
                    thread = f"past-{cached}-{n}"
                    if cached: await self.post("CPU 상태 확인", thread=thread)
                    self.mcp.calls.clear()
                    r = (await self.post(query, thread=thread)).json()
                    self.assertEqual(self.mcp.calls, [("server-b", "top")])
                    self.assertEqual(r["workflow_budget"]["command_calls"], 1)
                    self.assertEqual(r["final_validation"]["status"], "passed")
                    self.assertNotIn("새 조회를 하지 말라는 요청", r["reply"])

    async def test_model_reuse_policy_keeps_cached_scope(self):
        queries = ["CPU 상태를 새로 확인하지 마", "CPU는 조회하지 말고 기존 결과로 설명해줘",
                   "CPU는 측정 없이 기존 결과로 설명해줘", "CPU 조회 안 해줘",
                   "CPU는 조회하지 마시고 기존 결과로 설명해 주세요", "CPU 상태는 조회하지 마십시오",
                   "CPU는 조회하지 말아주시고 기존 결과로 설명해 주세요", "CPU 상태는 조회하지 말아주십시오"]
        for n, query in enumerate(queries):
            with self.subTest(query=query):
                self.status_quality("refresh")
                thread = f"prohibit-{n}"
                await self.post("CPU 상태 확인", thread=thread)
                self.status_quality("reuse")
                self.mcp.calls.clear()
                r = (await self.post(query, thread=thread)).json()
                self.assertEqual(self.mcp.calls, [])
                self.assertEqual(r["workflow_budget"]["command_calls"], 0)
                self.assertTrue(r["evidence"])
                self.assertEqual({i for t in r["tasks"] for i in t["information_ids"]}, {"cpu"})

    async def test_honorific_prohibitions_without_observations_never_fetch(self):
        for n, ending in enumerate(["마시고 기존 결과로 설명해 주세요", "마십시오",
                                    "말아주시고 기존 결과로 설명해 주세요", "말아주십시오"]):
            with self.subTest(ending=ending):
                self.status_quality("reuse")
                r = (await self.post("CPU 상태는 조회하지 " + ending, thread=f"no-history-prohibit-{n}")).json()
                self.assertEqual(self.mcp.calls, [])
                self.assertEqual(r["workflow_budget"]["command_calls"], 0)
                self.assertEqual(r["final_validation"]["status"], "partial")
                self.assertEqual(r["evidence"], [])

    async def test_explicit_cpu_measurement_does_not_run_extra_investigation(self):
        base = self.status_quality("refresh").__class__
        class SnapshotOnly(base):
            def monitoring_json(self, purpose, payload, schema):
                if purpose == "review":
                    self.inputs.append((purpose, payload))
                    return json.dumps(dict(tasks=[dict(server_id="server-b", goal="불필요한 추가 원인 조사",
                        information_ids=["process", "memory.detail", "log.content"])])), {"usd": 0}
                return super().monitoring_json(purpose, payload, schema)
        self.deps.quality_llm = SnapshotOnly()
        self.deps.monitoring_llm = self.deps.quality_llm
        r = (await self.post("서버 B 아직 조회 안 했으니 CPU 상태를 새로 측정해줘", thread="cpu-no-expansion")).json()
        self.assertEqual(self.mcp.calls, [("server-b", "top")])
        self.assertEqual(r["workflow_budget"]["command_calls"], 1)
        self.assertEqual(r["final_validation"]["status"], "passed")
        self.assertEqual({i for t in r["tasks"] for i in t["information_ids"]}, {"cpu"})
        self.assertFalse(any(p == "review" for p, _ in self.deps.quality_llm.inputs))

    async def test_lookup_scope_keeps_unfulfilled_explanation_requirements(self):
        for n, ids in enumerate([["cpu"], []]):
            with self.subTest(information_ids=ids):
                base = self.status_quality().__class__
                class MissingExplanation(base):
                    def monitoring_json(self, purpose, payload, schema):
                        if purpose == "final_review":
                            return json.dumps(dict(checks=[dict(requirement_id=r["requirement_id"],
                                status="unmet" if r["kind"] == "explanation" else "satisfied",
                                reason="사용률의 의미 설명 누락" if r["kind"] == "explanation" else "값 확인")
                                for r in payload["requirements"]], repair_tasks=[],
                                missing_requirements=[] if payload["repair_attempt"] else [dict(
                                    description="CPU 사용률의 의미 설명", server_ids=["server-b"],
                                    information_ids=ids, kind="explanation")], rewrite_instructions="의미 설명을 보완")), {"usd": 0}
                        return super().monitoring_json(purpose, payload, schema)
                self.deps.quality_llm = MissingExplanation()
                self.deps.monitoring_llm = self.deps.quality_llm
                self.mcp.calls.clear()
                r = (await self.post("CPU 상태 확인하고 사용률의 의미도 설명해줘", thread=f"explanation-{n}")).json()
                self.assertTrue(any(q["kind"] == "explanation" for q in r["requirements"]))
                self.assertFalse(r["final_validation"]["request_fully_satisfied"])
                self.assertEqual(r["final_validation"]["status"], "partial")
                self.assertEqual(self.mcp.calls, [("server-b", "top")])

    async def test_basic_status_repair_cannot_expand_scope_to_other_metrics(self):
        base = self.status_quality().__class__
        class NoisyReview(base):
            def monitoring_json(self, purpose, payload, schema):
                if purpose == "review":
                    self.inputs.append((purpose, payload))
                    raise AssertionError("A basic snapshot must not run an open-ended followup planner")
                if purpose == "final_review" and payload["repair_attempt"] == 0:
                    self.inputs.append((purpose, payload))
                    return json.dumps(dict(checks=[dict(requirement_id=r["requirement_id"],
                        status="unmet", reason="기본 세 지표의 설명을 보완") for r in payload["requirements"]],
                        repair_tasks=[dict(server_id="server-b", goal="범위 밖 로그 조사", information_ids=["log.content"])],
                        missing_requirements=[dict(description="범위 밖 메모리 상세", server_ids=["server-b"],
                            information_ids=["memory.detail"], kind="lookup")],
                        rewrite_instructions="기본 세 지표만 다시 설명")), {"usd": 0}
                return super().monitoring_json(purpose, payload, schema)
        self.deps.quality_llm = NoisyReview()
        self.deps.monitoring_llm = self.deps.quality_llm
        r = (await self.post("상태 확인해줘", thread="bounded-repair")).json()
        self.assertEqual(self.mcp.calls, [("server-b", "top"), ("server-b", "df")])
        self.assertEqual(r["workflow_budget"]["repair_rounds"], 1)
        self.assertEqual(r["final_validation"]["status"], "passed")
        self.assertEqual({i for t in r["tasks"] for i in t["information_ids"]}, {"cpu", "memory.summary", "disk"})
        self.assertEqual({i for q in r["requirements"] for i in q["information_ids"]}, {"cpu", "memory.summary", "disk"})
        self.assertFalse(any(p == "review" for p, _ in self.deps.quality_llm.inputs))

    async def test_basic_lookup_defaults_do_not_change_raw_command_or_reexplanation(self):
        self.status_quality()
        first = (await self.post("CPU 상태 확인", thread="no-expansion")).json()
        self.mcp.calls.clear()
        r = (await self.post("새로 확인하지 말고 기존 결과로 다시 설명해줘", thread="no-expansion")).json()
        self.assertEqual(self.mcp.calls, [])
        self.assertEqual({i for t in r["tasks"] for i in t["information_ids"]}, {"cpu"})
        self.assertEqual(r["evidence"], first["evidence"])
        self.mcp.calls.clear()
        raw = (await self.post("top JSON만", thread="raw-no-default")).json()
        self.assertEqual(self.mcp.calls, [("server-b", "top")])
        self.assertEqual(json.loads(raw["reply"])["command_id"], "top")
        self.assertNotIn("다른 지표가 필요하면", raw["reply"])

    async def test_unnamed_api_request_uses_default_resource_state(self):
        r=(await self.client.post("/chat",json=dict(message="CPU 상태 확인해줘",thread_id="single-default"))).json()
        self.assertEqual(r["scenario_id"],"server-a")
        self.assertEqual(self.mcp.calls,[("server-a","top")])
        self.assertIn("19.0%",r["reply"])
        self.assertIn("모니터링 서버",r["reply"])
        self.assertNotIn("어느 서버",r["reply"])
        self.assertNotIn("server-a",r["reply"])

    async def test_selected_states_use_one_server_without_name_questions(self):
        for scenario,value in [("server-a","19.0%"),("server-b","96.0%"),("server-g","25.0%")]:
            with self.subTest(scenario=scenario):
                self.mcp.calls.clear()
                r=(await self.post("CPU 상태 확인해줘",scenario,"one-host-"+scenario)).json()
                self.assertEqual(self.mcp.calls,[(scenario,"top")])
                self.assertIn(value,r["reply"])
                self.assertIn("모니터링 서버",r["reply"])
                self.assertNotIn("어느 서버",r["reply"])
                self.assertNotIn(scenario,r["reply"])
                self.assertEqual(r["final_validation"]["status"],"passed")

    async def test_default_state_is_also_returned_in_sse(self):
        r=await self.client.get("/chat/stream",params=dict(message="CPU 상태 확인해줘",thread_id="single-default-sse"))
        events=[json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: {")]
        final=next(e for e in events if e["type"]=="FinalAnswer")
        self.assertEqual(final["scenario_id"],"server-a")
        self.assertIn("19.0%",final["reply"])
        self.assertNotIn("어느 서버",final["reply"])

    async def test_single_server_intent_has_no_legacy_service_targets(self):
        model=self.status_quality()
        await self.post("CPU 상태 확인해줘")
        intent=next(payload for purpose,payload in model.inputs if purpose=="intent")
        self.assertTrue(intent["single_server_mode"])
        self.assertEqual(intent["legacy_systems"],[])
        self.assertEqual(intent["selected_scenario"],"server-b")

    async def test_switching_state_cannot_reuse_another_state_observation(self):
        await self.post("CPU 상태 확인해줘","server-a","single-switch")
        before=len(self.mcp.calls)
        r=(await self.post("기존 결과로 다시 설명해줘","server-b","single-switch")).json()
        self.assertEqual(len(self.mcp.calls),before)
        self.assertEqual(r["evidence"],[])
        self.assertNotIn("19.0%",r["reply"])
        self.assertNotIn("어느 서버",r["reply"])

    async def test_final_validation_receives_selected_saved_state_policy(self):
        class SnapshotReviewer(QualityLLM):
            reviews=[]
            def monitoring_json(self,purpose,payload,schema):
                if purpose=="final_review":self.reviews.append(payload)
                return super().monitoring_json(purpose,payload,schema)
        model=SnapshotReviewer("test",0);self.deps.quality_llm=model;self.deps.monitoring_llm=model
        r=(await self.post("CPU 상태 확인해줘")).json()
        self.assertEqual(len(model.reviews),1)
        context=model.reviews[0]["observation_context"]
        self.assertEqual(context["source"],"fixture")
        self.assertEqual(context["current_scope"],"selected_saved_state")
        self.assertFalse(context["live_measurement_supported"])
        self.assertEqual(model.reviews[0]["resource_state_id"],"server-b")
        self.assertIn("실시간 측정값은 아닙니다",r["reply"])
        self.assertEqual(r["final_validation"]["status"],"passed")

    async def test_verified_hostname_in_uname_output_is_preserved(self):
        r=(await self.post("운영체제와 커널 아키텍처 확인해줘")).json()
        original=json.loads((ROOT/"data/command_scenarios/server-b.json").read_text())["commands"]["uname"]["stdout"].strip()
        self.assertIn(original,r["reply"])
        self.assertIn("모니터링 서버 · uname",r["reply"])
        self.assertEqual(r["final_validation"]["status"],"passed")

    async def test_intent_failure_does_not_ask_for_a_server(self):
        class InvalidIntent(QualityLLM):
            def response_json(self,*args,**kwargs):raise ValueError("invalid intent")
        self.deps.quality_llm=InvalidIntent("test",0)
        r=(await self.post("CPU 상태 확인해줘")).json()
        self.assertEqual(self.mcp.calls,[])
        self.assertIn("요청 의도",r["reply"])
        self.assertNotIn("확인할 서버",r["reply"])
        self.assertNotIn("어느 서버",r["reply"])

    async def test_scenario_labels_describe_resource_states(self):
        rows=(await self.client.get("/monitoring/scenarios")).json()
        names={r["scenario_id"]:r["display_name"] for r in rows}
        self.assertIn("정상",names["server-a"])
        self.assertIn("CPU",names["server-b"])
        self.assertIn("메모리",names["server-c"])
        self.assertTrue(all("서버" not in name for name in names.values()))

    async def test_scenario_inventory_has_only_public_selection_metadata(self):
        r=await self.client.get("/monitoring/scenarios");self.assertEqual(r.status_code,200)
        rows=r.json();self.assertEqual({r["scenario_id"] for r in rows},{"server-"+c for c in "abcdefgh"})
        self.assertTrue(all(set(r)=={"scenario_id","display_name","os"} for r in rows))
    async def test_selected_resource_state_wins_over_server_like_query_labels(self):
        r=await self.post("CPU 상태 확인");self.assertEqual(r.status_code,200);b=r.json()
        self.assertEqual(b["scenario_id"],"server-b");self.assertEqual(self.mcp.calls,[("server-b","top")]);self.assertIn("96.0%",b["reply"])
        for query in ["서버 A CPU 상태 확인","시나리오 A CPU 상태 확인","scenario-a CPU 상태 확인"]:
            self.mcp.calls.clear();r=await self.post(query)
            self.assertEqual(self.mcp.calls,[("server-b","top")])
            self.assertIn("96.0%",r.json()["reply"])
            self.assertNotIn("어느 서버",r.json()["reply"])
        self.assertEqual((await self.post("CPU 상태 확인","server-z")).status_code,404)
    async def test_scenario_sessions_are_isolated_for_chat_and_sse(self):
        a=(await self.post("CPU 상태 확인","server-a","same")).json()
        b=(await self.post("CPU 상태와 대처방안 알려줘","server-b","same")).json()
        self.assertEqual([e["server_id"] for e in b["evidence"]],["server-b"])
        before=len(self.mcp.calls)
        for scenario in ["server-a","server-b"]:
            r=await self.client.get("/chat/stream",params=dict(message="그 서버 CPU 대처방안",thread_id="same",scenario_id=scenario))
            events=[json.loads(s[6:]) for s in r.text.splitlines() if s.startswith('data: {')]
            final=next(e for e in events if e["type"]=="FinalAnswer")
            self.assertEqual(final["scenario_id"],scenario);self.assertTrue(final["evidence"])
            self.assertTrue(all(e["server_id"]==scenario for e in final["evidence"]))
            self.assertEqual(final["workflow_budget"]["command_calls"],0)
        self.assertEqual(len(self.mcp.calls),before)
        self.assertEqual(a["evidence"][0]["server_id"],"server-a")
    async def test_short_followup_keeps_previous_observations_and_manual_route(self):
        first=(await self.post("CPU 상태와 대처방안 알려줘")).json()
        self.assertEqual(first["final_validation"]["status"],"passed");before=len(self.mcp.calls)
        for message in ["다시 설명해줘","기존 결과만 사용해서 다시 설명해줘"]:
            r=(await self.post(message)).json()
            self.assertEqual(r["metadata"]["route"],"investigate_remediate")
            self.assertEqual(r["evidence"],first["evidence"]);self.assertEqual(r["workflow_budget"]["command_calls"],0)
            self.assertEqual(r["final_validation"]["status"],"passed")
        self.assertEqual(len(self.mcp.calls),before)

    async def test_model_marks_reference_to_keep_source_route(self):
        from test_answer_completion import Model
        class WrongChat(Model):
            def response_json(self,purpose,payload,schema):
                if purpose=="intent":return json.dumps(intent_payload(route="investigate_remediate",style="brief",reason="repeat",observation_policy="reuse",continue_previous_task=True)),{"usd":0}
                return super().response_json(purpose,payload,schema)
        await self.post("CPU 상태와 대처방안 알려줘")
        self.deps.quality_llm=WrongChat();before=len(self.mcp.calls)
        r=(await self.post("새로 확인하지 말고 기존 결과로 다시 설명해줘.")).json()
        self.assertEqual(r["metadata"]["route"],"investigate_remediate");self.assertTrue(r["evidence"])
        self.assertEqual(r["final_validation"]["status"],"passed");self.assertEqual(len(self.mcp.calls),before)

    async def test_bare_reference_in_legacy_api_keeps_its_single_target(self):
        first=(await self.client.post("/chat",json=dict(message="서버 B CPU 상태와 대처방안 알려줘",thread_id="legacy-reference"))).json()
        before=len(self.mcp.calls)
        second=(await self.client.post("/chat",json=dict(message="다시 설명해줘",thread_id="legacy-reference"))).json()
        self.assertEqual(second["metadata"]["route"],"investigate_remediate")
        self.assertEqual(second["evidence"],first["evidence"]);self.assertEqual(second["final_validation"]["status"],"passed")
        self.assertEqual(len(self.mcp.calls),before)

    async def test_short_followup_after_a_concept_does_not_revive_old_observation(self):
        await self.post("CPU 상태와 대처방안 알려줘")
        await self.post("메모리와 디스크의 차이가 뭐야?")
        before=len(self.mcp.calls);searches=self.mcp.searches
        r=(await self.post("다시 설명해줘")).json()
        self.assertEqual(r["metadata"]["route"],"chat");self.assertEqual(r["evidence"],[])
        self.assertEqual(len(self.mcp.calls),before);self.assertEqual(self.mcp.searches,searches)

    async def test_selected_scenario_does_not_add_tools_to_concepts_or_generic_manuals(self):
        r=(await self.post("CPU 사용률이 뭐야?")).json();self.assertEqual(r["metadata"]["route"],"chat");self.assertEqual(self.mcp.calls,[]);self.assertEqual(self.mcp.searches,0)
        r=(await self.post("Linux CPU 과부하 대처방안")).json();self.assertEqual(r["metadata"]["route"],"remediation");self.assertEqual(self.mcp.calls,[]);self.assertEqual(self.mcp.searches,1)
    async def test_selected_scenario_raw_json_preserves_original_output(self):
        r=(await self.post("top JSON만")).json();raw=json.loads(r["reply"])
        self.assertEqual(raw["server_id"],"server-b");self.assertEqual(raw["stdout"],json.loads((ROOT/"data/command_scenarios/server-b.json").read_text())["commands"]["top"]["stdout"])
        self.assertEqual(self.mcp.searches,0);self.assertEqual(r["presentation"]["status"],"raw")
    async def test_parallel_scenarios_with_same_thread_keep_distinct_sources(self):
        import asyncio
        a,b=await asyncio.gather(self.post("CPU 상태 확인","server-a","parallel"),self.post("CPU 상태 확인","server-b","parallel"))
        for response,scenario,value in [(a,"server-a","19.0%"),(b,"server-b","96.0%")]:
            body=response.json();self.assertEqual(body["scenario_id"],scenario)
            self.assertTrue(all(e["server_id"]==scenario for e in body["evidence"]))
            self.assertIn(value,body["reply"])

    async def test_proxy_forwards_scenario_to_both_api_forms(self):
        import httpx,web_demo_server
        from unittest.mock import patch
        original=httpx.AsyncClient
        def upstream(**kwargs):return original(transport=httpx.ASGITransport(app=self.server.app),**kwargs)
        async with original(transport=httpx.ASGITransport(app=web_demo_server.app),base_url="http://proxy") as proxy:
            with patch.object(web_demo_server.httpx,"AsyncClient",side_effect=upstream):
                self.assertEqual((await proxy.get("/monitoring/scenarios")).status_code,200)
                r=await proxy.get("/chat/stream",params=dict(message="top JSON만",thread_id="proxy",scenario_id="server-b"))
                events=[json.loads(s[6:]) for s in r.text.splitlines() if s.startswith('data: {')]
                f=next(e for e in events if e["type"]=="FinalAnswer");self.assertEqual(json.loads(f["reply"])["server_id"],"server-b")

class ReviewRegressionTests(unittest.IsolatedAsyncioTestCase):
    setUp=CombinedWorkflowTests.setUp
    run_flow=CombinedWorkflowTests.run_flow
    cached_context=CombinedWorkflowTests.cached_context
    requirement=CombinedWorkflowTests.requirement
    # Reuse fixture setup only; inherited tests remain part of the base class.
    async def test_unsupported_and_unclassified_manual_cannot_pass_a_permissive_judge(self):
        from agent.remediation import manual_requirement_covered
        supported=await self.run_flow("Linux CPU 대처방안")
        manual=supported["remediation"]
        for ids in [["network"],["process"],["log.content"],[]]:
            self.assertFalse(manual_requirement_covered(dict(requirement_id="r1",server_ids=[],information_ids=ids),manual,[]))
        from test_answer_completion import Model
        class Network(Model):
            def response_json(self,purpose,payload,schema):
                if purpose=="intent":return json.dumps(intent_payload(route="remediation",style="detailed",reason="network",requested_information_ids=[],manual_information_ids=["network"],needs_diagnosis=False,needs_calculation=False)),{"usd":0}
                return super().response_json(purpose,payload,schema)
        self.quality=Network();self.mcp.searches.clear()
        r=await self.run_flow("Linux 네트워크 장애 대처방안")
        self.assertNotEqual(r["final_validation"]["status"],"passed");self.assertEqual(r["remediation"]["steps"],[]);self.assertEqual(self.mcp.searches,[])
    async def test_unclassified_manual_does_not_fall_back_to_an_incidental_cpu_mention(self):
        from test_answer_completion import Model
        class Unclassified(Model):
            def response_json(self,purpose,payload,schema):
                if purpose=="intent":return json.dumps(intent_payload(route="remediation",style="detailed",reason="unsupported DB",requested_information_ids=[],manual_information_ids=[],needs_diagnosis=False,needs_calculation=False)),{"usd":0}
                return super().response_json(purpose,payload,schema)
        self.quality=Unclassified();r=await self.run_flow("Linux CPU는 정상이고 DB 교착 대처방안을 알려줘")
        # Contradictory typed intent (manual route without any manual topic) is
        # rejected before tool access; no regex silently invents a CPU topic.
        self.assertEqual(r["route"],"chat")
        self.assertEqual(self.mcp.searches,[])
        self.assertEqual(self.mcp.calls,[])
        self.assertIn("요청 의도",r["final"])

    async def test_supported_manual_survives_mixed_unsupported_topic(self):
        from test_answer_completion import Model
        class Mixed(Model):
            def response_json(self,purpose,payload,schema):
                if purpose=="intent":return json.dumps(intent_payload(route="remediation",style="detailed",reason="mixed",requested_information_ids=[],manual_information_ids=["cpu","network"],needs_diagnosis=False,needs_calculation=False)),{"usd":0}
                return super().response_json(purpose,payload,schema)
        self.quality=Mixed();r=await self.run_flow("Linux CPU와 네트워크 장애 대처방안")
        self.assertEqual(r["final_validation"]["status"],"partial");self.assertTrue(r["remediation"]["steps"])
        self.assertTrue(any(q["information_ids"]==["network"] and q["status"]=="unsupported" for q in r["requirements"]))
        self.assertTrue(all(a["requirement_ids"]==["r1"] for a in r["remediation"]["steps"]))
    async def test_supported_cpu_and_unknown_db_topics_are_both_preserved(self):
        r=await self.run_flow("Linux CPU와 DB 교착 대처방안")
        self.assertEqual(r["final_validation"]["status"],"partial");self.assertTrue(r["remediation"]["steps"])
        self.assertTrue(any("DB" in q["description"] and q["status"]=="unsupported" for q in r["requirements"]))

    async def test_explicit_command_keeps_supported_manual_in_a_mixed_request(self):
        r=await self.run_flow("서버 B top CPU와 DB 교착 대처방안")
        self.assertEqual(self.mcp.calls,[("server-b","top")]);self.assertEqual(r["final_validation"]["status"],"partial")
        self.assertTrue(r["remediation"]["steps"])
        self.assertTrue(any(q["kind"]=="remediation" and q["information_ids"]==["cpu"] and q["status"]=="satisfied" for q in r["requirements"]))
        self.assertTrue(any("DB" in q["description"] and q["status"]=="unsupported" for q in r["requirements"]))

    async def test_model_reuse_policy_never_refreshes(self):
        from test_answer_completion import Model
        context=await self.cached_context()
        class Refresh(Model):
            def response_json(self,purpose,payload,schema):
                if purpose=="intent":return json.dumps(intent_payload(route="investigate_remediate",style="detailed",reason="reuse",requested_information_ids=["cpu"],manual_information_ids=["cpu"],needs_diagnosis=False,needs_calculation=False,observation_policy="reuse")),{"usd":0}
                return super().response_json(purpose,payload,schema)
        self.quality=Refresh()
        r=await self.run_flow("그 서버 CPU 대처방안. 새로 확인하지 말고 기존 결과를 사용해줘",context,[dict(role="user",content="서버 A CPU 상태")])
        self.assertEqual(self.mcp.calls,[]);self.assertEqual(r["workflow_budget"]["command_calls"],0)
    async def test_reuse_without_observation_is_conditional_and_partial(self):
        r=await self.run_flow("서버 A CPU 대처방안. 새로 확인하지 말고 기존 결과만 사용해줘")
        self.assertEqual(self.mcp.calls,[]);self.assertEqual(r["final_validation"]["status"],"partial")
        self.assertEqual(r["remediation"]["status"],"provided");self.assertTrue(all(a["conditional"] for a in r["remediation"]["steps"]))
    async def test_cached_hypotheses_cannot_cross_scenario(self):
        context=await self.cached_context()
        context["hypotheses"]=[dict(text="서버 B의 CPU 부하가 높을 가능성이 있습니다",evidence_ids=["e99"],quotes=["93.0 us"])]
        r=await self.run_flow("서버 A CPU 대처방안",context)
        self.assertEqual(r["hypotheses"],[]);self.assertNotIn("서버 B의 CPU",r["final"])


if __name__=="__main__": unittest.main()


class ManualTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_timeout_preserves_prior_observation(self):
        class BrokenMCP(StubMCP):
            async def call_tool(self, name, args):
                if name == "search_remediation_manuals":
                    raise TimeoutError("private-provider-payload")
                return await super().call_tool(name, args)
        observation = dict(server_id="server-b", observed_at="fixture", facts=[dict(summary="CPU 96%")])
        result = await remediation_node(dict(query="서버 B CPU 대처 방법", events=[],
            investigation_context=dict(evidence=[observation], query="CPU 상태")),
            SimpleNamespace(mcp=BrokenMCP(), fast_llm=FastLLM("test", 0), quality_llm=QualityLLM("test", 0)))
        self.assertEqual(result["evidence"], [observation])
        self.assertEqual(result["remediation"]["status"], "unavailable")
        self.assertNotIn("private-provider-payload", json.dumps(result))


class RequestDeadlineTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = ScenarioSelectionTests.asyncSetUp
    asyncTearDown = ScenarioSelectionTests.asyncTearDown
    status_quality = ScenarioSelectionTests.status_quality

    async def test_post_and_sse_timeout_preserve_real_graph_observations(self):
        from time import perf_counter
        self.status_quality()
        original = self.mcp.call_tool
        cancelled = []
        async def slow(name, args):
            if name == "get_command_result" and args["command"] == "ps":
                self.mcp.calls.append((args["server_id"], "ps"))
                try: await asyncio.sleep(1)
                except asyncio.CancelledError:
                    cancelled.append("ps"); raise
            return await original(name, args)
        self.mcp.call_tool = slow
        for streaming in (False, True):
            self.mcp.calls.clear()
            started = perf_counter()
            with patch.dict(os.environ, {"WORKFLOW_TIMEOUT_SECONDS":"0.08"}):
                params=dict(message="CPU와 프로세스 상태 확인",scenario_id="server-a",thread_id="deadline-"+str(streaming))
                if streaming:
                    response=await self.client.get("/chat/stream",params=params)
                    rows=[json.loads(x[6:]) for x in response.text.splitlines() if x.startswith("data: ") and x!="data: [DONE]"]
                    result=next(x for x in rows if x.get("type")=="FinalAnswer")
                    self.assertIn("data: [DONE]",response.text)
                else:
                    response=await self.client.post("/chat",json=params);result=response.json()
            self.assertEqual(response.status_code,200)
            self.assertLess(perf_counter()-started,.8)
            self.assertEqual(result["final_validation"]["status"],"unverified")
            self.assertTrue(result["evidence"][0]["facts"])
            self.assertIn("19.0%",result["reply"])
            self.assertIn("시간 제한",result["reply"])
            self.assertEqual(self.mcp.calls,[("server-a","top"),("server-a","ps")])
        self.assertEqual(cancelled,["ps","ps"])

    async def test_concurrent_deadline_snapshots_and_cancellation_are_isolated(self):
        from agent.deps import run_graph, save_progress
        class Graph:
            async def astream(self,state,stream_mode):
                save_progress({"limitations":[state["query"]]})
                await asyncio.sleep(state["delay"])
                yield "updates", {"answer":{"final":state["query"]}}
        slow,fast=await asyncio.gather(run_graph(Graph(),{"query":"slow","delay":1},timeout_seconds=.02),
                                       run_graph(Graph(),{"query":"fast","delay":0},timeout_seconds=1))
        self.assertEqual(fast["final"],"fast")
        self.assertEqual(slow["final_validation"]["status"],"unverified")
        self.assertIn("slow",slow["limitations"])
        self.assertNotIn("fast",slow["limitations"])
        task=asyncio.create_task(run_graph(Graph(),{"query":"cancel","delay":1},timeout_seconds=1))
        await asyncio.sleep(.01);task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task


    async def test_slow_sse_consumer_does_not_receive_deadline_cancellation(self):
        from agent.deps import stream_graph
        cancelled=[]
        class Graph:
            async def astream(self,state,stream_mode):
                yield "custom", {"type":"Started"}
                try:await asyncio.sleep(1)
                except asyncio.CancelledError:cancelled.append(True);raise
        stream=stream_graph(Graph(),{"query":"slow browser"},timeout_seconds=.02)
        try:
            first=await anext(stream)
            self.assertEqual(first[0],"custom")
            await asyncio.sleep(.05)
            self.assertEqual(cancelled,[True])
            rest=[item async for item in stream]
            result=rest[-1][1]["deadline"]
            self.assertEqual(result["final_validation"]["status"],"unverified")
        finally:await stream.aclose()


    async def test_timeout_during_format_keeps_checked_answer(self):
        from agent.deps import run_graph
        from agent.investigation import investigate
        from test_answer_completion import Model
        from test_information_investigation import FixtureMCP
        answer=dict(text="시스템 CPU는 19%입니다",requirement_ids=["r1"],evidence_ids=["e1"],quotes=["16.0 us"])
        model=Model(report=[dict(hypotheses=[],answers=[answer])])
        deps=SimpleNamespace(mcp=FixtureMCP(),quality_llm=model,fast_llm=Model(),monitoring_llm=Model())
        entered=[]
        async def slow_format(state,deps):
            entered.extend(state["answers"])
            await asyncio.sleep(1)
        class Graph:
            async def astream(self,state,stream_mode):
                yield "updates", {"investigate":await investigate(state,deps)}
        state=dict(query="현재 CPU를 설명해줘",scenario_id="server-a",single_server_mode=True,events=[],
            request_decision=dict(requested_information_ids=["cpu"],needs_explanation=True,
                requested_results=[dict(description="현재 CPU 설명",kind="explanation",information_ids=["cpu"])]))
        with patch("agent.response_policy.present_response",side_effect=slow_format):
            result=await run_graph(Graph(),state,timeout_seconds=.05)
        self.assertTrue(entered)
        self.assertIn(answer["text"],result["final"])
        self.assertTrue(result["answers"])
        self.assertEqual(result["final_validation"]["status"],"unverified")

    async def test_timeout_keeps_verified_manual_text_and_source(self):
        from agent.deps import run_graph
        deps=SimpleNamespace(mcp=StubMCP([match_for("kernel-cpu-control")]),
            fast_llm=FastLLM("test",0),quality_llm=QualityLLM("test",0))
        completed=[]
        class Graph:
            async def astream(self,state,stream_mode):
                result=await remediation_node(state,deps)
                completed.append(result)
                await asyncio.sleep(1)
                yield "updates", {"manual":result}
        result=await run_graph(Graph(),dict(query="Linux CPU 대응 방법",events=[]),timeout_seconds=.05)
        self.assertEqual(completed[0]["remediation"]["status"],"provided")
        step=completed[0]["remediation"]["steps"][0]
        self.assertIn(step["text"],result["final"])
        self.assertIn(completed[0]["remediation"]["sources"][0]["title"],result["final"])
        self.assertEqual(result["final_validation"]["status"],"unverified")

    async def test_upstream_cancellation_is_not_swallowed_as_empty_success(self):
        from agent.deps import run_graph
        class Graph:
            async def astream(self,state,stream_mode):
                raise asyncio.CancelledError("upstream cancelled")
                yield
        with self.assertRaises(asyncio.CancelledError):
            await run_graph(Graph(),{"query":"cancel"},timeout_seconds=1)


    async def test_timeout_in_second_specialist_keeps_first_checked_result(self):
        from agent.deps import run_graph
        from agent.investigation import investigate
        from test_answer_completion import Model
        from test_information_investigation import FixtureMCP
        import time
        class Expert(Model):
            def diagnostic_json(self,payload,schema):
                if payload["role"]=="logs":time.sleep(.2)
                eid=payload["observations"][0]["evidence_id"]
                return json.dumps(dict(diagnoses=[dict(text="메트릭 전문 판단으로 높은 CPU 전제가 관측되지 않았습니다",
                    requirement_ids=["r1"],evidence_ids=[eid],quote_ids=[eid+":l3"],symptom_observed=False)],tasks=[])),{"usd":0}
        deps=SimpleNamespace(mcp=FixtureMCP(),quality_llm=Expert(),fast_llm=Model(),monitoring_llm=Model())
        class Graph:
            async def astream(self,state,stream_mode):
                yield "updates", {"investigate":await investigate(state,deps)}
        state=dict(query="CPU와 로그 원인을 설명해줘",scenario_id="server-a",single_server_mode=True,events=[],
            request_decision=dict(requested_information_ids=["cpu","log.content"],needs_diagnosis=True,
                requested_results=[dict(description="두 영역 원인 설명",kind="diagnosis",information_ids=["cpu","log.content"])]))
        result=await run_graph(Graph(),state,timeout_seconds=.05)
        self.assertEqual(result["final_validation"]["status"],"unverified")
        self.assertIn("메트릭 전문 판단",result["final"])
        self.assertEqual(len(result["answers"]),1)


    async def test_repair_timeout_preserves_unchanged_role_cached_result(self):
        from agent.deps import run_graph
        from agent.investigation import investigate
        from test_answer_completion import Model, review, chk, task
        from test_information_investigation import FixtureMCP
        import time
        class Expert(Model):
            def __init__(self):
                super().__init__(final_review=[review(chk("r1","unmet"),tasks=[task("server-a","process")])])
                self.roles=[]
            def diagnostic_json(self,payload,schema):
                role=payload["role"];self.roles.append(role)
                if role=="metrics" and self.roles.count(role)==2:time.sleep(.25)
                eid=payload["observations"][0]["evidence_id"]
                return json.dumps(dict(diagnoses=[dict(text=role+" 전문 판단에 따르면 높은 부하의 전제가 관측되지 않았습니다",
                    requirement_ids=["r1"],evidence_ids=[eid],quote_ids=[eid+(":l3" if role=="metrics" else ":l1")],symptom_observed=False)],tasks=[])),{"usd":0}
        model=Expert();deps=SimpleNamespace(mcp=FixtureMCP(),quality_llm=model,fast_llm=Model(),monitoring_llm=Model())
        class Graph:
            async def astream(self,state,stream_mode):
                yield "updates", {"investigate":await investigate(state,deps)}
        state=dict(query="CPU와 로그 원인을 설명해줘",scenario_id="server-a",single_server_mode=True,events=[],
            request_decision=dict(requested_information_ids=["cpu","log.content"],needs_diagnosis=True,
                requested_results=[dict(description="두 영역 원인 설명",kind="diagnosis",information_ids=["cpu","log.content"])]))
        result=await run_graph(Graph(),state,timeout_seconds=.1)
        self.assertEqual(model.roles,["metrics","logs","metrics"])
        self.assertEqual(result["final_validation"]["status"],"unverified")
        self.assertIn("logs 전문 판단",result["final"])
        self.assertTrue(any("logs 전문 판단" in a["text"] for a in result["answers"]))


class ManualSpecialistTests(unittest.IsolatedAsyncioTestCase):
    setUp=CombinedWorkflowTests.setUp
    run_flow=CombinedWorkflowTests.run_flow
    requirement=CombinedWorkflowTests.requirement

    def configure_two_topics(self):
        from test_answer_completion import Model
        class ScopedSelection(Model):
            def remediation_json(self,payload,schema):
                # The baseline fake class selects topics from all requested words;
                # this controlled model follows the supplied candidate contract.
                raw,usage=super().remediation_json(payload,schema)
                value=json.loads(raw)
                allowed=schema["properties"]["incident_types"]["items"]["enum"]
                value["incident_types"]=[t for t in value["incident_types"] if t in allowed]
                return json.dumps(value),usage
        self.quality=ScopedSelection()
        self.mcp.matches=[[match_for("kernel-cpu-control")],[match_for("logrotate"),match_for("kernel-cpu-control",section="cpu.idle")]]
        requirements=[self.requirement("server-d","lookup",["cpu","disk"],"두 상태 확인"),
            self.requirement("server-d","remediation",["cpu"],"CPU 대처"),
            self.requirement("server-d","remediation",["disk"],"디스크 대처")]
        self.planner.scripts["plan"]=[dict(tasks=[dict(server_id="server-d",goal="상태 조회",information_ids=["cpu","disk"])],requirements=requirements)]

    async def test_manual_specialist_fills_missing_topic_before_final_validation(self):
        self.configure_two_topics()
        result=await self.run_flow("서버 D CPU와 디스크 상태 및 각각의 대처방안")
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual(len(self.mcp.searches),2)
        self.assertIn("디스크",self.mcp.searches[1]["query"])
        self.assertEqual(self.mcp.calls,[("server-d","top"),("server-d","df")])
        events=result["events"]
        assigned=[e for e in events if e["type"]=="SpecialistDelegated" and e["role"]=="manual"]
        self.assertEqual(len(assigned),1)
        self.assertEqual(assigned[0]["requirement_ids"],["r2","r3"])
        returned=next(e for e in events if e["type"]=="SpecialistReturned" and e["role"]=="manual")
        self.assertEqual(returned["rounds"],2)
        self.assertEqual(len(result["final_reviews"]),1)
        self.assertEqual(len(result["manual_state"]["matches"]),3)

    async def test_second_search_failure_preserves_first_topic_steps(self):
        self.configure_two_topics()
        original=self.mcp.call_tool
        async def fail_second(name,args):
            if name=="search_remediation_manuals" and self.mcp.searches:
                self.mcp.searches.append(args)
                raise ValueError("second search unavailable")
            return await original(name,args)
        self.mcp.call_tool=fail_second
        result=await self.run_flow("서버 D CPU와 디스크 상태 및 각각의 대처방안")
        self.assertEqual(len(self.mcp.searches),2)
        self.assertEqual(result["remediation"]["status"],"provided")
        self.assertTrue(result["remediation"]["steps"])
        self.assertTrue(all(s["doc_id"]=="kernel-cpu-control" for s in result["remediation"]["steps"]))
        self.assertIn("추가", " ".join(result["remediation"]["unconfirmed"]))
        self.assertEqual(result["requirements"][1]["status"],"satisfied")
        self.assertNotEqual(result["requirements"][2]["status"],"satisfied")
        self.assertEqual(result["final_validation"]["status"],"partial")

    async def test_missing_manual_topic_keeps_information_id_in_followup_search(self):
        from agent.remediation import build_manual_agent
        queries=[]
        async def retrieve(query):
            queries.append(query)
            return dict(matches=[dict(chunk_id="memory",incident_types=["memory-pressure"])],timings_ms={})
        async def select(matches):return [],dict(model_calls=0,cost_usd=0,summary_ms=0,incident_types=[])
        graph=build_manual_agent(retrieve,select,lambda x:None,lambda *a,**k:None)
        await graph.ainvoke(dict(query="CPU 대응",requirements=[dict(requirement_id="r2",description="높은 사용률에 어떻게 대응할지 알려줘.",information_ids=["cpu"])],rounds=0,matches=[],selected=[]))
        self.assertEqual(len(queries),2)
        self.assertIn("cpu",queries[1].lower())

    async def test_local_graph_has_independent_retrieve_assess_loop(self):
        from agent.remediation import build_manual_agent
        queries=[]; snapshots=[]; events=[]
        async def retrieve(query):
            queries.append(query)
            return dict(matches=[dict(chunk_id="c1",incident_types=["cpu-spike"])],timings_ms={})
        async def select(matches):return [],dict(model_calls=0,cost_usd=0,summary_ms=0,incident_types=[])
        def checkpoint(local):snapshots.append(local)
        graph=build_manual_agent(retrieve,select,checkpoint,lambda kind,**data:events.append((kind,data)))
        self.assertTrue({"retrieve","assess"}<=set(graph.get_graph().nodes))
        result=await graph.ainvoke(dict(query="CPU 대처",requirements=[],rounds=0,matches=[],selected=[]))
        self.assertEqual(queries,["CPU 대처"])
        self.assertEqual(result["rounds"],1)
        self.assertEqual(result["status"],"not_found")


    async def test_deadline_during_second_manual_search_keeps_checked_first_steps(self):
        from agent.deps import run_graph
        self.configure_two_topics()
        original=self.mcp.call_tool
        cancelled=[]
        async def slow_second(name,args):
            if name=="search_remediation_manuals" and self.mcp.searches:
                self.mcp.searches.append(args)
                try:await asyncio.sleep(1)
                except asyncio.CancelledError:cancelled.append(True);raise
            return await original(name,args)
        self.mcp.call_tool=slow_second
        deps=SimpleNamespace(mcp=self.mcp,fast_llm=FastLLM("test",0),quality_llm=self.quality,monitoring_llm=self.planner)
        result=await run_graph(build_graph(deps),dict(query="서버 D CPU와 디스크 상태 및 각각의 대처방안",history=[],events=[]),timeout_seconds=.1)
        self.assertEqual(cancelled,[True])
        self.assertEqual(result["final_validation"]["status"],"unverified")
        self.assertEqual(result["remediation"]["status"],"provided")
        self.assertTrue(result["remediation"]["steps"])
        for step in result["remediation"]["steps"]:self.assertIn(step["text"],result["final"])
        self.assertEqual(result["manual_state"]["search_calls"],2)

    async def test_corrupted_second_source_never_replaces_verified_first_steps(self):
        self.configure_two_topics()
        self.mcp.matches[1][0]["sha256"]="corrupted"
        result=await self.run_flow("서버 D CPU와 디스크 상태 및 각각의 대처방안")
        self.assertEqual(len(self.mcp.searches),2)
        self.assertEqual(result["final_validation"]["status"],"partial")
        self.assertEqual(result["remediation"]["status"],"provided")
        self.assertTrue(all(s["doc_id"]=="kernel-cpu-control" for s in result["remediation"]["sources"]))
        self.assertEqual(len(result["manual_state"]["matches"]),1)

    async def test_missing_topic_stops_after_shared_two_search_budget(self):
        self.configure_two_topics()
        self.mcp.matches=[self.mcp.matches[0]]
        result=await self.run_flow("서버 D CPU와 디스크 상태 및 각각의 대처방안")
        self.assertEqual(len(self.mcp.searches),2)
        self.assertEqual(result["workflow_budget"]["manual_searches"],2)
        self.assertEqual(result["final_validation"]["status"],"partial")
        self.assertTrue(result["remediation"]["steps"])

    async def test_second_selection_failure_preserves_verified_first_steps(self):
        self.configure_two_topics()
        original=self.quality.remediation_json
        def fail_second(payload,schema):
            if any("disk-full" in a["incident_types"] for a in payload["actions"]):return "not-json",{"usd":0}
            return original(payload,schema)
        self.quality.remediation_json=fail_second
        result=await self.run_flow("서버 D CPU와 디스크 상태 및 각각의 대처방안")
        self.assertEqual(result["final_validation"]["status"],"partial")
        self.assertEqual(result["remediation"]["status"],"provided")
        self.assertTrue(all(s["doc_id"]=="kernel-cpu-control" for s in result["remediation"]["steps"]))
        self.assertEqual(len(self.mcp.searches),2)


    async def test_repair_deadline_keeps_previous_checked_manual_selection(self):
        import time
        from agent.deps import run_graph
        class SlowRepair(QualityLLM):
            calls=0
            def remediation_json(self,payload,schema):
                self.calls+=1
                if self.calls>1:time.sleep(.25)
                return super().remediation_json(payload,schema)
        model=SlowRepair("test",0)
        deps=SimpleNamespace(mcp=StubMCP([match_for("kernel-cpu-control")]),quality_llm=model)
        state=dict(query="Linux CPU 대처",events=[],workflow_active=True,targets=[],manual_os="linux",
            requirements=[dict(requirement_id="r1",description="CPU 대처",kind="remediation",information_ids=["cpu"],server_ids=[],status="information_ready",reason="최종 검증 대기")])
        first=await remediation_node(state,deps)
        class Graph:
            async def astream(self,initial,stream_mode):
                yield "updates", {"previous":first}
                result=await remediation_node({**state,"manual_state":first["manual_state"],
                    "manual_feedback":{"manual_only":True,"rewrite_instructions":"조건 설명 보완"}},deps)
                yield "updates", {"manual":result}
        result=await run_graph(Graph(),state,timeout_seconds=.08)
        self.assertEqual(model.calls,2)
        self.assertEqual(result["final_validation"]["status"],"unverified")
        self.assertEqual(result["remediation"]["status"],"provided")
        self.assertEqual(result["remediation"]["steps"],first["remediation"]["steps"])
        for step in first["remediation"]["steps"]:self.assertIn(step["text"],result["final"])


class NoActionDispositionTests(unittest.IsolatedAsyncioTestCase):
    def evidence(self,server="server-d",command="top"):
        from monitoring.api import ScenarioStore,CommandRequest
        from monitoring.evidence import extract_facts
        e=ScenarioStore().result(CommandRequest(server_id=server,command=command)).model_dump(mode="json")
        return {**e,"evidence_id":"e1","facts":extract_facts(e)}

    def requirement(self,server="server-d",info="cpu"):
        return dict(requirement_id="r1",kind="remediation",action_condition="if_needed",description="확인 후 필요한 조치만",server_ids=[server],information_ids=[info])

    def test_no_action_requires_current_matching_low_pressure_evidence(self):
        from agent.remediation import bind_no_action,manual_requirement_covered
        spec=dict(requirement_id="r1",evidence_ids=["e1"])
        req=self.requirement();e=self.evidence()
        outcomes=bind_no_action([spec],[req],[e])
        self.assertEqual(len(outcomes),1);self.assertIn("18.0%",outcomes[0]["text"])
        self.assertIn("조치를 권하지",outcomes[0]["text"])
        manual=dict(status="provided",steps=[],sources=[],no_action=outcomes)
        self.assertTrue(manual_requirement_covered(req,manual,[e]))
        manual["no_action"][0]["text"]="항상 정상이며 영구적으로 조치 불필요"
        self.assertFalse(manual_requirement_covered(req,manual,[e]))
        for bad_req,bad_evidence in [
            (req,[]),(dict(req,action_condition="always",description="정상이어도 예방 절차를 설명"),[e]),(req,[self.evidence("server-b")]),
            (self.requirement("server-b"),[self.evidence("server-b")]),
            (self.requirement(info="disk"),[e]),
            (dict(req,server_ids=[]),[e])]:
            with self.subTest(req=bad_req,evidence=bool(bad_evidence)):
                with self.assertRaises(ValueError):bind_no_action([spec],[bad_req],bad_evidence)
        e["exit_code"]=1
        with self.assertRaises(ValueError):bind_no_action([spec],[req],[e])

    def test_macos_unused_alone_cannot_justify_skipping_memory_guidance(self):
        from agent.remediation import bind_no_action
        for sid in ("server-e","server-g"):
            with self.subTest(sid=sid):
                with self.assertRaises(ValueError):
                    bind_no_action([dict(requirement_id="r1",evidence_ids=["e1"])],
                        [self.requirement(sid,"memory.summary")],[self.evidence(sid)])

    def test_raw_output_overrides_tampered_facts_and_pressure_cannot_be_skipped(self):
        from agent.remediation import bind_no_action
        spec=dict(requirement_id="r1",evidence_ids=["e1"])
        e=self.evidence("server-b")
        for f in e["facts"]:
            if f["information_id"]=="cpu":f["fields"]["usage_percent"]=0
        with self.assertRaises(ValueError):bind_no_action([spec],[self.requirement("server-b")],[e])
        disk=self.evidence(command="df")
        with self.assertRaises(ValueError):bind_no_action([spec],[self.requirement(info="disk")],[disk])
        normal=self.evidence("server-a","df")
        self.assertTrue(bind_no_action([spec],[self.requirement("server-a","disk")],[normal]))

    async def test_selection_and_render_keep_no_action_separate_from_manual_steps(self):
        from agent.remediation import remediation_node,manual_requirement_covered
        class NoAction(QualityLLM):
            def remediation_json(self,payload,schema):
                return json.dumps(dict(applicable=False,incident_types=[],action_ids=[],assignments=[],
                    no_action=[dict(requirement_id="r1",evidence_ids=["e1"])])),{"usd":0}
        req=self.requirement();e=self.evidence()
        mcp=StubMCP([match_for("kernel-cpu-control")]);model=NoAction("test",0)
        result=await remediation_node(dict(query="CPU를 확인하고 필요한 조치만 알려줘",events=[],history=[],
            workflow_active=True,targets=["server-d"],manual_os="linux",requirements=[req],evidence=[e]),
            SimpleNamespace(mcp=mcp,quality_llm=model))
        self.assertEqual(result["remediation"]["status"],"provided")
        self.assertEqual(result["remediation"]["steps"],[])
        self.assertTrue(result["remediation"]["no_action"])
        self.assertTrue(manual_requirement_covered(req,result["remediation"],[e]))
        self.assertIn(result["remediation"]["no_action"][0]["text"],result["final"])
        self.assertEqual(sum(n=="search_remediation_manuals" for n,_ in mcp.calls),1)
        snapshots=[]
        with patch("agent.remediation.save_progress",side_effect=lambda value:snapshots.append(value)):
            again=await remediation_node(dict(query="CPU를 확인하고 필요한 조치만 알려줘",events=[],history=[],
                workflow_active=True,targets=["server-d"],manual_os="linux",requirements=[req],evidence=[e],
                manual_state=result["manual_state"]),SimpleNamespace(mcp=mcp,quality_llm=model))
        self.assertTrue(snapshots[0]["remediation"]["no_action"])
        self.assertEqual(again["remediation"]["no_action"],result["remediation"]["no_action"])


    async def test_no_action_survives_unrelated_retrieval_with_empty_action_schema(self):
        from agent.remediation import remediation_node
        class NoAction(QualityLLM):
            def remediation_json(self,payload,schema):
                assert payload["actions"]==[]
                assert schema["properties"]["action_ids"]["maxItems"]==0
                assert schema["properties"]["assignments"]["maxItems"]==0
                assert '"enum": []' not in json.dumps(schema)
                return json.dumps(dict(applicable=False,incident_types=[],action_ids=[],assignments=[],
                    no_action=[dict(requirement_id="r1",evidence_ids=["e1"])])),{"usd":.01}
        mcp=StubMCP([match_for("logrotate")])
        result=await remediation_node(dict(query="CPU를 확인하고 필요한 조치만 알려줘",events=[],history=[],
            workflow_active=True,targets=["server-d"],manual_os="linux",requirements=[self.requirement()],evidence=[self.evidence()]),
            SimpleNamespace(mcp=mcp,quality_llm=NoAction("test",0)))
        self.assertEqual(result["remediation"]["status"],"provided")
        self.assertTrue(result["remediation"]["no_action"])
        self.assertEqual(result["remediation"]["steps"],[])
        self.assertEqual(result["remediation"]["model_calls"],1)
        self.assertAlmostEqual(result["cost_usd"],.01)
        self.assertEqual(sum(n=="search_remediation_manuals" for n,_ in mcp.calls),1)

    async def test_manual_agent_does_not_search_again_for_a_verified_no_action_topic(self):
        from agent.remediation import build_manual_agent
        calls=[];checkpoints=[]
        async def retrieve(query):calls.append(query);return {"matches":[match_for("logrotate")],"timings_ms":{}}
        async def select(matches):return [],{"no_action":[dict(requirement_id="r1",evidence_ids=["e1"],text="verified") ]}
        agent=build_manual_agent(retrieve,select,checkpoints.append,lambda *a,**k:None)
        result=await agent.ainvoke(dict(query="CPU 조건부 조치",requirements=[self.requirement()],rounds=0,matches=[],selected=[]))
        self.assertEqual(len(calls),1);self.assertTrue(checkpoints);self.assertEqual(result["status"],"assessed")

    async def test_conditional_no_action_reaches_final_requirement_validation(self):
        from test_answer_completion import Model
        CombinedWorkflowTests.setUp(self)
        class Conditional(Model):
            def response_json(self,purpose,payload,schema):
                if purpose=="intent":return json.dumps(intent_payload(route="investigate_remediate",style="detailed",reason="조건부 조치",
                    requested_information_ids=["cpu"],manual_information_ids=["cpu"],requested_results=[
                    dict(kind="lookup",description="CPU 확인",information_ids=["cpu"]),
                    dict(kind="remediation",description="필요한 경우에만 조치",information_ids=["cpu"],action_condition="if_needed")])),{"usd":0}
                return super().response_json(purpose,payload,schema)
            def remediation_json(self,payload,schema):
                option=payload["no_action_options"][0]
                return json.dumps(dict(applicable=False,incident_types=[],action_ids=[],assignments=[],
                    no_action=[{k:option[k] for k in ("requirement_id","evidence_ids")}])),{"usd":0}
        self.quality=Conditional()
        result=await CombinedWorkflowTests.run_flow(self,"서버 D CPU를 확인하고 필요한 조치만 알려줘")
        self.assertEqual(result["final_validation"]["status"],"passed")
        self.assertEqual(result["remediation"]["steps"],[])
        outcome=result["remediation"]["no_action"][0]
        self.assertIn(outcome["text"],result["final"])
        payload=[p for purpose,p in self.quality.inputs if purpose=="final_review"][-1]
        self.assertTrue(payload["requirement_coverage"][outcome["requirement_id"]]["block_ids"])
        self.assertTrue(payload["requirement_coverage"][outcome["requirement_id"]]["evidence_ids"])


class DiagnosticAssignmentContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_cpu_assignment_has_one_assessment_and_explicit_linux_environment(self):
        from agent.investigation import investigate
        from test_answer_completion import Model
        from test_information_investigation import FixtureMCP
        received=[]
        class Expert(Model):
            def diagnostic_json(self,payload,schema):
                received.append((payload,schema))
                eid=payload["observations"][0]["evidence_id"]
                return json.dumps(dict(diagnoses=[dict(text="CPU 사용률 19.0%이며 높은 사용량 전제가 관측되지 않았습니다.",
                    requirement_ids=["r1"],evidence_ids=[eid],quote_ids=[eid+":l3"],symptom_observed=False)],tasks=[])),{"usd":0}
        model=Expert();mcp=FixtureMCP()
        result=await investigate(dict(query="CPU가 높으면 원인 설명",scenario_id="server-a",single_server_mode=True,events=[],
            request_decision=dict(requested_information_ids=["cpu"],needs_diagnosis=True,
                requested_results=[dict(description="CPU 원인",kind="diagnosis",information_ids=["cpu"])])),
            SimpleNamespace(mcp=mcp,quality_llm=model,fast_llm=Model(),monitoring_llm=Model()))
        self.assertTrue(received)
        payload,schema=received[0]
        self.assertEqual(payload["target_os"],{"server-a":"linux"})
        self.assertEqual(payload["original_query"],"CPU가 높으면 원인 설명")
        self.assertEqual(schema["properties"]["diagnoses"]["maxItems"],1)
        self.assertEqual(next(iter(schema["properties"])),"tasks")
        self.assertEqual(payload["requirements"][0]["information_ids"],["cpu"])
        self.assertEqual(mcp.calls,[("server-a","top")])
        self.assertFalse(any(e["type"]=="SpecialistFailed" for e in result["events"]))


class ResourceDiagnosisGroundingTests(unittest.TestCase):
    def fixture(self, server, command, info, text, observed=True):
        from monitoring.api import ScenarioStore, CommandRequest
        from monitoring.evidence import extract_facts
        from agent.investigation_models import DiagnosisAssessment
        evidence=ScenarioStore().result(CommandRequest(server_id=server,command=command)).model_dump(mode="json")
        evidence["evidence_id"]="e1"
        evidence["facts"]=extract_facts(evidence)
        facts=[f for f in extract_facts(evidence) if f["information_id"] in info]
        assessment=DiagnosisAssessment(text=text,requirement_ids=["r1"],evidence_ids=["e1"],
            quotes=[line for f in facts for line in f["quote"].splitlines()],symptom_observed=observed)
        requirement=dict(requirement_id="r1",kind="diagnosis",server_ids=[server],information_ids=info)
        return assessment,[evidence],[requirement]

    def test_cpu_observation_cannot_assert_process_causality(self):
        from agent.investigation import ground_resource_assessment, verify_report
        from agent.investigation_models import InvestigationReport
        a,e,r=self.fixture("server-b","top",["cpu"],"높은 CPU 사용률 때문에 시스템 과부하가 발생했습니다.")
        ground_resource_assessment(a,e,r)
        self.assertIn("96.0%",a.text)
        self.assertNotIn("batch-worker",a.text)
        self.assertIn("확정할 수 없습니다",a.text)
        self.assertTrue(a.symptom_observed)
        verify_report(InvestigationReport(hypotheses=[a.model_dump(exclude={"symptom_observed","requirement_ids"})]),e)

    def test_linux_memory_observation_cannot_assert_pressure_or_oom(self):
        from agent.investigation import ground_resource_assessment
        a,e,r=self.fixture("server-c","free",["memory.summary"],"메모리 압박이 관측되며 OOM이 임박했습니다.")
        ground_resource_assessment(a,e,r)
        self.assertIn("7680",a.text)
        self.assertNotIn("임박",a.text)
        self.assertNotIn("압박이 관측",a.text)
        self.assertIn("확정할 수 없습니다",a.text)
        self.assertTrue(a.symptom_observed)

    def test_discarded_memory_ratio_prose_does_not_fail_grounded_observation(self):
        from agent.investigation import ground_resource_assessment, verify_report
        from agent.investigation_models import InvestigationReport
        # Actual C history: a ratio in diagnostic prose was rejected before the
        # prose could be replaced with raw fields. Ratios belong to calculations.
        a,e,r=self.fixture("server-c","free",["memory.summary"],"현재 메모리 사용률은 93.75%입니다.")
        ground_resource_assessment(a,e,r)
        self.assertNotIn("93.75",a.text)
        self.assertIn("7680",a.text)
        self.assertIn("8192",a.text)
        verify_report(InvestigationReport(hypotheses=[a.model_dump(exclude={"symptom_observed","requirement_ids"})]),e)

    def test_discarding_prose_does_not_authorize_invented_source_quotes(self):
        from agent.investigation import ground_resource_assessment
        a,e,r=self.fixture("server-c","free",["memory.summary"],"현재 메모리 사용률은 93.75%입니다.")
        a.quotes.append("Mem: 8192 8000 0 0 0 0")
        with self.assertRaises(ValueError):ground_resource_assessment(a,e,r)

    def test_missing_resource_citation_fails_without_substituting_another_source(self):
        from agent.investigation import ground_resource_assessment
        a,e,r=self.fixture("server-b","top",["cpu"],"CPU가 높습니다.")
        a.quotes=[]
        with self.assertRaises(ValueError):ground_resource_assessment(a,e,r)

    def test_mac_usage_contributor_does_not_require_proven_pressure(self):
        from agent.investigation import ground_resource_assessment, cause_catalog, bind_cause
        from agent.investigation_models import CauseSelection
        from monitoring.api import ScenarioStore, CommandRequest
        from monitoring.evidence import extract_facts
        a,e,r=self.fixture("server-g","top",["memory.summary"],"물리 메모리 사용량을 관측했습니다.")
        ground_resource_assessment(a,e,r)
        self.assertIn("압박 여부를 확정할 수 없습니다",a.text)
        p=ScenarioStore().result(CommandRequest(server_id="server-g",command="ps")).model_dump(mode="json")
        p.update(evidence_id="e2",facts=extract_facts(p));e.append(p)
        factor=next(f for f in cause_catalog(e,r) if f["kind"]=="process_memory")
        selection=CauseSelection(requirement_id="r1",factor_id=factor["factor_id"])
        candidate=bind_cause(selection,e,r,[a.model_dump()])
        self.assertIn("후보",candidate.text)
        self.assertIn("e2",candidate.evidence_ids)
        a.symptom_observed=False
        with self.assertRaises(ValueError):bind_cause(selection,e,r,[a.model_dump()])

    def test_mixed_resource_process_observation_cannot_deny_quoted_process(self):
        from agent.investigation import ground_resource_assessment, verify_report
        from agent.investigation_models import InvestigationReport, DiagnosisAssessment
        from monitoring.api import ScenarioStore, CommandRequest
        from monitoring.evidence import extract_facts
        a,e,r=self.fixture("server-f","top",["cpu","process"],"CPU 사용률은 96%이며 프로세스 정보는 없습니다.")
        p=ScenarioStore().result(CommandRequest(server_id="server-f",command="ps")).model_dump(mode="json")
        p.update(evidence_id="e2",facts=extract_facts(p));e.append(p)
        a.evidence_ids.append("e2")
        a.quotes.append(p["facts"][0]["quote"].splitlines()[0])
        ground_resource_assessment(a,e,r)
        self.assertNotIn("정보는 없습니다",a.text)
        self.assertIn("프로세스",a.text)
        self.assertIn("96.0%",a.text)
        DiagnosisAssessment.model_validate(a.model_dump())
        verify_report(InvestigationReport(hypotheses=[a.model_dump(exclude={"symptom_observed","requirement_ids"})]),e)

    def test_mixed_resource_process_requires_actual_process_quote(self):
        from agent.investigation import ground_resource_assessment
        a,e,r=self.fixture("server-f","top",["cpu","process"],"현재 CPU 사용률은 96%입니다.")
        with self.assertRaises(ValueError):ground_resource_assessment(a,e,r)

    def test_mixed_log_or_disk_assessment_is_not_replaced(self):
        from agent.investigation import ground_resource_assessment
        a,e,r=self.fixture("server-b","top",["cpu","log.content"],"로그와 CPU를 함께 판단합니다.")
        ground_resource_assessment(a,e,r)
        self.assertEqual(a.text,"로그와 CPU를 함께 판단합니다.")

    def test_multi_target_assessment_retains_existing_contract(self):
        from agent.investigation import ground_resource_assessment
        a,e,r=self.fixture("server-b","top",["cpu"],"여러 대상의 원문을 비교합니다.")
        r[0]["server_ids"].append("server-c")
        ground_resource_assessment(a,e,r)
        self.assertEqual(a.text,"여러 대상의 원문을 비교합니다.")

    def test_single_host_all_resource_fields_fit_statement_contract(self):
        from agent.investigation import ground_resource_assessment, verify_report
        from agent.investigation_models import InvestigationReport, DiagnosisAssessment
        from monitoring.api import ScenarioStore, CommandRequest
        from monitoring.evidence import extract_facts
        for sid in ("server-a","server-c","server-e","server-g"):
            with self.subTest(sid=sid):
                evidence=[];quotes=[]
                for index,command in enumerate(("top","free") if sid in ("server-a","server-c") else ("top","vm_stat")):
                    source=ScenarioStore().result(CommandRequest(server_id=sid,command=command)).model_dump(mode="json")
                    source.update(evidence_id=f"e{index}",facts=extract_facts(source));evidence.append(source)
                    quotes += [line for f in source["facts"] for line in f["quote"].splitlines()]
                a=DiagnosisAssessment(text="원문 수치를 확인했습니다.",requirement_ids=["r1"],evidence_ids=[e["evidence_id"] for e in evidence],quotes=list(dict.fromkeys(quotes)),symptom_observed=True)
                r=[dict(requirement_id="r1",server_ids=[sid],information_ids=["cpu","memory.summary","memory.detail"])]
                ground_resource_assessment(a,evidence,r)
                DiagnosisAssessment.model_validate(a.model_dump())
                verify_report(InvestigationReport(hypotheses=[a.model_dump(exclude={"symptom_observed","requirement_ids"})]),evidence)


class MemoryProcessPresentationTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self):
        from monitoring.api import ScenarioStore, CommandRequest
        from monitoring.evidence import extract_facts
        from agent.investigation import render_answer
        e=ScenarioStore().result(CommandRequest(server_id="server-c",command="ps")).model_dump(mode="json")
        e["stdout"]="PID PPID COMMAND %CPU %MEM RSS\n41 1 tiny 0.1 1.0 100\n42 1 largest 1.0 60.9 5111808\n43 1 medium 0.2 15.0 1000000\n44 1 other 0.1 3.0 200000\n"
        e.update(evidence_id="e1",facts=extract_facts(e))
        requirements=[dict(requirement_id="r1",kind="lookup",server_ids=["server-c"],information_ids=["process"],description="점유 프로세스",status="satisfied",reason=""),dict(requirement_id="r2",kind="diagnosis",server_ids=["server-c"],information_ids=["memory.summary"],description="메모리 원인 후보",status="information_ready",reason="")]
        candidate=dict(text="PID 42 largest의 메모리 점유율 60.9%는 관측된 메모리 사용의 기여 후보입니다.",requirement_ids=["r2"],evidence_ids=["e1"],quotes=[e["stdout"].splitlines()[2]])
        state=dict(query="현재 메모리 사용 비율과 점유 프로세스, 원인 후보",requirements=requirements,evidence=[e],hypotheses=[candidate],tasks=[dict(server_id="server-c",information_ids=["memory.summary","process"])])
        state["final"]=render_answer([], [e], [candidate], ["단일 표본으로 누수 여부를 확인할 수 없습니다."])
        return state

    def layout(self, blocks, groups, count):
        from agent.response_policy import Layout
        optional={r["block_id"] for g in groups for r in g["rows"]}
        chosen={r["block_id"] for g in groups for r in g["rows"][:count]}
        sections={}
        for b in blocks:
            if b["block_id"] in optional and b["block_id"] not in chosen:continue
            sections.setdefault(b["section"],[]).append(b["block_id"])
        non_background=[i for k,ids in sections.items() if k!="배경" for i in ids]
        return Layout(sections=[dict(title=k,block_ids=v) for k,v in sections.items()],summary_ids=non_background[:3])

    def test_selected_process_count_is_flexible_and_rows_sort_by_memory(self):
        from agent.response_policy import source_blocks,memory_process_groups,validate_layout,render_layout
        state=self.fixture();blocks=source_blocks(state,"detailed");groups=memory_process_groups(blocks,state)
        self.assertEqual([r["pid"] for r in groups[0]["rows"]],[42,43,44,41])
        self.assertIn("조회된 프로세스 목록 기준", " ".join(b["text"] for b in blocks))
        for count in (2,4):
            layout=self.layout(blocks,groups,count)
            validate_layout(layout,blocks,"detailed",groups)
            text=render_layout(layout,blocks,"detailed",groups).split("**요약**")[0]
            self.assertLess(text.index("PID 42 largest:"),text.index("PID 43 medium:"))
            self.assertEqual("PID 41 tiny:" in text,count==4)
            self.assertTrue(all("r1" in b["requirement_ids"] and "e1" in b["evidence_ids"] for b in blocks if b["block_id"] in {r["block_id"] for r in groups[0]["rows"]}))

    def test_cannot_omit_all_processes_candidate_or_source_blocks(self):
        from agent.response_policy import source_blocks,memory_process_groups,validate_layout
        state=self.fixture();blocks=source_blocks(state,"detailed");groups=memory_process_groups(blocks,state)
        for count,remove in [(0,None),(2,"원인"),(2,"미확인 사항")]:
            layout=self.layout(blocks,groups,count)
            if remove:layout.sections=[s for s in layout.sections if s.title!=remove]
            with self.assertRaises(ValueError):validate_layout(layout,blocks,"detailed",groups)
        layout=self.layout(blocks,groups,2);layout.sections[0].block_ids.append("invented")
        with self.assertRaises(ValueError):validate_layout(layout,blocks,"detailed",groups)

    def test_cause_process_must_be_visible_and_summary_cannot_revive_omitted_rows(self):
        from agent.response_policy import source_blocks,memory_process_groups,validate_layout
        state=self.fixture();blocks=source_blocks(state,"detailed");groups=memory_process_groups(blocks,state)
        layout=self.layout(blocks,groups,2);largest=groups[0]["rows"][0]["block_id"]
        for s in layout.sections:s.block_ids=[i for i in s.block_ids if i!=largest]
        with self.assertRaises(ValueError):validate_layout(layout,blocks,"detailed",groups)
        layout=self.layout(blocks,groups,2);layout.summary_ids[0]=groups[0]["rows"][-1]["block_id"]
        with self.assertRaises(ValueError):validate_layout(layout,blocks,"detailed",groups)

    def test_cpu_only_scope_does_not_enable_optional_memory_rows(self):
        from agent.response_policy import source_blocks,memory_process_groups
        state=self.fixture();state["requirements"][1]["information_ids"]=["cpu"]
        self.assertEqual(memory_process_groups(source_blocks(state,"detailed"),state),[])

    async def test_existing_formatter_selects_rows_without_an_extra_model_call(self):
        from agent.response_policy import present_response,memory_process_groups
        state=self.fixture();state.update(route="commands",response_style="detailed",events=[],cost_usd=0)
        owner=self
        class Formatter:
            name="process-selector"
            calls=0
            def response_json(self,purpose,payload,schema):
                self.calls+=1
                self.assert_payload=payload
                return json.dumps(owner.layout(payload["blocks"],payload["process_groups"],2).model_dump()),{"usd":0}
        model=Formatter();result=await present_response(state,SimpleNamespace(quality_llm=model))
        self.assertEqual(result["presentation"]["status"],"formatted")
        self.assertEqual(model.calls,1)
        self.assertIn("PID 42 largest:",result["final"])
        self.assertNotIn("PID 41 tiny:",result["final"])
        self.assertTrue(result["requirement_coverage"]["r1"]["block_ids"])

    def test_memory_diagnosis_fallback_reports_missing_work_instead_of_generic_caution(self):
        from agent.investigation import render_answer
        state=self.fixture();r=state["requirements"]
        text=render_answer([],state["evidence"],[],[],requirements=r)
        self.assertIn("원인 후보 설명을 완성하지 못했습니다",text)
        self.assertNotIn("확보한 정보만으로 원인을 확정하지 않습니다",text)
        text=render_answer([],[],[],[],requirements=r)
        self.assertIn("프로세스 점유 정보를 확보하지 못했습니다",text)


    def test_identical_process_rows_on_two_targets_keep_separate_sources(self):
        from copy import deepcopy
        from agent.response_policy import source_blocks,memory_process_groups,validate_layout
        from agent.investigation import render_answer
        state=self.fixture();other=deepcopy(state["evidence"][0]);other.update(server_id="server-g",evidence_id="e2")
        state["evidence"].append(other)
        for r in state["requirements"]:r["server_ids"].append("server-g")
        state["final"]=render_answer([],state["evidence"],state["hypotheses"],[])
        blocks=source_blocks(state,"detailed");groups=memory_process_groups(blocks,state)
        self.assertEqual(len(groups),2)
        self.assertFalse({r["block_id"] for r in groups[0]["rows"]}&{r["block_id"] for r in groups[1]["rows"]})
        for g in groups:
            expected="e1" if g["server_id"]=="server-c" else "e2"
            self.assertTrue(all(b["evidence_ids"]==[expected] for b in blocks if b["block_id"] in {r["block_id"] for r in g["rows"]}))
        layout=self.layout(blocks,groups,2);validate_layout(layout,blocks,"detailed",groups)

    def test_absent_symptom_does_not_become_a_generation_failure(self):
        from agent.investigation import render_answer
        state=self.fixture()
        text=render_answer([],state["evidence"],[],[],requirements=state["requirements"],
            assessments=[dict(requirement_ids=["r2"],symptom_observed=False)])
        self.assertIn("현재 관측에서 확인되지 않았습니다",text)
        self.assertNotIn("완성하지 못했습니다",text)

    async def test_format_schema_allows_four_rows_and_brief_subset(self):
        from agent.response_policy import source_blocks,memory_process_groups,model_json,Layout,validate_layout
        state=self.fixture();blocks=source_blocks(state,"detailed");groups=memory_process_groups(blocks,state)
        layout=self.layout(blocks,groups,2)
        selected=[i for s in layout.sections for i in s.block_ids]
        class Formatter:
            def response_json(self,purpose,payload,schema):
                self.schema=schema
                return json.dumps(dict(brief_ids=selected)),{"usd":0}
        model=Formatter()
        result,_,calls=await model_json(model,"format",dict(style="brief",blocks=blocks,process_groups=groups),Layout,
            lambda value:validate_layout(value,blocks,"brief",groups))
        self.assertEqual(calls,1)
        prop=model.schema["properties"]["brief_ids"]
        self.assertEqual(prop["maxItems"],len(blocks))
        self.assertLess(prop["minItems"],len(blocks))
        self.assertEqual(result.brief_ids,selected)
