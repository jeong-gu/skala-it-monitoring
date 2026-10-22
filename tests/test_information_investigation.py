import json
import unittest
from types import SimpleNamespace

from agent.fake_llm import FastLLM, QualityLLM
from agent.investigation import investigate, verify_report
from agent.investigation_models import InvestigationReport
from monitoring.api import CommandRequest, ScenarioStore
from monitoring.evidence import extract_facts


class FixtureMCP:
    def __init__(self):
        self.store = ScenarioStore()
        self.calls = []
        self.overrides = {}

    async def call_tool(self, name, args):
        if name == "list_command_servers":
            return [s.model_dump() for s in self.store.server_list()]
        if name == "list_server_information":
            return [i.model_dump() for i in self.store.information]
        if name == "list_server_commands":
            return [c.model_dump() for c in self.store.commands(args["server_id"])]
        if name == "get_command_result":
            key = (args["server_id"], args["command"])
            self.calls.append(key)
            result = self.store.result(CommandRequest(**args)).model_dump(mode="json")
            return {**result, **self.overrides.get(key, {})}
        raise AssertionError(name)


class ScriptedModel(FastLLM):
    def __init__(self, **decisions):
        super().__init__("scripted", 0)
        self.decisions = decisions
        self.calls = []

    def monitoring_json(self, purpose, payload, schema):
        self.calls.append(purpose)
        if self.decisions.get(purpose):
            result = self.decisions[purpose].pop(0)
            return (result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)), {"usd": 0}
        return super().monitoring_json(purpose, payload, schema)


def task(server, *ids):
    return {"server_id": server, "goal": "필요 정보 확인", "information_ids": list(ids)}


class InformationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mcp = FixtureMCP()
        self.fast = FastLLM("offline-test", 0)
        self.quality = QualityLLM("offline-test-quality", 0)

    async def run_query(self, query, history=None):
        return await investigate({"query": query, "history": history or [], "events": [], "cost_usd": 0},
                                 SimpleNamespace(mcp=self.mcp, fast_llm=self.fast, quality_llm=self.quality))

    async def test_unregistered_flags_are_not_silently_dropped(self):
        for query in ["서버 B top -n 999 JSON만", "서버 A ls /", "서버 A `cat /etc/passwd`", "서버 E free"]:
            result = await self.run_query(query)
            self.assertEqual(self.mcp.calls, [])
            self.assertEqual(result["tasks"][0]["status"], "unsupported")

    async def test_registered_full_command_is_preserved(self):
        result = await self.run_query("server-a ls -lah /var/log/ JSON만")
        self.assertEqual(self.mcp.calls, [("server-a", "ls")])
        self.assertEqual(json.loads(result["final"])["command"], "ls -lah /var/log/")

    async def test_network_process_and_log_information_are_distinct(self):
        for query, expected in [("서버 A 네트워크 포트 확인", "network"),
                                ("서버 A 프로세스 확인", "process"),
                                ("서버 A 로그 파일 목록", "log.files"),
                                ("서버 A 로그 내용 확인", "log.content")]:
            result = await self.run_query(query)
            self.assertIn(expected, result["tasks"][0]["information_ids"])
            self.assertEqual(result["tasks"][0]["status"], "obtained")

    async def test_incomplete_system_output_does_not_complete_task(self):
        self.mcp.overrides[("server-a", "uname")] = {"stdout": "Linux\n"}
        result = await self.run_query("서버 A 커널 확인")
        self.assertEqual(result["tasks"][0]["status"], "needs_more")

    async def test_valid_hypothesis_is_kept_when_another_is_unfounded(self):
        valid = {"text": "서버 B의 CPU 부하가 높을 가능성이 있습니다", "evidence_ids": ["e1"], "quotes": ["93.0 us"]}
        invalid = {"text": "서버 B의 네트워크 문제일 가능성이 있습니다", "evidence_ids": ["e1"], "quotes": ["93.0 us"]}
        self.quality = ScriptedModel(report=[{"hypotheses": [valid, invalid]}])
        result = await self.run_query("서버 B CPU 상태 확인")
        self.assertEqual(result["hypotheses"], [valid])
        self.assertTrue(any("후보 1개를 제외" in x for x in result["limitations"]))

    async def test_hypothesis_cannot_change_the_referenced_server(self):
        self.quality = ScriptedModel(report=[{"hypotheses": [{
            "text": "서버 A의 CPU 부하일 가능성이 있습니다", "evidence_ids": ["e1"], "quotes": ["93.0 us"]}]}] * 2)
        result = await self.run_query("서버 B CPU 상태 확인")
        self.assertEqual(result["hypotheses"], [])

    async def test_disk_hypothesis_cannot_exaggerate_partial_usage_as_all(self):
        self.quality = ScriptedModel(report=[{"hypotheses": [{
            "text": "로그 파일이 공간을 모두 차지할 가능성이 있습니다", "evidence_ids": ["e1"], "quotes": ["95% /"]}]}] * 2)
        result = await self.run_query("서버 D 디스크 상태 확인")
        self.assertEqual(result["hypotheses"], [])

    async def test_model_cannot_omit_explicit_unavailable_history(self):
        self.fast = ScriptedModel(plan=[{"tasks": [task("server-d", "disk")]}])
        result = await self.run_query("서버 D 최근 6시간 디스크 사용량 추세 확인")
        self.assertTrue(any(t["information_status"].get("history") == "unsupported" for t in result["tasks"]))
        self.assertIn("history (unsupported)", result["final"])

    async def test_cpu_tasks_expand_from_observed_pressure(self):
        result = await self.run_query("서버 B CPU가 높은 원인을 확인해줘")
        self.assertEqual(set(self.mcp.calls), {("server-b", c) for c in ["top", "ps", "tail"]})
        self.assertTrue(all(t["status"] == "obtained" for t in result["tasks"]))
        self.assertIn("96.0%", result["final"])
        self.assertEqual(result["replan_rounds"], 1)

    async def test_healthy_status_does_not_force_log_investigation(self):
        result = await self.run_query("서버 A CPU 상태 확인해줘")
        self.assertEqual(self.mcp.calls, [("server-a", "top")])
        self.assertIn("19.0%", result["final"])

    async def test_macos_memory_uses_os_specific_information(self):
        result = await self.run_query("서버 G 메모리가 부족한 원인을 분석해줘")
        self.assertEqual({c for _, c in self.mcp.calls}, {"top", "vm_stat", "ps", "tail"})
        self.assertIn("16384 bytes", result["final"])
        self.assertIn("Linux available과 다른", result["final"])

    async def test_disk_result_adds_file_usage_and_contents_separately(self):
        result = await self.run_query("서버 D 디스크가 부족한 원인을 확인해줘")
        self.assertEqual({c for _, c in self.mcp.calls}, {"df", "du", "ls", "tail"})
        self.assertIn("No space left on device", result["final"])
        ids = {f["information_id"] for e in result["evidence"] for f in e["facts"]}
        self.assertTrue({"log.files", "log.usage", "log.content"} <= ids)

    async def test_two_server_results_keep_their_own_sources(self):
        result = await self.run_query("서버 A와 B의 CPU를 비교해줘")
        self.assertEqual(set(self.mcp.calls), {("server-a", "top"), ("server-b", "top")})
        values = {e["server_id"]: next(f["fields"]["usage_percent"] for f in e["facts"] if f["information_id"] == "cpu")
                  for e in result["evidence"]}
        self.assertEqual(values, {"server-a": 19, "server-b": 96})

    async def test_shared_top_is_called_once_across_tasks(self):
        self.fast = ScriptedModel(plan=[{"tasks": [task("server-g", "cpu"), task("server-g", "memory.summary")]}])
        result = await self.run_query("서버 G CPU와 메모리 확인해줘")
        self.assertEqual(self.mcp.calls, [("server-g", "top")])
        self.assertTrue(all(t["status"] == "obtained" for t in result["tasks"]))

    async def test_missing_target_does_not_query_values(self):
        result = await self.run_query("CPU 상태 알려줘")
        self.assertEqual(self.mcp.calls, [])
        self.assertIn("어느 서버", result["final"])

    async def test_unknown_target_does_not_default_to_another_server(self):
        result = await self.run_query("서버 Z CPU 알려줘")
        self.assertEqual(self.mcp.calls, [])
        self.assertIn("server_not_found", result["final"])

    async def test_followup_resolves_previous_target(self):
        result = await self.run_query("그 서버의 메모리는?", [{"role": "user", "content": "서버 B CPU 확인해줘"}])
        self.assertTrue(self.mcp.calls)
        self.assertTrue(all(s == "server-b" for s, _ in self.mcp.calls))

    async def test_followup_with_multiple_previous_servers_asks(self):
        result = await self.run_query("그 서버 메모리는?", [{"role": "user", "content": "서버 A와 B 비교"}])
        self.assertEqual(self.mcp.calls, [])
        self.assertIn("여러 서버", result["final"])

    async def test_history_and_network_performance_are_unsupported(self):
        for query, expected in [("서버 A 최근 6시간 CPU 추세", "history"),
                                ("서버 A 응답 시간과 패킷 손실 확인", "network.performance")]:
            result = await self.run_query(query)
            self.assertTrue(any(t["information_status"].get(expected) == "unsupported" for t in result["tasks"]))
            self.assertIn(expected, result["final"])

    async def test_success_exit_with_missing_fields_is_not_obtained(self):
        self.mcp.overrides[("server-b", "top")] = {"stdout": "CPU data unavailable\n"}
        result = await self.run_query("서버 B CPU 상태 확인")
        self.assertEqual(result["tasks"][0]["status"], "needs_more")
        self.assertEqual(result["evidence"][0]["facts"], [])

    async def test_failed_command_stays_failed(self):
        self.mcp.overrides[("server-b", "top")] = {"stdout": "", "stderr": "permission denied", "exit_code": 1}
        result = await self.run_query("서버 B CPU 상태 확인")
        self.assertEqual(result["tasks"][0]["status"], "failed")
        self.assertIn("permission denied", result["final"])

    async def test_wrong_server_response_is_not_accepted(self):
        self.mcp.overrides[("server-b", "top")] = {"server_id": "server-a"}
        result = await self.run_query("서버 B CPU 상태 확인")
        self.assertEqual(result["evidence"][0]["facts"], [])
        self.assertIn("일치하지", result["evidence"][0]["error"])

    async def test_two_additional_rounds_limit(self):
        self.fast = ScriptedModel(plan=[{"tasks": [task("server-b", "cpu")]}], review=[
            {"tasks": [task("server-b", "process")]}, {"tasks": [task("server-b", "log.content")]},
            {"tasks": [task("server-b", "disk")]}])
        result = await self.run_query("서버 B 원인 확인")
        self.assertEqual(result["replan_rounds"], 2)
        self.assertEqual(self.fast.calls.count("review"), 2)
        self.assertEqual(len(self.mcp.calls), 3)

    async def test_twelve_command_limit(self):
        ids = [i.information_id for i in self.mcp.store.information]
        servers = list(self.mcp.store.servers)
        self.fast = ScriptedModel(plan=[{"tasks": [task(s, *ids) for s in servers]}])
        result = await self.run_query(" ".join(servers) + " 전체 정보 확인")
        self.assertEqual(len(self.mcp.calls), 12)
        self.assertEqual(result["command_call_count"], 12)
        self.assertIn("조회 한도", result["final"])

    async def test_invalid_plan_retries_once_without_commands(self):
        self.fast = ScriptedModel(plan=["null", "not JSON"])
        result = await self.run_query("서버 B CPU 확인")
        self.assertEqual(self.fast.calls, ["plan", "plan"])
        self.assertEqual(self.mcp.calls, [])
        self.assertIn("계획을 만들지 못", result["final"])

    async def test_partial_target_plan_is_rejected(self):
        self.fast = ScriptedModel(plan=[{"tasks": [task("server-a", "cpu")]}] * 2)
        await self.run_query("서버 A와 B CPU 확인")
        self.assertEqual(self.mcp.calls, [])

    async def test_json_only_preserves_api_result(self):
        result = await self.run_query("서버 B top 결과 JSON만")
        self.assertEqual(json.loads(result["final"]), self.mcp.store.result(CommandRequest(server_id="server-b", command="top")).model_dump(mode="json"))

    async def test_report_hallucinated_numbers_are_rejected(self):
        bad = {"hypotheses": [{"text": "CPU 99.99% 문제 가능성", "evidence_ids": ["e1"], "quotes": ["93.0 us"]}]}
        self.quality = ScriptedModel(report=[bad, bad])
        result = await self.run_query("서버 B CPU 상태 확인")
        self.assertEqual(result["hypotheses"], [])
        self.assertNotIn("99.99%", result["final"])
        self.assertIn("해석 검증 실패", result["final"])


class FactExtractionTests(unittest.TestCase):
    def test_socket_limitations_are_not_positive_health_claims(self):
        store = ScenarioStore()
        raw = store.result(CommandRequest(server_id='server-a', command='ss')).model_dump(mode='json')
        evidence = [{**raw, 'evidence_id':'e1', 'facts':extract_facts(raw)}]
        def check(text):
            report = InvestigationReport(hypotheses=[dict(text=text, evidence_ids=['e1'], quotes=[raw['stdout'].splitlines()[0]])])
            verify_report(report, evidence)
        for text in ['소켓 정보만으로 외부 서비스 응답 시간을 확인할 수 없습니다.',
                     '현재 소켓 정보로는 외부 서비스 정상 응답 여부를 확인할 수 없습니다.',
                     '이 출력만으로 응답 시간·패킷 손실·외부 서비스 정상 여부는 확인할 수 없습니다.',
                     '소켓 출력\\n이 출력만으로 응답 시간·패킷 손실·외부 서비스 정상 여부는 확인할 수 없습니다.']:
            with self.subTest(text=text): check(text)
        for text in ['외부 서비스 응답 시간이 정상입니다.',
                     '네트워크가 정상입니다. 소켓 정보만으로 외부 서비스 응답 시간을 확인할 수 없습니다.',
                     '소켓 정보만으로 외부 서비스 응답 시간을 확인할 수 없습니다. 따라서 외부 서비스는 정상입니다.',
                     '소켓 정보만으로 외부 서비스 응답 시간을 확인할 수 없습니다. 응답 시간은 정상입니다.']:
            with self.subTest(text=text), self.assertRaises(ValueError): check(text)

    def test_catalog_capabilities_have_parsable_fields_in_all_fixtures(self):
        store = ScenarioStore()
        for sid in store.servers:
            for command in store.commands(sid):
                with self.subTest(server=sid, command=command.command_id):
                    raw = store.result(CommandRequest(server_id=sid, command=command.command_id)).model_dump(mode="json")
                    facts = extract_facts(raw)
                    self.assertEqual({f["information_id"] for f in facts}, set(command.information_ids))
                    self.assertTrue(all(f["quote"] in raw["stdout"] or all(line in raw["stdout"] for line in f["quote"].splitlines()) for f in facts))

    def test_empty_or_unbounded_logs_are_not_completed_information(self):
        for stdout in ["", "\n".join(["2026-10-20T12:00:00+09:00 INFO ok"] * 101)]:
            self.assertEqual(extract_facts({"command_id": "tail", "os": "linux", "exit_code": 0, "stdout": stdout}), [])

    def test_report_quote_must_exist(self):
        report = InvestigationReport.model_validate({"hypotheses": [{"text": "가능성", "evidence_ids": ["e1"], "quotes": ["made up"]}]})
        with self.assertRaises(ValueError):
            verify_report(report, [{"evidence_id": "e1", "stdout": "real output", "facts": []}])


class DiagnosticSubgraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_role_cannot_query_another_domain_or_target(self):
        from agent.graph import build_diagnostic_agent
        for target,info in [("server-a","cpu"),("server-b","log.content")]:
            calls=[]
            async def decide(role,assignment,evidence,allowed):
                return dict(diagnoses=[],tasks=[task(target,info)])
            async def observe(items):
                calls.append(items);return [],True
            graph=build_diagnostic_agent("logs",decide,observe,lambda *a,**k:None)
            with self.assertRaises(ValueError):
                await graph.ainvoke(dict(assignment=[dict(requirement_id="r1",server_ids=["server-a"],information_ids=["log.content"])],evidence=[]))
            self.assertEqual(calls,[])

    async def test_local_loop_is_bounded_and_assignment_is_not_mutated(self):
        from agent.graph import build_diagnostic_agent
        assignment=[dict(requirement_id="r1",server_ids=["server-a"],information_ids=["cpu"])]
        seen=[];observations=[]
        async def decide(role,assigned,evidence,allowed):
            self.assertNotIn("log.content",allowed)
            assigned[0]["server_ids"].append("mutated-copy")
            seen.append(role)
            return dict(diagnoses=[],tasks=[task("server-a","process")])
        async def observe(items):
            observations.append(items);return [],True
        graph=build_diagnostic_agent("metrics",decide,observe,lambda *a,**k:None)
        result=await graph.ainvoke(dict(assignment=assignment,evidence=[]))
        self.assertEqual(len(seen),3);self.assertEqual(len(observations),2)
        self.assertEqual(result["rounds"],2)
        self.assertEqual(assignment[0]["server_ids"],["server-a"])

    async def test_supervisor_budget_refusal_ends_local_collection(self):
        from agent.graph import build_diagnostic_agent
        calls=[]
        async def decide(*args):
            calls.append("decide");return dict(diagnoses=[],tasks=[task("server-a","process")])
        async def observe(items):
            calls.append("observe");return [],False
        result=await build_diagnostic_agent("metrics",decide,observe,lambda *a,**k:None).ainvoke(
            dict(assignment=[dict(requirement_id="r1",server_ids=["server-a"],information_ids=["cpu"])],evidence=[]))
        self.assertEqual(calls,["decide","observe"])
        self.assertEqual(result["status"],"limited")


class TransportPreservationTests(unittest.IsolatedAsyncioTestCase):
    async def test_ps_timeout_keeps_top_and_failed_task(self):
        class BrokenMCP(FixtureMCP):
            async def call_tool(self, name, args):
                if name == "get_command_result" and args["command"] == "ps":
                    self.calls.append((args["server_id"], "ps"))
                    raise TimeoutError("private-provider-payload")
                return await super().call_tool(name, args)
        mcp = BrokenMCP()
        planner = ScriptedModel(plan=[dict(tasks=[task("server-a", "cpu", "process")])])
        result = await investigate(dict(query="서버 A CPU와 프로세스 확인", events=[]),
            SimpleNamespace(mcp=mcp, fast_llm=planner, quality_llm=QualityLLM("test", 0)))
        self.assertEqual(mcp.calls, [("server-a", "top"), ("server-a", "ps")])
        self.assertTrue(result["evidence"][0]["facts"])
        self.assertTrue(result["evidence"][1]["error"])
        self.assertIn("19.0%", result["final"])
        self.assertNotIn("private-provider-payload", json.dumps(result))
        self.assertNotEqual(result["final_validation"]["status"], "passed")


class MemoryCauseCitationTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_symptom_and_process_candidate_need_both_sources(self):
        mcp=FixtureMCP();evidence=[]
        for eid,command in [("e1","free"),("e2","ps")]:
            raw=await mcp.call_tool("get_command_result",dict(server_id="server-c",command=command))
            evidence.append({**raw,"evidence_id":eid,"command_id":command,"facts":extract_facts(raw)})
        item=dict(text="메모리 부족은 cache-worker가 메모리 60.9%를 사용하는 것과 관련될 가능성이 있습니다",
                  requirement_ids=["r1"],evidence_ids=["e2"],quote_ids=["e2:l2"])
        with self.assertRaisesRegex(ValueError,"memory.summary.*e1"):
            verify_report(InvestigationReport(hypotheses=[item]),evidence)
        item.update(evidence_ids=["e1","e2"],quote_ids=["e1:l2","e2:l2"])
        verify_report(InvestigationReport(hypotheses=[item]),evidence)


class SupportingContextBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_disk_context_does_not_grant_logs_disk_query_permission(self):
        from agent.graph import build_diagnostic_agent
        mcp=FixtureMCP();raw=await mcp.call_tool("get_command_result",dict(server_id="server-d",command="df"))
        evidence=[{**raw,"evidence_id":"e1","facts":extract_facts(raw)}];seen=[];queried=[]
        async def decide(role,assignment,observations,allowed):
            seen.extend(f["information_id"] for e in observations for f in e["facts"])
            self.assertNotIn("disk",allowed)
            return dict(diagnoses=[],tasks=[task("server-d","disk")])
        async def observe(items):queried.extend(items);return evidence,True
        graph=build_diagnostic_agent("logs",decide,observe,lambda *a,**k:None)
        with self.assertRaises(ValueError):
            await graph.ainvoke(dict(assignment=[dict(requirement_id="r1",server_ids=["server-d"],information_ids=["log.usage"],
                supporting_for=["disk"],supporting_server_ids=["server-d"])],evidence=evidence))
        self.assertEqual(seen,["disk"])
        self.assertEqual(queried,[])


    async def test_handoff_requires_positive_matching_metrics_quote_and_target(self):
        from agent.investigation import disk_log_support
        mcp=FixtureMCP();raw=await mcp.call_tool("get_command_result",dict(server_id="server-d",command="df"))
        e={**raw,"evidence_id":"e1","facts":extract_facts(raw)}
        requirement=dict(requirement_id="r1",kind="diagnosis",description="디스크 원인",server_ids=["server-d"],information_ids=["disk"])
        row=e["facts"][0]["quote"].splitlines()[0]
        assessment=dict(source_role="metrics",symptom_observed=True,requirement_ids=["r1"],evidence_ids=["e1"],quotes=[row])
        support=disk_log_support([requirement],[e],[assessment])
        self.assertEqual(support[0]["supporting_server_ids"],["server-d"])
        self.assertEqual(requirement["information_ids"],["disk"])
        for delta in [dict(source_role="logs"),dict(symptom_observed=False),dict(requirement_ids=["r2"]),
                      dict(evidence_ids=["other"]),dict(quotes=["not an observed row"])]:
            with self.subTest(delta=delta):self.assertEqual(disk_log_support([requirement],[e],[{**assessment,**delta}]),[])
        self.assertEqual(disk_log_support([{**requirement,"server_ids":["server-a"]}],[e],[assessment]),[])
        self.assertEqual(disk_log_support([requirement],[{**e,"exit_code":1}],[assessment]),[])


    def test_support_preserves_all_selected_log_rows_and_their_cause_binding(self):
        from agent.investigation import ground_storage_support, bind_cause, cause_catalog
        from agent.investigation_models import DiagnosisAssessment, CauseSelection
        evidence=[]
        for eid,command,stdout in [("d","df","Filesystem Size Used Avail Use% Mounted on\n/dev/root 100G 95G 5G 95% /\n"),
            ("u","du","83886080 /var/log/\n"),
            ("l","tail","2026-10-22T01:00:00Z INFO ready\n2026-10-22T01:01:00Z ERROR failed reason=connection_refused\n")]:
            e=dict(evidence_id=eid,server_id="server-x",command_id=command,os="linux",exit_code=0,error=None,stdout=stdout)
            e["facts"]=extract_facts(e);evidence.append(e)
        requirement=dict(requirement_id="r1",kind="diagnosis",description="디스크와 로그 원인",server_ids=["server-x"],information_ids=["disk","log.content"])
        assignment={**requirement,"information_ids":["log.usage","log.content"],"supporting_for":["disk"],"supporting_server_ids":["server-x"]}
        rows=evidence[-1]["stdout"].splitlines()
        assessment=DiagnosisAssessment(text="선택한 로그 관측",requirement_ids=["r1"],evidence_ids=["l"],quotes=rows,symptom_observed=True)
        factor=next(c for c in cause_catalog(evidence,[requirement]) if c["kind"]=="log_reason")
        selected=CauseSelection(requirement_id="r1",factor_id=factor["factor_id"])
        bind_cause(selected,evidence,[requirement],[assessment.model_dump()])
        ground_storage_support(assessment,evidence,[assignment])
        self.assertTrue(set(rows)<=set(assessment.quotes))
        candidate=bind_cause(selected,evidence,[requirement],[assessment.model_dump()])
        self.assertIn(rows[1],candidate.quotes)


    def test_support_retains_full_model_quote_budget_and_bounded_derived_sources(self):
        from agent.investigation import ground_storage_support, verify_report
        from agent.investigation_models import DiagnosisAssessment, InvestigationReport
        from pydantic import ValidationError
        rows=[f"2026-10-22T01:{i:02}:00Z INFO sample-{i}" for i in range(11)] + ["2026-10-22T01:12:00Z ERROR failed reason=connection_refused"]
        evidence=[]
        for eid,command,stdout in [("u","du","83886080 /var/log/\n"),("l","tail","\n".join(rows)+"\n")]:
            item=dict(evidence_id=eid,server_id="server-x",command_id=command,os="linux",exit_code=0,error=None,stdout=stdout)
            item["facts"]=extract_facts(item);evidence.append(item)
        assignment=dict(requirement_id="r1",kind="diagnosis",description="디스크와 로그",server_ids=["server-x"],
            information_ids=["log.usage","log.content"],supporting_for=["disk"],supporting_server_ids=["server-x"])
        for selected in [rows,["\n".join(rows)]]:
            with self.subTest(selected_quotes=len(selected)):
                assessment=DiagnosisAssessment(text="선택 로그",requirement_ids=["r1"],evidence_ids=["l"],quotes=selected,symptom_observed=True)
                ground_storage_support(assessment,evidence,[assignment])
                self.assertTrue(set(selected)<=set(assessment.quotes))
                self.assertEqual(len(assessment.quotes),len(selected)+1)
                report=InvestigationReport(hypotheses=[assessment.model_dump(exclude={"symptom_observed"})])
                verify_report(report,evidence)
        with self.assertRaises(ValidationError):
            DiagnosisAssessment(text="선택 로그",requirement_ids=["r1"],evidence_ids=["l"],quote_ids=[f"l:l{i}" for i in range(13)],symptom_observed=True)


class MacMemoryEvidenceBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def evidence(self,sid):
        result=ScenarioStore().result(CommandRequest(server_id=sid,command="top")).model_dump(mode="json")
        result.update(evidence_id="e1",facts=extract_facts(result))
        return result

    def report(self,text,evidence):
        quote=next(f["quote"] for f in evidence["facts"] if f["information_id"]=="memory.summary")
        return InvestigationReport(hypotheses=[dict(text=text,evidence_ids=["e1"],quotes=[quote])])

    def test_mac_memory_counts_do_not_prove_pressure_or_health(self):
        for sid in ("server-e","server-g"):
            e=self.evidence(sid)
            for text in ("이는 메모리 부족 상태가 아닙니다.","메모리 압박 상태입니다.","메모리는 정상입니다.","메모리가 충분합니다.","메모리 상태는 양호합니다.","메모리가 넉넉합니다."):
                with self.subTest(sid=sid,text=text):
                    with self.assertRaises(ValueError):verify_report(self.report(text,e),[e])

    def test_mac_memory_uncertainty_does_not_license_an_adjacent_claim(self):
        e=self.evidence("server-g")
        text="macOS의 사용·미사용 메모리와 페이지 통계만으로 메모리 압박 여부를 확정할 수 없습니다. 메모리 부족 상태가 아닙니다."
        with self.assertRaises(ValueError):verify_report(self.report(text,e),[e])

    def test_mac_memory_observations_and_explicit_limit_are_allowed(self):
        e=self.evidence("server-g")
        text="메모리 사용량은 15360 MiB이고 미사용 메모리는 1024 MiB입니다. macOS의 사용·미사용 메모리와 페이지 통계만으로 메모리 압박 여부를 확정할 수 없습니다."
        verify_report(self.report(text,e),[e])

    async def test_specialist_retries_false_health_claim_before_final_output(self):
        class Expert(QualityLLM):
            def __init__(self):super().__init__("scripted",0);self.calls=[]
            def diagnostic_json(self,payload,schema):
                self.calls.append(payload.copy())
                text="이는 메모리 부족 상태가 아닙니다." if len(self.calls)==1 else (
                    "메모리 사용량은 15360 MiB이고 미사용 메모리는 1024 MiB입니다. "
                    "macOS의 사용·미사용 메모리와 페이지 통계만으로 메모리 압박 여부를 확정할 수 없습니다.")
                quote=next(q for q in payload["quote_catalog"] if "PhysMem:" in q["text"])
                return json.dumps(dict(diagnoses=[dict(text=text,requirement_ids=["r1"],evidence_ids=[quote["evidence_id"]],
                    quote_ids=[quote["quote_id"]],symptom_observed=False)],tasks=[])),{"usd":0}
        model=Expert();mcp=FixtureMCP()
        result=await investigate(dict(query="현재 메모리 상태를 확인해줘",scenario_id="server-g",single_server_mode=True,
            events=[],history=[],request_decision=dict(requested_information_ids=["memory.summary"],needs_diagnosis=True,
                requested_results=[dict(description="메모리 상태 진단",kind="diagnosis",information_ids=["memory.summary"])])),
            SimpleNamespace(mcp=mcp,quality_llm=model,fast_llm=FastLLM("fake",0),monitoring_llm=FastLLM("fake",0)))
        self.assertEqual(len(model.calls),2)
        self.assertIn("단정할 수 없습니다",model.calls[1]["validation_feedback"])
        self.assertNotIn("부족 상태가 아닙니다",result["final"])
        self.assertIn("메모리 압박 여부를 확정할 수 없습니다",result["final"])
        self.assertEqual(mcp.calls,[("server-g","top")])
        self.assertEqual(result["hypotheses"],[])

    def test_unquoted_pressure_log_cannot_license_a_memory_health_claim(self):
        e=self.evidence("server-g")
        log=ScenarioStore().result(CommandRequest(server_id="server-g",command="tail")).model_dump(mode="json")
        log.update(evidence_id="e2",facts=extract_facts(log))
        report=self.report("이는 메모리 부족 상태가 아닙니다.",e)
        report.hypotheses[0].evidence_ids.append("e2")
        with self.assertRaises(ValueError):verify_report(report,[e,log])
