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
