"""Run with: python -m unittest discover -s tests -v (no LLM/API credentials)."""
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.command_query import parse_command_query
from monitoring.api import ScenarioStore, create_router
from monitoring.client import request_local_api

ROOT = Path(__file__).resolve().parent.parent


class CommandAPITests(unittest.TestCase):
    def setUp(self):
        self.store = ScenarioStore()
        app = FastAPI()
        app.include_router(create_router(self.store))
        self.client = TestClient(app)

    def test_server_inventory_does_not_expose_scenario_answers(self):
        response = self.client.get("/monitoring/servers")
        self.assertEqual(response.status_code, 200)
        servers = response.json()
        self.assertEqual(len(servers), 8)
        self.assertEqual({s["os"] for s in servers}, {"linux", "macos"})
        self.assertTrue(all(set(s) == {"server_id", "display_name", "os"} for s in servers))

    def test_os_specific_command_catalogs(self):
        linux = self.client.get("/monitoring/servers/server-a/commands").json()
        macos = self.client.get("/monitoring/servers/server-e/commands").json()
        self.assertIn("free", {c["command_id"] for c in linux})
        self.assertNotIn("vm_stat", {c["command_id"] for c in linux})
        self.assertIn("vm_stat", {c["command_id"] for c in macos})
        self.assertNotIn("free", {c["command_id"] for c in macos})

    def test_every_command_returns_saved_raw_output_and_timestamp(self):
        for server in self.store.servers.values():
            for command in self.store.commands(server.server_id):
                with self.subTest(server=server.server_id, command=command.command_id):
                    response = self.client.post("/monitoring/commands", json={
                        "server_id": server.server_id, "command": command.command_id})
                    self.assertEqual(response.status_code, 200)
                    result = response.json()
                    self.assertEqual(result["stdout"], server.commands[command.command_id].stdout)
                    self.assertEqual(result["source"], "fixture")
                    self.assertEqual(result["command"], command.command)
                    self.assertEqual(result["observed_at"], "2026-10-20T12:00:01+09:00")
                    self.assertNotIn("scenario", result)

    def test_complete_command_is_accepted_without_modification(self):
        command = "ls -lah /var/log/"
        response = self.client.post("/monitoring/commands", json={
            "server_id": "server-a", "command": command})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["command"], command)

    def test_unknown_server_is_not_replaced(self):
        response = self.client.post("/monitoring/commands", json={
            "server_id": "server-z", "command": "top"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["code"], "server_not_found")

    def test_injection_paths_and_unsupported_options_are_rejected(self):
        for command in ["top; touch /tmp/should-not-exist", "$(id)", "cat /etc/passwd", "ls /", "top -n 999"]:
            with self.subTest(command=command):
                response = self.client.post("/monitoring/commands", json={
                    "server_id": "server-a", "command": command})
                self.assertEqual(response.status_code, 422)
        response = self.client.post("/monitoring/commands", json={
            "server_id": "../../etc/passwd", "command": "top"})
        self.assertEqual(response.status_code, 404)

    def test_other_os_command_is_rejected(self):
        response = self.client.post("/monitoring/commands", json={
            "server_id": "server-e", "command": "free"})
        self.assertEqual(response.status_code, 422)

    def test_missing_empty_and_extra_fields_are_rejected(self):
        for payload in [{}, {"server_id": "server-a", "command": ""},
                        {"server_id": "server-a", "command": "top", "shell": True}]:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/monitoring/commands", json=payload).status_code, 422)

    def test_failed_command_is_preserved(self):
        result = self.store.servers["server-a"].commands["top"]
        result.stdout, result.stderr, result.exit_code = "", "permission denied", 1
        response = self.client.post("/monitoring/commands", json={
            "server_id": "server-a", "command": "top"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["exit_code"], 1)
        self.assertEqual(response.json()["stderr"], "permission denied")
        self.assertEqual(response.json()["stdout"], "")

    def test_distinct_scenarios_are_deterministic(self):
        for server_id, expected in [("server-a", "81.0 id"), ("server-b", "4.0 id"),
                                    ("server-g", "15360M used"), ("server-h", "95% /")]:
            command = "df" if server_id == "server-h" else "top"
            payload = {"server_id": server_id, "command": command}
            first = self.client.post("/monitoring/commands", json=payload).json()
            second = self.client.post("/monitoring/commands", json=payload).json()
            self.assertEqual(first, second)
            self.assertIn(expected, first["stdout"])

    def test_explicit_chat_query_parser(self):
        for query in ["서버 B top 조회해줘", "server-b top", "서버B에서 top 결과 보여줘", "서버 B top 조회해줘."]:
            self.assertEqual(parse_command_query(query), ("server-b", "top"))
        self.assertIsNone(parse_command_query("payments-service 상태 확인"))
        self.assertIsNone(parse_command_query("서버 ABC top"))


class LocalClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_connection_failure_is_not_fabricated_as_a_normal_result(self):
        def offline(request):
            raise httpx.ConnectError("offline", request=request)
        client = httpx.AsyncClient(transport=httpx.MockTransport(offline))
        with patch("monitoring.client.httpx.AsyncClient", return_value=client):
            with self.assertRaisesRegex(ValueError, "연결할 수 없습니다"):
                await request_local_api("GET", "/monitoring/servers")

    async def test_timeout_is_reported_separately(self):
        def timeout(request):
            raise httpx.ReadTimeout("timeout", request=request)
        client = httpx.AsyncClient(transport=httpx.MockTransport(timeout))
        with patch("monitoring.client.httpx.AsyncClient", return_value=client):
            with self.assertRaisesRegex(ValueError, "시간이 초과"):
                await request_local_api("GET", "/monitoring/servers")

    async def test_nonlocal_base_url_is_rejected_before_request(self):
        for url in ["https://example.com", "http://127.0.0.1.evil.invalid", "http://user@localhost:8100"]:
            with self.subTest(url=url), patch.dict(os.environ, {"LOCAL_MONITORING_API_URL": url}):
                with self.assertRaisesRegex(ValueError, "이 컴퓨터"):
                    await request_local_api("GET", "/monitoring/servers")


class LocalRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.base_url = f"http://127.0.0.1:{port}"
        cls.temp = tempfile.TemporaryDirectory()
        cls.log = open(Path(cls.temp.name) / "server.log", "w+")
        cls.process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(port),
        ], cwd=ROOT, stdout=cls.log, stderr=cls.log, env={
            **os.environ, "LLM_BACKEND": "fake", "SUPERVISOR_VERSION": "v2",
            "LOCAL_MONITORING_API_URL": cls.base_url,
            "SESSION_DB_PATH": str(Path(cls.temp.name) / "session.db"),
            "LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false",
        })
        for _ in range(100):
            try:
                if httpx.get(cls.base_url + "/health", timeout=0.5, trust_env=False).status_code == 200:
                    return
            except httpx.RequestError:
                pass
            if cls.process.poll() is not None:
                break
            time.sleep(0.1)
        cls.process.terminate()
        cls.process.wait(timeout=10)
        cls.log.seek(0)
        message = cls.log.read()
        cls.log.close()
        cls.temp.cleanup()
        raise RuntimeError(f"Test FastAPI failed to start: {message}")

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait(timeout=5)
        if os.environ.get("LOCAL_COMMAND_TEST_DEBUG") == "1":
            cls.log.seek(0)
            print(cls.log.read())
        cls.log.close()
        cls.temp.cleanup()

    def test_chat_graph_mcp_http_round_trip_preserves_json(self):
        response = httpx.post(self.base_url + "/chat", json={
            "message": "서버 B top 조회해줘", "thread_id": "local-command-tests",
        }, timeout=10, trust_env=False)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        actual = json.loads(body["reply"])
        expected = httpx.post(self.base_url + "/monitoring/commands", json={
            "server_id": "server-b", "command": "top"}, trust_env=False).json()
        self.assertEqual(actual, expected)
        self.assertEqual(body["metadata"]["route"], "commands")
        self.assertEqual(body["metadata"]["cost_usd"], 0)

    def test_macos_chat_command(self):
        body = httpx.post(self.base_url + "/chat", json={
            "message": "서버 G vm_stat 조회해줘", "thread_id": "mac-command-tests",
        }, timeout=10, trust_env=False).json()
        self.assertEqual(json.loads(body["reply"])["os"], "macos")
        self.assertIn("page size of 16384", json.loads(body["reply"])["stdout"])

    def test_unknown_server_survives_mcp_error_as_clear_answer(self):
        body = httpx.post(self.base_url + "/chat", json={
            "message": "서버 Z top 조회해줘", "thread_id": "unknown-command-tests",
        }, timeout=10, trust_env=False).json()
        self.assertIn("server_not_found", body["reply"])
        self.assertNotIn("auth-service", body["reply"])

    def test_stream_completes_and_returns_same_raw_result(self):
        response = httpx.get(self.base_url + "/chat/stream", params={
            "message": "서버 D df 조회해줘", "thread_id": "stream-command-tests",
        }, timeout=10, trust_env=False)
        events = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        final = next(e for e in events if e["type"] == "FinalAnswer")
        self.assertEqual(json.loads(final["reply"])["command_id"], "df")
        self.assertIn("95% /", json.loads(final["reply"])["stdout"])
        self.assertIn("data: [DONE]", response.text)

    def test_basic_legacy_chat_still_works(self):
        response = httpx.post(self.base_url + "/chat", json={
            "message": "안녕하세요", "thread_id": "legacy-command-tests",
        }, timeout=10, trust_env=False)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["metadata"]["route"], "chat")

    def test_legacy_incident_pipeline_still_works(self):
        response = httpx.post(self.base_url + "/chat", json={
            "message": "auth-service 장애 원인 진단해줘", "thread_id": "legacy-incident-tests",
        }, timeout=10, trust_env=False)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["metadata"]["route"], "incident")
        self.assertIn("CPU", response.json()["reply"])

    def test_two_chat_requests_can_call_back_into_same_fastapi(self):
        async def run():
            async with httpx.AsyncClient(base_url=self.base_url, timeout=10, trust_env=False) as client:
                return await asyncio.gather(*[
                    client.post("/chat", json={"message": query, "thread_id": f"concurrent-{i}"})
                    for i, query in enumerate(["서버 B top", "서버 G vm_stat"])
                ])
        responses = asyncio.run(run())
        self.assertEqual([r.status_code for r in responses], [200, 200])
        self.assertEqual([json.loads(r.json()["reply"])["server_id"] for r in responses],
                         ["server-b", "server-g"])


if __name__ == "__main__":
    unittest.main()
