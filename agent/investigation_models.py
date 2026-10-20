"""Typed model contracts and an offline planner for integration tests."""
import json
import re

from pydantic import BaseModel, ConfigDict, Field


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server_id: str
    goal: str = Field(min_length=1, max_length=200)
    information_ids: list[str] = Field(min_length=1, max_length=10)


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tasks: list[TaskSpec] = Field(max_length=12)


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=400)
    evidence_ids: list[str] = Field(min_length=1)
    quotes: list[str] = Field(min_length=1)


class InvestigationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypotheses: list[Hypothesis] = Field(max_length=2)


INSTRUCTIONS = {
    "plan": """서버 조사 계획을 JSON으로 작성하라. tasks 배열의 각 항목은
server_id, goal(한국어 조사 목적), information_ids(확인할 정보 ID 목록)이다.
명령어를 직접 생성하지 말고 제공된 정보 ID를 선택하라. targets의 모든 서버를 포함하라.
CPU면 cpu, 프로세스면 process, 메모리면 memory.summary/memory.detail,
디스크면 disk, 소켓이면 network, 로그 내용이면 log.content,
로그 파일 목록이면 log.files, 로그 저장량이면 log.usage, OS면 system을 사용하라.
느리다는 포괄적 요청은 우선 CPU·메모리를 확인하고, 관련 정보는 후속 조사로 나눠라.
사용자가 특정 정보들을 명시하면 모두 포함하라. 시계열은 history, 배포 이력은 deployment,
응답시간·패킷 손실은 network.performance로 기록하라(현재 미지원).
예: {"tasks":[{"server_id":"server-b","goal":"CPU 병목 확인","information_ids":["cpu"]}]}.
현재 요청을 우선하고 이전 이력의 다른 요청을 반복하지 마라. JSON 객체만 출력하라.""",
    "review": """수집된 사실로 사용자 요청을 해결하기 위해 추가로 확인할 서버 정보를 선택하라.
JSON tasks 배열의 항목은 server_id, goal, information_ids이다. 새로 필요한 task만 반환하라.
obtained_information은 서버별로 이미 확보한 모든 정보 ID다. 그 ID들은 절대 다시 요구하지 마라. 추가 확인이 불필요하면 {"tasks":[]}를 반환하라.
CPU 과다 시 process와 log.content, 메모리 부족 시 memory.detail/process/log.content,
디스크 부족 시 log.usage/log.files/log.content를 검토하되 요청 목적에 필요한 것만 선택하라.
정상 상태 조회에 무관한 조사를 추가하지 마라. targets 안의 서버만 사용하라.
고정된 명령어 목록을 만들지 말고 정보 ID를 선택하라. JSON 객체만 출력하라.""",
    "report": """관측 사실에 근거한 원인 후보를 한국어 JSON으로 작성하라.
형식: {"hypotheses":[{"text":"...일 가능성이 있습니다", "evidence_ids":["e1"],
"quotes":["해당 evidence quote_candidates에서 복사한 정확한 한 줄"]}]}.
사실과 후보를 구분하라. 원인을 확정하거나 조회하지 않은 사실의 부재를 단정하지 마라.
가장 강한 근거가 있는 후보 하나를 우선하라. 정상 지표인 영역을 원인 후보로 나열하지 마라. 로그 용량이 크더라도 디스크 전체를 모두 차지했다고 과장하지 마라. 단일 표본으로 증가·감소·추세를 주장하지 마라. 수치를 새로 만들지 마라. 후보마다 유효한 evidence ID와 실제 원문 인용이 필요하다. quotes는 quote_candidates의 한 줄을 그대로 복사하라. 여러 줄을 합치거나 공백·수치를 고치지 마라.
근거가 부족하거나 원인 분석 요청이 아니면 {"hypotheses":[]}도 올바른 답이다.
소켓만으로 외부 서비스 응답·RTT·패킷 손실을 확인했다고 말하지 마라.
최근 로그 발췌에 오류가 없다고 전체 기간에 오류가 없다고 단정하지 마라. JSON 객체만 출력하라.""",
}


def fake_decision(purpose: str, payload: dict) -> dict:
    """Deterministic test backend. Real backends always use their model."""
    if purpose == "report":
        return {"hypotheses": []}
    query = payload["query"].lower()
    tasks = []
    for server_id in payload["targets"]:
        ids = []
        if purpose == "plan":
            if re.search(r"시간별|추세|최근\s*\d+\s*시간", query): ids.append("history")
            if re.search(r"배포.*이력|이력.*배포", query): ids.append("deployment")
            if re.search(r"응답\s*시간|패킷\s*손실|rtt", query): ids.append("network.performance")
            if "cpu" in query: ids.append("cpu")
            if re.search(r"메모리|memory|swap|페이지", query): ids.extend(["memory.summary", "memory.detail"])
            if re.search(r"프로세스|process", query): ids.append("process")
            if re.search(r"디스크|disk|용량", query): ids.append("disk")
            if re.search(r"네트워크|network|소켓|포트", query): ids.append("network")
            if re.search(r"로그.*(파일|목록|크기)", query): ids.append("log.files")
            elif "로그" in query or "log" in query: ids.append("log.content")
            if re.search(r"os|운영체제|커널|아키텍처", query): ids.append("system")
            if not ids: ids = ["cpu", "memory.summary"]
        else:
            facts = [f for e in payload["evidence"] if e["server_id"] == server_id for f in e["facts"]]
            obtained = {f["information_id"] for f in facts}
            wants_diagnosis = bool(re.search(r"원인|느리|느려|부족|과부하|장애|분석|왜", query))
            if wants_diagnosis:
                for fact in facts:
                    fields = fact["fields"]
                    if fact["information_id"] == "cpu" and fields.get("usage_percent", 0) >= 80:
                        ids.extend(["process", "log.content"])
                    if fact["information_id"] == "memory.summary":
                        low = (fields.get("available_mib", 1e9) < 1024 or fields.get("unused_mib", 1e9) <= 1024)
                        if low: ids.extend(["memory.detail", "process", "log.content"])
                    if fact["information_id"] == "disk" and any(p["usage_percent"] >= 90 for p in fields["entries"]):
                        ids.extend(["log.usage", "log.files", "log.content"])
            ids = [i for i in ids if i not in obtained]
        if ids:
            tasks.append({"server_id": server_id, "goal": "요청 정보 확인" if purpose == "plan" else "관측 결과 추가 확인",
                          "information_ids": list(dict.fromkeys(ids))})
    return {"tasks": tasks}
