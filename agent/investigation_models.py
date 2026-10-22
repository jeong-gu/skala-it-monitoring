"""Typed model contracts and an offline planner for integration tests."""
import json
import re

from pydantic import BaseModel, ConfigDict, Field

from .request_ledger import RequirementSpec
from .answer_contracts import AnswerStatement, Calculation, RequirementCheck


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server_id: str
    goal: str = Field(min_length=1, max_length=200)
    information_ids: list[str] = Field(min_length=1, max_length=10)


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tasks: list[TaskSpec] = Field(max_length=12)
    requirements: list[RequirementSpec] = Field(default_factory=list, max_length=16)


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=400)
    evidence_ids: list[str] = Field(min_length=1)
    # 12 model-selected quotes plus at most 12 command-derived support sources.
    quotes: list[str] = Field(default_factory=list, max_length=24)
    quote_ids: list[str] = Field(default_factory=list, max_length=12, exclude=True)
    requirement_ids: list[str] = Field(default_factory=list, max_length=16)


class DiagnosisAssessment(AnswerStatement):
    """Observation of the requested symptom before selecting its cause."""
    symptom_observed: bool


class CauseSelection(BaseModel):
    """Internal selection; source-bound prose is rendered by the program."""
    model_config = ConfigDict(extra="forbid")
    requirement_id: str
    factor_id: str


class InvestigationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    diagnoses: list[DiagnosisAssessment] = Field(default_factory=list, max_length=8)
    hypotheses: list[Hypothesis] = Field(max_length=8)
    causes: list[CauseSelection] = Field(default_factory=list, max_length=8)
    answers: list[AnswerStatement] = Field(default_factory=list, max_length=8)
    calculations: list[Calculation] = Field(default_factory=list, max_length=8)


class DiagnosticDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    diagnoses: list[DiagnosisAssessment] = Field(default_factory=list, max_length=8)
    tasks: list[TaskSpec] = Field(default_factory=list, max_length=6)


DIAGNOSTIC_INSTRUCTIONS = """너는 배정된 영역의 증상 확인 전문 에이전트다. 제한된 관측만 검토해 JSON으로 답한다.
작업은 원인 추측이나 대처방안 작성이 아니라, 요청한 증상이 현재 표본에 나타나는지 판단하고 부족한 정보만 요청하는 것이다.
먼저 tasks에 필요한 추가 관측을 결정하고, diagnoses에 배정된 요구별 판단을 기록한다. CPU·메모리 증상이 있는데 process 근거가 없으면 process task를 빠뜨리지 마라.
requirements에 배정된 정보만 판단한다. 동일 requirement_id는 전체 diagnoses에서 한 번만 사용한다. CPU 요구에 top의 부수적 메모리 사실이 있더라도 메모리 진단을 추가하지 마라. target_os의 실제 OS를 사용하고, Linux에 macOS 한계 문장을 적용하지 마라.
requirements의 diagnosis 요구마다 diagnoses에 text(관측된 원본 수치와 판단 이유), requirement_ids, evidence_ids, quote_ids, symptom_observed를 기록한다.
코어 수가 없으면 Load Average 수치만으로 과부하를 단정하지 마라. 시스템 CPU는 cpu.usage_percent 기준이고 process.cpu_percent는 코어 합산이라 서로 다르다. 전체 CPU 사용량이 낮고 대부분 idle이면 전체 CPU가 높다는 전제는 관측되지 않은 것이다. 가장 큰 프로세스가 있다는 사실만으로 과부하라고 판단하지 마라.
메모리는 available·swap·사용 용량과 할당 오류 등 확보한 사실을 구분하라. used 비율을 암산해 text에 쓰지 마라. 페이지·unused와 Linux available을 동일시하지 마라.
디스크는 파일시스템 사용률, 로그는 관측 구간의 실제 메시지를 보라. 시간 추세·누수·원인 확정은 현재 표본으로 판단하지 않는다.
macOS의 top·vm_stat에서 사용/미사용 용량·페이지 수만 확인했으면 메모리 부족 또는 정상 상태를 단정하지 마라. 조사 목적을 구분하라: 메모리 사용 상태·점유 원인을 묻고 사용량이 관측되면 symptom_observed=true로 사용량의 기여 프로세스를 조사한다. 메모리 압박 자체를 묻고 압력 근거가 없으면 false(미확인)다. 압박 미확인을 사용량 관측이나 점유 원인 조사 생략의 이유로 쓰지 마라. 원본 수치와 별도 문장 "macOS의 사용·미사용 메모리와 페이지 통계만으로 메모리 압박 여부를 확정할 수 없습니다."를 사용하라.
증상이 없으면 symptom_observed=false, 현재 값과 단일 표본의 한계를 설명한다. 이 경우 불필요한 원인 수집 tasks는 []다.
증상이 있고 CPU·메모리 원인 후보 조사에 process 근거가 아직 없으면 allowed_information_ids 중 process를 tasks로 요청한다. 요약 사용량만으로 원인 후보를 조사했다고 끝내지 마라. 이미 얻은 정보는 다시 요청하지 마라.
supporting_for가 있으면 다른 전문 에이전트의 검증된 증상을 위한 추가 근거 수집이다. read_only_information_ids의 관측은 맥락과 인용에만 쓰고 해당 정보를 조회하는 tasks는 만들지 마라.
supporting_for에 disk가 있으면 supporting_server_ids의 log.usage가 없을 때 tasks로 요청한다. 확보한 뒤 원본 로그 저장량을 인용해 diagnoses에 기록하고, 그것만으로 전체 디스크 부족의 주원인을 확정하지 마라. 원래 요구 ID를 유지한다.
role=logs이면 로그 파일·저장량·내용만 요청한다. role=metrics이면 수치·프로세스·소켓 정보만 요청한다. 명령문을 생성하지 않는다.
text의 모든 수치는 인용 근거의 원본 값이어야 한다. 계산·환산·퍼센트를 새로 만들지 마라. quote_ids에는 quote_catalog의 실제 행 ID만 선택한다.
조회가 더 필요해도 이미 관측 가능한 증상은 판단하라. 증상 판단의 근거 자체가 없을 때만 diagnoses=[]로 두고 필요한 조회를 요청한다.
증상 관측 여부와 원인 확정 여부를 혼동하지 마라. 고부하가 관측돼도 그 원인은 아직 확정되지 않았다.
"""


class AnswerReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checks: list[RequirementCheck] = Field(max_length=32)
    missing_requirements: list[RequirementSpec] = Field(default_factory=list, max_length=8)
    repair_tasks: list[TaskSpec] = Field(default_factory=list, max_length=8)
    rewrite_instructions: str = Field(default="", max_length=1000)


INSTRUCTIONS = {
    "plan": """서버 조사 계획을 JSON으로 작성하라. tasks 배열의 각 항목은
server_id, goal(한국어 조사 목적), information_ids(확인할 정보 ID 목록)이다.
requirements 배열에 원래 질문에서 반드시 답해야 하는 요구를 각각 기록하라. 각 항목은 description, server_ids, information_ids, kind(lookup/calculation/comparison/diagnosis/explanation/remediation)이다.
대처 요구는 remediation으로 별도 기록하고, CPU·메모리·디스크 등 명시된 문제별로 나눠라. 각 대처 요구의 information_ids에는 해당 주제의 정보 ID를 넣어라.
서버 대처 전에 확인할 상태 조회 요구도 lookup으로 별도 기록하라. 매뉴얼 검색을 위한 가짜 정보 ID나 Task는 만들지 마라.
조회 task와 별개로 계산·비교·원인 후보·확인 불가능한 요청도 빠뜨리지 마라. 요구한 시점/기간·수치의 의미·출력 단위는 description에 보존하라.
사용/전체 메모리의 퍼센트는 단순 lookup이 아니라 kind=calculation이다. 계산이라는 단어가 없어도 비율·차이·환산 결과를 요구하면 calculation으로 기록하라. 원본 수치 확보만으로 계산 요구를 대체하지 마라.
명령어를 직접 생성하지 말고 제공된 정보 ID를 선택하라. targets의 모든 서버를 포함하라.
single_server_mode=true이면 targets는 한 모니터링 서버의 선택된 자원 상태 키다. 다른 서버를 고르거나 이름을 묻지 말고 그 키만 사용하라. goal과 설명에는 서버 A/B 대신 모니터링 서버라고 작성하라.
같은 서버의 같은 목적은 한 task로 합치고 필요한 최소 목록만 작성하라. 권한·도구 설치·설정·종료 영향은 별도 관측 Task를 만들지 말고 매뉴얼 적용 전 미확인 조건으로 남긴다.
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
obtained_information은 서버별로 이미 확보한 모든 정보 ID다. 그 ID들은 절대 다시 요구하지 마라. 추가 확인이 불필요하면 {"tasks":[],"requirements":[]}를 반환하라. review의 requirements는 항상 []이며 새 사용자 요구를 만들지 않는다.
추가 task는 현재 query에서 요구한 조사 목적과 관측된 이상을 설명하는 데 직접 필요한 경우만 선택하라. CPU 요청을 디스크·로그 저장량 조사로 넓히지 마라.
메모리.detail이 이미 free 결과에 포함돼 있으면 그것만 새 task로 요청하고 종료하지 마라. 프로세스 원인 후보가 필요하고 아직 process가 없으면 이를 선택하라. 사용량이 많다는 사실 자체를 같은 증상의 원인 설명으로 대신하지 마라.
CPU 과다 시 process와 log.content, 메모리 부족 시 memory.detail/process/log.content,
디스크 부족 시 log.usage/log.files/log.content를 검토하되 요청 목적에 필요한 것만 선택하라.
정상 상태 조회에 무관한 조사를 추가하지 마라. targets 안의 서버만 사용하라.
request_decision.needs_diagnosis=false인 계산·값 조회는 확보된 피연산자로 끝내라. CPU 원인 질문에 메모리·디스크·로그 용량 조사를 덧붙이지 마라. CPU가 높다는 전제가 현재 관측에서 뒷받침되지 않으면 불필요한 원인 수집을 멈추고 그 관측과 한계를 설명하라.
고정된 명령어 목록을 만들지 말고 정보 ID를 선택하라. JSON 객체만 출력하라.""",
    "final_review": """원래 질문 query와 실제 답변 draft_answer를 대조하라. JSON만 반환한다.
CPU Load Average만으로 과부하를 확정하는 문장은 코어 수가 없으면 근거 부족이다. CPU 사용률의 높은 값과 Load Average를 혼동하지 않았는지도 검토한다.
먼저 query만 기준으로 요청 동작을 읽고 requirements에서 빠지거나 다른 kind에 합쳐진 동작을 missing_requirements에 기록한 뒤 checks를 작성하라. requirements와 request_decision은 오류가 있을 수 있는 초안이지 원문 요구의 완전성을 보증하지 않는다.
예: '상태와 원인을 조사하고 해결 방법'은 lookup·diagnosis·remediation 세 종류다. lookup description에 '원인'이 포함돼 있어도 diagnosis 요구가 따로 없으면 누락이다. 실제 수치와 매뉴얼만으로 원인 조사까지 충족했다고 판정하지 마라. '원인을 확정할 수 없다'는 한계 문장만으로 요구된 후보 조사를 수행한 것은 아니다.
반대로 '상태를 확인하고 필요한 조치'는 원인 분석을 별도로 요청한 것이 아니므로 diagnosis를 추가하지 않는다. '원인은 조사하지 말고 상태와 조치만'의 명시적 제외를 존중한다. 동작 판단은 전체 원문의 의미로 하며 단어 등장만으로 요구를 추가하지 않는다.
각 requirements.evaluation_focus는 관측된 조건에 맞춘 검토 대상이다. 원래 description은 유지하되 조건이 성립하지 않으면 실제 답변의 현재 관측·조건 불충족·한계 설명을 검사한다. 조건부 조사가 적용되지 않는 것과 사용자의 질문에 답하지 않은 것은 다르다. 높은 경우에만 후보를 묻는 질문에서 높지 않다고 정확히 답했으면 satisfied다.
single_server_mode=true이면 resource_state_id와 targets의 내부 ID는 한 서버의 테스트 상태 키이며, 답변의 모니터링 서버는 그 대상이다. 예전 서버명이나 A/B를 별도 대상 요청으로 판단하지 마라.
observation_context.current_scope=selected_saved_state이면 '현재', '상태 확인'은 선택한 저장 상태를 조회하는 뜻이다. 그 상태의 값·근거와 모의 데이터임을 정확히 밝힌 답변은 충족이다. 실시간 장비 값이 아니라는 이유만으로 반려하거나 재측정을 요구하지 마라. 관측 시각은 저장된 시각을 유지한다. 단, 실제 운영 장비의 실시간 실측이나 과거 추세를 명시적으로 요구했다면 시나리오 값으로 대신 확인했다고 주장하면 안 되며 지원 한계를 밝혀야 한다.
requirement_coverage와 answer_blocks는 실제 최종 본문에 남은 요구별 블록·대처 단계·관측·출처 연결이다. 먼저 해당 요구의 블록을 찾아 읽고 누락을 판정하라. 연결이 있다는 이유만으로 통과시키지 말고 내용·적용 조건·OS가 질문과 맞는지 검사하라. 정상 영역에 장애나 즉각 조치 필요성을 만들지 마라. CPU가 정상인 경우 조건부 추가 확인 절차와 한계를 분명히 제공했다면 불필요한 장애 조치를 요구하지 마라.
calculation_rejections에 있는 계산은 이미 검증한 원본의 집계 기준이 달라 수행할 수 없다. 다만 먼저 거절된 연산·피연산자가 원래 query에서 요구한 계산에 실제 해당하는지 검사하라. 다른 계산을 잘못 선택한 경우 해당 check에 rejection_matches_request=false, status=unmet과 올바른 보완 내용을 기록하라. 거절 기록만으로 정상적인 다른 계산도 불가능하다고 판단하지 마라. 해당 요청에 맞는 거절이면 rejection_matches_request=true로 기록할 수 있다. 같은 계산을 missing_requirements에 다른 이름으로 추가하거나 임의 코어 수로 계산하라고 요구하지 마라. 동일 연산·동일 피연산자의 거절 계산을 다시 기록해야 한다면 calculation_fingerprint에 해당 거절의 fingerprint를 연결하라. 다른 연산·피연산자의 새 계산은 null로 두어 구분하라. 거절 설명의 적절성을 확인하라.
request_decision은 현재 질문에서 이미 결정한 의도·조회 정책이다. observation_policy=reuse이면 새 명령 조회를 하지 않고 기존 관측을 설명하는 요청이다. 조회 금지는 CPU 등 해당 지표를 답변에서 제외하라는 뜻이 아니다. 과거 관측을 실제 새 측정으로 표현하지 않는지 검사하되, 기존 CPU 수치를 설명했다는 이유만으로 unmet으로 판정하지 마라.
네트워크 상태와 소켓만으로 외부 응답 정상 여부를 알 수 있는지 묻는 요청은 관측된 포트·연결 상태와 확인 불가 한계를 설명하면 충족이다. 외부 점검 방법이나 도구 사용법을 요청하지 않았다면 그런 새 요구를 missing_requirements에 추가하지 마라.
1. requirements의 모든 requirement_id를 checks에서 하나씩 판정한다.
2. 답변에 요구한 값/계산/비교/설명/후보가 실제 있으면 satisfied, 빠지면 unmet이다.
조건부 진단은 전제가 관측됐는지 먼저 보라. 부하가 높지 않은 표본에서 수치·조건 불충족·확인 한계를 설명했으면 불필요한 원인 후보를 강요하지 마라. 정상 관측에 과다 점유·병목을 지어내면 unmet이다.
3. status는 satisfied 또는 unmet만 사용한다. 요구한 정보가 unavailable/unsupported/failed라면 그 한계를 답변에 정확히 안내했을 때 satisfied, 안내가 빠졌으면 unmet이다.
information_ready는 정보가 확보됐다는 뜻이다. 답변이 실제 요구를 충족했는지만 판정한다. 미지원 여부는 코드가 별도로 처리한다.
remediation 요구는 대응 단계·적용 조건·출처가 해당 질문과 관측에 맞는지 검사한다. 정상 영역의 장애를 지어내거나 조회하지 않은 조건을 확인했다고 말하면 unmet이다.
서버 관측과 매뉴얼 권장은 구분해야 한다. 제공된 remediation 자료 밖의 조치·명령·조건을 답변에 넣으면 unmet이다.
이미 확인한 정보는 다시 조회하지 말고, 필요한 대처 주제가 누락되면 missing_requirements의 remediation 요구로 기록하라.
macOS top/vm_stat의 용량·페이지 수만으로 메모리 부족 또는 정상 상태를 단정하면 unmet이다. 압박 미확인 설명은 압박 질문에 대한 한계 답변이다. 사용 상태·점유 원인 후보도 요청했다면 그 요구는 별도로 검사한다. 압박 미확인 설명이나 조건부 매뉴얼만으로 점유 원인 요구를 충족 처리하지 마라. 프로세스별 사용량에 연결된 조건부 기여 후보는 압박 확정과 다르다.
4. 현재 값으로 과거 시점/추세/배포 관계를 주장하면 unmet이다. 대상·단위·시점이 맞아야 한다.
5. query의 요구가 목록에서 빠졌으면 missing_requirements에 추가한다. 기존 요구를 표현만 바꿔 중복 추가하거나 조사에 도움이 된다는 이유로 새 출력 요구를 만들지 마라. CPU %는 원본에 이미 있으므로 원인 후보 질문에 별도 CPU 비율 계산 요구를 추가하지 마라. 메모리 사용 비율은 used/total 연산이므로 원본 용량만으로 충족하지 않는다. query가 % 기호 없이 사용 비율·점유율을 물어도 calculation 요구가 빠졌으면 missing_requirements에 추가한다. 기존 lookup·diagnosis 요구로 대체하지 않는다.
6. 추가 조회는 repair_tasks(server_id,goal,information_ids), 다시 쓸 내용은 rewrite_instructions에 쓴다. 확보된 정보는 다시 조회하지 않는다.
진단 요청(주범/병목/원인)은 관측에 근거한 후보와 한계를 설명하는 작업이다. '주범' 표현만 보고 실험적 원인 확정을 추가 요구하지 마라. 사용자가 확정 증명을 명시적으로 요구한 경우에는 그 증명을 못 한 한계를 판정한다. 후보+실제 관측+확인 한계를 제공하면 합리적 진단 답변이다. 불확실성 명시는 결함이 아니라 단일 표본의 필수 제한이다.
unmet이면 부족한 내용과 필요한 보완을 구체적으로 써라. 보완 task가 없는 unmet 판정도 rewrite_instructions에는 실제 본문의 누락과 수정 지시를 반드시 써야 한다. 정상 관측으로 조건이 성립하지 않은 질문을 반려하기 전에 evaluation_focus에 맞게 이미 답했는지 다시 확인하라. 검증된 원인 후보·실측 수치·확인 한계가 이미 있으면 원인 확정을 추가 요구하거나 추상적으로 '설명 부족'이라고 반복하지 마라. 정상 표본에 장애 후보를 억지로 요구하지 마라.
출력: {"checks":[{"requirement_id":"r1","status":"satisfied 또는 unmet","reason":"판정 이유"}],
"missing_requirements":[],"repair_tasks":[],"rewrite_instructions":""}.
missing_requirements 항목은 description,server_ids,information_ids,kind다. 현재 targets만 사용한다.
targets=[]인 OS 일반 매뉴얼 질문은 서버 상태 요청이 아니다. repair_tasks=[]이며 추가 요구의 server_ids도 []다. OS 이름을 server_id로 만들지 마라.
remediation의 action_condition=always는 일반·예방 절차를 제공하는 요구다. 해당 OS의 출처 있는 단계와 적용 전 조건이 있으면 서버 관측이나 '조치 불필요' 설명을 요구하지 않는다. action_condition=if_needed에서만 관측에 따른 조치 또는 생략을 검토한다. 둘을 혼동하여 조건부 조치라는 새 요구를 추가하지 마라.
모든 요구가 충족이면 repair_tasks=[]와 rewrite_instructions=""를 반환한다.
판정은 현재 draft_answer와 원문 근거로만 하라. 이전 검사에서 미충족이었다는 이유로 현재 완성된 답변을 반려하지 마라.
마지막으로 diagnosis 종류를 다음 기준으로 판정하라. diagnosis_assessments가 해당 증상을 false로 판단했고 질문이 '높은/부족한 경우' 등 조건부이면: 실제 본문에 현재 값·조건 불충족 설명·표본 한계가 있으면 satisfied다. 원인 후보가 없다는 이유로 unmet으로 만들지 마라. true인 경우에는 실제 원인을 설명할 후보가 필요하다. '메모리 부족이 성능 저하를 일으킬 수 있다'는 부족의 결과이지 부족의 원인 후보가 아니다. 질문이 요구한 인과 방향을 확인하라.""",
    "report": """원래 query를 충족하는 한국어 답변을 JSON으로 작성하라.
single_server_mode=true이면 서버 A/B를 구별하지 말고 모니터링 서버의 선택 상태를 설명하라. evidence/requirement ID와 원문 인용은 그대로 보존하라.
request_decision의 현재 조회 정책을 유지하라. reuse는 새 조회 금지이며 기존 근거의 설명은 허용한다. 지표 자체를 답변에서 제외하는 요청으로 바꾸지 마라.
specialist_assessments가 제공되면 전문 에이전트의 증상 판단과 근거는 이미 검증됐으며 코드가 답변에 포함한다. diagnoses=[]로 두고 이 판단을 다시 생성하지 마라. 같은 요구에 영역별 판단이 여러 개 있으면 모든 영역이 false일 때 원인 후보를 생성하지 않는다. true인 영역의 후보만 실제 점유·로그 근거로 설명하고 정상 영역을 원인으로 돌리지 마라. 이미 제공된 diagnosis 내용은 answers에도 복제하지 마라. answers는 별도의 계산·비교·설명 요구를 위해서만 사용한다.
requirements와 requirement_issues를 먼저 대조하여 계산·비교·설명 요구를 빠뜨리지 마라. 이전 final_feedback가 있으면 반드시 고려하라.
calculation_rejections는 기록된 연산·피연산자 조합만 차단한다. 최종 피드백에서 잘못된 계산 연결을 지적했다면 다른 올바른 연산·피연산자로 보완하라. 거절된 조합을 반복 생성하지 마라.
verified_results는 이미 원문 검증을 통과한 결과다. repair_requirement_ids가 있으면 해당 요구의 부족한 결과만 작성하라. 다른 요구의 검증 결과는 코드가 유지하므로 반복 생성하지 마라.
remediation 요구의 대응 절차는 뒤의 공식 매뉴얼 단계에서 처리한다. 이 report에서는 서버 관측·계산·원인 후보만 작성하고 매뉴얼 조치를 만들지 마라.
answers에는 {text, requirement_ids, evidence_ids, quote_ids} 형태의 직접 답변을 넣어라. 원문에 있는 수치만 text에 써라.
quote_catalog는 원문의 실제 행이다. quote_ids에는 해당 evidence_id에 속한 quote_id만 선택하라. quotes나 원문 복사문은 생성하지 않는다. 코드가 ID를 실제 원문 행으로 복원한다.
계산은 직접 숫자를 만들어 text에 쓰지 말고 calculations에 맡겨라. 항목은 label, requirement_ids, operation(sum/difference/product/ratio/percent),
operands:[{evidence_id,information_id,field}], output_unit(native/B/KiB/MiB/GiB/%/percentage_points/ratio)다.
field는 facts.fields의 경로(예: entries.0.rss_kib). difference/ratio/percent는 앞 값과 뒤 값을 순서대로 계산한다.
메모리 단위 환산은 sum의 피연산자1개와 output_unit을 사용한다. 페이지 수 환산은 product에 페이지 수와 실제 page_size_bytes를 참조한다.
전체 대비 사용 물리 메모리 비율은 memory.summary의 used_mib / total_mib, operation=percent, output_unit="%"다. 원본에 해당 필드가 없으면 다른 의미의 free/available 필드로 바꾸지 마라. 계산 결과 숫자를 answers.text나 hypotheses.text에 미리 쓰지 마라. 계산 결과는 별도로 출력되므로 원인 설명에는 실제 원본 점유 수치와 후보의 관계를 사용하라.
코드가 계산·단위 변환·공식·출처를 검증하고 렌더링한다. 계산이 필요 없으면 calculations=[]이다.
원인 분석 전에 diagnoses에 각 diagnosis 요구의 증상이 현재 관측에서 확인되는지 기록하라. 항목은 text(현재 원본 수치와 증상 판단 이유), requirement_ids, evidence_ids, quote_ids, symptom_observed이다.
코어 수 없는 Load Average만으로 과부하를 단정하지 마라. CPU 사용률로 판단한 높은 사용량과 Load Average 해석은 구분한다.
시스템 전체의 CPU 증상은 cpu의 usage_percent로 판단하고 process의 코어 합산 %를 시스템 전체 %로 읽지 마라. 예를 들어 전체 CPU가 20% 근처라면 가장 큰 프로세스가 있다는 이유만으로 전체 CPU 과부하의 증상으로 판단하지 않는다. 단일 표본에서 원인 확정·추세는 알 수 없다.
조건부 요청에서 증상이 관측되지 않으면 symptom_observed=false, hypotheses=[]로 하고 현재 수치와 조건이 성립하지 않는다는 설명을 제공한다. 코드가 diagnoses.text를 답변에 포함하므로 answers에 같은 설명을 복제하지 마라. 계산값은 diagnoses.text에서도 새로 계산하지 않는다.
원인 후보는 causes에 {"requirement_id":"진단 요구 ID", "factor_id":"cause_catalog의 실제 ID"}로 선택한다. hypotheses=[]로 두고 원인 설명을 새로 쓰지 마라. 코드가 선택한 관측 요인과 증상 근거를 연결해 조건부 기여 후보와 확인 한계를 렌더링한다.
eligible_cause_choices는 증상·대상·출처 검증을 통과한 선택지다. diagnosis 요구별로 관련 후보를 causes에 선택하고, 별도의 explanation 요구만 답하며 진단 후보를 빠뜨리지 마라. diagnoses=[]는 전문 증상 판단의 중복 생성을 막는 것이며 causes까지 생략하라는 뜻이 아니다.
cause_catalog의 requirement_ids에 해당 요구 ID가 있는 항목만 선택하라. 같은 requirement_id·server_id·kind에서는 반드시 하나만 선택한다. 가장 관련 있고 큰 점유 등 근거가 강한 후보를 선택하라. 해당 시나리오 이름이나 알려진 프로세스 이름으로 판단하지 마라.
CPU 사용·메모리 압박·디스크 사용률 같은 증상 자체나 그로 인한 성능 저하를 원인으로 쓰지 마라. process_memory는 실제 프로세스 점유, process_cpu는 코어 합산 사용, log_storage는 해당 파일시스템에 위치한 로그의 공간 점유다. log_reason은 로그에 기록된 이유이며 사실로 확정한 근본 원인이 아니다.
전문 판단에서 증상이 관측되지 않으면 causes=[]다. 관측된 증상이 있어도 적합한 요인이 없으면 causes=[]로 남기고 근거 없는 원인을 만들지 마라. 이 경우 최종 검증은 미충족을 기록하며 추가 관측이 필요할 수 있다.
최소한의 JSON을 작성하고 불필요한 answers·calculations·hypotheses·causes는 []로 반환하라. 수치를 직접 계산하거나 전체 출력 원문을 복사하지 마라.
macOS의 사용/미사용 용량이나 페이지 수를 근거로 메모리 부족 또는 정상 상태를 단정하지 마라. 원본 수치와 별도 문장 "macOS의 사용·미사용 메모리와 페이지 통계만으로 메모리 압박 여부를 확정할 수 없습니다."를 사용한다.
소켓만으로 외부 서비스 응답·RTT·패킷 손실을 확인했다고 말하지 마라.
소켓 확인 한계를 설명할 때는 별도 문장으로 "소켓 정보만으로 외부 서비스 응답 시간을 확인할 수 없습니다." 또는 "현재 소켓 정보로는 외부 서비스 정상 응답 여부를 확인할 수 없습니다."를 사용한다. 이 문장도 해당 소켓 evidence_ids와 원문 quote_ids를 연결한다. 관측 사실과 한계는 다른 문장으로 분리한다.
최근 로그 발췌에 오류가 없다고 전체 기간에 오류가 없다고 단정하지 마라. JSON 객체만 출력하라.""",
}


def fake_decision(purpose: str, payload: dict) -> dict:
    """Deterministic test backend. Real backends always use their model."""
    if purpose == "report":
        return {"hypotheses": [], "answers": [], "calculations": []}
    if purpose == "final_review":
        return {"checks": [{"requirement_id": r["requirement_id"],
                            "status": "satisfied" if r["status"] in {"information_ready", "unsupported", "failed"} else "unmet",
                            "reason": "오프라인 테스트: 정보 상태 기반 검토"}
                           for r in payload["requirements"]],
                "missing_requirements": [], "repair_tasks": [], "rewrite_instructions": "미충족 요구에 필요한 결과를 보완하세요."}
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
    result = {"tasks": tasks}
    intent = payload.get("request_decision", {})
    if purpose == "plan" and intent:
        result["requirements"] = [dict(description=t["goal"], server_ids=[t["server_id"]],
            information_ids=t["information_ids"], kind="lookup") for t in tasks]
        for flag, kind in [("needs_calculation", "calculation"), ("needs_diagnosis", "diagnosis"), ("needs_explanation", "explanation")]:
            if intent.get(flag):
                result["requirements"] += [dict(description=query, server_ids=[t["server_id"]],
                    information_ids=t["information_ids"], kind=kind) for t in tasks]
    return result
