"""Information-first investigations. Models choose tasks; code resolves commands."""
import asyncio
import json
import re
from copy import deepcopy
from time import perf_counter
from decimal import Decimal

from langgraph.config import get_stream_writer
from pydantic import ValidationError

from monitoring.api import CommandResult
from monitoring.evidence import extract_facts, causal_factors
from .investigation_models import AnswerReview, InvestigationReport, TaskPlan, TaskSpec, DiagnosticDecision, DiagnosisAssessment, Hypothesis, CauseSelection
from .request_ledger import RequestLedger, RequirementSpec
from .answer_calculations import calculate, CalculationRejected
from .answer_contracts import AnswerStatement
from .mcp_client import call_tool_result
from .deps import save_progress

MAX_ROUNDS = 3
MAX_COMMAND_CALLS = 12
MAX_ANSWER_REPAIRS = 2
SERVER = re.compile(r"(?:(?:server|scenario)[- ]([a-z0-9]+)|(?:서버|시나리오)\s*([a-z])(?![a-z0-9_-]))", re.I)
MONITORING_WORDS = re.compile(r"서버|cpu|메모리|memory|프로세스|디스크|네트워크|소켓|로그|운영체제|커널|swap", re.I)


def mentioned_servers(text):
    found = ["server-" + (m[1] or m[2]).lower() for m in SERVER.finditer(text)]
    # Also accept '서버 A와 B', without scanning unrelated letters elsewhere.
    shorthand = re.search(r"(?:서버|시나리오)\s*[a-z]((?:\s*(?:와|과|및|,|하고)\s*[a-z])+)", text, re.I)
    if shorthand:
        found += ["server-" + c.lower() for c in re.findall(r"[a-z]", shorthand[1], re.I)]
    return list(dict.fromkeys(found))


def monitoring_request(query):
    return bool(MONITORING_WORDS.search(query) or mentioned_servers(query))


def resolve_targets(query, history, servers, scenario_id=None, single_server_mode=False):
    targets = mentioned_servers(query)
    if scenario_id is not None:
        if scenario_id not in {s["server_id"] for s in servers}:return [], "등록되지 않은 시나리오입니다."
        if not single_server_mode and targets and set(targets)!={scenario_id}:return [], "선택한 시나리오와 요청 대상이 다릅니다. 시나리오를 변경한 뒤 요청해주세요."
        return [scenario_id],None
    if not targets and re.search(r"그\s*서버|해당\s*서버|그럼|거기|이어서", query):
        for message in reversed(history):
            targets = mentioned_servers(message["content"])
            if targets:
                break
    registered = {s["server_id"] for s in servers}
    if not targets:
        return [], "어느 서버를 확인할까요? 등록된 서버: " + ", ".join(sorted(registered))
    if not mentioned_servers(query) and len(targets) != 1:
        return [], "이전 대화에 여러 서버가 있습니다. 확인할 서버를 지정해주세요."
    unknown = set(targets) - registered
    if unknown:
        return [], "server_not_found: 등록되지 않은 서버 " + ", ".join(sorted(unknown))
    return targets, None









def requested_unsupported(query):
    """Capability guards: a model may not replace history/performance with a snapshot."""
    requirements = []
    if re.search(r"시간별|추세|시계열|지난\s*\d+\s*(?:시간|일|분)|최근\s*\d+\s*(?:시간|일|분)", query):
        requirements.append(("history", "시계열·과거 지표 확인"))
    if re.search(r"배포.*(?:이력|기록)|(?:이력|기록).*배포", query):
        requirements.append(("deployment", "배포 이력 확인"))
    if re.search(r"응답\s*시간|패킷\s*손실|\brtt\b|네트워크\s*지연", query, re.I):
        requirements.append(("network.performance", "외부 응답 시간·네트워크 성능 확인"))
    return requirements


def task_status(task, evidence, information, catalogs):
    sid = task["server_id"]
    relevant = [e for e in evidence if e["server_id"] == sid]
    have = {f["information_id"] for e in relevant for f in e["facts"]}
    available = {c["command_id"] for c in catalogs[sid]}
    statuses = {}
    for iid in task["information_ids"]:
        if iid in have:
            statuses[iid] = "obtained"
            continue
        commands = information.get(iid, {}).get("commands_by_os", {}).get(task["os"], [])
        if not commands or not set(commands) <= available:
            statuses[iid] = "unsupported"
        elif any(e["command_id"] in commands and (e.get("error") or e["exit_code"] != 0) for e in relevant):
            statuses[iid] = "failed"
        elif any(e["command_id"] in commands for e in relevant):
            statuses[iid] = "needs_more"
        else:
            statuses[iid] = "pending"
    task["information_status"] = statuses
    values = set(statuses.values())
    task["status"] = ("obtained" if values == {"obtained"} else
                      "unsupported" if values == {"unsupported"} else
                      "failed" if "failed" in values else
                      "needs_more" if values != {"pending"} else "pending")


def evidence_quote_catalog(evidence):
    """Stable line identities preserve source whitespace and cannot cross evidence."""
    return [{"quote_id": f"{e['evidence_id']}:l{number}",
             "evidence_id": e["evidence_id"], "text": line}
            for e in evidence for number, line in enumerate(e.get("stdout", "").splitlines(), 1)
            if line.strip()]


def bind_report_sources(report, evidence):
    catalog = {q["quote_id"]: q for q in evidence_quote_catalog(evidence)}
    for statement in [*report.diagnoses, *report.answers, *report.hypotheses]:
        resolved = []
        for quote_id in statement.quote_ids:
            source = catalog.get(quote_id)
            if source is None or source["evidence_id"] not in statement.evidence_ids:
                raise ValueError("인용 ID가 없거나 선택한 근거와 일치하지 않습니다.")
            resolved.append(source["text"])
        statement.quotes = list(dict.fromkeys(statement.quotes + resolved))
        if not statement.quotes:
            raise ValueError("관측 설명에는 원문 인용 ID 또는 실제 인용이 필요합니다.")


MAC_MEMORY_LIMITATION = "macOS의 사용·미사용 메모리와 페이지 통계만으로 메모리 압박 여부를 확정할 수 없습니다."


SOCKET_LIMITATIONS = {
    "소켓 정보만으로 외부 서비스 응답 시간을 확인할 수 없습니다.",
    "현재 소켓 정보로는 외부 서비스 정상 응답 여부를 확인할 수 없습니다.",
    "이 출력만으로 응답 시간·패킷 손실·외부 서비스 정상 여부는 확인할 수 없습니다.",
}


def verify_report(report, evidence, calculated_numbers=()):
    bind_report_sources(report, evidence)
    by_id = {e["evidence_id"]: e for e in evidence}
    for hypothesis in report.hypotheses:
        if not set(hypothesis.evidence_ids) <= by_id.keys():
            raise ValueError("Unknown evidence ID")
        selected = [by_id[i] for i in hypothesis.evidence_ids]
        from monitoring.evidence import verify_metric_values
        verify_metric_values(hypothesis.text, selected)
        if not all(q.strip() and any(q in e["stdout"] for e in selected) for q in hypothesis.quotes):
            raise ValueError("인용이 원문에 없습니다. quote_candidates의 한 줄을 공백까지 그대로 복사하세요.")
        if not set(mentioned_servers(hypothesis.text)) <= {e["server_id"] for e in selected}:
            raise ValueError("Hypothesis refers to a different server")
        allowed_numbers = {Decimal(n) for n in re.findall(r"\d+(?:\.\d+)?", " ".join(
            e["stdout"] + " " + " ".join(f["summary"] for f in e["facts"]) for e in selected))}
        allowed_numbers.update(Decimal(str(n)) for n in calculated_numbers)
        if not {Decimal(n) for n in re.findall(r"\d+(?:\.\d+)?", hypothesis.text)} <= allowed_numbers:
            raise ValueError("Unsupported number in hypothesis")
        facts = [f for e in selected for f in e["facts"]]
        process_names={p["name"] for e in evidence for f in e["facts"] if f["information_id"]=="process" for p in f["fields"].get("entries",[])}
        named=[name for name in process_names if len(name)>2 and name in hypothesis.text]
        if named and re.search(r"CPU|cpu|메모리|점유|과도",hypothesis.text):
            process_sources=[e for e in selected if any(f["information_id"]=="process" for f in e["facts"])]
            if any(not any(name in quote and quote in e["stdout"] for e in process_sources for quote in hypothesis.quotes) for name in named):
                matching = [q["quote_id"] for q in evidence_quote_catalog(evidence)
                            if any(name in q["text"] for name in named)
                            and any(e["evidence_id"] == q["evidence_id"] and any(f["information_id"] == "process" for f in e["facts"]) for e in evidence)]
                raise ValueError("프로세스 점유 문장의 evidence_ids와 quote_ids는 실제 ps 행을 가리켜야 합니다. 사용할 수 있는 해당 프로세스 행 ID: " + ", ".join(matching))
        log_text = " ".join(e["stdout"] for e in selected if e["command_id"] == "tail")
        # Only complete, canonical limitations may bypass the unsupported-claim
        # guard. A negative clause must never license a positive clause beside it.
        socket_source = any(f["information_id"] == "network" for f in facts)
        clauses = re.split(r"(?<=[.!?。])\s+|\n", hypothesis.text.strip().replace("\\n", "\n"))
        mac_memory_counts = any(e.get("os") == "macos" and any(
            f["information_id"] in {"memory.summary", "memory.detail"} for f in e["facts"]) for e in selected)
        claims = " ".join(c for c in clauses if not (
            (socket_source and c.strip() in SOCKET_LIMITATIONS)
            or (mac_memory_counts and c.strip() == MAC_MEMORY_LIMITATION)))
        if (mac_memory_counts
                and re.search(r"(?:메모리|RAM).{0,32}(?:부족|압박|정상|충분|양호|넉넉)|(?:부족|압박).{0,16}(?:없|아니)",claims,re.I)):
            raise ValueError("macOS 메모리 용량·페이지 수만으로 압박 상태나 정상 여부를 단정할 수 없습니다. "
                             "원본 수치를 설명하고 별도 문장으로 다음 한계를 사용하세요: " + MAC_MEMORY_LIMITATION)
        if re.search(r"누수|배포.*원인|패킷.*손실|응답\s*시간|증가|감소|급증|추세", claims):
            raise ValueError("현재 표본뿐이므로 증가·감소·추세·누수·응답 시간은 확인할 수 없습니다. 현재 관측에만 근거하세요.")
        if re.search(r"외부\s*서비스.*(?:정상|응답|지연|성공)", claims):
            raise ValueError("소켓 관측은 외부 서비스의 응답 성공·성능 근거가 아닙니다. 표준 소켓 한계 문장만 사용하세요.")
        if re.search(r"네트워크.*(?:문제|장애|병목|정상|지연)", claims) and not re.search(r"ERROR.*(?:connect|network|timeout)", log_text, re.I):
            raise ValueError("Sockets do not establish a network fault")
        if re.search(r"메모리\s*(?:부족|압박)", claims):
            pressure = any((f["information_id"] == "memory.summary" and
                            (f["fields"].get("available_mib", 1e9) < max(1024, f["fields"].get("total_mib", 0) * .1) or
                             f["fields"].get("unused_mib", 1e9) <= 1024)) or
                           (f["information_id"] == "memory.detail" and f["fields"].get("swap_used_mib", 0) > 0) for f in facts)
            if not pressure and not re.search(r"allocation failed|memory pressure", log_text, re.I):
                memory_ids = [e["evidence_id"] for e in evidence if any(
                    f["information_id"] in {"memory.summary", "memory.detail"} for f in e["facts"])]
                raise ValueError("메모리 부족을 주장한 문장에는 memory.summary/detail의 증상 근거도 필요합니다. "
                                 "프로세스 점유만으로 부족을 확인할 수 없습니다. 확보된 메모리 근거 ID: "
                                 + ", ".join(memory_ids) + ". 실제로 뒷받침하는 근거와 quote_id를 함께 선택하거나 주장을 좁히세요.")
        if re.search(r"디스크.*(?:부족|높|포화)", hypothesis.text):
            pressure = any(f["information_id"] == "disk" and any(p["usage_percent"] >= 90 for p in f["fields"]["entries"]) for f in facts)
            if not pressure and "No space left" not in log_text:
                raise ValueError("Disk pressure lacks referenced evidence")
        if re.search(r"원인(?:이다|입니다)|입증|증명|문제\s*없|이상\s*없|오류\s*없|모두\s*차지|전부\s*차지|완전히|가득", hypothesis.text):
            raise ValueError("확정·과장 표현입니다. 모두 차지, 가득, 완전히, 오류 없음 등의 표현을 제거하고 관측된 부분 사용량의 기여 가능성만 기술하세요.")


def cause_catalog(evidence, requirements):
    """Restrict observed factor choices by requested topic, host and filesystem."""
    catalog = []
    for factor in causal_factors(evidence):
        item = deepcopy(factor)
        if item["kind"] == "log_storage":
            mounts = [(e,entry,line) for e in evidence if e["server_id"]==item["server_id"] and not e.get("error")
                      for f in extract_facts(e) if f["information_id"]=="disk"
                      for entry,line in zip(f["fields"]["entries"], f["quote"].splitlines())
                      if item["path"].rstrip("/")==entry["mount"].rstrip("/")
                      or item["path"].startswith(entry["mount"].rstrip("/")+"/")]
            if not mounts:continue
            source,mount,line = max(mounts,key=lambda m:len(m[1]["mount"]))
            # Same disk-pressure guard already used in verify_report. Never
            # attribute a full root filesystem to logs on a separate healthy mount.
            if mount["usage_percent"] < 90:continue
            item["mount"] = mount["mount"]
            item["evidence_ids"].append(source["evidence_id"])
            item["quotes"].append(line)
            item["text"] += f" 해당 디렉터리가 위치한 {mount['mount']} 파일시스템의 공간 점유에 기여할 수 있는 후보입니다. 전체 사용량의 주된 원인인지는 확인되지 않았습니다."
        topics = {"process_cpu":{"cpu"}, "process_memory":{"memory.summary","memory.detail"},
                  "log_storage":{"disk"}, "log_reason":{"log.content"}}[item["kind"]]
        item["requirement_ids"] = [r["requirement_id"] for r in requirements
            if r["kind"]=="diagnosis" and item["server_id"] in r["server_ids"] and topics & set(r["information_ids"])]
        if item["requirement_ids"]:catalog.append(item)
    return catalog


def ground_resource_assessment(assessment, evidence, requirements):
    """Render resource observations separately from source-bound causal candidates."""
    assigned = [r for r in requirements if r["requirement_id"] in assessment.requirement_ids]
    resource_ids = {"cpu", "memory.summary", "memory.detail", "process"}
    topics = {i for r in assigned for i in r["information_ids"]}
    if (not assigned or not topics or not topics <= resource_ids or not assessment.symptom_observed
            or len({sid for r in assigned for sid in r["server_ids"]}) != 1):return
    # This free-form observation is replaced below, so it must not fail on
    # model-computed ratios that never reach the answer. Validate its sources
    # first, then validate the canonical raw-field text in the normal caller.
    source_claim = assessment.model_dump(exclude={"symptom_observed", "requirement_ids"})
    source_claim["text"] = "인용된 원문 관측을 확인했습니다."
    verify_report(InvestigationReport(hypotheses=[source_claim]), evidence)
    required = {(sid, i) for r in assigned for sid in r["server_ids"] for i in r["information_ids"]}
    covered, summaries = set(), []
    for source in evidence:
        if source.get("error") or source["evidence_id"] not in assessment.evidence_ids:continue
        for fact in extract_facts(source):
            key = (source["server_id"], fact["information_id"])
            if key not in required:continue
            lines = fact["quote"].splitlines()
            if not any(line in quote for quote in assessment.quotes for line in lines):continue
            selected_lines = [line for line in lines if any(line in quote for quote in assessment.quotes)]
            assessment.quotes = list(dict.fromkeys([*assessment.quotes, *(selected_lines if fact["information_id"] == "process" else lines)]))
            covered.add(key)
            # Process rows are already rendered in the factual evidence block.
            # Keep this bounded assessment factual without repeating every row.
            summary = ("인용한 프로세스 행에서 CPU·메모리 점유와 RSS를 확인했습니다."
                       if fact["information_id"] == "process" else fact["summary"])
            if summary not in summaries:summaries.append(summary)
    if covered != required:
        raise ValueError("자원·프로세스 판단에는 배정된 정보의 원문 행을 빠짐없이 인용하세요.")
    assessment.text = " ".join(summaries) + " 단일 표본으로 원인을 확정할 수 없습니다."
    if topics & {"memory.summary", "memory.detail"} and any(
            e.get("os") == "macos" and e["evidence_id"] in assessment.evidence_ids for e in evidence):
        assessment.text += " macOS의 사용·미사용 메모리와 페이지 통계만으로 메모리 압박 여부를 확정할 수 없습니다."


def ground_disk_absence(assessment, evidence, requirements):
    """Ground a typed disk-absence verdict; do not infer negation from prose."""
    if assessment.symptom_observed:return
    requested = [r for r in requirements if r["requirement_id"] in assessment.requirement_ids]
    if not requested or any(set(r["information_ids"]) != {"disk"} for r in requested):return
    targets = {sid for r in requested for sid in r["server_ids"]}
    sources = [e for e in evidence if e["server_id"] in targets and not e.get("error")
               and any(f["information_id"] == "disk" for f in extract_facts(e))]
    if targets != {e["server_id"] for e in sources}:
        raise ValueError("디스크 증상 미관측 판단에는 배정된 대상의 파일시스템 관측이 필요합니다.")
    summaries = []
    for source in sources:
        for fact in extract_facts(source):
            if fact["information_id"] != "disk":continue
            if any(row["usage_percent"] >= 90 for row in fact["fields"]["entries"]):
                raise ValueError("파일시스템 사용률이 높은 관측이 있으므로 공간 압박 전제가 없다고 처리할 수 없습니다. 해당 관측을 포함해 증상을 다시 판단하세요.")
            if source["evidence_id"] not in assessment.evidence_ids or not all(
                    any(row in quote for quote in assessment.quotes) for row in fact["quote"].splitlines()):
                raise ValueError("증상 미관측 판단에는 확인한 파일시스템 행의 quote_ids를 포함하세요.")
            summaries.append(fact["summary"])
    # Replace free prose with measured facts and the validated typed verdict.
    # Downstream numerical/source verification still checks this canonical text.
    assessment.text = " ".join(summaries) + " 현재 인용한 파일시스템에서는 공간 압박 전제가 관측되지 않았습니다. 단일 표본으로 과거 상태는 확인하지 않았습니다."


def ground_storage_support(assessment, evidence, requirements):
    """Render supporting observations; free prose cannot redefine their cause."""
    assigned = [r for r in requirements if r["requirement_id"] in assessment.requirement_ids]
    support = [r for r in assigned if "disk" in r.get("supporting_for", [])]
    if not support:return
    if len(support) != len(assigned):
        raise ValueError("지원 조사와 독립 로그 판단은 요구 ID별로 구분해 기록하세요.")
    required = {sid for r in support for sid in r["supporting_server_ids"]}
    wanted = {i for r in assigned for i in r["information_ids"]}
    seen, summaries, sources, quotes = set(), [], [], []
    for source in evidence:
        if source.get("error"):continue
        for fact in extract_facts(source):
            info = fact["information_id"]
            if source["server_id"] in required and info == "log.usage":
                seen.add(source["server_id"])
                summaries.append(fact["summary"])
                sources.append(source["evidence_id"]);quotes.append(fact["quote"])
            elif info in wanted and info.startswith("log.") and source["evidence_id"] in assessment.evidence_ids:
                # Preserve separately requested log observations as excerpts,
                # without reusing unsupported causal interpretation around them.
                rows = [row for row in fact["quote"].splitlines() if any(row in q for q in assessment.quotes)]
                if rows:
                    # Preserve original excerpts (including multiline quotes),
                    # rather than splitting them or choosing an arbitrary first row.
                    summaries.append("별도로 요청한 로그의 선택 원문도 함께 확인했습니다.")
                    sources.append(source["evidence_id"])
                    quotes.extend(q for q in assessment.quotes if q in source["stdout"]
                                  and any(row in q for row in rows))
    if seen != required:
        raise ValueError("지원 조사에는 배정된 대상의 실제 log.usage 관측이 필요합니다.")
    assessment.text = " ".join(summaries) + " 이 추가 관측만으로 전체 공간 사용의 원인을 확정하지 않습니다."
    assessment.evidence_ids = list(dict.fromkeys(sources))
    assessment.quotes = list(dict.fromkeys(quotes))
    assessment.quote_ids = []


def disk_log_support(requirements, evidence, assessments):
    """Delegate only logs on the filesystem whose pressure was actually cited."""
    result = []
    for requirement in requirements:
        if requirement["kind"] != "diagnosis" or "disk" not in requirement["information_ids"]:continue
        targets = []
        for sid in requirement["server_ids"]:
            mounts = [(e, entry, row) for e in evidence if e["server_id"] == sid and not e.get("error")
                for f in extract_facts(e) if f["information_id"] == "disk"
                for entry, row in zip(f["fields"]["entries"], f["quote"].splitlines())
                if "/var/log/".startswith(entry["mount"].rstrip("/")+"/")]
            if not mounts:continue
            source, mount, row = max(mounts, key=lambda m:len(m[1]["mount"]))
            if mount["usage_percent"] < 90:continue
            if any(a.get("source_role") == "metrics" and a["symptom_observed"]
                   and requirement["requirement_id"] in a["requirement_ids"]
                   and source["evidence_id"] in a["evidence_ids"]
                   and any(row in q and q in source["stdout"] for q in a.get("quotes", [])) for a in assessments):
                targets.append(sid)
        if targets:
            result.append({**deepcopy(requirement), "server_ids":targets, "information_ids":["log.usage"],
                "supporting_for":["disk"], "supporting_server_ids":targets})
    return result


def bind_cause(selection, evidence, requirements, assessments):
    """Bind a selected contributor and verified symptom without free causal prose."""
    item = next((c for c in cause_catalog(evidence,requirements)
                 if c["factor_id"]==selection.factor_id and selection.requirement_id in c["requirement_ids"]),None)
    if item is None:raise ValueError("원인 요인 ID가 없거나 요구의 대상·정보 범위와 일치하지 않습니다.")
    by_id = {e["evidence_id"]:e for e in evidence}
    topic = {"process_cpu":{"cpu"},"process_memory":{"memory.summary","memory.detail"},
             "log_storage":{"disk"},"log_reason":{"log.content"}}[item["kind"]]
    relevant = []
    for assessment in assessments or []:
        if not assessment["symptom_observed"] or selection.requirement_id not in assessment["requirement_ids"]:continue
        sources = [by_id.get(eid) for eid in assessment["evidence_ids"]]
        quotes = assessment.get("quotes",[])
        if (not sources or any(e is None or e["server_id"]!=item["server_id"] for e in sources)
                or not quotes or any(not q.strip() or not any(q in e["stdout"] for e in sources) for q in quotes)):
            continue
        # A command may contain several topics (top: CPU and memory). Require
        # the actual quoted observation row, not merely another fact in its file.
        def quoted_topic(e, fact):
            if fact["information_id"] not in topic or not fact.get("quote"):return False
            rows = fact["quote"].splitlines()
            if fact["information_id"] == "cpu":
                rows = rows[:1]  # load-average alone does not establish CPU utilization.
            elif item["kind"] in {"log_storage", "log_reason"}:
                rows = [row for row in rows if row in item["quotes"]]
            elif e["command_id"] == "vm_stat":
                rows = [row for row in rows if re.match(r"Pages (?:free|active|wired down|occupied by compressor):",row)]
            return any(row and row in q and q in e["stdout"] for row in rows for q in quotes)
        if not any(quoted_topic(e,f) for e in sources for f in extract_facts(e)):continue
        relevant.append(assessment)
    if not relevant:raise ValueError("원인 요인 선택 전에 같은 대상·주제의 검증된 증상 관측이 필요합니다.")
    source_ids = list(dict.fromkeys(item["evidence_ids"]+[eid for a in relevant for eid in a["evidence_ids"]]))
    quotes = list(dict.fromkeys(item["quotes"]+[q for a in relevant for q in a["quotes"]]))
    text = item["text"] + " 단일 표본으로 인과관계나 발생 경위를 확정할 수 없습니다."
    result = Hypothesis(text=text,requirement_ids=[selection.requirement_id],evidence_ids=source_ids,quotes=quotes)
    verify_report(InvestigationReport(hypotheses=[result]),evidence)
    return result


def eligible_cause_choices(evidence, requirements, assessments):
    """Offer only source-bound contributors to an observed symptom; never select one."""
    choices = []
    for factor in cause_catalog(evidence, requirements):
        for rid in factor["requirement_ids"]:
            selection = CauseSelection(requirement_id=rid, factor_id=factor["factor_id"])
            try:
                bind_cause(selection, evidence, requirements, assessments)
            except ValueError:
                continue
            choices.append(selection.model_dump())
    return choices


def merge_verified_results(previous: list[dict], updates: list[dict], repair_ids: set[str] | None) -> list[dict]:
    if repair_ids is None:return deepcopy(updates)
    selected=[]
    for update in updates:
        ids=set(update["requirement_ids"]) & repair_ids
        if ids:selected.append({**deepcopy(update),"requirement_ids":sorted(ids)})
    replaced={rid for item in selected for rid in item["requirement_ids"]}
    retained=[]
    for item in previous:
        ids=set(item["requirement_ids"]) - replaced
        if ids:retained.append({**deepcopy(item),"requirement_ids":sorted(ids)})
    return retained+selected


def withdraw_absent_hypotheses(items, assessments):
    """Apply source-verified absence only where no assigned role observes it."""
    assessed = {rid for a in assessments for rid in a["requirement_ids"]}
    supported = {rid for a in assessments if a["symptom_observed"] for rid in a["requirement_ids"]}
    withdrawn = assessed - supported
    retained = []
    for item in items:
        if not item.get("requirement_ids"):
            retained.append(deepcopy(item))
            continue
        remaining = set(item["requirement_ids"]) - withdrawn
        if remaining:retained.append({**deepcopy(item), "requirement_ids":sorted(remaining)})
    return retained


def render_answer(tasks, evidence, hypotheses, limitations, answers=(), calculations=(), requirements=()):
    lines = ["모의 시나리오의 저장된 출력 기준입니다. 실제 장비의 실시간 측정값은 아닙니다.", "", "확인한 사실"]
    for e in evidence:
        for fact in e["facts"]:
            lines.append(f"- [{e['evidence_id']}] {e['server_id']} · {e['command']}: {fact['summary']}")
    if not any(e["facts"] for e in evidence):
        lines.append("- 확인 가능한 정보를 확보하지 못했습니다.")
    if answers or calculations:
        lines += ["", "요청에 대한 답변"]
        lines.extend(f"- {a['text']} (근거: {', '.join(a['evidence_ids'])})" for a in answers)
        for c in calculations:
            lines.append(f"- {c['label']}: {Decimal(c['value']):f} {c['unit']} "
                         f"(계산: {c['equation']}; 근거: {', '.join(c['evidence_ids'])})")
    lines += ["", "원인 후보 — 확정되지 않음"]
    if hypotheses:
        for item in hypotheses:
            lines.append(f"- {item['text']} (근거: {', '.join(item['evidence_ids'])})")
    else:
        lines.append("- 확보한 정보만으로 원인을 확정하지 않습니다.")
    unresolved = [f"{t['server_id']} · {t['goal']}: {i} ({s})" for t in tasks
                  for i, s in t["information_status"].items() if s != "obtained"]
    unresolved += [f"{r['description']}: {r['reason']}" for r in requirements
                   if r['status'] != 'satisfied']
    lines += ["", "확인 한계·미확인 항목"]
    lines.extend("- " + text for text in dict.fromkeys(limitations + unresolved))
    if not limitations and not unresolved:
        lines.append("- 단일 시나리오 표본이며 실제 운영 원인에 대한 검증은 별도로 필요합니다.")
    return "\n".join(lines)


def should_generate_report(intent: dict, requirements: list[dict], feedback: dict | None) -> bool:
    if intent.get("needs_explanation", True) or intent.get("needs_diagnosis") or intent.get("needs_calculation"):
        return True
    if not requirements or any(r["kind"] != "lookup" for r in requirements):
        return True
    return bool(feedback and (feedback.get("rewrite_instructions") or feedback.get("missing_requirements")))


async def investigate(state, deps):
    from .remediation import remediation_node, specified_os
    from .response_policy import present_response
    query, history = state["query"], state.get("history", [])
    bound_target=state.get("scenario_id") or state.get("reference_target")
    started=perf_counter();timings={}
    observation_policy=(state.get("request_decision") or {}).get("observation_policy","auto")
    lookup_disabled=observation_policy in {"reuse", "clarify"}
    manual_requested = state.get("workflow_manual", False) or state.get("route") == "investigate_remediate"
    generic_manual = manual_requested and not mentioned_servers(query) and not re.search(r"그\s*서버|해당\s*서버|거기|이어서", query) and (not bound_target or state.get("route")=="remediation")
    manual_os = specified_os(query)
    manual_result, manual_state, presentation = None, {}, {}
    manual_text, terminal_note, text = "", "", ""
    targets, server_map, catalogs, information, definitions = [], {}, {}, {}, []
    planner = getattr(deps, "monitoring_llm", None) or deps.fast_llm
    events = list(state.get("events", []))
    evidence, tasks, limitations, hypotheses = [], [], [], []
    report_failure_notes = {}
    ledger = RequestLedger()
    answers, calculations, final_reviews = [], [], []
    diagnosis_assessments = None  # None preserves legacy injected writer contracts.
    use_specialists = callable(getattr(deps.quality_llm, "diagnostic_json", None))
    specialist_assessments, specialist_key = None, None
    specialist_cache, specialist_failures = {}, set()
    calculation_notes = {}
    lookup_scope = set()
    final_validation = {"status": "not_run", "request_fully_satisfied": False,
                        "reason": "대상 확인·지원 명령 확인 또는 원문 전용 응답은 별도 처리합니다."}
    cost, prompt, calls, rounds = state.get("cost_usd", 0), None, 0, 0
    try:
        writer = get_stream_writer()
    except RuntimeError:
        writer = lambda _: None

    def emit(kind, **data):
        event = {"type": kind, "event_id": f"investigation-{len(events)}", **data}
        events.append(event)
        save_progress({"tasks":tasks, "evidence":evidence, "hypotheses":hypotheses,
            "limitations":limitations, "events":events, "cost_usd":round(cost, 6),
            "answers":answers, "calculation_results":calculations, **ledger.payload(),
            "command_call_count":calls, "remediation":manual_result, "manual_state":manual_state,
            "workflow_budget":{"query_rounds":rounds, "command_calls":calls,
                               "manual_searches":manual_state.get("search_calls",0)},
            "workflow_timings_ms":timings})
        writer(event)

    def finish(text):
        emit("WorkflowCompleted" if final_reviews else "InvestigationCompleted", pipeline="information-investigation")
        return {"final": text, "tasks": tasks, "evidence": evidence, "hypotheses": hypotheses,
                "limitations": list(dict.fromkeys(limitations)), "events": events, "cost_usd": round(cost, 6),
                "replan_rounds": max(0, rounds - 1), "command_call_count": calls,
                "last_prompt": prompt, **ledger.payload(), "answers": answers,
                "calculation_results": calculations, "final_reviews": final_reviews,
                "final_validation": final_validation, "validation_attempts": len(final_reviews),
                "remediation": manual_result, "manual_state":manual_state,
                "presentation": presentation, "_presentation_done":bool(presentation),
                "workflow_timings_ms":{**{k:round(v,2) for k,v in timings.items()},"total":round((perf_counter()-started)*1000,2)},
                "workflow_budget":{"query_rounds":rounds,"command_calls":calls,
                                   "manual_searches":manual_state.get("search_calls",0),
                                   "review_count":len(final_reviews),"repair_rounds":max(0,len(final_reviews)-1)}}

    def pending_calculation_ids(verified=()):
        have = {(e["server_id"], f["information_id"]) for e in evidence for f in e.get("facts", [])}
        covered = {rid for c in list(verified)+calculations for rid in c["requirement_ids"]}
        pending = []
        for requirement in ledger.requirements:
            rid = requirement["requirement_id"]
            needed = {(server, info) for server in requirement["server_ids"] for info in requirement["information_ids"]}
            if (requirement["kind"] == "calculation" and needed and needed <= have
                    and requirement["status"] not in {"failed", "unsupported"}
                    and not ledger.is_calculation_blocked(rid) and rid not in covered):
                pending.append(rid)
        return pending

    async def ask(model, purpose, payload, contract, check=None):
        nonlocal cost, prompt
        payload = dict(payload)
        payload["single_server_mode"]=bool(state.get("single_server_mode"))
        payload["resource_state_id"]=state.get("scenario_id")
        if state.get("single_server_mode"):
            payload["observation_context"]={"source":"fixture", "current_scope":"selected_saved_state",
                                            "live_measurement_supported":False}
        schema = contract.model_json_schema()
        if contract is TaskPlan:
            spec = schema["$defs"]["TaskSpec"]["properties"]
            spec["server_id"]["enum"] = payload["targets"]
            spec["information_ids"]["items"]["enum"] = [i["information_id"] for i in payload["information"]] + [
                "history", "deployment", "network.performance"]
        if contract is DiagnosticDecision:
            # At most one assessment per assigned requirement. Incidental fields
            # in a shared command must not create another symptom assessment.
            schema["properties"]["diagnoses"]["maxItems"]=min(8,len(payload["requirements"]))
            props=schema["properties"]
            schema["properties"]={"tasks":props["tasks"],"diagnoses":props["diagnoses"]}
            spec = schema["$defs"]["TaskSpec"]["properties"]
            spec["server_id"]["enum"] = payload["targets"]
            spec["information_ids"]["items"]["enum"] = payload["allowed_information_ids"]
            source = schema["$defs"]["DiagnosisAssessment"]
            source["properties"].pop("quotes")
            source["properties"]["quote_ids"].update(minItems=1)
            if payload["quote_catalog"]:
                source["properties"]["quote_ids"]["items"]["enum"] = [q["quote_id"] for q in payload["quote_catalog"]]
            source["properties"]["requirement_ids"]["items"]["enum"] = [r["requirement_id"] for r in payload["requirements"]]
            if payload["observations"]:
                source["properties"]["evidence_ids"]["items"]["enum"] = [e["evidence_id"] for e in payload["observations"]]
            if not payload["observations"]:schema["properties"]["diagnoses"]["maxItems"]=0
        if contract is AnswerReview:
            # Audit the user's unparsed request before grading the existing list;
            # grading first can anchor the model on an incomplete contract.
            props=schema["properties"]
            schema["properties"]={k:props[k] for k in ("missing_requirements","checks","repair_tasks","rewrite_instructions")}
            # The validator judges answer adequacy only; capability state belongs to code.
            schema["$defs"]["RequirementCheck"]["properties"]["status"]["enum"] = ["satisfied", "unmet"]
            if not ledger.calculation_rejections:
                schema["$defs"]["RequirementCheck"]["properties"].pop("rejection_matches_request",None)
            if lookup_scope:
                for name in ("TaskSpec", "RequirementSpec"):
                    schema["$defs"][name]["properties"]["information_ids"]["items"]["enum"] = sorted(lookup_scope)
            if payload["targets"]:
                schema["$defs"]["TaskSpec"]["properties"]["server_id"]["enum"]=payload["targets"]
                schema["$defs"]["RequirementSpec"]["properties"]["server_ids"]["items"]["enum"]=payload["targets"]
            else:
                # A generic OS manual has no server target; never invent one for repair.
                schema["properties"]["repair_tasks"]["maxItems"]=0
                schema["$defs"]["RequirementSpec"]["properties"]["server_ids"]["maxItems"]=0
        if contract is InvestigationReport:
            kinds=[r["kind"] for r in ledger.requirements if r["kind"]!="remediation"]
            diagnosis_ids = [r["requirement_id"] for r in ledger.requirements if r["kind"] == "diagnosis"]
            schema["properties"]["hypotheses"]["maxItems"]=min(8, len(diagnosis_ids))
            schema["properties"]["diagnoses"]["maxItems"]=0 if specialist_assessments is not None else min(8, len(diagnosis_ids))
            if specialist_assessments is not None:
                supported = {rid for a in specialist_assessments if a["symptom_observed"] for rid in a["requirement_ids"]}
                schema["properties"]["hypotheses"]["maxItems"] = len(supported)
            if diagnosis_ids:
                schema["properties"]["diagnoses"]["minItems"]=0 if specialist_assessments is not None else 1
                ref = schema["$defs"]["DiagnosisAssessment"]["properties"]["requirement_ids"]
                ref["items"]["enum"]=diagnosis_ids
            if diagnosis_ids:
                ref = schema["$defs"]["Hypothesis"]["properties"]["requirement_ids"]
                ref.update(minItems=1)
                ref["items"]["enum"] = diagnosis_ids
            schema["properties"]["calculations"]["maxItems"]=min(8,sum(k in {"calculation","comparison"} for k in kinds))
            pending = pending_calculation_ids()
            if pending:
                schema["properties"]["calculations"]["minItems"] = 1
                payload["required_calculation_ids"] = pending
                payload["calculation_instruction"] = "이 요구들은 원본 정보가 확보됐습니다. calculations에 연산·원본 필드·요구 ID를 연결하세요. 원본 용량을 answers에 반복하는 것으로 계산 결과를 대신하지 마세요."
            answer_ids = [r["requirement_id"] for r in ledger.requirements if r["kind"] != "remediation"
                          and (specialist_assessments is None or r["kind"] != "diagnosis")]
            repair_ids = payload.get("repair_requirement_ids")
            if repair_ids is not None:
                answer_ids = [rid for rid in answer_ids if rid in repair_ids]
            factors = [c for c in payload["cause_catalog"]
                       if any(rid in diagnosis_ids and (repair_ids is None or rid in repair_ids)
                              for rid in c["requirement_ids"])]
            if specialist_assessments is not None:
                eligible = {(c["requirement_id"], c["factor_id"]) for c in payload["eligible_cause_choices"]}
                factors = [{**c, "requirement_ids":[rid for rid in c["requirement_ids"] if (rid, c["factor_id"]) in eligible]}
                           for c in factors]
                factors = [c for c in factors if c["requirement_ids"]]
            schema["properties"]["causes"]["maxItems"] = min(8, len({
                (rid, c["server_id"], c["kind"]) for c in factors for rid in c["requirement_ids"]}))
            if factors:
                properties = schema["$defs"]["CauseSelection"]["properties"]
                properties["factor_id"]["enum"] = [c["factor_id"] for c in factors]
                properties["requirement_id"]["enum"] = sorted({rid for c in factors for rid in c["requirement_ids"]
                    if rid in diagnosis_ids and (repair_ids is None or rid in repair_ids)})
            else:
                schema["properties"]["causes"]["maxItems"] = 0
            schema["properties"]["answers"]["maxItems"]=min(8,len(answer_ids))
            if answer_ids:
                schema["$defs"]["AnswerStatement"]["properties"]["requirement_ids"]["items"]["enum"] = answer_ids
            quote_ids=[q["quote_id"] for q in payload["quote_catalog"]]
            ids=[e["evidence_id"] for e in payload["evidence"]]
            for name in ("Hypothesis","AnswerStatement","DiagnosisAssessment"):
                definition=schema["$defs"][name]
                properties=definition["properties"]
                properties.pop("quotes")  # Legacy callers can still supply exact text.
                properties["quote_ids"].update(minItems=1)
                properties["quote_ids"]["items"]["enum"]=quote_ids
                definition["required"]=[k for k in definition["required"] if k!="quotes"]+["quote_ids"]
                properties["evidence_ids"]["items"]["enum"]=ids
        for attempt in range(2):
            tick=perf_counter()
            try:
                # Existing LLM adapters are synchronous; keep the FastAPI loop free
                # for the MCP subprocess's callbacks to this same server.
                raw, usage = await asyncio.to_thread(
                    model.monitoring_json, purpose, payload, schema)
                cost += usage.get("usd", 0)
                prompt = json.dumps({"purpose": purpose, "input": payload}, ensure_ascii=False)
                answer = contract.model_validate_json(raw)
                if check:
                    check(answer)
                return answer
            except (ValueError, ValidationError) as exc:
                payload["validation_feedback"] = str(exc)[:500]
                if contract is InvestigationReport:
                    pending = pending_calculation_ids()
                    payload["required_calculation_ids"] = pending
                    payload["verified_results"].update(calculations=deepcopy(calculations),
                                                       answers=deepcopy(answers),hypotheses=deepcopy(hypotheses))
                    schema["properties"]["calculations"]["minItems"] = 1 if pending else 0
                ledger.note(purpose, str(exc)[:500])
            except Exception:
                # Do not leak provider credentials or request headers in errors.
                raise ValueError(f"{purpose} 모델 호출 실패") from None
            finally:
                timings[purpose]=timings.get(purpose,0)+(perf_counter()-tick)*1000
        raise ValueError(f"{purpose} 응답이 두 번 모두 검증을 통과하지 못했습니다.")

    if generic_manual:
        # Unsupported capability is already known from the typed intent. OS is
        # relevant only if at least one requested manual topic can be served.
        from .remediation import MANUAL_INFORMATION
        intent = state.get("request_decision") or {}
        ids = intent.get("manual_information_ids", [])
        specs = [RequirementSpec(**r, server_ids=[]) for r in intent.get("requested_results", []) if r["kind"] == "remediation"]
        if not specs:
            specs = [RequirementSpec(description="공식 대처방안: "+i,information_ids=[i],kind="remediation") for i in ids]
            specs += [RequirementSpec(description="미지원 대처 요구: "+str(t)[:260],kind="remediation") for t in intent.get("unsupported_manual_topics",[])]
        requested = set(ids) | {i for r in specs for i in r.information_ids}
        if specs and not requested & MANUAL_INFORMATION:
            note = "요청한 대처 주제는 미지원입니다. 공식 매뉴얼 지원 범위는 CPU·메모리·디스크입니다. 서버 상태는 조회하지 않았습니다."
            ledger.add(specs, [], "plan")
            for r in ledger.requirements:
                ledger.update(r,"unsupported",note,"capability")
                ledger.note("capability",note,[r["requirement_id"]])
            limitations.append(note)
            manual_result = {"status":"not_found","steps":[],"sources":[],"unconfirmed":[note],
                             "unsupported_requirements":[r["requirement_id"] for r in ledger.requirements],
                             "search_calls":0,"model_calls":0}
            final_validation = {"status":"partial","request_fully_satisfied":False,"reason":note}
            emit("RequestRequirements",**ledger.payload())
            return finish(note)
        if not manual_os:
            return finish("Linux 또는 macOS 중 어떤 환경의 대처방안인지 알려주세요.")
    else:
        try:
            servers = await call_tool_result(deps.mcp, "list_command_servers", {})
            targets, clarification = resolve_targets(query, history, servers, bound_target, state.get("single_server_mode",False))
            if clarification:return finish(clarification)
            if manual_requested and len(targets)!=1:return finish("조사와 대처를 함께 진행할 서버 한 대를 지정해주세요.")
            server_map = {s["server_id"]: s for s in servers}
            if manual_requested:
                actual_os=server_map[targets[0]]["os"]
                if manual_os and actual_os!=manual_os:return finish(f"{targets[0]}의 등록 OS는 {actual_os}입니다. 요청한 OS와 달라 대상 환경을 확인해주세요.")
                manual_os=actual_os
            definitions = await call_tool_result(deps.mcp, "list_server_information", {})
            information = {i["information_id"]: i for i in definitions}
            catalogs = {sid: await call_tool_result(deps.mcp, "list_server_commands", {"server_id": sid}) for sid in targets}
        except ValueError as exc:
            return finish(f"서버 정보 목록을 조회하지 못했습니다: {exc}")

    context=state.get("investigation_context") or {}
    if targets and (lookup_disabled or (manual_requested and observation_policy!="refresh")):
        evidence=deepcopy([e for e in context.get("evidence",[]) if e["server_id"] in targets])
        if evidence:
            tasks=deepcopy([t for t in context.get("tasks",[]) if t["server_id"] in targets])
            retained={e["evidence_id"] for e in evidence}
            for candidate in context.get("hypotheses",[]):
                if not set(candidate.get("evidence_ids",[]))<=retained:continue
                try:verify_report(InvestigationReport(hypotheses=[candidate]),evidence)
                except ValueError:continue
                hypotheses.append(deepcopy(candidate))
            rounds=1
            emit("RemediationContextReused",server_id=targets[0])

    emit("RequirementsIdentified", targets=targets, query=query)
    base = {"query": query, "request_decision":state.get("request_decision", {}), "targets": targets,
            "servers": [server_map[s] for s in targets],
            "information": [{"information_id": i["information_id"], "description": i["description"],
                             "limitations": i["limitations"]} for i in definitions],
            "history": history[-6:]}

    all_commands = {c["command_id"] for cs in catalogs.values() for c in cs}
    # Include other OS commands to report an explicit unsupported request honestly.
    all_commands |= {c for i in definitions for cs in i["commands_by_os"].values() for c in cs}
    explicit = [cid for cid in sorted(all_commands)
                if re.search(rf"(?<![a-zA-Z0-9_]){re.escape(cid)}(?![a-zA-Z0-9_])", query)]
    if explicit and re.search(r"[;|&$<>]", query):
        return finish("복합 셸 구문은 지원하지 않습니다. 등록된 명령 ID를 지정해주세요.")
    # Exact full command strings are allowed; never silently discard flags/paths.
    quoted = re.findall(r"`([^`]+)`", query)
    unsupported = [q for q in quoted if not any(q in (c["command_id"], c["command"])
                   for cs in catalogs.values() for c in cs)]
    for cid in explicit:
        argument = re.search(rf"(?<![a-zA-Z0-9_]){re.escape(cid)}\s+((?:-|/)[a-zA-Z0-9_.,/:=+*% -]*)", query)
        if argument:
            supplied = cid + " " + argument[1].strip()
            supplied = re.split(r"\s+(?:JSON|json)(?:\s|$)", supplied)[0].strip()
            if not all(any(c["command"] == supplied for c in catalogs[sid]) for sid in targets):
                unsupported.append(supplied)
    unknown = re.search(r"\b(cat|head|grep|curl|ping|rm|kill|sudo|systemctl)\b", query)
    if unknown:
        unsupported.append(unknown[1])
    missing = [(sid, cid) for sid in targets for cid in explicit
               if cid not in {c["command_id"] for c in catalogs[sid]}]
    if unsupported or missing:
        for sid in targets:
            ids = list(dict.fromkeys(["command." + x.split()[0] for x in unsupported] +
                                     ["command." + cid for server, cid in missing if server == sid]))
            if ids:
                tasks.append({"task_id": f"t{len(tasks)+1}", "server_id": sid, "os": server_map[sid]["os"],
                              "goal": "지정한 명령 확인", "information_ids": ids,
                              "information_labels": {i: i for i in ids}, "command_ids": [],
                              "information_status": {i: "unsupported" for i in ids}, "status": "unsupported"})
        return finish("지정한 명령 또는 옵션은 해당 서버에서 지원하지 않습니다. "
                      "명령 ID 또는 명령 카탈로그의 정확한 명령문을 사용해주세요. " +
                      ", ".join(unsupported + [f"{sid}: {cid}" for sid, cid in missing]))
    intent=state.get("request_decision") or {}
    simple_lookup = bool(intent and not manual_requested and not explicit
                         and not intent.get("needs_diagnosis"))
    result_scope = {iid for r in intent.get("requested_results", []) if r["kind"] != "remediation"
                    for iid in r["information_ids"]}
    lookup_scope = (set(intent.get("requested_information_ids", [])) | result_scope |
                    {iid for iid, _ in requested_unsupported(query)}) if simple_lookup else set()
    from .response_policy import explicit_information_scope
    manual_ids=intent.get("manual_information_ids",sorted(explicit_information_scope(query)))
    unsupported_specs=[RequirementSpec(description="미지원 대처 요구: "+str(topic)[:260],server_ids=targets,kind="remediation")
                       for topic in intent.get("unsupported_manual_topics",[]) if str(topic).strip()]
    def manual_specs(ids):
        grouped={}
        for iid in ids:grouped.setdefault(iid.split(".")[0],[]).append(iid)
        return [RequirementSpec(description="공식 대처방안: "+", ".join(group),server_ids=targets,information_ids=group,kind="remediation") for group in grouped.values()]+unsupported_specs
    if generic_manual:
        # The serverless route uses the same original typed contract. Replacing
        # its prose with an ID label weakens retrieval and creates duplicate needs.
        plan=TaskPlan(tasks=[],requirements=specs or [RequirementSpec(description=query,server_ids=[],kind="remediation")])
    elif intent.get("requested_results"):
        # Bind the classified result contract exactly once. Investigation may
        # request supporting observations without rewriting user requirements.
        requirements = [RequirementSpec(**r, server_ids=targets) for r in intent["requested_results"]]
        ids = list(dict.fromkeys(intent.get("requested_information_ids", []) + [
            iid for r in requirements if r.kind != "remediation" for iid in r.information_ids]))
        if explicit:
            # The command controls observation selection, not the answer contract.
            ids = list(dict.fromkeys(ids + [iid for commands in catalogs.values()
                       for c in commands if c["command_id"] in explicit for iid in c["information_ids"]]))
        specs = [TaskSpec(server_id=sid, goal="요청한 결과의 근거 확인", information_ids=ids[start:start+10])
                 for sid in targets for start in range(0, len(ids), 10)]
        if len(specs) > 12:
            return finish("한 번에 처리할 조사 항목이 많습니다. 요청 범위를 나눠주세요.")
        plan = TaskPlan(tasks=specs, requirements=requirements)
    elif explicit:
        specs = []
        for sid in targets:
            selected = [c for c in catalogs[sid] if c["command_id"] in explicit]
            specs.append({"server_id": sid, "goal": "지정한 명령의 정보 확인",
                          "information_ids": list(dict.fromkeys(i for c in selected for i in c["information_ids"]))})
        plan = TaskPlan.model_validate({"tasks": specs})
    elif (manual_requested or state.get("request_decision")) and intent.get("requested_information_ids") and not (intent.get("needs_calculation") or intent.get("needs_diagnosis")):
        # Reuse the typed intent's lookup scope instead of planning aliases again.
        ids=list(dict.fromkeys(intent["requested_information_ids"]))
        specs=[TaskSpec(server_id=sid,goal="요청한 서버 정보 확인",information_ids=ids) for sid in targets]
        requirements=[RequirementSpec(description="요청한 상태 확인: "+", ".join(ids),server_ids=targets,information_ids=ids)]
        if intent.get("needs_explanation"):
            requirements.append(RequirementSpec(description="요청한 정보의 의미·해석 설명",server_ids=targets,information_ids=ids,kind="explanation"))
        if intent.get("needs_diagnosis"):
            requirements.append(RequirementSpec(description="관측에 근거한 원인 후보 설명",server_ids=targets,information_ids=ids,kind="diagnosis"))
        if manual_requested:
            requirements+=manual_specs(manual_ids) or [RequirementSpec(description=query,server_ids=targets,kind="remediation")]
        plan=TaskPlan(tasks=specs,requirements=requirements)
    else:
        def initial_check(plan):
            if intent.get("needs_diagnosis") and not any(r.kind == "diagnosis" for r in plan.requirements):
                raise ValueError("원래 원인 조사 요구와 조건을 diagnosis 요구로 보존하세요. 상태 lookup만으로 대신하지 마세요.")
            if intent.get("needs_calculation") and not any(r.kind in {"calculation", "comparison"} for r in plan.requirements):
                raise ValueError("의도 분류에서 계산·비교를 요청했습니다. 그 결과 요구를 calculation/comparison으로 보존하세요. 원본 용량 조회 lookup만으로 비율·환산을 대신할 수 없습니다.")
            if any(not set(r.server_ids) <= set(targets) for r in plan.requirements):
                raise ValueError("요구사항 대상이 원래 대상과 다릅니다.")
            if {t.server_id for t in plan.tasks} != set(targets):
                raise ValueError("지정된 모든 대상 서버의 task가 필요합니다. 다른 서버로 대체할 수 없습니다.")
        try:
            plan = await ask(planner, "plan", base, TaskPlan, initial_check)
        except ValueError as exc:
            return finish(f"조사 계획을 만들지 못했습니다: {exc}")

    # Preserve explicit unavailable requirements even when the model omits them.
    # This guards capability boundaries; supported investigation tasks remain model-selected.
    for sid in targets:
        selected_ids = {iid for t in plan.tasks if t.server_id == sid for iid in t.information_ids}
        for iid, goal in requested_unsupported(query):
            if iid not in selected_ids:
                plan.tasks.append(TaskSpec(server_id=sid, goal=goal, information_ids=[iid]))

    ledger.add(plan.requirements or [RequirementSpec(
        description=t.goal, server_ids=[t.server_id], information_ids=t.information_ids)
        for t in plan.tasks], targets, "plan")
    # Explicit unavailable requests remain user requirements even when the typed
    # intent omitted them from its initial lookup scope.
    for sid in targets:
        for iid, goal in requested_unsupported(query):
            if not any(sid in r["server_ids"] and iid in r["information_ids"] for r in ledger.requirements):
                ledger.add([RequirementSpec(description=goal, server_ids=[sid], information_ids=[iid])], targets, "plan")
    if manual_requested:
        from .remediation import manual_scope_supported
        has_supported=any(r["kind"]=="remediation" and manual_scope_supported(r) for r in ledger.requirements)
        missing_specs=[] if has_supported else [spec for spec in manual_specs(manual_ids) if spec.information_ids]
        ledger.add(missing_specs+unsupported_specs,targets,"plan")
        if not any(r["kind"]=="remediation" for r in ledger.requirements):
            ledger.add([RequirementSpec(description=query,server_ids=targets,kind="remediation")],targets,"plan")
    if manual_requested and targets:
        # Observation failures must remain visible even when a model lists only advice.
        needed=list(dict.fromkeys(i for t in plan.tasks for i in t.information_ids))
        covered={i for r in ledger.requirements if r["kind"]!="remediation" for i in r["information_ids"]}
        if set(needed)-covered:
            ledger.add([RequirementSpec(description="대처 전 필요한 서버 상태 확인",server_ids=targets,information_ids=[i for i in needed if i not in covered])],targets,"plan")

    def add_tasks(plan):
        # Investigation steps support the original request; only the initial plan and
        # final query comparison may register user requirements.
        for spec in plan.tasks:
            if spec.server_id not in targets:
                raise ValueError("요청하지 않은 서버의 task")
            if not all(re.fullmatch(r"[a-z][a-z0-9_.-]{0,79}", i) for i in spec.information_ids):
                raise ValueError("잘못된 정보 ID")
            if lookup_scope:
                ids = [i for i in spec.information_ids if i in lookup_scope]
                if not ids:continue
                spec = spec.model_copy(update={"information_ids": ids})
            key = (spec.server_id, tuple(sorted(spec.information_ids)))
            if any((t["server_id"], tuple(sorted(t["information_ids"]))) == key for t in tasks):
                continue
            task = {**spec.model_dump(), "task_id": f"t{len(tasks)+1}",
                    "os": server_map[spec.server_id]["os"], "command_ids": [],
                    "information_labels": {i: information.get(i, {}).get("name", i) for i in spec.information_ids}}
            # Show which already collected command supplies a newly requested field.
            task["command_ids"] = list(dict.fromkeys(e["command_id"] for e in evidence
                if e["server_id"] == spec.server_id and any(f["information_id"] in spec.information_ids for f in e["facts"])))
            task_status(task, evidence, information, catalogs)
            tasks.append(task)
        ledger.sync(tasks, evidence, "collection")
        emit("RequestRequirements", **ledger.payload())
        emit("InvestigationTasks", tasks=json.loads(json.dumps(tasks, ensure_ascii=False)))

    attempted = {(e["server_id"],e["command_id"]) for e in evidence}

    async def collect_impl(plan):
        nonlocal calls, rounds
        add_tasks(plan)
        if lookup_disabled:
            if any(status!="obtained" for t in tasks for status in t["information_status"].values()):
                note="새 조회를 하지 말라는 요청에 따라 이전 관측에 없는 정보는 확인하지 않았습니다."
                if note not in limitations:limitations.append(note)
                ledger.note("observation_policy",note)
            return False
        jobs = {}
        for task in tasks:
            sid = task["server_id"]
            by_id = {c["command_id"]: c for c in catalogs[sid]}
            if explicit and rounds == 0:
                required_commands = explicit
            else:
                required_commands = [c for iid, status in task["information_status"].items()
                                     if status == "pending" for c in
                                     information.get(iid, {}).get("commands_by_os", {}).get(task["os"], [])]
            for cid in required_commands:
                if cid in by_id and (sid, cid) not in attempted:
                    jobs[(sid, cid)] = by_id[cid]
        if not jobs:
            return False
        if rounds >= MAX_ROUNDS or calls >= MAX_COMMAND_CALLS:
            ledger.note("collection_limit", "추가 조회 2회 또는 명령 조회 12건 한도에 도달했습니다.")
            limitations.append("조회 한도에 도달해 추가 정보를 확보하지 못했습니다.")
            return False
        rounds += 1
        ordered=list(jobs.items())
        if not explicit:
            pending={(t["server_id"],iid) for t in tasks for iid,status in t["information_status"].items() if status=="pending"}
            ordered.sort(key=lambda job:-sum((job[0][0],iid) in pending for iid in job[1]["information_ids"]))
        for (sid, cid), command in ordered:
            # Earlier jobs in the same plan can already satisfy this information.
            wanted = {iid for t in tasks if t["server_id"] == sid for iid, status in t["information_status"].items()
                      if status == "pending"}
            if not (explicit and rounds == 1) and not wanted.intersection(command["information_ids"]):
                continue
            if calls >= MAX_COMMAND_CALLS:
                limitations.append("요청당 명령 조회 한도에 도달해 추가 정보를 확보하지 못했습니다.")
                ledger.note("collection_limit", limitations[-1])
                break
            calls += 1
            attempted.add((sid, cid))
            for t in tasks:
                if t["server_id"] == sid and set(t["information_ids"]) & set(command["information_ids"]):
                    if cid not in t["command_ids"]:t["command_ids"].append(cid)
            emit("CommandSelected", server_id=sid, command=command["command"],
                 information_ids=command["information_ids"], round=rounds)
            try:
                raw = await call_tool_result(deps.mcp, "get_command_result", {"server_id": sid, "command": cid})
                checked = CommandResult.model_validate(raw)
                if (checked.server_id, checked.os, checked.command_id, checked.command) != (
                        sid, server_map[sid]["os"], cid, command["command"]):
                    raise ValueError("요청 대상과 명령 응답이 일치하지 않습니다.")
                result = checked.model_dump(mode="json")
                result["facts"] = extract_facts(result)
                result["error"] = None
            except ValueError as exc:
                result = {"server_id": sid, "os": server_map[sid]["os"], "command_id": cid,
                          "command": command["command"], "stdout": "", "stderr": "", "exit_code": None,
                          "observed_at": None, "source": "unavailable", "facts": [], "error": str(exc)}
            result["evidence_id"] = f"e{len(evidence)+1}"
            evidence.append(result)
            emit("CommandCompleted", server_id=sid, command=command["command"], exit_code=result["exit_code"])
            for t in tasks:task_status(t, evidence, information, catalogs)
            ledger.sync(tasks, evidence, f"collection_{rounds}")
            emit("RequestRequirements", **ledger.payload())
            emit("InvestigationTasks", tasks=json.loads(json.dumps(tasks, ensure_ascii=False)))
        return True

    async def collect(plan):
        tick=perf_counter()
        try:return await collect_impl(plan)
        finally:timings["collection"]=timings.get("collection",0)+(perf_counter()-tick)*1000

    try:
        for _ in range(MAX_ROUNDS):
            if not await collect(plan):break
            if calls >= MAX_COMMAND_CALLS or rounds >= MAX_ROUNDS:
                break
            # A state lookup stays within the requested metrics. Root-cause
            # investigations keep their dynamic followups; missing lookups can be repaired.
            if lookup_scope:break
            if use_specialists and any(r["kind"] == "diagnosis" for r in ledger.requirements):break
            if state.get("request_decision"):
                if (explicit or manual_requested) and not intent.get("needs_diagnosis"):break
            else:
                if explicit and not re.search(r"원인|분석|왜|진단", query):break
                if manual_requested and not re.search(r"원인|분석|왜|진단|느리|느려|문제",query):break
            def review_check(answer):
                if not {t.server_id for t in answer.tasks} <= set(targets):
                    raise ValueError("추가 조사 대상이 원래 대상과 다릅니다.")
            obtained = {sid: sorted({f["information_id"] for e in evidence if e["server_id"] == sid for f in e["facts"]}) for sid in targets}
            plan = await ask(planner, "review", {
                **base, "tasks": tasks, "obtained_information": obtained,
                "evidence": [{k: e[k] for k in ("evidence_id", "server_id", "command_id", "facts", "error")} for e in evidence],
                **ledger.payload(),
            }, TaskPlan, review_check)
            for spec in plan.tasks:
                spec.information_ids = [i for i in spec.information_ids if i not in obtained[spec.server_id]]
            plan.tasks = [spec for spec in plan.tasks if spec.information_ids]
            if not plan.tasks:break
            emit("InvestigationReview", round=rounds, additional_task_count=len(plan.tasks))
    except ValueError as exc:
        limitations.append(str(exc))
        ledger.note("collection", str(exc))

    for task in tasks:
        task_status(task, evidence, information, catalogs)
        for iid in task["information_ids"]:
            if iid in information and information[iid]["limitations"]:
                limitations.append(information[iid]["limitations"])
    for e in evidence:
        if e.get("error"):
            limitations.append(f"{e['server_id']} · {e['command_id']} 조회 실패: {e['error']}")
        elif e["exit_code"] != 0:
            limitations.append(f"{e['server_id']} · {e['command_id']} 종료 코드 {e['exit_code']}: {e['stderr']}")

    ledger.sync(tasks, evidence, "collection_complete")
    emit("RequestRequirements", **ledger.payload())
    raw_only = bool(re.search(r"json\s*만|원문\s*만", query, re.I))
    if raw_only:
        presentation={"status":"raw","model_calls":0}
        results = [{k: v for k, v in e.items() if k not in {"facts", "evidence_id"} and (k != "error" or v)} for e in evidence]
        return finish(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2))
    async def run_diagnostics():
        nonlocal specialist_assessments, specialist_key, specialist_failures, answers, hypotheses
        assigned = [r for r in ledger.requirements if r["kind"] == "diagnosis"]
        if not use_specialists or not assigned:return
        key = (tuple(e["evidence_id"] for e in evidence), tuple(r["requirement_id"] for r in assigned))
        if key == specialist_key:return
        from .graph import build_diagnostic_agent, diagnostic_scope, diagnostic_observations
        class Adapter:
            name = getattr(deps.quality_llm, "name", "diagnostic_model")
            def monitoring_json(self, purpose, payload, schema):
                return deps.quality_llm.diagnostic_json(payload, schema)
        async def decide(role, requirements, observations, allowed):
            payload = {"role": role, "original_query": query, "target_os":{sid:server_map[sid]["os"] for sid in targets}, "requirements": [{**{k:r[k] for k in ("requirement_id", "description", "server_ids", "information_ids")},
                    **{k:r[k] for k in ("supporting_for", "supporting_server_ids") if k in r}} for r in requirements],
                       "targets": targets, "allowed_information_ids": allowed,
                       "read_only_information_ids":sorted(diagnostic_scope(role, requirements)[1]),
                       "observations": [{"evidence_id":e["evidence_id"], "server_id":e["server_id"], "facts":e["facts"]} for e in observations],
                       "quote_catalog": evidence_quote_catalog(observations)}
            def check(decision):
                selected = [rid for a in decision.diagnoses for rid in a.requirement_ids]
                known = {r["requirement_id"] for r in requirements}
                if not set(selected) <= known or len(selected) != len(set(selected)):
                    raise ValueError("전문 에이전트가 배정되지 않은 요구 또는 중복 판단을 반환했습니다.")
                assigned_targets = {sid for r in requirements for sid in r["server_ids"]}
                if any(t.server_id not in assigned_targets or not set(t.information_ids) <= set(allowed) for t in decision.tasks):
                    raise ValueError("전문 에이전트의 조회가 배정된 대상·정보 범위를 벗어났습니다.")
                obtained = {(e["server_id"], f["information_id"]) for e in observations for f in e["facts"]}
                effective_tasks = [t for t in decision.tasks if any((t.server_id, i) not in obtained for i in t.information_ids)]
                support_needed = {(sid, "log.usage") for r in requirements if "disk" in r.get("supporting_for", [])
                                  for sid in r.get("supporting_server_ids", [])} - obtained
                requested = {(t.server_id, i) for t in decision.tasks for i in t.information_ids}
                if support_needed - requested:
                    raise ValueError("디스크 증상의 추가 근거 수집이 배정됐습니다. supporting_server_ids의 아직 없는 log.usage를 tasks로 요청하세요. 읽기 전용 disk 관측만 반복해서 완료하지 마세요.")
                if not effective_tasks and set(selected) != known:
                    raise ValueError("배정된 모든 증상을 판단하거나 아직 얻지 못한 정보의 추가 조회를 요청하세요. 이미 확보한 정보는 유효한 추가 task가 아닙니다.")
                facts = {f["information_id"] for e in observations for f in e["facts"]}
                symptom_ids = {rid for a in decision.diagnoses if a.symptom_observed for rid in a.requirement_ids}
                requires_process = any(r["requirement_id"] in symptom_ids and
                    set(r["information_ids"]) & {"cpu", "memory.summary", "memory.detail"} for r in requirements)
                if role == "metrics" and requires_process and "process" not in facts and not any("process" in t.information_ids for t in decision.tasks):
                    raise ValueError("CPU·메모리의 증상 요약만으로 원인 후보 조사가 끝나지 않습니다. 아직 없는 process 정보를 추가 task로 요청하세요. 같은 증상을 원인으로 되풀이하지 마세요.")
                verified = []
                for a in decision.diagnoses:
                    try:
                        bind_report_sources(InvestigationReport(diagnoses=[a], hypotheses=[]), observations)
                        ground_resource_assessment(a, observations, requirements)
                        ground_disk_absence(a, observations, requirements)
                        ground_storage_support(a, observations, requirements)
                        verify_report(InvestigationReport(hypotheses=[a.model_dump(exclude={"symptom_observed", "requirement_ids"})]), observations)
                    except ValueError as exc:
                        # A valid observation request is independent of its accompanying
                        # prose. Drop unverified prose, collect scoped new information,
                        # and reassess; without a new task the normal retry still applies.
                        if not effective_tasks:raise
                        ledger.note("diagnostic_"+role, "추가 조회 전 검증되지 않은 진단을 제외했습니다: "+str(exc), a.requirement_ids)
                    else:
                        verified.append(a)
                decision.diagnoses = verified
            decision = await ask(Adapter(), "diagnostic_"+role, payload, DiagnosticDecision, check)
            return decision.model_dump()
        async def observe(items):
            progressed = await collect(TaskPlan(tasks=items))
            return evidence, progressed
        def scoped_requirements(role):
            result = []
            for r in assigned:
                scoped = [i for i in r["information_ids"] if i.startswith("log.") == (role == "logs")]
                if scoped or (not r["information_ids"] and role == "metrics"):
                    result.append({**r, "information_ids":scoped})
            return result

        def input_key(role, requirements):
            observations = [{k:e.get(k) for k in ("evidence_id", "server_id", "facts", "stdout", "error")}
                            for e in diagnostic_observations(role, requirements, evidence)]
            assignment = [{k:r[k] for k in ("requirement_id", "description", "server_ids", "information_ids", "kind",
                                           "supporting_for", "supporting_server_ids") if k in r} for r in requirements]
            return json.dumps([assignment, observations], sort_keys=True, ensure_ascii=False)

        def supporting_logs(requirements, assessments):
            by_id = {r["requirement_id"]:deepcopy(r) for r in requirements}
            for support in disk_log_support(assigned, evidence, assessments):
                rid = support["requirement_id"]
                if rid in by_id:
                    current = by_id[rid]
                    for key in ("information_ids", "server_ids"):
                        current[key] = list(dict.fromkeys(current[key]+support[key]))
                    current.update({k:support[k] for k in ("supporting_for", "supporting_server_ids")})
                else:by_id[rid] = support
            return list(by_id.values())

        assignments = {role:scoped_requirements(role) for role in ("metrics", "logs")}
        combined = []
        previous_metrics = (specialist_cache.get("metrics") or {}).get("result") or {}
        prior_assessments = [{**a, "source_role":"metrics"} for a in previous_metrics.get("diagnoses", [])]
        for role, requirements in assignments.items():
            if role == "logs":requirements = supporting_logs(requirements, prior_assessments)
            cached = specialist_cache.get(role)
            if requirements and cached and cached["result"] is not None and cached["key"] == input_key(role, requirements):
                combined += [{**a, "source_role":role} for a in cached["result"].get("diagnoses", [])]
        assigned_ids = {r["requirement_id"] for r in assigned}
        retained = [{**a, "requirement_ids":[rid for rid in a["requirement_ids"] if rid not in assigned_ids]}
                    for a in answers if set(a["requirement_ids"]) - assigned_ids]
        # Seed unchanged roles before awaiting any changed role. A deadline during
        # that reassessment must not erase already verified independent results.
        answers = retained + [AnswerStatement.model_validate({k:v for k,v in a.items() if k not in {"source_role", "symptom_observed"}}).model_dump() for a in combined]
        save_progress({"answers":answers})
        specialist_failures = set()
        for role, requirements in assignments.items():
            if role == "logs":
                # Assess metrics first, then delegate source-bound supporting work.
                # These are role-local assignments, not new user requirements.
                requirements = supporting_logs(requirements, combined)
            if not requirements:continue
            emit("SpecialistDelegated", role=role, requirement_ids=[r["requirement_id"] for r in requirements])
            required = {r["requirement_id"] for r in requirements}
            cached = specialist_cache.get(role)
            if cached and cached["key"] == input_key(role, requirements):
                result = cached["result"]
                if result is None:
                    specialist_failures.update(required)
                    continue
                emit("SpecialistReused", role=role, status=result["status"])
            else:
                graph = build_diagnostic_agent(role, decide, observe, emit)
                try:
                    result = await graph.ainvoke({"assignment":deepcopy(requirements), "evidence":deepcopy(evidence), "rounds":0})
                    returned = {rid for a in result.get("diagnoses", []) for rid in a["requirement_ids"]}
                    if required != returned:
                        raise ValueError("전문 에이전트가 조회 한도 안에 배정된 증상 판단을 완료하지 못했습니다.")
                except ValueError as exc:
                    specialist_cache[role] = {"key":input_key(role, requirements), "result":None}
                    specialist_failures.update(required)
                    hypotheses = [h for h in hypotheses if not set(h.get("requirement_ids", [])) & required]
                    ledger.note("diagnostic_"+role, str(exc), sorted(required))
                    note = ("메트릭" if role == "metrics" else "로그") + " 전문 판단을 완료하지 못했습니다. 다른 역할의 결과와 확보된 관측은 유지합니다."
                    if note not in limitations:limitations.append(note)
                    emit("SpecialistFailed", role=role, requirement_ids=sorted(required), reason=str(exc))
                    continue
                specialist_cache[role] = {"key":input_key(role, requirements), "result":deepcopy(result)}
            recovered_note = ("메트릭" if role == "metrics" else "로그") + " 전문 판단을 완료하지 못했습니다. 다른 역할의 결과와 확보된 관측은 유지합니다."
            if recovered_note in limitations:limitations.remove(recovered_note)
            combined = [a for a in combined if a["source_role"] != role]
            combined += [{**a, "source_role":role} for a in result.get("diagnoses", [])]
            retained = [{**a, "requirement_ids":[rid for rid in a["requirement_ids"] if rid not in {r["requirement_id"] for r in assigned}]}
                        for a in answers if set(a["requirement_ids"]) - {r["requirement_id"] for r in assigned}]
            saved_answers = [AnswerStatement.model_validate({k:v for k,v in a.items() if k not in {"source_role", "symptom_observed"}}).model_dump() for a in combined]
            answers = retained + saved_answers
            emit("SpecialistReturned", role=role, status=result["status"], rounds=result.get("rounds",0))
        specialist_assessments = combined
        specialist_key = (tuple(e["evidence_id"] for e in evidence), tuple(r["requirement_id"] for r in assigned))

    def writer_payload(feedback):
        # Put measured process rows before ancillary logs; keep original IDs and text.
        process_sources=[{"evidence_id":e["evidence_id"],"rows":list(dict.fromkeys(
            line for line in e["stdout"].splitlines() if any(
                p["name"] in line for f in e["facts"] if f["information_id"]=="process"
                for p in f["fields"].get("entries",[]))))}
            for e in evidence if any(f["information_id"]=="process" for f in e["facts"])]
        return {"query": query, "request_decision":intent, "specialist_assessments":specialist_assessments,
                "measured_process_sources":process_sources,
                "cause_catalog":cause_catalog(evidence,ledger.requirements),
                "eligible_cause_choices":eligible_cause_choices(evidence,ledger.requirements,diagnosis_assessments),
                "quote_catalog":evidence_quote_catalog(evidence),
                "targets": targets, "tasks": tasks, **ledger.payload(),
                "final_feedback": feedback, "calculation_rejections":deepcopy(ledger.calculation_rejections),
                "verified_results":{"answers":deepcopy(answers),"calculations":deepcopy(calculations),"hypotheses":deepcopy(hypotheses)},
                "repair_requirement_ids":None if feedback is None else feedback.get("repair_requirement_ids",[]),
                "evidence": [{"evidence_id": e["evidence_id"], "server_id": e["server_id"],
                              "command_id": e["command_id"], "observed_at": e["observed_at"],
                              "facts": e["facts"], "error": e["error"], "exit_code": e["exit_code"]} for e in evidence]}

    async def compose(feedback):
        nonlocal hypotheses, answers, calculations, diagnosis_assessments
        if not any(e["facts"] for e in evidence):return
        await run_diagnostics()
        if specialist_assessments is not None:
            # These role results were already source-checked. A writer failure
            # must not discard successful independent specialist work.
            diagnosis_assessments = [{k:v for k,v in a.items() if k != "source_role"} for a in specialist_assessments]
            sourced = [AnswerStatement.model_validate({k:v for k,v in a.items() if k not in {"source_role", "symptom_observed"}}).model_dump()
                       for a in specialist_assessments]
            diagnosis_ids = {r["requirement_id"] for r in ledger.requirements if r["kind"] == "diagnosis"}
            retained = [{**a, "requirement_ids":[rid for rid in a["requirement_ids"] if rid not in diagnosis_ids]}
                        for a in answers if set(a["requirement_ids"]) - diagnosis_ids]
            answers = retained + sourced
            hypotheses = [h for h in hypotheses if not set(h.get("requirement_ids", [])) & specialist_failures]
            save_progress({"answers":answers, "hypotheses":hypotheses})
        repair_ids=None if feedback is None else set(feedback.get("repair_requirement_ids",[]))
        verified_attempt_hypotheses = []
        def checked_report(report):
            nonlocal calculations, answers, hypotheses, verified_attempt_hypotheses, diagnosis_assessments
            item_errors = []
            if specialist_assessments is not None:
                report.diagnoses = [DiagnosisAssessment.model_validate({k:v for k,v in a.items() if k != "source_role"})
                                    for a in specialist_assessments]
            # Source binding is per item below. An invalid quote in one item
            # must not prevent independent formulas or explanations being kept.
            ids = [r["requirement_id"] for r in ledger.requirements]
            diagnosis_ids = {r["requirement_id"] for r in ledger.requirements if r["kind"] == "diagnosis"}
            if "diagnoses" in report.model_fields_set and diagnosis_ids:
                assessed = [rid for a in report.diagnoses for rid in a.requirement_ids]
                if (set(assessed) | specialist_failures) != diagnosis_ids or (specialist_assessments is None and len(assessed) != len(set(assessed))):
                    raise ValueError("원인 후보 전에 각 diagnosis 요구의 증상 관측 여부를 중복·누락 없이 판단하세요.")
                # A mixed requirement may have one assessment per assigned role.
                # A normal metric must not suppress an observed log symptom.
                supported = {rid for a in report.diagnoses if a.symptom_observed for rid in a.requirement_ids}
                absent = set(assessed) - supported
                if any(set(h.requirement_ids) & absent for h in report.hypotheses):
                    raise ValueError("증상이 관측되지 않은 진단에는 원인 후보를 만들지 마세요. 현재 수치와 확인 한계를 설명하세요.")
                for assessment in report.diagnoses:
                    try:
                        bind_report_sources(InvestigationReport(diagnoses=[assessment],hypotheses=[]),evidence)
                        # Specialist assessments were already checked in their actual role scope.
                        if specialist_assessments is None:
                            ground_resource_assessment(assessment, evidence, ledger.requirements)
                            ground_disk_absence(assessment, evidence, ledger.requirements)
                    except ValueError as exc:
                        item_errors.append(str(exc))
                    else:
                        report.answers.append(AnswerStatement.model_validate(assessment.model_dump(exclude={"symptom_observed"})))

            computed = []
            for spec in report.calculations:
                try:
                    for rid in spec.requirement_ids:
                        requirement=next((r for r in ledger.requirements if r["requirement_id"]==rid),None)
                        if requirement is None:raise ValueError("계산에 알 수 없는 요구사항 ID가 있습니다.")
                        for ref in spec.operands:
                            source=next((e for e in evidence if e["evidence_id"]==ref.evidence_id),None)
                            if ref.information_id not in requirement["information_ids"] or not source or source["server_id"] not in requirement["server_ids"]:
                                raise ValueError("계산 피연산자가 해당 요구의 대상·정보 범위와 다릅니다. 요청에 맞는 원본 필드로 수정하세요.")
                    computed.append(calculate(spec, evidence, ids))
                except CalculationRejected as rejection:
                    ledger.record_calculation_rejection(rejection.record())
                    refs = ", ".join(f"{i['evidence_id']}:{i['field']}={i['value']} {i['unit']}" for i in rejection.operand_refs)
                    note = str(rejection) + " 원본: " + refs
                    calculation_notes[rejection.fingerprint] = note
                    if note not in limitations:limitations.append(note)
                except ValueError as exc:
                    item_errors.append(str(exc))
            # Each formula has already passed source, unit and basis checks.
            # Keep it even when a sibling answer is missing or its prose fails.
            updated_ids = {rid for c in computed for rid in c["requirement_ids"]}
            calculations = merge_verified_results(calculations, computed,
                repair_ids if repair_ids is not None else updated_ids)
            save_progress({"calculation_results":calculations, **ledger.payload()})
            missing_calculations = pending_calculation_ids()
            if missing_calculations:
                item_errors.append("원본 정보가 확보된 계산 요구가 누락됐습니다: " + ", ".join(missing_calculations)
                                 + ". calculations에 해당 requirement_ids, operation, 원본 operands를 작성하세요. 원본 수치 재설명으로 계산을 대체할 수 없습니다.")
            # Canonical calculated text comes from checked operands. A redundant
            # model paraphrase must not introduce a rounded or conflicting value.
            calculated_ids = {rid for c in computed+calculations for rid in c["requirement_ids"]}
            calculation_only_ids = {r["requirement_id"] for r in ledger.requirements if r["kind"] == "calculation"}
            report.answers = [a.model_copy(update={"requirement_ids":[rid for rid in a.requirement_ids if not ledger.is_calculation_blocked(rid)]})
                              for a in report.answers if any(not ledger.is_calculation_blocked(rid) for rid in a.requirement_ids)]
            valid_answers = []
            for a in report.answers:
                if not set(a.requirement_ids) <= set(ids):
                    item_errors.append("알 수 없는 답변 요구사항 ID")
                    continue
                # Exact sources/numbers are checked here; semantic scope is checked by final_review.
                verified_numbers = [c["value"] for c in computed+calculations
                                    if set(c["evidence_ids"]) <= set(a.evidence_ids)
                                    and set(c["requirement_ids"]) & set(a.requirement_ids)]
                try:
                    bind_report_sources(InvestigationReport(answers=[a],hypotheses=[]),evidence)
                    verify_report(InvestigationReport(hypotheses=[a.model_dump(exclude={"requirement_ids"})]),
                                  evidence, calculated_numbers=verified_numbers)
                except ValueError as exc:
                    if not set(a.requirement_ids) <= calculated_ids & calculation_only_ids:
                        item_errors.append(str(exc))
                    # The rejected paraphrase is never rendered; the validated
                    # formula still answers this calculation-only requirement.
                else:
                    valid_answers.append(a)
            report.answers = valid_answers
            answer_updates = [a.model_dump() for a in valid_answers]
            updated_answer_ids = {rid for a in answer_updates for rid in a["requirement_ids"]}
            answers = merge_verified_results(answers, answer_updates,
                updated_answer_ids if repair_ids is None else repair_ids)
            save_progress({"answers":answers, "calculation_results":calculations, **ledger.payload()})
            valid, rejected, hypothesis_errors = [], 0, []
            diagnosis_ids = {r["requirement_id"] for r in ledger.requirements if r["kind"] == "diagnosis"}
            verified_assessments = [a.model_dump() for a in report.diagnoses if any(
                a.text == v.text and a.requirement_ids == v.requirement_ids and a.evidence_ids == v.evidence_ids
                for v in valid_answers)]
            if verified_assessments:
                diagnosis_assessments = verified_assessments
            choices = {c["factor_id"]:c for c in cause_catalog(evidence,ledger.requirements)}
            selected_groups = {}
            for selection in report.causes:
                factor = choices.get(selection.factor_id)
                if factor:
                    selected_groups.setdefault((selection.requirement_id,factor["server_id"],factor["kind"]),set()).add(selection.factor_id)
            for selection in report.causes:
                try:
                    factor = choices.get(selection.factor_id)
                    if factor and len(selected_groups[(selection.requirement_id,factor["server_id"],factor["kind"])]) > 1:
                        raise ValueError("같은 요구의 같은 종류 요인은 가장 관련성이 큰 주요 후보 하나만 선택하세요. 작은 점유 프로세스를 동등한 후보로 나열하지 마세요.")
                    candidate = bind_cause(selection,evidence,ledger.requirements,verified_assessments)
                    if candidate.model_dump() not in [h.model_dump() for h in valid]:valid.append(candidate)
                except ValueError as exc:
                    rejected += 1
                    hypothesis_errors.append(str(exc))
            for item in report.hypotheses:
                if diagnosis_ids:
                    rejected += 1
                    hypothesis_errors.append("진단 원인 후보는 자유 문장이 아니라 causes의 requirement_id/factor_id로 선택해야 합니다.")
                    continue
                if not item.requirement_ids and len(diagnosis_ids) == 1:
                    item.requirement_ids = list(diagnosis_ids)  # legacy injected writers
                if diagnosis_ids and (not item.requirement_ids or not set(item.requirement_ids) <= diagnosis_ids):
                    item_errors.append("각 원인 후보는 실제 diagnosis 요구사항 ID를 지정해야 합니다.")
                    continue
                try:
                    verify_report(InvestigationReport(hypotheses=[item]), evidence)
                    valid.append(item)
                except ValueError as exc:
                    rejected += 1
                    hypothesis_errors.append(str(exc))
            if (report.hypotheses or report.causes) and not valid:item_errors.extend(hypothesis_errors)
            report.hypotheses = valid
            candidate_updates = [h.model_dump() for h in valid if h.requirement_ids]
            candidate_ids = {rid for h in candidate_updates for rid in h["requirement_ids"]}
            # Only this compose attempt's verified candidates survive a sibling
            # error. An explicit absent-symptom reassessment can withdraw them.
            verified_attempt_hypotheses = withdraw_absent_hypotheses(verified_attempt_hypotheses,verified_assessments)
            hypotheses = withdraw_absent_hypotheses(hypotheses,verified_assessments)
            verified_attempt_hypotheses = merge_verified_results(verified_attempt_hypotheses,candidate_updates,candidate_ids)
            hypotheses = merge_verified_results(hypotheses,candidate_updates,candidate_ids)
            save_progress({"hypotheses":hypotheses})
            # Symptoms and sibling explanations do not answer a cause request.
            # Only demand coverage where the existing source/scope checks can
            # bind a contributor; absent/unknown symptoms remain honest partials.
            eligible = eligible_cause_choices(evidence,ledger.requirements,verified_assessments)
            covered = {rid for h in verified_attempt_hypotheses for rid in h["requirement_ids"]}
            missing = [c for c in eligible if c["requirement_id"] not in covered
                       and (repair_ids is None or c["requirement_id"] in repair_ids)]
            if missing:
                item_errors.append("검증된 증상에 연결 가능한 원인 후보가 누락됐습니다. causes에 각 요구별 관련 요인을 선택하세요: "
                                   + json.dumps(missing,ensure_ascii=False)
                                   + ". 증상 설명만으로 원인 후보를 대신하지 마세요. 근본 원인 확정은 허용하지 않습니다.")
            if rejected:limitations.append(f"근거 검증을 통과하지 못한 원인 후보 {rejected}개를 제외했습니다.")
            if item_errors:raise ValueError("; ".join(dict.fromkeys(item_errors)))
        try:
            if (specialist_assessments and not any(a["symptom_observed"] for a in specialist_assessments)
                    and all(r["kind"] in {"lookup", "diagnosis"} for r in ledger.requirements)):
                report = InvestigationReport(hypotheses=[])
                checked_report(report)
            else:
                report = await ask(deps.quality_llm, "report", writer_payload(feedback), InvestigationReport, checked_report)
            diagnosis_ids={r["requirement_id"] for r in ledger.requirements if r["kind"]=="diagnosis"}
            if "diagnoses" in report.model_fields_set:
                diagnosis_assessments = [a.model_dump() for a in report.diagnoses]
            fresh_hypotheses = verified_attempt_hypotheses + [h.model_dump(exclude={"requirement_ids"})
                                for h in report.hypotheses if not h.requirement_ids]
            if repair_ids is None:
                hypotheses = fresh_hypotheses
            elif repair_ids & diagnosis_ids:
                # A successful reassessment may withdraw a former candidate.
                # Keep other requirements, but never revive a rejected cause just
                # because the corrected report has an empty hypotheses array.
                retained = []
                for h in hypotheses:
                    remaining = set(h.get("requirement_ids", [])) - repair_ids
                    if remaining:retained.append({**h, "requirement_ids":sorted(remaining)})
                hypotheses = retained + fresh_hypotheses
            answer_updates = [a.model_dump() for a in report.answers]
            answers = merge_verified_results(answers,answer_updates,
                {rid for a in answer_updates for rid in a["requirement_ids"]} if repair_ids is None else repair_ids)
            completed_ids = (repair_ids if repair_ids is not None else
                             {r["requirement_id"] for r in ledger.requirements if r["kind"] != "remediation"})
            # A partial compose may already have preserved a valid sibling.
            # The prior full-answer review, after programmatic downgrades,
            # identifies those siblings; do not keep blaming them after repair.
            if final_reviews:
                completed_ids = completed_ids | {c["requirement_id"] for c in final_reviews[-1]["checks"]
                    if c["status"] == "satisfied" and c["requirement_id"] not in specialist_failures}
            for note, affected in list(report_failure_notes.items()):
                remaining = affected - completed_ids
                if remaining:
                    report_failure_notes[note] = remaining
                else:
                    report_failure_notes.pop(note)
                    if note in limitations:limitations.remove(note)
            save_progress({"answers":answers, "hypotheses":hypotheses, "calculation_results":calculations, **ledger.payload()})
        except ValueError as exc:
            affected = (repair_ids if repair_ids is not None else
                        {r["requirement_id"] for r in ledger.requirements if r["kind"] != "remediation"})
            ledger.note("answer_generation", str(exc), sorted(affected))
            note = f"답변 일부의 근거 검증 실패: {exc}. 확보한 결과와 미충족 요구를 함께 표시합니다."
            report_failure_notes.setdefault(note, set()).update(affected)
            if note not in limitations:limitations.append(note)

    def draft(include_requirement_notes=True):
        # Rebuild notes after repair, including newly failed commands and unsupported fields.
        for t in tasks:
            for iid in t["information_ids"]:
                note = information.get(iid, {}).get("limitations")
                if note and note not in limitations:limitations.append(note)
        for e in evidence:
            if e.get("error") or e.get("exit_code") != 0:
                note = f"{e['server_id']} · {e['command_id']} 조회 실패: {e.get('error') or e['stderr']}"
                if note not in limitations:limitations.append(note)
        return render_answer(tasks, evidence, hypotheses, limitations, answers, calculations, ledger.requirements if include_requirement_notes else ()) if targets else ""

    feedback = None
    seen_review_states = set()
    for repair in range(MAX_ANSWER_REPAIRS + 1):
        manual_only_repair = bool(feedback and feedback.get("manual_only"))
        presentation_only=bool(feedback and feedback.get("presentation_only"))
        needs_interpretation = (any(r["kind"] not in {"lookup","remediation"} for r in ledger.requirements) if manual_requested
                                else should_generate_report(intent, ledger.requirements, feedback))
        if not manual_only_repair and not presentation_only and needs_interpretation:await compose(feedback)
        if manual_requested and not presentation_only:
            context_state={**state,"workflow_active":True,"query":query,"targets":targets,"manual_os":manual_os,
                "evidence":evidence,"tasks":tasks,"hypotheses":hypotheses,"limitations":limitations,
                "command_call_count":calls,"events":events,"cost_usd":cost,**ledger.payload(),
                "manual_state":manual_state,"manual_feedback":feedback}
            tick=perf_counter();rem=await remediation_node(context_state,deps)
            timings["manual"]=timings.get("manual",0)+(perf_counter()-tick)*1000
            manual_result=rem["remediation"];manual_state=rem.get("manual_state",manual_state)
            manual_text=rem["final"];events=rem["events"];cost=rem.get("cost_usd",cost)
            from .remediation import manual_requirement_covered
            for requirement in ledger.requirements:
                if requirement["kind"]!="remediation":continue
                from .remediation import manual_scope_supported
                if not manual_scope_supported(requirement):
                    ledger.update(requirement,"unsupported","대처 주제가 미지원이거나 분명하지 않습니다. CPU·메모리·디스크 범위의 주제를 지정해주세요.","manual")
                    ledger.note("manual",requirement["reason"],[requirement["requirement_id"]])
                    continue
                if manual_requirement_covered(requirement,manual_result,evidence):
                    ledger.update(requirement,"information_ready","공식 매뉴얼 근거를 확보했습니다. 전체 답변 검증이 남았습니다.","manual")
                elif manual_result["status"]=="unavailable":
                    ledger.update(requirement,"failed","공식 매뉴얼 검색·생성에 실패했습니다.","manual")
                else:
                    ledger.update(requirement,"needs_more","요청한 대처의 공식 근거를 충분히 확보하지 못했습니다.","manual")
                    ledger.note("manual",requirement["reason"],[requirement["requirement_id"]])
        text = draft(include_requirement_notes=False)
        if manual_requested:text="\n\n".join(x for x in [text,manual_text] if x)
        tick=perf_counter()
        rendered=await present_response({**state,"route":state.get("route","commands"),"final":text,
            "evidence":evidence,"tasks":tasks,"answers":answers,"hypotheses":hypotheses,"calculation_results":calculations,"remediation":manual_result,**ledger.payload(),
            "events":events,"cost_usd":cost},deps)
        timings["presentation"]=timings.get("presentation",0)+(perf_counter()-tick)*1000
        text=rendered.get("final",text);presentation=rendered.get("presentation",{})
        events=rendered["events"];cost=rendered.get("cost_usd",cost)
        emit("WorkflowBudget",query_rounds=rounds,command_calls=calls,manual_searches=manual_state.get("search_calls",0),repair_rounds=repair)
        expected_ids = {r["requirement_id"] for r in ledger.requirements}
        def checked_review(review):
            model_unmet = {c.requirement_id for c in review.checks if c.status == "unmet"}
            ids = [c.requirement_id for c in review.checks]
            if set(ids) != expected_ids or len(ids) != len(set(ids)):
                raise ValueError("모든 원래 요구사항을 중복·누락 없이 검증해야 합니다.")
            if any(t.server_id not in targets for t in review.repair_tasks):
                raise ValueError("검증 후 추가 조회 대상이 원래 대상과 다릅니다.")
            if len(ledger.requirements) + len(review.missing_requirements) > 32:
                raise ValueError("요구사항 기록 한도에 도달했습니다.")
            if any(not set(r.server_ids) <= set(targets) for r in review.missing_requirements):
                raise ValueError("추가 요구사항 대상이 원래 대상과 다릅니다.")
            if lookup_scope:
                review.repair_tasks = [t.model_copy(update={"information_ids": [i for i in t.information_ids if i in lookup_scope]})
                                       for t in review.repair_tasks if set(t.information_ids) & lookup_scope]
                review.missing_requirements = [r for r in review.missing_requirements
                    if set(r.information_ids) <= lookup_scope
                    and (not intent.get("basic_status") or (r.kind == "lookup" and r.information_ids))]
            by_id = {r["requirement_id"]: r for r in ledger.requirements}
            for c in review.checks:
                r = by_id[c.requirement_id]
                if c.requirement_id in specialist_failures:
                    c.status, c.reason = "unmet", "배정된 전문 역할의 판단이 일부 실패했습니다. 성공한 역할의 결과만으로 전체 진단을 충족할 수 없습니다."
                if ledger.is_calculation_blocked(c.requirement_id):
                    if c.rejection_matches_request is False:
                        ledger.allow_calculation_repair(c.requirement_id)
                        c.status="unmet"
                        for rejection in ledger.calculation_rejections:
                            if all(not ledger.is_calculation_blocked(rid) for rid in rejection["requirement_ids"]):
                                note=calculation_notes.get(rejection["fingerprint"])
                                if note in limitations:limitations.remove(note)
                    else:
                        c.status, c.reason = "unavailable", r["reason"]
                if c.status == "satisfied" and r["status"] in {"failed", "unsupported"}:
                    c.status = "unavailable"
                elif c.status == "satisfied" and r["information_ids"] and r["status"] in {"needs_more", "pending"}:
                    if not lookup_disabled:raise ValueError("확보하지 못한 필수 정보는 충족으로 처리할 수 없습니다.")
                    c.status,c.reason="unmet","확보하지 못한 필수 정보는 충족으로 처리할 수 없습니다."
                if c.status == "satisfied" and r["kind"] == "calculation" and not any(c.requirement_id in x["requirement_ids"] for x in calculations):
                    c.status, c.reason = "unmet", "요청한 계산 결과·계산식·원문 출처가 답변에 없습니다."
                if (c.status == "satisfied" and r["kind"] == "comparison"
                        and c.requirement_id in ledger.repairable_calculation_requirements
                        and not any(c.requirement_id in x["requirement_ids"] for x in calculations)
                        and not any(c.requirement_id in a["requirement_ids"] and a["text"] in text for a in answers)):
                    c.status, c.reason = "unmet", "거절한 비교를 대체할 검증된 계산이나 원문 근거가 있는 답변이 없습니다."
                if c.status == "satisfied" and r["kind"] == "diagnosis" and not any(
                        c.requirement_id in h.get("requirement_ids", []) and h["text"] in text for h in hypotheses):
                    # A conditional cause request can be answered by sourced
                    # evidence that its premise was not observed. Do not force
                    # the writer to invent a fault in a normal snapshot.
                    relevant = [a for a in diagnosis_assessments or [] if c.requirement_id in a["requirement_ids"]]
                    explained = (bool(relevant) and all(not a["symptom_observed"] for a in relevant)
                                 and any(c.requirement_id in a["requirement_ids"] and a["text"] in text for a in answers))
                    if not explained:
                        c.status, c.reason = "unmet", "원인 후보 또는 후보를 제시할 수 없는 이유의 원문 근거 설명이 없습니다."
                if c.status=="satisfied" and r["kind"]=="remediation":
                    from .remediation import manual_requirement_covered
                    visible_steps=[a for a in (manual_result or {}).get("steps",[]) if c.requirement_id in a.get("requirement_ids",[])]
                    visible=bool(visible_steps) and all(a["text"] in text and a["precondition"] in text for a in visible_steps)
                    visible=visible and all(s["title"] in text and s["section"] in text for s in (manual_result or {}).get("sources",[]) if s["chunk_id"] in {a["chunk_id"] for a in visible_steps})
                    dispositions=[d for d in (manual_result or {}).get("no_action",[]) if d.get("requirement_id")==c.requirement_id]
                    visible=visible or bool(dispositions) and all(d["text"] in text for d in dispositions)
                    if not visible or not manual_requirement_covered(r,manual_result or {},evidence):
                        c.status,c.reason="unmet","대처 요구에 연결된 공식 출처·대응 단계가 부족합니다."
                if c.status == "unavailable" and r["status"] not in {"failed", "unsupported"}:
                    raise ValueError("추가 확인이 가능한 요구를 임의로 미지원 처리할 수 없습니다.")
            # Programmatic guards can reject a judge's false positive. Feed the
            # precise deficit back to the writer instead of an empty repair.
            code_rejections = [c for c in review.checks if c.status == "unmet" and c.requirement_id not in model_unmet]
            if code_rejections:
                review.rewrite_instructions += "\n" + "\n".join(c.requirement_id+": "+c.reason for c in code_rejections)
                if any(by_id[c.requirement_id]["kind"] == "diagnosis" for c in code_rejections):
                    review.rewrite_instructions += "\ncause_catalog와 관측된 증상을 대조해 적합한 원인 후보를 causes의 requirement_id/factor_id로 선택하세요. 증상 설명만 반복하지 마세요."
            repairable_unmet = [c for c in review.checks if c.requirement_id in model_unmet and c.status == "unmet"
                and by_id[c.requirement_id]["status"] not in {"failed", "unsupported"}
                and not ledger.is_calculation_blocked(c.requirement_id)
                and c.requirement_id not in specialist_failures]
            if (repairable_unmet and not review.rewrite_instructions.strip()
                    and not review.repair_tasks and not review.missing_requirements):
                raise ValueError("unmet 판정에는 실제 본문에서 빠진 내용과 구체적인 보완 지시가 필요합니다. "
                                 "조건부 요구는 evaluation_focus와 증상 관측 결과를 먼저 대조하세요. "
                                 "이미 답한 내용이면 satisfied, 실제 부족하면 rewrite_instructions에 그 부족을 구체적으로 적으세요.")
        try:
            review = await ask(deps.quality_llm, "final_review", {
                "query": query, "request_decision":intent, "targets": targets, "draft_answer": text,
                # Judge the current rendered answer; historical rejection prose
                # is audit data, not a fresh deficit to repeat after a repair.
                "requirements": [{**{k:r[k] for k in ("requirement_id", "description", "server_ids", "information_ids", "kind", "status", "action_condition")},
                    "evaluation_focus": (
                        "원래 질문의 조사 목적과 미관측 판정이 같은지 먼저 확인. 조건부 질문의 전제가 미관측이면 값과 한계 설명으로 충족할 수 있다. 압박 미확인은 메모리 점유 원인 질문의 생략 근거가 아니며, 별도로 요청한 후보가 없으면 unmet이다."
                        if r["kind"] == "diagnosis" and diagnosis_assessments and
                           any(r["requirement_id"] in a["requirement_ids"] for a in diagnosis_assessments) and
                           all(not a["symptom_observed"] for a in diagnosis_assessments if r["requirement_id"] in a["requirement_ids"])
                        else "관측에 연결된 원인 후보와 확인 한계가 본문에 있는지 판단. 조건부 기여 후보는 허용하며 확정 원인이나 후보 수 증가를 요구하지 않는다. unmet이면 이미 있는 후보와 다른 구체적 누락을 지목한다."
                        if r["kind"] == "diagnosis"
                        else "일반 매뉴얼·예방 절차 요청이다. 해당 OS·주제의 공식 단계, 적용 전 조건, 출처가 본문에 있는지 검토한다. 서버 관측이나 조치 생략 설명은 요구하지 않는다."
                        if r["kind"] == "remediation" and r["action_condition"] == "always"
                        else "조건부 조치 요청이다. 관측 조건이 성립하면 공식 단계·조건·출처를, 성립하지 않으면 원문 관측과 조치 생략 설명을 확인한다. 두 답변을 동시에 요구하지 않는다."
                        if r["kind"] == "remediation" else "원래 요구의 결과가 현재 본문에 실제 있는지 확인")}
                                 for r in ledger.requirements],
                "hypotheses": hypotheses, "diagnosis_assessments":diagnosis_assessments,
                "evidence": [{k:e[k] for k in ("evidence_id", "server_id", "observed_at", "facts", "error", "exit_code")} for e in evidence],
                "calculations": calculations, "repair_attempt": repair,
                "remediation":manual_result, "presentation":presentation,
                "calculation_rejections":deepcopy(ledger.calculation_rejections),
                "requirement_coverage":rendered.get("requirement_coverage",{}), "answer_blocks":rendered.get("answer_blocks",[]),
                "remaining_manual_searches":max(0,2-manual_state.get("search_calls",0)),
                "remaining_query_rounds": max(0, MAX_ROUNDS - rounds),
                "remaining_command_calls": max(0, MAX_COMMAND_CALLS - calls),
            }, AnswerReview, checked_review)
        except ValueError as exc:
            final_reviews.append({"attempt":repair+1,"model":getattr(deps.quality_llm,"name","quality_llm"),
                                  "error":str(exc),"checks":[],"status":"unverified"})
            ledger.note("final_validation", str(exc), list(expected_ids))
            final_validation = {"status":"unverified", "request_fully_satisfied":False,
                                "reason":str(exc), "model":getattr(deps.quality_llm, "name", "quality_llm"),
                                "repair_rounds":repair}
            terminal_note="최종 답변 검증을 완료하지 못했습니다. 요청 충족을 보장하지 않습니다."
            break
        record = review.model_dump()
        record.update(attempt=repair + 1, model=getattr(deps.quality_llm, "name", "quality_llm"))
        final_reviews.append(record)
        ledger.apply_checks(review.checks, f"final_validation_{repair+1}")
        ledger.add(review.missing_requirements, targets, "final_validation")
        for requirement in ledger.requirements:
            if requirement["requirement_id"] not in expected_ids:
                ledger.note("final_validation","원래 요청에서 누락된 요구를 추가했습니다: "+requirement["description"],[requirement["requirement_id"]])
        for c in review.checks:
            if c.status != "satisfied":ledger.note("final_validation", c.reason, [c.requirement_id])
        unmet = [c for c in review.checks if c.status == "unmet"]
        bad_presentation=presentation.get("status")=="fallback"
        complete = not unmet and not review.missing_requirements and not bad_presentation and all(c.status == "satisfied" for c in review.checks)
        final_validation = {"status":"passed" if complete else "partial", "request_fully_satisfied":complete,
                            "model":record["model"], "repair_rounds":repair,
                            "reason":"모든 원래 요구사항의 답변 충족 확인" if complete else "미충족 또는 제공 불가능한 요구가 남았습니다."}
        emit("FinalAnswerValidation", **final_validation, checks=record["checks"])
        emit("RequestRequirements", **ledger.payload())
        if not unmet and not review.missing_requirements and not bad_presentation:break
        if (unmet and all(c.requirement_id in specialist_failures for c in unmet)
                and not review.repair_tasks and not review.missing_requirements and not bad_presentation):
            terminal_note="동일한 관측으로 실패한 전문 판단을 반복하지 않습니다. 확보된 다른 결과는 유지합니다."
            break
        have={(e["server_id"],f["information_id"]) for e in evidence for f in e.get("facts",[])}
        unmet_specs=[r for r in ledger.requirements if r["requirement_id"] in {c.requirement_id for c in unmet}]
        blocked_only = bool(ledger.calculation_rejections and not unmet_specs
                            and not any(r["status"] == "pending" for r in ledger.requirements))
        if blocked_only and not bad_presentation:
            break
        if lookup_disabled and unmet_specs and all(any((sid,i) not in have for sid in r["server_ids"] for i in r["information_ids"]) for r in unmet_specs):
            terminal_note="이전 관측에 없는 정보는 확인하지 못했습니다. 새 조회를 요청하면 해당 항목을 조사할 수 있습니다."
            break
        signature = json.dumps([text, sorted((c.requirement_id,c.status,c.reason) for c in unmet),
                                [e["evidence_id"] for e in evidence], review.model_dump()["missing_requirements"]],sort_keys=True,ensure_ascii=False)
        if signature in seen_review_states and not review.repair_tasks and unmet_specs and all(r["kind"] == "remediation" for r in unmet_specs):
            ledger.note("final_validation", "본문·근거·미충족 사유가 반복되어 추가 보완을 중단했습니다.", [c.requirement_id for c in unmet])
            break
        seen_review_states.add(signature)
        if repair >= MAX_ANSWER_REPAIRS:
            terminal_note="최종 답변 보완은 최대 2회까지 수행했습니다. 남은 요구는 미충족으로 남깁니다."
            break
        feedback = {"checks":record["checks"], "rewrite_instructions":review.rewrite_instructions,
                    "missing_requirements":record["missing_requirements"],
                    "repair_requirement_ids":sorted({c.requirement_id for c in unmet}|{r["requirement_id"] for r in ledger.requirements if r["requirement_id"] not in expected_ids}),
                    "presentation_only":bad_presentation and not unmet and not review.missing_requirements,
                    "manual_only":bool(unmet) and all(next(r for r in ledger.requirements if r["requirement_id"]==c.requirement_id)["kind"]=="remediation" for c in unmet)}
        next_tasks = list(review.repair_tasks)
        # Every unmet recorded requirement is reconsidered, even if the small model omitted repair_tasks.
        missing_ids = {c.requirement_id for c in unmet}
        for r in ledger.requirements:
            if r["requirement_id"] in missing_ids or r["status"] == "pending":
                if r["kind"]=="remediation":continue
                for sid in r["server_ids"]:
                    if r["information_ids"]:
                        next_tasks.append(TaskSpec(server_id=sid, goal=r["description"][:200], information_ids=r["information_ids"]))
        if next_tasks:
            try:
                await collect(TaskPlan(tasks=next_tasks[:12]))
            except ValueError as exc:
                ledger.note("answer_repair", str(exc))
                limitations.append(str(exc))
        emit("AnswerRepair", attempt=repair+1, reason=review.rewrite_instructions or "미충족 요구에 필요한 조회·계산·설명 보완")
    emit("InvestigationTasks", tasks=tasks)
    emit("RequestRequirements", **ledger.payload())
    if terminal_note:
        # Post-validation control information belongs to the existing status/limitations
        # fields. Never mutate the body the final validator actually reviewed.
        if terminal_note not in limitations:limitations.append(terminal_note)
    return finish(text)
