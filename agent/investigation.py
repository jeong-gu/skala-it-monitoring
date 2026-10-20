"""Information-first investigations. Models choose tasks; code resolves commands."""
import asyncio
import json
import re
from decimal import Decimal

from langgraph.config import get_stream_writer
from pydantic import ValidationError

from monitoring.api import CommandResult
from monitoring.evidence import extract_facts
from .investigation_models import AnswerReview, InvestigationReport, TaskPlan, TaskSpec
from .request_ledger import RequestLedger, RequirementSpec
from .answer_calculations import calculate

MAX_ROUNDS = 3
MAX_COMMAND_CALLS = 12
MAX_ANSWER_REPAIRS = 2
SERVER = re.compile(r"(?:server[- ]([a-z0-9]+)|서버\s*([a-z])(?![a-z0-9_-]))", re.I)
MONITORING_WORDS = re.compile(r"서버|cpu|메모리|memory|프로세스|디스크|네트워크|소켓|로그|운영체제|커널|swap", re.I)


def mentioned_servers(text):
    found = ["server-" + (m[1] or m[2]).lower() for m in SERVER.finditer(text)]
    # Also accept '서버 A와 B', without scanning unrelated letters elsewhere.
    shorthand = re.search(r"서버\s*[a-z]((?:\s*(?:와|과|및|,|하고)\s*[a-z])+)", text, re.I)
    if shorthand:
        found += ["server-" + c.lower() for c in re.findall(r"[a-z]", shorthand[1], re.I)]
    return list(dict.fromkeys(found))


def monitoring_request(query):
    return bool(MONITORING_WORDS.search(query) or mentioned_servers(query))


def resolve_targets(query, history, servers):
    targets = mentioned_servers(query)
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


def verify_report(report, evidence, calculated_numbers=()):
    by_id = {e["evidence_id"]: e for e in evidence}
    for hypothesis in report.hypotheses:
        if not set(hypothesis.evidence_ids) <= by_id.keys():
            raise ValueError("Unknown evidence ID")
        selected = [by_id[i] for i in hypothesis.evidence_ids]
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
        log_text = " ".join(e["stdout"] for e in selected if e["command_id"] == "tail")
        if re.search(r"누수|배포.*원인|패킷.*손실|응답\s*시간|증가|감소|급증|추세", hypothesis.text):
            raise ValueError("현재 표본뿐이므로 증가·감소·추세·누수·응답 시간은 확인할 수 없습니다. 현재 관측에만 근거하세요.")
        if re.search(r"네트워크.*(?:문제|장애|병목|정상|지연)", hypothesis.text) and not re.search(r"ERROR.*(?:connect|network|timeout)", log_text, re.I):
            raise ValueError("Sockets do not establish a network fault")
        if re.search(r"메모리\s*(?:부족|압박)", hypothesis.text):
            pressure = any((f["information_id"] == "memory.summary" and
                            (f["fields"].get("available_mib", 1e9) < max(1024, f["fields"].get("total_mib", 0) * .1) or
                             f["fields"].get("unused_mib", 1e9) <= 1024)) or
                           (f["information_id"] == "memory.detail" and f["fields"].get("swap_used_mib", 0) > 0) for f in facts)
            if not pressure and not re.search(r"allocation failed|memory pressure", log_text, re.I):
                raise ValueError("Memory pressure lacks referenced evidence")
        if re.search(r"디스크.*(?:부족|높|포화)", hypothesis.text):
            pressure = any(f["information_id"] == "disk" and any(p["usage_percent"] >= 90 for p in f["fields"]["entries"]) for f in facts)
            if not pressure and "No space left" not in log_text:
                raise ValueError("Disk pressure lacks referenced evidence")
        if re.search(r"원인(?:이다|입니다)|입증|증명|문제\s*없|이상\s*없|오류\s*없|모두\s*차지|전부\s*차지|완전히|가득", hypothesis.text):
            raise ValueError("확정·과장 표현입니다. 모두 차지, 가득, 완전히, 오류 없음 등의 표현을 제거하고 관측된 부분 사용량의 기여 가능성만 기술하세요.")


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


async def investigate(state, deps):
    query, history = state["query"], state.get("history", [])
    planner = getattr(deps, "monitoring_llm", None) or deps.fast_llm
    events = list(state.get("events", []))
    evidence, tasks, limitations, hypotheses = [], [], [], []
    ledger = RequestLedger()
    answers, calculations, final_reviews = [], [], []
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
        writer(event)

    def finish(text):
        emit("Finalize", pipeline="information-investigation")
        return {"final": text, "tasks": tasks, "evidence": evidence, "hypotheses": hypotheses,
                "limitations": list(dict.fromkeys(limitations)), "events": events, "cost_usd": round(cost, 6),
                "replan_rounds": max(0, rounds - 1), "command_call_count": calls,
                "last_prompt": prompt, **ledger.payload(), "answers": answers,
                "calculation_results": calculations, "final_reviews": final_reviews,
                "final_validation": final_validation, "validation_attempts": len(final_reviews)}

    async def ask(model, purpose, payload, contract, check=None):
        nonlocal cost, prompt
        payload = dict(payload)
        schema = contract.model_json_schema()
        if contract is TaskPlan:
            spec = schema["$defs"]["TaskSpec"]["properties"]
            spec["server_id"]["enum"] = payload["targets"]
            spec["information_ids"]["items"]["enum"] = [i["information_id"] for i in payload["information"]] + [
                "history", "deployment", "network.performance"]
        if contract is AnswerReview:
            # The small model judges answer adequacy only; capability state belongs to code.
            schema["$defs"]["RequirementCheck"]["properties"]["status"]["enum"] = ["satisfied", "unmet"]
        for attempt in range(2):
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
                ledger.note(purpose, str(exc)[:500])
            except Exception:
                # Do not leak provider credentials or request headers in errors.
                raise ValueError(f"{purpose} 모델 호출 실패") from None
        raise ValueError(f"{purpose} 응답이 두 번 모두 검증을 통과하지 못했습니다.")

    try:
        servers = await deps.mcp.call_tool("list_command_servers", {})
        definitions = await deps.mcp.call_tool("list_server_information", {})
        targets, clarification = resolve_targets(query, history, servers)
        if clarification:
            return finish(clarification)
        server_map = {s["server_id"]: s for s in servers}
        information = {i["information_id"]: i for i in definitions}
        catalogs = {sid: await deps.mcp.call_tool("list_server_commands", {"server_id": sid}) for sid in targets}
    except ValueError as exc:
        return finish(f"서버 정보 목록을 조회하지 못했습니다: {exc}")

    emit("RequirementsIdentified", targets=targets, query=query)
    base = {"query": query, "targets": targets,
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
    if explicit:
        specs = []
        for sid in targets:
            selected = [c for c in catalogs[sid] if c["command_id"] in explicit]
            specs.append({"server_id": sid, "goal": "지정한 명령의 정보 확인",
                          "information_ids": list(dict.fromkeys(i for c in selected for i in c["information_ids"]))})
        plan = TaskPlan.model_validate({"tasks": specs})
    else:
        def initial_check(plan):
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

    def add_tasks(plan):
        # Investigation steps support the original request; only the initial plan and
        # final query comparison may register user requirements.
        for spec in plan.tasks:
            if spec.server_id not in targets:
                raise ValueError("요청하지 않은 서버의 task")
            if not all(re.fullmatch(r"[a-z][a-z0-9_.-]{0,79}", i) for i in spec.information_ids):
                raise ValueError("잘못된 정보 ID")
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

    attempted = set()

    async def collect(plan):
        nonlocal calls, rounds
        add_tasks(plan)
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
        for (sid, cid), command in jobs.items():
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
                raw = await deps.mcp.call_tool("get_command_result", {"server_id": sid, "command": cid})
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

    try:
        for _ in range(MAX_ROUNDS):
            if not await collect(plan):break
            if calls >= MAX_COMMAND_CALLS or rounds >= MAX_ROUNDS:
                break
            if explicit and not re.search(r"원인|분석|왜|진단", query):break
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
        results = [{k: v for k, v in e.items() if k not in {"facts", "evidence_id"} and (k != "error" or v)} for e in evidence]
        return finish(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2))
    def writer_payload(feedback):
        return {"query": query, "targets": targets, "tasks": tasks, **ledger.payload(),
                "final_feedback": feedback,
                "evidence": [{"evidence_id": e["evidence_id"], "server_id": e["server_id"],
                              "command_id": e["command_id"], "observed_at": e["observed_at"],
                              "facts": e["facts"], "error": e["error"], "exit_code": e["exit_code"],
                              "quote_candidates": [line for line in e["stdout"].splitlines() if line.strip()]} for e in evidence]}

    async def compose(feedback):
        nonlocal hypotheses, answers, calculations
        hypotheses, answers, calculations = [], [], []
        if not any(e["facts"] for e in evidence):return
        def checked_report(report):
            nonlocal calculations
            ids = [r["requirement_id"] for r in ledger.requirements]
            computed = [calculate(c, evidence, ids) for c in report.calculations]
            for a in report.answers:
                if not set(a.requirement_ids) <= set(ids):raise ValueError("알 수 없는 답변 요구사항 ID")
                # Exact sources/numbers are checked here; semantic scope is checked by final_review.
                verified_numbers = [c["value"] for c in computed
                                    if set(c["evidence_ids"]) <= set(a.evidence_ids)
                                    and set(c["requirement_ids"]) & set(a.requirement_ids)]
                verify_report(InvestigationReport(hypotheses=[a.model_dump(exclude={"requirement_ids"})]),
                              evidence, calculated_numbers=verified_numbers)
            valid, rejected = [], 0
            for item in report.hypotheses:
                try:
                    verify_report(InvestigationReport(hypotheses=[item]), evidence)
                    valid.append(item)
                except ValueError:rejected += 1
            if report.hypotheses and not valid:verify_report(report, evidence)
            report.hypotheses = valid
            calculations = computed
            if rejected:limitations.append(f"근거 검증을 통과하지 못한 원인 후보 {rejected}개를 제외했습니다.")
        try:
            report = await ask(deps.quality_llm, "report", writer_payload(feedback), InvestigationReport, checked_report)
            hypotheses = [h.model_dump() for h in report.hypotheses]
            answers = [a.model_dump() for a in report.answers]
        except ValueError as exc:
            calculations = []
            ledger.note("answer_generation", str(exc))
            limitations.append(f"원인 후보 해석 검증 실패: {exc}. 관측 사실만 제공합니다.")

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
        return render_answer(tasks, evidence, hypotheses, limitations, answers, calculations, ledger.requirements if include_requirement_notes else ())

    feedback = None
    for repair in range(MAX_ANSWER_REPAIRS + 1):
        await compose(feedback)
        text = draft(include_requirement_notes=False)
        expected_ids = {r["requirement_id"] for r in ledger.requirements}
        def checked_review(review):
            ids = [c.requirement_id for c in review.checks]
            if set(ids) != expected_ids or len(ids) != len(set(ids)):
                raise ValueError("모든 원래 요구사항을 중복·누락 없이 검증해야 합니다.")
            if any(t.server_id not in targets for t in review.repair_tasks):
                raise ValueError("검증 후 추가 조회 대상이 원래 대상과 다릅니다.")
            if len(ledger.requirements) + len(review.missing_requirements) > 32:
                raise ValueError("요구사항 기록 한도에 도달했습니다.")
            if any(not set(r.server_ids) <= set(targets) for r in review.missing_requirements):
                raise ValueError("추가 요구사항 대상이 원래 대상과 다릅니다.")
            by_id = {r["requirement_id"]: r for r in ledger.requirements}
            for c in review.checks:
                r = by_id[c.requirement_id]
                if c.status == "satisfied" and r["status"] in {"failed", "unsupported"}:
                    c.status = "unavailable"
                elif c.status == "satisfied" and r["information_ids"] and r["status"] in {"needs_more", "pending"}:
                    raise ValueError("확보하지 못한 필수 정보는 충족으로 처리할 수 없습니다.")
                if c.status == "satisfied" and r["kind"] == "calculation" and not any(c.requirement_id in x["requirement_ids"] for x in calculations):
                    c.status, c.reason = "unmet", "요청한 계산 결과·계산식·원문 출처가 답변에 없습니다."
                if c.status == "satisfied" and r["kind"] == "diagnosis" and not hypotheses:
                    c.status, c.reason = "unmet", "원인 후보를 요청했지만 검증된 후보를 제시하지 못했습니다."
                if c.status == "unavailable" and r["status"] not in {"failed", "unsupported"}:
                    raise ValueError("추가 확인이 가능한 요구를 임의로 미지원 처리할 수 없습니다.")
        try:
            review = await ask(deps.fast_llm, "final_review", {
                "query": query, "targets": targets, "draft_answer": text,
                **ledger.payload(), "evidence": [{k:e[k] for k in ("evidence_id", "server_id", "observed_at", "facts", "error", "exit_code")} for e in evidence],
                "calculations": calculations, "repair_attempt": repair,
                "remaining_query_rounds": max(0, MAX_ROUNDS - rounds),
                "remaining_command_calls": max(0, MAX_COMMAND_CALLS - calls),
            }, AnswerReview, checked_review)
        except ValueError as exc:
            final_reviews.append({"attempt":repair+1,"model":getattr(deps.fast_llm,"name","fast_llm"),
                                  "error":str(exc),"checks":[],"status":"unverified"})
            ledger.note("final_validation", str(exc), list(expected_ids))
            final_validation = {"status":"unverified", "request_fully_satisfied":False,
                                "reason":str(exc), "model":getattr(deps.fast_llm, "name", "fast_llm"),
                                "repair_rounds":repair}
            limitations.append("최종 답변 검증을 완료하지 못했습니다. 요청 충족을 보장하지 않습니다.")
            break
        record = review.model_dump()
        record.update(attempt=repair + 1, model=getattr(deps.fast_llm, "name", "fast_llm"))
        final_reviews.append(record)
        ledger.apply_checks(review.checks, f"final_validation_{repair+1}")
        ledger.add(review.missing_requirements, targets, "final_validation")
        for c in review.checks:
            if c.status != "satisfied":ledger.note("final_validation", c.reason, [c.requirement_id])
        unmet = [c for c in review.checks if c.status == "unmet"]
        complete = not unmet and not review.missing_requirements and all(c.status == "satisfied" for c in review.checks)
        final_validation = {"status":"passed" if complete else "partial", "request_fully_satisfied":complete,
                            "model":record["model"], "repair_rounds":repair,
                            "reason":"모든 원래 요구사항의 답변 충족 확인" if complete else "미충족 또는 제공 불가능한 요구가 남았습니다."}
        emit("FinalAnswerValidation", **final_validation, checks=record["checks"])
        emit("RequestRequirements", **ledger.payload())
        if not unmet and not review.missing_requirements:break
        if repair >= MAX_ANSWER_REPAIRS:
            limitations.append("최종 답변 보완은 최대 2회까지 수행했습니다. 남은 요구는 미충족으로 남깁니다.")
            break
        feedback = {"checks":record["checks"], "rewrite_instructions":review.rewrite_instructions,
                    "missing_requirements":record["missing_requirements"]}
        next_tasks = list(review.repair_tasks)
        # Every unmet recorded requirement is reconsidered, even if the small model omitted repair_tasks.
        missing_ids = {c.requirement_id for c in unmet}
        for r in ledger.requirements:
            if r["requirement_id"] in missing_ids or r["status"] == "pending":
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
    return finish(draft())
