"""RAG route added alongside, rather than rewriting, the investigation pipeline."""
import re
import json
import hashlib
from copy import deepcopy
from typing import TypedDict
from langgraph.graph import StateGraph, END

from langgraph.config import get_stream_writer

from .investigation import mentioned_servers, resolve_targets
from remediation.actions import candidates
from remediation.model import summarize
from .mcp_client import call_tool_result
from .deps import save_progress


MANUAL_INFORMATION = {"cpu", "memory.summary", "memory.detail", "disk", "log.usage", "log.files"}


def manual_scope_supported(requirement: dict) -> bool:
    ids=set(requirement.get("information_ids",[]))
    return bool(ids) and ids<=MANUAL_INFORMATION


def manual_topic_query(requirements):
    """Search the already typed subject, not opaque IDs or a missing pronoun.

    These topics come from the model's requirement contract, never a keyword
    classifier. Preserve the user's clause for context without treating it as fact.
    """
    subjects={"cpu-spike":"high CPU utilization", "memory-pressure":"memory pressure and RAM usage",
              "disk-full":"disk space exhaustion"}
    topics=sorted({subjects[t] for r in requirements for t in requirement_types(r)})
    descriptions=list(dict.fromkeys(r["description"] for r in requirements))
    return "Remediation for " + "; ".join(topics) + ". Request: " + "; ".join(descriptions)


def requirement_types(requirement):
    ids=set(requirement.get("information_ids",[]));types=set()
    if "cpu" in ids:types.add("cpu-spike")
    if ids & {"memory.summary","memory.detail"}:types.add("memory-pressure")
    if ids & {"disk","log.usage","log.files"}:types.add("disk-full")
    return types


def bind_no_action(items, requirements, evidence):
    """Bind a model's conditional no-action choice to reparsed observations.

    These conservative rejection guards are not universal incident thresholds or
    proof of health. A separate final judge still checks the original condition.
    """
    from monitoring.evidence import extract_facts
    known={r["requirement_id"]:r for r in requirements}
    refs={e["evidence_id"]:e for e in evidence}
    result=[];seen=set()
    for item in items:
        rid=item["requirement_id"];r=known.get(rid)
        if rid in seen or not r or r["kind"]!="remediation" or not r["server_ids"] or r.get("action_condition","always")!="if_needed":
            raise ValueError("조치 생략에는 대상이 있는 실제 대처 요구가 필요합니다.")
        seen.add(rid);ids=set(r["information_ids"])
        if not ids or not ids <= {"cpu","memory.summary","disk"}:
            raise ValueError("해당 정보만으로 조치 불필요를 확인할 수 없습니다.")
        selected=[refs.get(eid) for eid in item["evidence_ids"]]
        if not selected or any(e is None or e.get("error") or e.get("exit_code")!=0 for e in selected):
            raise ValueError("조치 생략의 관측이 없거나 조회에 실패했습니다.")
        if {e["server_id"] for e in selected}!=set(r["server_ids"]):
            raise ValueError("조치 생략 근거의 대상이 다릅니다.")
        # Check all current observations for contradictions, not only the rows
        # the model preferred. Cached facts can never replace the raw output.
        facts=[(e,f) for e in evidence if e["server_id"] in r["server_ids"] and not e.get("error") and e.get("exit_code")==0
               for f in extract_facts(e) if f["information_id"] in ids]
        bound=[(e,f) for e in selected for f in extract_facts(e) if f["information_id"] in ids]
        if {(e["server_id"],f["information_id"]) for e,f in bound} != {(s,i) for s in r["server_ids"] for i in ids}:
            raise ValueError("조치 생략에 필요한 원본 지표가 없습니다.")
        for e,f in facts:
            v=f["fields"];iid=f["information_id"]
            if iid=="cpu" and v["usage_percent"]>50:
                raise ValueError("대부분 유휴인 CPU 표본이 아니므로 조치 불필요로 확정하지 마세요.")
            if iid=="disk" and any(row["usage_percent"]>=90 for row in v["entries"]):
                raise ValueError("공간 압박 표본이 있으므로 디스크 조치를 생략할 수 없습니다.")
            if iid=="memory.summary":
                if e["os"] == "macos":
                    raise ValueError("macOS 미사용 메모리는 가용 메모리나 메모리 압력 지표가 아니므로 조치 생략의 근거가 될 수 없습니다.")
                remaining=v.get("available_mib")
                if remaining is None or remaining<=max(1024,v.get("total_mib",0)*.1):
                    raise ValueError("메모리 여유 근거가 충분하지 않아 조치를 생략할 수 없습니다.")
        if "memory.summary" in ids:
            details=[f for e in evidence if e["server_id"] in r["server_ids"] and not e.get("error") and e.get("exit_code")==0
                     for f in extract_facts(e) if f["information_id"]=="memory.detail"]
            if any(f["fields"].get("swap_used_mib",0)>0 for f in details):
                raise ValueError("Swap 관측이 있으므로 메모리 조치 생략을 보수적으로 제외합니다.")
        summaries=list(dict.fromkeys(f["summary"] for e,f in bound))
        result.append({"requirement_id":rid,"requirement_ids":[rid],"evidence_ids":list(dict.fromkeys(item["evidence_ids"])),
            "text":" ".join(summaries)+" 현재 관측 기준으로 해당 항목의 즉시 조치를 권하지 않습니다. 단일 표본이므로 지속 상태나 서비스 정상 여부를 보장하지 않습니다."})
    return result


def manual_requirement_covered(requirement, manual, evidence):
    """Evidence/source presence is enforced by code, not the final judge."""
    if not manual_scope_supported(requirement) or manual.get("status")!="provided":return False
    dispositions=[d for d in manual.get("no_action",[]) if d.get("requirement_id")==requirement["requirement_id"]]
    if dispositions:
        try:
            rebound=bind_no_action([{k:d[k] for k in ("requirement_id","evidence_ids")} for d in dispositions],[requirement],evidence)
            return rebound==dispositions and not any(requirement["requirement_id"] in s.get("requirement_ids",[]) for s in manual.get("steps",[]))
        except (ValueError,KeyError,TypeError):return False
    from remediation.index import ROOT
    try:corpus={c["chunk_id"]:c for c in json.loads((ROOT/"data/manuals/chunks.json").read_text())}
    except (OSError,ValueError):return False
    sources={s["chunk_id"]:s for s in manual.get("sources",[])}
    refs={e["evidence_id"]:e for e in evidence}
    steps=[a for a in manual.get("steps",[]) if requirement["requirement_id"] in a.get("requirement_ids",[])]
    if not steps:return False
    types=set()
    for step in steps:
        chunk=corpus.get(step.get("chunk_id"));source=sources.get(step.get("chunk_id"))
        if not chunk or not source or not step.get("precondition") or not step.get("quote"):return False
        canonical=next((a for a in candidates([chunk]) if a["action_id"]==step.get("action_id")),None)
        if not canonical or any(step.get(k)!=canonical.get(k) for k in ("text","precondition","quote","risk","incident_types","doc_id","chunk_id")):return False
        if step["quote"] not in chunk["text"] or source["sha256"]!=chunk["sha256"] or source["os"]!=manual.get("os"):return False
        if source["doc_id"]!=step.get("doc_id") or chunk["os"]!=source["os"]:return False
        if requirement["server_ids"] and manual.get("server_id") not in requirement["server_ids"]:return False
        for eid in step.get("evidence_ids",[]):
            if eid not in refs or refs[eid]["server_id"] not in requirement["server_ids"] or not refs[eid].get("facts"):return False
        types.update(step["incident_types"])
    return requirement_types(requirement)<=types


async def remediation_workflow(state,deps):
    from .investigation import investigate
    return await investigate({**state,"workflow_manual":True},deps)


def wants_remediation(query):
    if re.search(r"json\s*만|원문\s*만", query, re.I): return False
    if re.search(r"(?:대처|조치|대응|해결)(?:방안|방법|책|법)?\s*(?:말고|제외|없이)|(?:대처|조치|대응|해결).*?(?:하지\s*마|필요\s*없|알려주지\s*마)",query): return False
    return bool(re.search(r"대처|조치|대응|해결|어떻게\s*(?:하|해|해야|처리|낮|줄)|낮추|줄이|줄여|remediat|mitigat|how\s+to\s+(?:fix|resolve|reduce)", query, re.I))


def specified_os(query):
    found=[]
    if re.search(r"linux|리눅스|ubuntu|rhel|red\s*hat",query,re.I): found.append("linux")
    if re.search(r"macos|mac\b|맥(?:북|에서|의|\s|$)|darwin",query,re.I): found.append("macos")
    return found[0] if len(found)==1 else None


class ManualAgentState(TypedDict, total=False):
    query: str
    requirements: list[dict]
    matches: list[dict]
    selected: list[dict]
    metrics: dict
    timings_ms: dict
    rounds: int
    pending_query: str
    status: str
    error: str


def build_manual_agent(retrieve, select, checkpoint, emit):
    """A scoped manual specialist with local retrieval/selection/coverage state.

    The supervisor injects only manual retrieval, not server command tools.
    The retrieval callback owns the request-wide budget and source integrity.
    """
    async def retrieve_node(local):
        try:
            found = await retrieve(local.get("pending_query") or local["query"])
            return {**found, "rounds":local.get("rounds",0)+1, "status":"retrieved", "error":""}
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return {"rounds":local.get("rounds",0)+1, "status":"partial" if local.get("selected") else "unavailable", "error":str(exc)}

    async def assess(local):
        try:
            selected, metrics = await select(deepcopy(local["matches"]))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            # Missing source topics need the remaining retrieval opportunity even
            # when selection cannot form a valid answer from the first hits.
            available={t for m in local["matches"] for t in m.get("incident_types",[])}
            missing=[r for r in local["requirements"] if requirement_types(r)-available]
            if missing:
                return {"status":"needs_information", "pending_query":manual_topic_query(missing), "error":str(exc)}
            return {"status":"partial" if local.get("selected") else "unavailable", "error":str(exc)}
        available = {t for m in local["matches"] for t in m.get("incident_types",[])}
        skipped={d["requirement_id"] for d in metrics.get("no_action",[])}
        missing = [r for r in local["requirements"] if r["requirement_id"] not in skipped and requirement_types(r)-available]
        pending = manual_topic_query(missing) if missing else ""
        delta = {"selected":selected, "metrics":metrics, "pending_query":pending,
                 "status":"needs_information" if pending else ("assessed" if selected or skipped else "not_found")}
        if selected or skipped:checkpoint({**local, **delta})
        emit("SpecialistAssessed", role="manual", chunks=len(local["matches"]), steps=len(selected),
             missing_requirement_ids=[r["requirement_id"] for r in missing])
        return delta

    graph = StateGraph(ManualAgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("assess", assess)
    graph.set_entry_point("retrieve")
    graph.add_conditional_edges("retrieve", lambda s: "assess" if s["status"]=="retrieved" else END,
                                {"assess":"assess", END:END})
    graph.add_conditional_edges("assess", lambda s: "retrieve" if s["status"]=="needs_information" and s["rounds"]<2 else END,
                                {"retrieve":"retrieve", END:END})
    return graph.compile(name="manual_remediation_agent")


def render_manual_selection(selected, matches, requirements, evidence, target, os_name, observations, integrated, unsupported, no_action=()):
    selected = deepcopy(selected)
    by_id={m["chunk_id"]:m for m in matches}
    if integrated:
        for step in selected:
            bound=[r for r in requirements if r["requirement_id"] in step.get("requirement_ids",[])]
            if any(requirement_types(r) and not requirement_types(r)&set(step["incident_types"]) for r in bound):
                raise ValueError("대처 단계의 주제와 원래 요구가 일치하지 않아 제외했습니다.")
            wanted={i for r in bound for i in r["information_ids"]}
            step["evidence_ids"]=[e["evidence_id"] for e in evidence if e.get("facts") and e["server_id"]==target and
                                  (not wanted or any(f["information_id"] in wanted for f in e["facts"]))]
            step["conditional"]=bool(target and not step["evidence_ids"])
    sources=[]
    for cid in dict.fromkeys(a["chunk_id"] for a in selected):
        m=by_id[cid]
        sources.append({k:m[k] for k in ("chunk_id","doc_id","title","section","source_url","sha256","collected_at","os","product","version")})
    unconfirmed=["문서의 권장 절차이며 실제 조치를 실행하거나 복구 완료를 확인한 것은 아닙니다."]
    unconfirmed += ["미지원 또는 미분류 대처 요구: "+r["description"]+". 해당 주제의 대응 절차는 제공하지 못했습니다." for r in unsupported]
    if observations: unconfirmed.append("서버 정보는 저장된 시나리오 또는 같은 세션의 이전 조회 결과 기준입니다. 조치 전 실제 상태를 다시 확인해야 합니다.")
    else: unconfirmed.append("현재 서버 상태를 조회하지 않았습니다. 요청에서 설명한 증상을 전제로 한 일반적인 매뉴얼 안내입니다.")
    text=("확인한 서버 정보 — 이전 조회 근거\n"+observations+"\n\n") if observations and not integrated else ""
    text+="대처방안 요약 — 공식 매뉴얼 기준\n"
    for disposition in no_action:
        text+="- "+disposition["text"]+" (관측 근거: "+", ".join(disposition["evidence_ids"])+")\n"
    labels={"cpu-spike":"CPU", "memory-pressure":"메모리", "disk-full":"디스크"}
    for i, step in enumerate(selected,1):
        source=next(s for s in sources if s["chunk_id"]==step["chunk_id"])
        topics="·".join(labels[t] for t in step["incident_types"] if t in labels)
        text+=f"{i}) [{topics} · 조건부 확인/대응] {step['text']} 적용 전: {step['precondition']} 출처: {source['title']} · {source['section']}\n"
    text+="\n\n조치 전 확인\n"+"\n".join("- "+a["precondition"] for a in selected)
    text+="\n\n미확인 사항\n"+"\n".join("- "+x for x in unconfirmed)
    text+="\n\n출처\n"+"\n".join("- "+s["title"]+" · "+s["section"] for s in sources)
    if integrated and target and not any(e.get("facts") for e in evidence):
        text="현재 서버 관측을 확보하지 못했습니다. 아래 내용은 요청한 증상을 전제로 한 조건부 매뉴얼 안내입니다.\n\n"+text
    return text, dict(steps=selected, sources=sources, no_action=list(no_action), unconfirmed=unconfirmed,
        os=os_name, server_id=target, incident_types=sorted({t for a in selected for t in a["incident_types"]}))


async def remediation_node(state, deps):
    query, history = state["query"], state.get("history", [])
    events = list(state.get("events", []))
    integrated=state.get("workflow_active",False)
    cache=deepcopy(state.get("manual_state") or {})
    try: writer=get_stream_writer()
    except RuntimeError: writer=lambda _:None
    result={"tasks":[],"evidence":[],"hypotheses":[],"limitations":[],"command_call_count":0,
            "cost_usd":state.get("cost_usd",0), "remediation":None}
    if integrated:
        result.update({k:deepcopy(state.get(k,[])) for k in ("tasks","evidence","hypotheses","limitations")})
        result["command_call_count"]=state.get("command_call_count",0)

    def emit(kind, **data):
        event={"type":kind,"event_id":f"remediation-{len(events)}",**data}
        events.append(event)
        save_progress({**result, "events":events, "manual_state":cache})
        writer(event)

    unsupported=[]
    def finish(text, status, **data):
        result["remediation"]={"status":status,"steps":[],"sources":[],"unconfirmed":[],
                               "unsupported_requirements":[r["requirement_id"] for r in unsupported],**data}
        if integrated:
            result["remediation"].update(search_calls=cache.get("search_calls",0),model_calls=cache.get("model_calls",0))
        save_progress({**result, "_verified_manual_text":text if status == "provided" else "", "events":events})
        emit("Finalize",pipeline="manual-remediation")
        return {**result,"final":text,"events":events,**({"manual_state":cache} if integrated else {})}

    target, os_name, context = None, specified_os(query), state.get("investigation_context")
    has_target = bool(mentioned_servers(query) or re.search(r"그\s*서버|해당\s*서버|거기|이어서",query))
    if integrated:
        target=next(iter(state.get("targets",[])),None);os_name=state.get("manual_os")
    elif has_target:
        try:
            servers = await call_tool_result(deps.mcp, "list_command_servers",{})
            targets, clarification = resolve_targets(query,history,servers)
        except ValueError:
            return finish("서버 목록을 확인하지 못해 대처방안을 정하지 못했습니다.","unavailable")
        if clarification: return finish(clarification,"needs_clarification")
        if len(targets)!=1: return finish("대처방안을 확인할 서버 하나를 지정해주세요.","needs_clarification")
        target=targets[0]
        actual_os=next(s["os"] for s in servers if s["server_id"]==target)
        if os_name and os_name!=actual_os:
            return finish(f"{target}의 등록 OS는 {actual_os}입니다. 요청한 OS와 달라 먼저 대상 환경을 확인해주세요.","needs_clarification")
        os_name=actual_os
        cached_evidence=[e for e in (context or {}).get("evidence",[]) if e["server_id"]==target]
        if cached_evidence:
            result.update({k:context.get(k,[]) for k in ("tasks","hypotheses","limitations")})
            result["tasks"]=[t for t in result["tasks"] if t["server_id"]==target]
            result["evidence"]=cached_evidence
            emit("RemediationContextReused",server_id=target)
    if not os_name:
        return finish("Linux 또는 macOS 중 어떤 환경의 대처방안이 필요한지 알려주세요. 특정 서버라면 서버 이름을 지정해주세요.","needs_clarification")

    observations="\n".join(f"{e['server_id']} · {e.get('observed_at')}: {f['summary']}" for e in result.get("evidence",[]) for f in e.get("facts",[]))[:3500]
    summary_query=state.get("request_subject") or query
    search_query=summary_query
    # Only supply the old symptom for a short, purely referential follow-up.
    if target and not re.search(r"cpu|메모리|memory|디스크|disk|저장|프로세스|네트워크",search_query,re.I):
        prior=(context or {}).get("query","") or next((m["content"] for m in reversed(history) if m.get("role")=="user" and target in mentioned_servers(m["content"])),"")
        if prior: search_query += "\n직전 요청(관측 사실 아님): " + prior[:500]
    requirements=[{**{k:r[k] for k in ("requirement_id","description","server_ids","information_ids","kind")},"action_condition":r.get("action_condition","always")} for r in state.get("requirements",[]) if r["kind"]=="remediation"] if integrated else []
    if integrated:
        unsupported=[r for r in requirements if not manual_scope_supported(r)]
        requirements=[r for r in requirements if manual_scope_supported(r)]
        if not requirements:
            return finish("요청한 대처 주제가 미지원이거나 분명하지 않습니다. 공식 매뉴얼 지원 범위는 CPU·메모리·디스크입니다.","not_found",os=os_name,server_id=target)
    if integrated:search_query=manual_topic_query(requirements)
    if integrated and state.get("manual_feedback"):
        known_ids=set(cache.get("requirement_ids",[]))
        needed_types=set().union(*(requirement_types(r) for r in requirements)) if requirements else set()
        cached_types={t for m in cache.get("matches",[]) for t in m.get("incident_types",[])}
        focus=[r["description"] for r in requirements if (requirement_types(r)-cached_types) or r["requirement_id"] not in known_ids]
        if focus and needed_types-cached_types:search_query=manual_topic_query([r for r in requirements if r["description"] in focus])
    if integrated:
        # A shared top response can include memory even for a CPU-only request.
        # Keep those facts in the answer, but do not let them dilute retrieval.
        requested_ids={i for r in requirements for i in r["information_ids"]}
        search_observations="\n".join(f["summary"] for e in result.get("evidence",[])
            for f in e.get("facts",[]) if f["information_id"] in requested_ids)[:3500]
    else:search_observations=observations
    async def retrieve(focused_query):
        key=os_name+":"+focused_query
        if cache.get("terminal_error"):raise ValueError(cache["terminal_error"])
        if key in cache.get("queries",[]) or cache.get("search_calls",0)>=2:
            emit("ManualSearchReused",search_calls=cache.get("search_calls",0))
            return {"matches":deepcopy(cache.get("matches",[])),"timings_ms":cache.get("timings_ms",{})}
        cache["search_calls"]=cache.get("search_calls",0)+1
        cache.setdefault("queries",[]).append(key)
        emit("ManualSearchStarted",os=os_name)
        arguments={"query":focused_query,"os":os_name,"evidence_summary":search_observations}
        if integrated:arguments["allow_mixed_topics"]=True
        try:
            found=await call_tool_result(deps.mcp,"search_remediation_manuals",arguments)
            matches=found["matches"]
            if any(m["os"]!=os_name for m in matches):raise ValueError("매뉴얼 OS 불일치")
            if integrated:
                from remediation.index import ROOT
                originals={c["chunk_id"]:c for c in json.loads((ROOT/"data/manuals/chunks.json").read_text())}
                for m in matches:
                    original=originals.get(m["chunk_id"])
                    if not original or any(original[k]!=m[k] for k in ("sha256","text","os","doc_id","source_url")):
                        raise ValueError("매뉴얼 출처·원문 무결성 검증에 실패했습니다.")
        except (ValueError, KeyError, TypeError, OSError) as exc:
            cache["terminal_error"]="공식 매뉴얼 검색에 실패했습니다: "+str(exc)
            raise ValueError(cache["terminal_error"]) from exc
        cache["matches"]=list({m["chunk_id"]:m for m in cache.get("matches",[])+matches}.values())
        times=cache.setdefault("timings_ms",{})
        for name,value in found.get("timings_ms",{}).items():times[name]=times.get(name,0)+value
        emit("ManualSearchCompleted",chunks=len(cache["matches"]))
        return {"matches":deepcopy(cache["matches"]),"timings_ms":times}

    async def select(matches):
        actions=candidates(matches)
        if integrated:
            needed_types=set().union(*(requirement_types(r) for r in requirements))
            actions=[a for a in actions if set(a["incident_types"]) & needed_types]
        feedback=state.get("manual_feedback") if (state.get("manual_feedback") or {}).get("manual_only") else None
        selection_key=hashlib.sha256(json.dumps([summary_query,observations,requirements,[a["action_id"] for a in actions],feedback],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if selection_key==cache.get("selection_key"):
            return deepcopy(cache["selected"]),{**cache["metrics"],"model_calls":0,"cost_usd":0}
        try:
            from monitoring.evidence import extract_facts
            observed=[{"evidence_id":e["evidence_id"],"server_id":e["server_id"],"facts":extract_facts(e)} for e in result["evidence"] if not e.get("error") and e.get("exit_code")==0]
            eligible=[]
            for requirement in requirements:
                refs=[e["evidence_id"] for e in result["evidence"] if e["server_id"] in requirement["server_ids"] and any(f["information_id"] in requirement["information_ids"] for f in extract_facts(e))]
                try:eligible+=bind_no_action([{"requirement_id":requirement["requirement_id"],"evidence_ids":refs}],requirements,result["evidence"])
                except ValueError:pass
            if not actions and not eligible:
                return [],{"model_calls":0,"cost_usd":0,"summary_ms":0,"incident_types":[]}
            selected, metrics=await summarize(deps.quality_llm,summary_query,observations,actions,requirements or None,feedback,
                observed_facts=observed,no_action_options=eligible,validate_no_action=(lambda items:bind_no_action(items,requirements,result["evidence"])) if integrated else None)
        except ValueError as exc:
            failed=getattr(exc,"metrics",{})
            cache["model_calls"]=cache.get("model_calls",0)+failed.get("model_calls",0)
            cache["summary_ms"]=cache.get("summary_ms",0)+failed.get("summary_ms",0)
            result["cost_usd"]+=failed.get("cost_usd",0)
            raise
        cache["model_calls"]=cache.get("model_calls",0)+metrics["model_calls"]
        cache["summary_ms"]=cache.get("summary_ms",0)+metrics["summary_ms"]
        result["cost_usd"]=round(result.get("cost_usd",0)+metrics["cost_usd"],6)
        # Validate bindings before caching or publishing any selected step.
        if selected or metrics.get("no_action"):render_manual_selection(selected,matches,requirements,result["evidence"],target,os_name,observations,integrated,unsupported,metrics.get("no_action",[]))
        cache.update(selection_key=selection_key,selected=deepcopy(selected),metrics=metrics,
                     requirement_ids=[r["requirement_id"] for r in requirements])
        return selected,metrics

    def checkpoint(local):
        text,details=render_manual_selection(local["selected"],local["matches"],requirements,result["evidence"],target,os_name,observations,integrated,unsupported,local.get("metrics",{}).get("no_action",[]))
        details.update(status="provided",unsupported_requirements=[r["requirement_id"] for r in unsupported],
                       search_calls=cache.get("search_calls",0),model_calls=cache.get("model_calls",0),
                       timings_ms={**cache.get("timings_ms",{}),"summary":cache.get("summary_ms",0)})
        result["remediation"]=details
        cache["verified_selected"]=deepcopy(local["selected"])
        cache["verified_no_action"]=deepcopy(local.get("metrics",{}).get("no_action",[]))
        save_progress({**result,"manual_state":cache,"events":events,"_verified_manual_text":text})

    # Re-entry may wait on a replacement selection. Restore previously verified
    # steps before the first event can overwrite the request deadline snapshot.
    previous=cache.get("verified_selected",[])
    previous_no_action=cache.get("verified_no_action",[])
    if previous or previous_no_action:
        known={r["requirement_id"] for r in requirements}
        canonical={a["action_id"]:a for a in candidates(cache.get("matches",[]))}
        try:
            for step in previous:
                action=canonical.get(step.get("action_id"))
                if not action or any(step.get(k)!=action.get(k) for k in ("text","precondition","quote","risk","incident_types","doc_id","chunk_id")):
                    raise ValueError("기존 매뉴얼 단계의 출처 검증에 실패했습니다.")
                if integrated and not set(step.get("requirement_ids",[]))<=known:
                    raise ValueError("기존 매뉴얼 단계의 요구 연결을 확인하지 못했습니다.")
            if previous_no_action:
                rebound=bind_no_action([{k:d[k] for k in ("requirement_id","evidence_ids")} for d in previous_no_action],requirements,result["evidence"])
                if rebound!=previous_no_action:raise ValueError("이전 조치 생략 결과의 관측 근거가 달라졌습니다.")
            checkpoint({"selected":previous,"matches":cache.get("matches",[]),"metrics":{"no_action":previous_no_action}})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            cache.pop("verified_selected",None)
            cache.pop("verified_no_action",None)
            cache["terminal_error"]=str(exc)
    emit("SpecialistDelegated",role="manual",requirement_ids=[r["requirement_id"] for r in requirements])
    specialist=build_manual_agent(retrieve,select,checkpoint,emit)
    local=await specialist.ainvoke({"query":search_query,"requirements":deepcopy(requirements),"rounds":0,
        "matches":deepcopy(cache.get("matches",[])),"selected":deepcopy(cache.get("verified_selected",[])),
        "metrics":cache.get("metrics",{}),"timings_ms":cache.get("timings_ms",{})})
    emit("SpecialistReturned" if local["status"]!="unavailable" else "SpecialistFailed",role="manual",
         status=local["status"],rounds=local["rounds"])
    if not local["selected"] and not local.get("metrics",{}).get("no_action"):
        if local.get("error"):
            return finish("기존 서버 조회 결과는 유지됩니다. 대처방안을 제공하지 못했습니다: "+local["error"],"unavailable")
        return finish("현재 상황에 적용할 수 있는 공식 대처 매뉴얼을 찾지 못했습니다. 지원 범위는 CPU·메모리·디스크입니다.","not_found",
                      timings_ms={**cache.get("timings_ms",{}),"summary":cache.get("summary_ms",0)})
    text,details=render_manual_selection(local["selected"],local["matches"],requirements,result["evidence"],target,os_name,observations,integrated,unsupported,local.get("metrics",{}).get("no_action",[]))
    if local.get("error"):
        note="추가 매뉴얼 확인을 완료하지 못했습니다: "+local["error"]
        details["unconfirmed"].append(note)
        text+="\n\n미확인 사항\n- "+note
    return finish(text,"provided",**details,
        timings_ms={**cache.get("timings_ms",{}),"summary":cache.get("summary_ms",0)},
        model_calls=cache.get("model_calls",0),search_calls=cache.get("search_calls",0))
