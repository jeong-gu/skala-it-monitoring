"""all_total 통합 실습 — FastAPI 백엔드 (Wave 1·3·12·14 축 + 전체 조립).

- Wave 1  : lifespan에서 Agent 번들(MCP 연결·LLM·그래프)을 프로세스당 정확히 1회 생성,
            app.state로 모든 요청이 공유 (creation_count/instance_id로 증명)
- Wave 2  : mcp_server.py를 subprocess(stdio)로 기동해 연결 유지
- Wave 3  : GET /chat/stream — 그래프 실행 과정을 SSE 이벤트로 스트리밍
- Wave 12 : 대화 이력을 SQLite(session.db)에 저장 — 재시작해도 유지
- Wave 14 : SUPERVISOR_VERSION(v1/v2) 환경변수로 legacy/신형 파이프라인 즉시 전환
- LangSmith: POST/SSE 공통 요청 루트와 모델 호출의 익명·수치 전용 추적
"""
import json
import os
import asyncio
import anyio
import math
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from time import perf_counter
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()   # LANGCHAIN_TRACING_V2 / API_KEY / PROJECT — import보다 먼저

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langsmith import Client, tracing_context
from pydantic import BaseModel, Field

from agent.deps import SupervisorDeps, run_graph, stream_graph, REQUEST_TELEMETRY
from agent.fake_llm import FastLLM, QualityLLM
from agent.graph import build_graph
from agent.mcp_client import MCPClient
from agent.session_store import SqliteSessionStore
from agent.supervisor_v1 import run_v1
from monitoring.api import create_router, ScenarioStore
from remediation.context import recall, remember

CREATION_COUNT = 0   # Wave 1 — 프로세스 내 Agent 번들 생성 횟수 (항상 1이어야 정상)


class ChatRequest(BaseModel):   # Wave 1 — Pydantic 요청 검증
    message: str
    thread_id: str = "default"
    scenario_id: str | None = Field(default=None, min_length=1, max_length=64)


def _build_llms():
    """LLM_BACKEND=fake(기본, 결정론) | ollama(로컬 실모델) | openai(실제 OpenAI API)."""
    backend = os.environ.get("LLM_BACKEND", "fake")
    if backend == "ollama":
        from agent.ollama_llm import OllamaFastLLM, OllamaQualityLLM
        fast = OllamaFastLLM(os.environ.get("OLLAMA_FAST_MODEL", "qwen3:0.6b"),
                             usd_per_1k_tokens=0.00015)
        quality = OllamaQualityLLM(os.environ.get("OLLAMA_QUALITY_MODEL", "qwen3:8b"),
                                   usd_per_1k_tokens=0.005)
        return backend, fast, quality
    if backend == "openai":
        from agent.openai_llm import OpenAIFastLLM, OpenAIQualityLLM
        fast = OpenAIFastLLM(os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini"),
                             usd_per_1k_tokens=0.00015)
        quality = OpenAIQualityLLM(os.environ.get("OPENAI_QUALITY_MODEL", "gpt-4o"),
                                   usd_per_1k_tokens=0.005)
        return backend, fast, quality
    fast = FastLLM("fast-model-stub", usd_per_1k_tokens=0.00015)
    quality = QualityLLM("quality-model-stub", usd_per_1k_tokens=0.005)
    return backend, fast, quality


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CREATION_COUNT
    CREATION_COUNT += 1
    mcp = MCPClient()
    await mcp.start()   # Wave 2 — MCP subprocess는 서버 수명 동안 딱 1개
    backend, fast_llm, quality_llm = _build_llms()
    app.state.llm_backend = backend
    deps = SupervisorDeps(
        fast_llm=fast_llm,
        quality_llm=quality_llm,
        monitoring_llm=fast_llm if backend == "fake" else quality_llm,
        information_catalog=[i.model_dump() for i in scenario_store.information],
        mcp=mcp,
        session_store=SqliteSessionStore(),
        version=os.environ.get("SUPERVISOR_VERSION", "v2"),
    )
    app.state.deps = deps
    app.state.graph = build_graph(deps) if deps.version == "v2" else None
    app.state.instance_id = id(deps)
    yield
    await mcp.stop()
    deps.session_store.close()
    try:   # 종료 전 LangSmith 배치 전송 플러시 — kill 시 run이 pending으로 남는 것 방지
        from langchain_core.tracers.langchain import wait_for_all_tracers
        wait_for_all_tracers()
    except Exception:
        pass


app = FastAPI(lifespan=lifespan)
scenario_store = ScenarioStore()
app.include_router(create_router(scenario_store))


@app.get("/health")
async def health():
    return {"status": "ok", "instance_id": app.state.instance_id,
            "creation_count": CREATION_COUNT,
            "supervisor_version": app.state.deps.version,
            "llm_backend": app.state.llm_backend,
            "fast_llm": app.state.deps.fast_llm.name,
            "quality_llm": app.state.deps.quality_llm.name}


@app.get("/version")
async def version():
    return {"supervisor_version": app.state.deps.version}


def _metadata(result: dict, run_id: str | None) -> dict:
    return {
        "route": result.get("route"),
        "scenario_id": result.get("scenario_id"),
        "task_count": len(result.get("tasks", [])),
        "gate_removed": result.get("gate_removed", 0),
        "replan_rounds": result.get("replan_rounds", 0),
        "retry_count": result.get("retry_count", 0),
        "parallel_elapsed": result.get("parallel_elapsed"),
        "validation_attempts": result.get("validation_attempts", 0),
        "cost_usd": result.get("cost_usd", 0.0),
        "command_call_count": result.get("command_call_count", 0),
        "final_validation": result.get("final_validation", {}),
        "presentation": result.get("presentation", {}),
        "workflow_budget": result.get("workflow_budget", {}),
        "workflow_timings_ms": result.get("workflow_timings_ms", {}),
        "llm_backend": app.state.llm_backend,
        "supervisor_version": app.state.deps.version,
        "instance_id": app.state.instance_id,
        "langsmith_run_id": run_id,
    }


def scenario_session_key(thread_id: str, scenario_id: str | None) -> str:
    return thread_id if scenario_id is None else "scenario:" + json.dumps([thread_id,scenario_id], separators=(",", ":"))


def chat_state(message: str, thread_id: str, scenario_id: str | None) -> dict:
    deps=app.state.deps
    if deps.version == "v2":
        # A fixture key selects a saved state of one logical monitored host.
        # An omitted key uses the first registered state (normal A in this corpus).
        scenario_id = scenario_id or scenario_store.scenario_list()[0].scenario_id
        scenario_store.server(scenario_id)  # Invalid explicit IDs never fall back.
    elif scenario_id is not None:
        raise HTTPException(409, detail="시나리오 선택은 v2에서 지원합니다.")
    key=scenario_session_key(thread_id,scenario_id)
    return {"query":message,"thread_id":key,"scenario_id":scenario_id,"single_server_mode":deps.version=="v2",
            "scenario_os":scenario_store.server(scenario_id).os if scenario_id else None,
            "history":deps.session_store.load(key),"investigation_context":recall(deps.session_store,key),
            "events":[],"cost_usd":0.0}


@lru_cache(maxsize=1)
def _metrics_client():
    enabled = os.environ.get("LANGSMITH_TRACING", os.environ.get("LANGCHAIN_TRACING_V2", "false")).lower() == "true"
    if not enabled or not (os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")):return None
    return Client(omit_traced_runtime_info=True, auto_batch_tracing=True, timeout_ms=2000)


@asynccontextmanager
async def _request_trace(transport):
    """One numeric-only root for either transport; raw nested tracing is off.

    Direct Client calls export only a constructed allowlist, never graph state.
    A failed telemetry service must neither fail nor repeat application work.
    """
    root_id = uuid.uuid4()
    started, clock = datetime.now(timezone.utc), perf_counter()
    order = started.strftime('%Y%m%dT%H%M%S%fZ') + str(root_id)
    trace_state={"run_id":None,"result":{}}
    records=[];token=REQUEST_TELEMETRY.set(records)
    client=None;error=None
    with tracing_context(enabled=False):
        try:
            try:
                client=_metrics_client()
                if client is not None:
                    await asyncio.to_thread(client.create_run, name="chat_request", run_type="chain", inputs={},
                        id=root_id, trace_id=root_id, dotted_order=order, start_time=started,
                        project_name=os.environ.get("LANGSMITH_PROJECT",os.environ.get("LANGCHAIN_PROJECT","default")),
                        extra={"metadata":{"metrics_only":True}}, tags=["metrics-only",transport])
                    trace_state["run_id"]=str(root_id)
            except Exception:client=None
            yield trace_state
        except asyncio.CancelledError:
            error="request_cancelled"
            raise
        except BaseException:
            error="request_failed"
            raise
        finally:
            try:
                if client is not None:
                    result=trace_state["result"]
                    metrics={"duration_ms":round((perf_counter()-clock)*1000,2),"model_calls":len(records),
                        "validation_code":{"passed":1,"partial":2,"unverified":3}.get((result.get("final_validation") or {}).get("status"),0)}
                    for key in ("cost_usd","command_call_count","replan_rounds","validation_attempts"):
                        value=result.get(key,0)
                        if isinstance(value,(int,float)) and math.isfinite(value):metrics[key]=value
                    metrics["cost_usd"] = max(metrics.get("cost_usd",0), sum(item["metrics"].get("usd",0) for item in records))
                    metrics["manual_searches"]=(result.get("remediation") or {}).get("search_calls",0)
                    # Disconnect cancellation is level-triggered in Starlette.
                    # Bound cleanup and shield it; child export failure must never
                    # leave an otherwise reachable request root unfinished.
                    with anyio.move_on_after(2, shield=True):
                        await asyncio.to_thread(client.update_run, root_id, trace_id=root_id,dotted_order=order,
                            end_time=datetime.now(timezone.utc),outputs={"metrics":metrics},error=error)
                        for item in records:
                            child_id=uuid.uuid4()
                            child_order=order+'.'+item['start'].strftime('%Y%m%dT%H%M%S%fZ')+str(child_id)
                            try:
                                await asyncio.to_thread(client.create_run, name="llm."+item["purpose"],run_type="llm",inputs={},
                                    outputs={"metrics":item["metrics"]}, id=child_id,parent_run_id=root_id,trace_id=root_id,
                                    dotted_order=child_order,start_time=item["start"],end_time=item["end"],
                                    extra={"metadata":{"metrics_only":True}})
                            except Exception:pass

            except Exception:pass
            finally:REQUEST_TELEMETRY.reset(token)


async def _handle_chat(message: str, thread_id: str, scenario_id: str | None = None) -> tuple[dict, str | None]:
    deps = app.state.deps
    state=chat_state(message,thread_id,scenario_id)
    thread_id=state["thread_id"];history=state["history"]
    async with _request_trace("post") as traced:
        if deps.version == "v2":
            result = await run_graph(app.state.graph, state)
        else:
            result = await run_v1(deps, message, thread_id, history)   # Wave 14
        traced["result"].update(result)
        remember(deps.session_store, thread_id, result, message)
        deps.session_store.append(thread_id, "user", message)
        deps.session_store.append(thread_id, "assistant", result["final"])
    return result, traced["run_id"]


@app.post("/chat")
async def chat(req: ChatRequest):
    result, run_id = await _handle_chat(req.message, req.thread_id, req.scenario_id)
    return {"reply": result["final"], "scenario_id":result.get("scenario_id"), "events": result.get("events", []),
            "workflow_timings_ms": result.get("workflow_timings_ms", {}),
            "workflow_budget": result.get("workflow_budget", {}),
            "presentation": result.get("presentation", {}),
            "remediation": result.get("remediation"),
            "metadata": _metadata(result, run_id), "final_validation": result.get("final_validation", {}),
            **{k: result.get(k, []) for k in ("tasks", "evidence", "hypotheses", "limitations", "requirements", "requirement_issues", "answers", "calculation_results", "final_reviews")}}


@app.get("/chat/stream")
async def chat_stream(message: str, thread_id: str = "default", scenario_id: str | None = None):
    """Wave 3 — 그래프의 노드별 진행 상황을 SSE로 스트리밍.

    이벤트마다 'data: {json}\\n\\n' 형태로 내려보내고 마지막에 data: [DONE] 센티널을
    보낸다 — 프론트엔드는 \\n\\n 경계로 이벤트를 분리해 파싱해야 한다(Wave 3b).
    """
    deps = app.state.deps
    state=chat_state(message,thread_id,scenario_id)
    thread_id=state["thread_id"]
    scenario_id=state["scenario_id"]

    async def events(traced):
        history = state["history"]
        if deps.version != "v2":
            result = await run_v1(deps, message, thread_id, history)
            traced["result"].update(result)
            for ev in result["events"]:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'FinalAnswer', 'reply': result['final'], 'metadata':_metadata(result,traced['run_id'])}, ensure_ascii=False)}\n\n"
        else:
            seen = 0
            seen_event_ids = set()
            final = ""
            investigation = {k: [] for k in ("tasks", "evidence", "hypotheses", "limitations", "requirements", "requirement_issues", "answers", "calculation_results", "final_reviews")}
            final_validation = {}
            presentation = {}
            investigation["scenario_id"] = scenario_id
            investigation["request_subject"] = message
            investigation["remediation"] = None
            investigation["workflow_budget"] = {}
            investigation["workflow_timings_ms"] = {}
            last_prompt = None   # finalize까지 살아남은 최종 write/chat 호출의 실제 프롬프트
            # stream_mode를 리스트로 주면 (mode, payload) 튜플이 나온다 — "updates"는
            # 노드가 끝날 때마다의 상태 델타, "custom"은 write_node/chat_answer_node가
            # get_stream_writer()로 실행 중에 즉시 흘려보내는 토큰(AnswerToken) 이벤트다.
            async for mode, payload in stream_graph(app.state.graph, state):
                if mode == "custom":
                    if payload.get("event_id"):
                        seen_event_ids.add(payload["event_id"])
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    continue
                for node_name, delta in payload.items():
                    if not delta:continue
                    traced["result"].update(delta)
                    for ev in (delta.get("events") or [])[seen:]:
                        if ev.get("event_id") in seen_event_ids:
                            continue
                        ev = {**ev, "node": node_name}
                        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    if delta.get("events"):
                        seen = len(delta["events"])
                    if delta.get("final"):
                        final = delta["final"]
                    if delta.get("last_prompt"):
                        last_prompt = delta["last_prompt"]
                    if "final_validation" in delta:
                        final_validation = delta["final_validation"]
                    if "presentation" in delta:
                        presentation = delta["presentation"]
                    for key in investigation:
                        if key in delta:
                            investigation[key] = delta[key]
            yield f"data: {json.dumps({'type': 'FinalAnswer', 'metadata':_metadata(traced['result'],traced['run_id']), 'reply': final, 'scenario_id':scenario_id, 'prompt': last_prompt, 'presentation': presentation, 'final_validation': final_validation, **investigation}, ensure_ascii=False)}\n\n"
            remember(deps.session_store, thread_id, investigation, message)
            deps.session_store.append(thread_id, "user", message)
            deps.session_store.append(thread_id, "assistant", final)
        yield "data: [DONE]\n\n"

    async def gen():
        async with _request_trace("sse") as traced:
            traced["result"]["scenario_id"] = scenario_id
            async for event in events(traced):
                yield event

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="warning")
