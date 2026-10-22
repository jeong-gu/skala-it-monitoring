"""Wave 13 — 의존성 주입.

노드 함수가 closure의 지역 변수에 암묵적으로 기대지 않도록, supervisor가 쓰는 모든
의존성을 SupervisorDeps 하나에 명시적으로 담아 functools.partial로 각 노드에 주입한다.
파일을 어디로 옮겨도, 단위 테스트에서 mock deps로 갈아 끼워도 동작이 같다.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SupervisorDeps:
    fast_llm: Any        # Wave 10a — 분류/계획 등 짧은 단계용 저비용 모델
    quality_llm: Any     # Wave 10a — 최종 답변 작성용 고품질 모델
    mcp: Any             # Wave 2  — MCP stdio 클라이언트 (Tool/Resource/Prompt)
    session_store: Any   # Wave 12 — SQLite 세션 저장소
    version: str         # Wave 14 — SUPERVISOR_VERSION (v1/v2)
    monitoring_llm: Any = None  # 정보 중심 조사는 별도 계획 모델 주입 가능
    information_catalog: list[dict] = field(default_factory=list)


# Request-local state is shared with child graph tasks through ContextVar copies.
# No model, session, or snapshot is stored in a process-global mutable singleton.
import asyncio
import math
import os
from contextvars import ContextVar
from copy import deepcopy
from time import perf_counter


_REQUEST_PROGRESS = ContextVar("monitoring_request_progress", default=None)
# Request-local numeric telemetry. The same mutable list is copied into provider
# worker threads; it never contains prompts, outputs, server names or exceptions.
REQUEST_TELEMETRY = ContextVar("monitoring_request_telemetry", default=None)


def save_progress(values: dict) -> None:
    progress = _REQUEST_PROGRESS.get()
    if progress is not None:
        progress["state"].update(deepcopy(values))


def _deadline_result(progress: dict) -> dict:
    from .investigation import render_answer
    from .response_policy import single_server_text
    result = deepcopy(progress["state"])
    note = f"요청 전체 시간 제한({progress['seconds']:g}초)에 도달했습니다. 확보된 관측만 제공하며 최종 답변 검증은 완료하지 못했습니다."
    result.setdefault("limitations", []).append(note)
    result["final_validation"] = {"status":"unverified", "request_fully_satisfied":False, "reason":note}
    result.setdefault("events", []).append({"type":"WorkflowTimedOut", "event_id":"workflow-timeout", "seconds":progress["seconds"]})
    result.setdefault("workflow_timings_ms", {})["total"] = round((perf_counter()-progress["started"])*1000, 2)
    result["presentation"] = {"status":"fallback", "reason":"요청 전체 시간 제한", "model_calls":0}
    # Reconstruct only from stored verified facts/results. Never ask a model to
    # produce or validate another answer after the execution deadline expires.
    text = render_answer(result.get("tasks", []), result.get("evidence", []),
        result.get("hypotheses", []), result["limitations"], result.get("answers", []),
        result.get("calculation_results", []), result.get("requirements", []))
    manual_text = result.pop("_verified_manual_text", "")
    if manual_text and (result.get("remediation") or {}).get("status") == "provided":
        text += "\n\n" + manual_text
    result["final"] = single_server_text(text, result)
    result["_presentation_done"] = True
    return result


async def stream_graph(graph, state: dict, *, timeout_seconds=None):
    seconds = float(os.environ.get("WORKFLOW_TIMEOUT_SECONDS", "120") if timeout_seconds is None else timeout_seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("WORKFLOW_TIMEOUT_SECONDS must be a finite positive number")
    progress = {"state":deepcopy(state), "seconds":seconds, "started":perf_counter()}
    queue = asyncio.Queue()

    async def produce():
        # The timer belongs to the execution task. Keeping a timeout context
        # across an async-generator yield would cancel the SSE consumer instead
        # when it is sending data or waiting on browser backpressure.
        token = _REQUEST_PROGRESS.set(progress)
        timeout = asyncio.timeout(seconds)
        try:
            try:
                async with timeout:
                    async for mode, payload in graph.astream(state, stream_mode=["updates", "custom"]):
                        if mode == "updates":
                            for delta in payload.values():
                                if delta:save_progress(delta)
                        queue.put_nowait((mode, payload))
            except TimeoutError:
                if not timeout.expired():raise
                queue.put_nowait(("updates", {"deadline":_deadline_result(progress)}))
        except asyncio.CancelledError as exc:
            # Upstream cancellation is not a successful empty result. A consumer
            # that deliberately cancelled this worker during cleanup ignores the
            # queued item; an active consumer receives the original cancellation.
            queue.put_nowait(("execution_error", exc))
            raise
        except Exception as exc:
            queue.put_nowait(("execution_error", exc))
        finally:
            _REQUEST_PROGRESS.reset(token)
            queue.put_nowait(None)

    worker = asyncio.create_task(produce())
    try:
        while True:
            item = await queue.get()
            if item is None:break
            if item[0] == "execution_error":raise item[1]
            yield item
    finally:
        if not worker.done():worker.cancel()
        try:await worker
        except asyncio.CancelledError:pass


async def run_graph(graph, state: dict, *, timeout_seconds=None) -> dict:
    result = deepcopy(state)
    async for mode, payload in stream_graph(graph, state, timeout_seconds=timeout_seconds):
        if mode == "updates":
            for delta in payload.values():
                if delta:result.update(delta)
    return result
