"""MCP-side HTTP client for the FastAPI running on this machine."""
import os
from urllib.parse import urlsplit

import httpx


async def request_local_api(method: str, path: str, **kwargs):
    base_url = os.environ.get("LOCAL_MONITORING_API_URL", "http://127.0.0.1:8100")
    parsed = urlsplit(base_url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("LOCAL_MONITORING_API_URL은 이 컴퓨터의 HTTP 주소여야 합니다.")
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=5.0, trust_env=False) as client:
            response = await client.request(method, path, **kwargs)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ValueError(f"조회 거부 ({exc.response.status_code}): {exc.response.text}") from exc
    except httpx.TimeoutException as exc:
        raise ValueError("로컬 모니터링 API 응답 시간이 초과되었습니다.") from exc
    except httpx.RequestError as exc:
        raise ValueError("로컬 FastAPI에 연결할 수 없습니다. 서버 주소와 실행 상태를 확인하세요.") from exc
    return response.json()
