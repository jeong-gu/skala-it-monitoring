# 로컬 명령어 시나리오 API

기존 8100 포트의 FastAPI 안에서 명령어 결과를 JSON으로 반환한다. 추가 외부 서버는 없다.

```text
채팅의 직접 명령 요청 → LangGraph → MCP(stdio/JSON-RPC)
                                  → 같은 로컬 FastAPI의 /monitoring
                                  → 서버별 JSON 파일 → 출력 원문
```

현재 단계는 저장된 명령어 출력 조회다. 셸 명령, SSH, 서버 재시작은 실행하지 않는다.
CPU·메모리 질문을 LLM이 명령으로 변환하는 기능은 이번 추가 경로에 포함하지 않는다.
채팅에서는 서버와 명령을 직접 지정한다. 기존 서비스 장애 진단 경로는 그대로 이용할 수 있다.

## 파일과 시나리오

- `data/command_scenarios/commands.json`: OS별 명령 ID, 정확한 명령문, 조회 목적.
- `data/command_scenarios/server-a.json` ~ `server-h.json`: 서버별 고정 스냅샷.
- 각 서버 파일의 `commands`에 `stdout`, `stderr`, `exit_code`를 저장한다.
- 출력은 실행 시점마다 바뀌지 않는다. `source: fixture`와 파일의 `observed_at`을 반환한다.
- `scenario`는 파일 설명용이며 서버 목록과 명령 결과에는 노출하지 않는다.
- 파일은 시작할 때 읽는다. 수정 후 FastAPI를 재시작하면 반영된다.

| 서버 | OS | 시나리오 | 예시 명령 |
|---|---|---|---|
| server-a / 서버 A | Linux | 정상 | top, free, df |
| server-b / 서버 B | Linux | CPU 과부하 | top, ps |
| server-c / 서버 C | Linux | 메모리 부족·Swap 사용 | free, top, ps |
| server-d / 서버 D | Linux | 디스크 부족·큰 로그 | df, ls, du |
| server-e / 서버 E | macOS | 정상 | top, vm_stat, df |
| server-f / 서버 F | macOS | CPU 과부하 | top, ps |
| server-g / 서버 G | macOS | 메모리 압박 | top, vm_stat, ps |
| server-h / 서버 H | macOS | 디스크 부족·큰 로그 | df, ls, du |

서버 목록 API는 ID·표시 이름·OS만 제공한다. 위 상태는 모의 시나리오이며 실제 장비 측정값이 아니다.

## API 사용

기존 FastAPI를 실행하면 라우트가 함께 열린다. 프로젝트 디렉터리에서:

```bash
LLM_BACKEND=ollama SUPERVISOR_VERSION=v2 .venv/bin/python3 server.py
```

위 명령은 이미 서버가 떠 있으면 중복 실행하지 않는다. DB를 삭제하는 기존 실행 스크립트 대신
서버 프로세스를 정상 종료한 뒤 위 방식으로 다시 실행하면 대화 DB를 보존할 수 있다.

```bash
# 서버 목록
curl http://127.0.0.1:8100/monitoring/servers

# Linux 서버 B의 지원 명령 목록
curl http://127.0.0.1:8100/monitoring/servers/server-b/commands

# 명령 ID 또는 카탈로그의 정확한 명령문으로 요청
curl -X POST http://127.0.0.1:8100/monitoring/commands \
  -H 'Content-Type: application/json' \
  -d '{"server_id":"server-b","command":"top"}'
```

응답 형태:

```json
{
  "server_id": "server-b",
  "os": "linux",
  "command_id": "top",
  "command": "top -b -n 2 -d 1",
  "stdout": "저장된 명령어 출력 원문",
  "stderr": "",
  "exit_code": 0,
  "observed_at": "2026-10-20T12:00:01+09:00",
  "source": "fixture"
}
```

- 미등록 서버: HTTP 404 / `server_not_found`.
- 지원하지 않는 명령·해당 OS에 없는 명령: HTTP 422 / `unsupported_command`.
- `top; ...`, 임의 경로, 등록되지 않은 옵션은 실행하거나 해석하지 않는다.
- 저장된 명령 자체가 실패한 사례는 HTTP 200에서 `exit_code != 0`과 `stderr`로 구분한다.
- 연결 실패·시간 초과는 MCP 도구 오류로 전달하며 다른 서버나 0% 지표로 대체하지 않는다.

## MCP와 채팅

추가 도구:

1. `list_command_servers()`
2. `list_server_commands(server_id)`
3. `get_command_result(server_id, command)`

MCP는 같은 컴퓨터의 FastAPI에 비동기로 요청한다. 기본 주소는 `http://127.0.0.1:8100`이다.
다른 로컬 포트를 사용하려면 `LOCAL_MONITORING_API_URL`을 지정한다. 외부 주소는 받지 않는다.

채팅 입력 예시:

```text
서버 B top 조회해줘
서버 C free 조회해줘
서버 D df 조회해줘
서버 G vm_stat 조회해줘
server-a ls -lah /var/log/
```

이 경로는 조회 결과를 모델이 재작성하지 않고 JSON 그대로 보여준다.
지원하지 않는 명령을 적거나 명령을 생략하면 해당 서버의 지원 명령을 안내한다.
서버가 미등록이면 조회 실패를 명시한다.

## 명령 해석 시 주의점

- Linux `top`: CPU `id`는 유휴 비율이다. 전체 사용률은 `100 - id`이며 `us`만 전체 사용률로 보지 않는다.
- macOS `top`: 두 표본 중 두 번째를 사용한다. 첫 표본의 프로세스별 CPU는 비교에 사용하지 않는다.
- 프로세스 CPU는 코어 합산으로 100%를 넘을 수 있다. 시스템 전체 CPU 비율과 직접 비교하지 않는다.
- Linux `free -m`: 출력 단위는 MiB다. `free`와 `available`은 서로 다른 열이다.
- macOS `vm_stat`: 페이지 수를 메모리 크기로 바꾸려면 헤더의 페이지 크기를 곱한다.
- `df`는 파일시스템 사용량, `ls`는 디렉터리 목록, `du`는 해당 경로의 사용량이다.
- Linux는 `ss`, macOS는 `lsof`로 네트워크 연결 예시를 제공한다.

## 검증

```bash
.venv/bin/python3 -m unittest discover -s tests -v
```

실제 LLM 없이 모의 모델과 임시 포트의 FastAPI를 사용해 API·MCP·LangGraph·SSE 왕복을 검사한다.
실행 중인 8100 서버를 종료하거나 원본 대화 DB에 테스트 대화를 쓰지 않는다.
