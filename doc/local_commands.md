# 정보 중심 서버 조사

사용자 요청을 task와 필요한 정보로 나눈 뒤, 프로그램이 정보 ID를 OS별 등록 명령으로 변환한다.
LLM이 임의의 셸 명령문을 만들지 않는다.

```text
사용자 요청 + 같은 세션 이력
 → 대상 확인 → 모델의 task·정보 선택 → 코드의 OS별 명령 해결·중복 제거
 → MCP(stdio JSON-RPC) → 동일 로컬 FastAPI → 서버별 JSON 출력
 → 필드 충족 검사 → 모델의 추가 task 선택(최대 2회)
 → 확인한 사실·원인 후보·미확인 항목 + 원문 근거
```

외부 서버, SSH, 실제 셸 실행은 포함하지 않는다. 모든 결과는 `source: fixture`인 고정 시나리오다.
기존 서비스 인시던트 경로와 구분되는 v2 경로이며 `SUPERVISOR_VERSION=v2`로 실행한다.

## 정보 정의와 명령

`data/command_scenarios/information.json`이 정보 ID, 의미, 필드·단위, OS별 조회 방법과 한계를 정의한다.
`commands.json`은 OS별 정확한 명령문과 제공하는 정보 ID를 정의한다.
서버 JSON에 지원 명령 전체의 `stdout`, `stderr`, `exit_code`, 공통 `observed_at`을 저장한다.
파일은 서버 시작 시 검증·로딩한다. 수정 후 FastAPI 재시작이 필요하다.

| 정보 ID | 의미 | Linux | macOS |
|---|---|---|---|
| system | OS·커널·아키텍처 | uname | uname |
| cpu | 시스템 CPU 비율·Load Average | top | top |
| process | PID·PPID·이름·CPU·메모리·RSS | ps | ps |
| memory.summary | OS별 메모리 요약 | free | top |
| memory.detail | Swap 또는 페이지·압축 통계 | free | vm_stat |
| disk | 파일시스템별 용량·사용률 | df | df |
| network | 소켓·포트·연결·프로세스 | ss | lsof |
| log.files | 로그 파일 목록·크기·수정 시각 | ls | ls |
| log.usage | /var/log/ 사용량 | du | du |
| log.content | 최근 애플리케이션 로그 | tail | tail |

`tail`은 `tail -n 100 /var/log/application.log`로 고정하며 각 파일은 100줄 이하의 발췌다.
Linux top도 메모리 요약을 제공하므로 이미 확보한 정보는 재사용한다.
Linux available, macOS unused 및 페이지 통계는 다른 지표다.
프로세스 CPU는 코어 합산 비율이며 시스템 CPU와 직접 비교하지 않는다.
CPU 계산값은 `100 - idle`과 원본 유휴 비율을 함께 표시한다.
소켓 정보는 외부 서비스 응답 성공·시간을 입증하지 않는다.
시계열(history), 배포 이력(deployment), 네트워크 성능(network.performance)은 미지원이다.
빈 로그·일부 로그로 전체 기간에 오류가 없다고 판단하지 않는다.

## 시나리오

| 서버 | OS | 파일에서 설정한 상태 |
|---|---|---|
| server-a | Linux | 정상 |
| server-b | Linux | CPU 과부하 |
| server-c | Linux | 메모리 부족·Swap 사용 |
| server-d | Linux | 디스크 부족·큰 로그 |
| server-e | macOS | 정상 |
| server-f | macOS | CPU 과부하 |
| server-g | macOS | 메모리 압박 |
| server-h | macOS | 디스크 부족·큰 로그 |

시나리오 정답은 서버 목록과 명령 응답에 노출하지 않는다. Agent는 출력에서 판단한다.

## API·MCP

- `GET /monitoring/information`: 전체 정보 정의·필드·단위·OS별 명령.
- `GET /monitoring/servers`: 서버 ID·표시 이름·OS.
- `GET /monitoring/servers/{server_id}/commands`: 정확한 명령문·제공 정보 ID.
- `POST /monitoring/commands`: 기존 원문 응답 계약 유지.

```json
{"server_id":"server-b","command":"top"}
```

응답은 `server_id`, `os`, `command_id`, `command`, `stdout`, `stderr`, `exit_code`,
`observed_at`, `source`를 포함한다. 명령 ID 또는 카탈로그의 정확한 전체 명령문만 받는다.
미등록 서버는 404 `server_not_found`, 미지원 명령은 422 `unsupported_command`다.
저장된 명령 실패는 `exit_code != 0`으로 유지한다. 연결 실패는 정상 수치로 대체하지 않는다.

MCP 도구는 `list_command_servers`, `list_server_information`, `list_server_commands`,
`get_command_result`다. 기본 주소는 `http://127.0.0.1:8100`이며
`LOCAL_MONITORING_API_URL`로 다른 루프백 포트를 설정할 수 있다.

`POST /chat`과 SSE `FinalAnswer`는 기존 `reply`에 `tasks`, `evidence`, `hypotheses`,
`limitations`를 추가한다. evidence에는 기존 원문과 `evidence_id`, 추출 사실·인용이 들어간다.
채팅 화면은 task의 목적·정보·선택 명령·진행 상태와 접을 수 있는 원문을 표시한다.

## 계획·검증 정책

- task는 서버·조사 목적·정보 ID 목록으로 동적으로 만든다. 고정 task 유형을 강제하지 않는다.
- 대상 누락·모호한 후속 대상은 조회 전에 확인한다. 미등록 서버를 대체하지 않는다.
- “그 서버”는 같은 세션의 가장 최근 서버를 사용한다. 복수 후보면 확인한다.
- 상태: pending(미조회), obtained(정보 확보), needs_more(추가 확인 필요), failed(조회 실패), unsupported(미지원).
- 성공 종료만으로 완료하지 않는다. 등록 형식의 실제 필드를 추출해야 확보로 인정한다.
- 같은 서버·명령은 요청 중 한 번만 호출한다. 확보한 출력은 여러 task가 공유한다.
- 첫 조회 후 추가 조사 최대 2회, 결과 조회 총 12건. 카탈로그 조회는 결과 조회 수에 포함하지 않는다.
- Ollama에서는 0.6B 모델이 추가 조사를 생략한 실측 결과에 따라 조사 계획·재검토·해석에 quality 모델(기본 qwen3:8b)을 사용한다. 기존 경로의 모델 분리는 유지한다.
- 구조화 모델 출력은 검증하고 1회 수정 재시도한다. 실패하면 미확인으로 남긴다.
- 확인 사실은 원문 파서로 작성한다. 모델 원인 후보는 참조 ID·원문 인용·수치·대상을 검증한다.
  이 검증이 의미상의 인과관계를 입증하지 않으므로 원인 후보는 확정하지 않는다.
- `JSON만` 또는 `원문만` 요청은 원문 결과 객체(복수면 배열)를 reply로 반환한다.

## 실행·예시

기존 서버가 실행 중이면 정상 종료 후 재시작한다. DB를 삭제하는 실행 스크립트 대신:

```bash
LLM_BACKEND=ollama SUPERVISOR_VERSION=v2 .venv/bin/python3 server.py
```

채팅 `http://localhost:8600/`:

```text
서버 B가 느린데 원인을 확인해줘
서버 G 메모리가 부족한 원인을 확인해줘
서버 D 디스크가 부족한 원인을 확인해줘
그 서버의 최근 애플리케이션 로그를 보여줘
서버 A와 B의 CPU를 비교해줘
서버 B top JSON만
```

## 검증

```bash
.venv/bin/python3 -m unittest discover -s tests -v
```

모의 모델로 동적 계획, 정보 충족, OS 대응, 중복 제거, 추가 조회·12건 제한과
오류·미지원·근거 검증을 검사한다. 임시 DB·포트의 API·MCP·LangGraph·SSE 왕복을 포함한다.
실제 Ollama 및 화면 검증은 별도로 실행한다.
