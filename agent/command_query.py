"""Explicit command queries use MCP and the local API without an LLM rewrite."""
import json
import re

SERVER_PREFIX = re.compile(
    r"^\s*(?:서버\s*([a-z])|server[- ]([a-z]))(?![a-z0-9_-])(?:에서|의)?\s*(.*?)\s*$",
    re.IGNORECASE,
)


def parse_command_query(query: str) -> tuple[str, str] | None:
    match = SERVER_PREFIX.match(query.strip().rstrip(".!?。"))
    if not match:
        return None
    server_id = "server-" + (match[1] or match[2]).lower()
    command = re.sub(r"\s+(?:결과\s*)?(?:조회해줘|보여줘|조회|실행해줘|실행)$", "", match[3]).strip()
    return server_id, command


async def command_query_node(state, deps):
    server_id, command = parse_command_query(state["query"])
    try:
        commands = await deps.mcp.call_tool("list_server_commands", {"server_id": server_id})
        if not any(command in (c["command_id"], c["command"]) for c in commands):
            supported = ", ".join(c["command_id"] for c in commands)
            return {"final": f"{server_id}의 지원 명령: {supported}\n"
                             "예: 서버 B top 조회해줘\n"
                             "명령 ID 또는 목록의 정확한 명령문을 지정해주세요.",
                    "events": state["events"] + [{"type": "Finalize", "pipeline": "local-command"}]}
        result = await deps.mcp.call_tool(
            "get_command_result", {"server_id": server_id, "command": command})
    except ValueError as exc:
        return {"final": f"명령어 조회 실패: {exc}",
                "events": state["events"] + [{"type": "Finalize", "pipeline": "local-command"}]}
    return {
        "final": json.dumps(result, ensure_ascii=False, indent=2),
        "events": state["events"] + [
            {"type": "CommandCompleted", "server_id": server_id,
             "command": result["command"], "exit_code": result["exit_code"]},
            {"type": "Finalize", "pipeline": "local-command"},
        ],
    }
