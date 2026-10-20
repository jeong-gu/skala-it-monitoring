"""Compatibility parser plus entry point to information-first investigations."""
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
    from .investigation import investigate
    return await investigate(state, deps)
