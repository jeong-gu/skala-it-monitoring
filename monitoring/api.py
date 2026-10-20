"""Serve saved command output; this module never executes shell commands."""
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "command_scenarios"


class CommandDefinition(BaseModel):
    command_id: str
    command: str
    description: str
    information_ids: list[str]


class InformationDefinition(BaseModel):
    information_id: str
    name: str
    description: str
    fields: dict[str, dict[str, str]]
    commands_by_os: dict[str, list[str]]
    limitations: str = ""


class SavedOutput(BaseModel):
    stdout: str
    stderr: str = ""
    exit_code: int = Field(default=0, ge=0, le=255)


class ServerScenario(BaseModel):
    server_id: str
    display_name: str
    os: Literal["linux", "macos"]
    scenario: str
    observed_at: datetime
    commands: dict[str, SavedOutput]


class ServerInfo(BaseModel):
    server_id: str
    display_name: str
    os: Literal["linux", "macos"]


class CommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server_id: str = Field(min_length=1, max_length=64)
    command: str = Field(min_length=1, max_length=200)


class CommandResult(SavedOutput):
    server_id: str
    os: Literal["linux", "macos"]
    command_id: str
    command: str
    observed_at: datetime
    source: Literal["fixture"] = "fixture"


class ScenarioStore:
    def __init__(self, directory: Path = DATA_DIR):
        self.directory = directory
        catalog = json.loads((directory / "commands.json").read_text())
        self.catalog = {
            os_name: [CommandDefinition.model_validate(c) for c in commands]
            for os_name, commands in catalog.items()
        }
        self.information = [InformationDefinition.model_validate(i) for i in
                            json.loads((directory / "information.json").read_text())]
        ids = {i.information_id for i in self.information}
        if len(ids) != len(self.information):
            raise ValueError("Duplicate information ID")
        for os_name, commands in self.catalog.items():
            by_id = {c.command_id: c for c in commands}
            if len(by_id) != len(commands):
                raise ValueError("Duplicate command ID")
            for command in commands:
                if not set(command.information_ids) <= ids:
                    raise ValueError(f"Undefined information: {command.command_id}")
            for information in self.information:
                for command_id in information.commands_by_os.get(os_name, []):
                    if command_id not in by_id or information.information_id not in by_id[command_id].information_ids:
                        raise ValueError(f"Invalid information-command mapping: {information.information_id}")
        self.servers: dict[str, ServerScenario] = {}
        for path in sorted(directory.glob("server-*.json")):
            server = ServerScenario.model_validate_json(path.read_text())
            if server.server_id in self.servers:
                raise ValueError(f"Duplicate server ID: {server.server_id}")
            supported = {c.command_id for c in self.catalog[server.os]}
            if set(server.commands) != supported:
                raise ValueError(f"Command catalog mismatch: {path.name}")
            if len(server.commands["tail"].stdout.splitlines()) > 100:
                raise ValueError(f"Tail scenario exceeds 100 lines: {path.name}")
            if server.observed_at.utcoffset() is None:
                raise ValueError(f"Timezone required: {path.name}")
            self.servers[server.server_id] = server
        if not self.servers:
            raise ValueError("No monitoring server scenarios found")

    def server(self, server_id: str) -> ServerScenario:
        # Look up registered IDs, never turn client input into a file path.
        try:
            return self.servers[server_id]
        except KeyError:
            raise HTTPException(404, detail={
                "code": "server_not_found", "server_id": server_id,
                "message": "등록되지 않은 서버입니다.",
            }) from None

    def server_list(self) -> list[ServerInfo]:
        return [ServerInfo.model_validate(s.model_dump()) for s in self.servers.values()]

    def commands(self, server_id: str) -> list[CommandDefinition]:
        return self.catalog[self.server(server_id).os]

    def result(self, request: CommandRequest) -> CommandResult:
        server = self.server(request.server_id)
        # Accept only a catalog ID or its complete, exact command string.
        definition = next((c for c in self.catalog[server.os]
                           if request.command in (c.command_id, c.command)), None)
        if definition is None:
            raise HTTPException(422, detail={
                "code": "unsupported_command", "server_id": server.server_id,
                "command": request.command,
                "message": "해당 OS에 등록된 명령 ID 또는 정확한 명령문을 사용하세요.",
                "supported_commands": [c.command_id for c in self.catalog[server.os]],
            })
        return CommandResult(
            server_id=server.server_id, os=server.os,
            command_id=definition.command_id, command=definition.command,
            observed_at=server.observed_at,
            **server.commands[definition.command_id].model_dump(),
        )


def create_router(store: ScenarioStore | None = None) -> APIRouter:
    store = store if store is not None else ScenarioStore()
    router = APIRouter(prefix="/monitoring", tags=["Local command scenarios"])

    @router.get("/information", response_model=list[InformationDefinition])
    async def list_information():
        """확인 가능한 서버 정보, 단위, OS별 명령과 한계를 제공합니다."""
        return store.information

    @router.get("/servers", response_model=list[ServerInfo])
    async def list_servers():
        """서버 목록. 시나리오 정답이나 진단 라벨은 제공하지 않습니다."""
        return store.server_list()

    @router.get("/servers/{server_id}/commands", response_model=list[CommandDefinition])
    async def list_commands(server_id: str):
        """OS별 지원 명령 ID, 실제 명령문, 조회 목적."""
        return store.commands(server_id)

    @router.post("/commands", response_model=CommandResult)
    async def command_result(request: CommandRequest):
        """셸 실행 없이 파일에 저장된 stdout/stderr/exit_code를 반환합니다."""
        return store.result(request)

    return router
