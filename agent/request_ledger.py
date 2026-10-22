"""Append-only requirement identities and durable, stage-level gap records.

A command completing is not the same as the original request being answered.
Only final validation can mark a requirement satisfied. Failed/unsupported
requirements remain visible to the writer, validator, API and user.
"""
from copy import deepcopy

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal


class RequirementSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=1, max_length=300)
    server_ids: list[str] = Field(default_factory=list, max_length=8)
    information_ids: list[str] = Field(default_factory=list, max_length=10)
    kind: Literal["lookup", "calculation", "comparison", "diagnosis", "explanation", "remediation"] = "lookup"
    action_condition: Literal["always", "if_needed"] = "always"
    calculation_fingerprint: str | None = Field(default=None, exclude=True, max_length=64)

    @model_validator(mode="after")
    def target_required(self):
        if not self.server_ids and self.kind != "remediation":
            raise ValueError("서버 없는 요구는 OS가 확인된 매뉴얼 안내에만 허용됩니다.")
        return self


class RequestLedger:
    def __init__(self, payload=None):
        self.requirements = deepcopy((payload or {}).get("requirements", []))
        self.issues = deepcopy((payload or {}).get("requirement_issues", []))
        self.calculation_rejections = deepcopy((payload or {}).get("calculation_rejections", []))
        self.repairable_calculation_requirements = set()

    def add(self, specs, targets, stage):
        for spec in specs:
            data = spec.model_dump()
            if not set(data["server_ids"]) <= set(targets):
                raise ValueError("요구사항 대상이 원래 조회 대상과 다릅니다.")
            data["server_ids"] = sorted(set(data["server_ids"]))
            data["information_ids"] = sorted(set(data["information_ids"]))
            if stage == "final_validation" and data["kind"] in {"calculation", "comparison"} and spec.calculation_fingerprint:
                # The fingerprint already binds the verified operation and resolved
                # operand sources. Declared scope cannot revive that same operation.
                blocked = [r for r in self.requirements if self.is_calculation_blocked(r["requirement_id"], spec.calculation_fingerprint)]
                if blocked:
                    self.note(stage, "이미 집계 기준 불일치로 거절한 동일 지표의 계산 요구가 재등록되었습니다.", [r["requirement_id"] for r in blocked])
                    continue
            if any(all(r[k] == data[k] for k in data) for r in self.requirements):
                continue
            if len(self.requirements) >= 32:
                raise ValueError("요구사항 기록 한도에 도달했습니다.")
            self.requirements.append({**data, "requirement_id": f"r{len(self.requirements)+1}",
                                      "status": "pending", "reason": "아직 답변 충족 여부를 확인하지 않았습니다.",
                                      "history": [{"stage": stage, "status": "pending", "reason": "요구사항 등록"}]})

    def note(self, stage, reason, requirement_ids=None):
        ids = sorted(set(requirement_ids or []))
        existing = next((x for x in self.issues if x["stage"] == stage and x["reason"] == reason
                         and x["requirement_ids"] == ids and not x["resolved"]), None)
        if existing:
            return existing
        issue = {"issue_id": f"g{len(self.issues)+1}", "stage": stage, "reason": reason,
                 "requirement_ids": ids, "resolved": False, "resolution": None}
        self.issues.append(issue)
        return issue

    def update(self, requirement, status, reason, stage):
        if (requirement["status"], requirement["reason"]) != (status, reason):
            requirement.update(status=status, reason=reason)
            requirement["history"].append({"stage": stage, "status": status, "reason": reason})

    def record_calculation_rejection(self, rejection: dict) -> None:
        record = deepcopy(rejection)
        previous = next((r for r in self.calculation_rejections if r["fingerprint"] == record["fingerprint"]), None)
        if previous:
            previous["requirement_ids"] = sorted(set(previous["requirement_ids"]) | set(record["requirement_ids"]))
        else:
            self.calculation_rejections.append(record)
        for r in self.requirements:
            if r["requirement_id"] in record["requirement_ids"]:
                self.repairable_calculation_requirements.discard(r["requirement_id"])
                self.update(r, "unsupported", record["reason"], "calculation")
                self.note("calculation", record["reason"], [r["requirement_id"]])

    def allow_calculation_repair(self, requirement_id: str) -> None:
        self.repairable_calculation_requirements.add(requirement_id)
        requirement=next(r for r in self.requirements if r["requirement_id"]==requirement_id)
        self.update(requirement,"needs_more","거절된 연산의 피연산자가 실제 요청과 달라 다른 계산으로 보완합니다.","calculation_binding_review")

    def is_calculation_blocked(self, requirement_id: str, fingerprint: str | None = None) -> bool:
        if fingerprint is None and requirement_id in self.repairable_calculation_requirements:return False
        return any(requirement_id in r["requirement_ids"] and not r["repairable"]
                   and (fingerprint is None or r["fingerprint"] == fingerprint) for r in self.calculation_rejections)

    def sync(self, tasks, evidence, stage):
        have = {(e["server_id"], f["information_id"]) for e in evidence for f in e.get("facts", [])}
        for r in self.requirements:
            if r["kind"] == "remediation" or self.is_calculation_blocked(r["requirement_id"]):
                # Observation readiness is not manual-answer readiness.
                continue
            needed = {(s, i) for s in r["server_ids"] for i in r["information_ids"]}
            missing = needed - have
            states = [t.get("information_status", {}).get(i, "pending") for s, i in missing
                      for t in tasks if t["server_id"] == s and i in t["information_ids"]]
            if not missing and needed:
                self.update(r, "information_ready", "정보는 확보했습니다. 최종 답변의 요구 충족 여부는 별도 검증합니다.", stage)
            elif "failed" in states:
                self.update(r, "failed", "필요 정보 조회가 실패했습니다.", stage)
            elif "unsupported" in states:
                self.update(r, "unsupported", "요청 범위 중 현재 지원하지 않는 정보가 있습니다.", stage)
            else:
                self.update(r, "needs_more", "필요 정보 또는 요청한 계산·설명의 충족 확인이 남았습니다.", stage)
            if missing:
                self.note(stage, r["reason"] + " " + ", ".join(f"{s}:{i}" for s, i in sorted(missing)), [r["requirement_id"]])

    def apply_checks(self, checks, stage):
        by_id = {r["requirement_id"]: r for r in self.requirements}
        for check in checks:
            r = by_id[check.requirement_id]
            if self.is_calculation_blocked(r["requirement_id"]):
                r["review_reason"] = check.reason
                continue
            reason = ("최종 검토에서 요청한 답변의 충족을 확인했습니다." if check.status == "satisfied" else check.reason)
            self.update(r, "satisfied" if check.status == "satisfied" else
                        (r["status"] if check.status == "unavailable" else "needs_more"), reason, stage)
            # Preserve the model's literal reason separately from the user-facing state label.
            r["review_reason"] = check.reason
        satisfied = {r["requirement_id"] for r in self.requirements if r["status"] == "satisfied"}
        for issue in self.issues:
            if issue["requirement_ids"] and set(issue["requirement_ids"]) <= satisfied:
                issue.update(resolved=True, resolution=f"{stage}: 최종 답변에서 요구 충족 확인")

    def payload(self):
        return {"requirements": deepcopy(self.requirements), "requirement_issues": deepcopy(self.issues)}
