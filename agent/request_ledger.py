"""Append-only requirement identities and durable, stage-level gap records.

A command completing is not the same as the original request being answered.
Only final validation can mark a requirement satisfied. Failed/unsupported
requirements remain visible to the writer, validator, API and user.
"""
from copy import deepcopy

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class RequirementSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=1, max_length=300)
    server_ids: list[str] = Field(min_length=1, max_length=8)
    information_ids: list[str] = Field(default_factory=list, max_length=10)
    kind: Literal["lookup", "calculation", "comparison", "diagnosis", "explanation"] = "lookup"


class RequestLedger:
    def __init__(self):
        self.requirements = []
        self.issues = []

    def add(self, specs, targets, stage):
        for spec in specs:
            data = spec.model_dump()
            if not set(data["server_ids"]) <= set(targets):
                raise ValueError("요구사항 대상이 원래 조회 대상과 다릅니다.")
            data["server_ids"] = sorted(set(data["server_ids"]))
            data["information_ids"] = sorted(set(data["information_ids"]))
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

    def sync(self, tasks, evidence, stage):
        have = {(e["server_id"], f["information_id"]) for e in evidence for f in e.get("facts", [])}
        for r in self.requirements:
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
