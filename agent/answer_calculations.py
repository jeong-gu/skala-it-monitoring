"""No model arithmetic or shell/eval: compute only referenced numeric fields."""
from decimal import Decimal, InvalidOperation

MEMORY = {"B": Decimal(1), "KiB": Decimal(1024), "MiB": Decimal(1024**2), "GiB": Decimal(1024**3)}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("계산 피연산자는 조회된 숫자 필드여야 합니다.")
    n = Decimal(str(value))
    if not n.is_finite():
        raise ValueError("유한한 수치만 계산할 수 있습니다.")
    return n


def calculate(spec, evidence, requirement_ids):
    if not set(spec.requirement_ids) <= set(requirement_ids):
        raise ValueError("계산에 알 수 없는 요구사항 ID가 있습니다.")
    inputs = []
    for ref in spec.operands:
        source = next((e for e in evidence if e["evidence_id"] == ref.evidence_id), None)
        fact = next((f for f in (source or {}).get("facts", []) if f["information_id"] == ref.information_id), None)
        if not fact:
            raise ValueError("계산의 원본 정보를 확보하지 못했습니다.")
        value = fact["fields"]
        try:
            for segment in ref.field.split("."):
                value = value[int(segment)] if isinstance(value, list) else value[segment]
        except (KeyError, IndexError, TypeError, ValueError):
            raise ValueError("계산의 원본 필드가 없습니다.") from None
        leaf = ref.field.split(".")[-1]
        unit = ("B/page" if leaf == "page_size_bytes" else
                next((u for suffix, u in [("_mib", "MiB"), ("_kib", "KiB"), ("_bytes", "B"),
                                         ("_pages", "pages"), ("_percent", "%")]
                      if leaf.endswith(suffix)), "number"))
        inputs.append({**ref.model_dump(), "server_id": source["server_id"],
                       "value": str(_number(value)), "unit": unit})
    vals = [Decimal(i["value"]) for i in inputs]; units = [i["unit"] for i in inputs]
    # Normalize memory operands before operations; do not equate OS-specific meanings.
    if all(u in MEMORY for u in units):
        vals = [v * MEMORY[u] for v, u in zip(vals, units)];units = ["B"] * len(units)
    if spec.operation in {"sum", "difference", "ratio", "percent"} and len(set(units)) != 1:
        raise ValueError("단위가 다른 필드는 직접 계산할 수 없습니다.")
    if spec.operation in {"difference", "ratio", "percent"} and len(vals) != 2:
        raise ValueError("차이·비율 계산에는 두 피연산자가 필요합니다.")
    unit = units[0]
    try:
        if spec.operation == "sum":result = sum(vals)
        elif spec.operation == "difference":
            result = vals[0] - vals[1]
            if unit == "%":unit = "percentage_points"
        elif spec.operation in {"ratio", "percent"}:
            if not vals[1]:raise ValueError("0으로 나눌 수 없습니다.")
            result = vals[0] / vals[1] * (100 if spec.operation == "percent" else 1)
            unit = "%" if spec.operation == "percent" else "ratio"
        elif spec.operation == "product":
            if len(vals) != 2 or sorted(units) != ["B/page", "pages"]:
                raise ValueError("곱셈은 페이지 수 × 실제 페이지 크기만 지원합니다.")
            result = vals[0] * vals[1];unit = "B"
    except (InvalidOperation, ZeroDivisionError):
        raise ValueError("계산을 수행할 수 없습니다.") from None
    output = unit if spec.output_unit == "native" else spec.output_unit
    if output in MEMORY and unit == "B":result /= MEMORY[output]
    elif output != unit:raise ValueError("요청한 출력 단위로 변환할 수 없습니다.")
    if not result.is_finite():raise ValueError("유한한 결과가 아닙니다.")
    symbol = {"sum":" + ","difference":" − ","product":" × ","ratio":" ÷ ","percent":" ÷ "}[spec.operation]
    equation = symbol.join(f"{i['value']} {i['unit']} ({i['evidence_id']}:{i['field']})" for i in inputs)
    if spec.operation == "percent":equation += " × 100"
    if output in MEMORY and unit == "B":equation += f"; {output} 환산은 bytes ÷ {MEMORY[output]}"
    return {"label":spec.label, "requirement_ids":spec.requirement_ids, "inputs":inputs,
            "value":str(result.normalize()), "unit":output, "equation":equation,
            "evidence_ids":list(dict.fromkeys(i["evidence_id"] for i in inputs))}
