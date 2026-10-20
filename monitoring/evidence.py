"""Conservative extraction for the registered command formats, not arbitrary shells.

Facts carry source quotes. Missing fields remain missing, and nonzero exit codes
never become healthy values. Numeric conversions are explicit in summaries.
"""
import re


def _last(pattern, text, flags=0):
    matches = list(re.finditer(pattern, text, flags))
    return matches[-1] if matches else None


def extract_facts(result: dict) -> list[dict]:
    if result.get("exit_code") != 0 or not result.get("stdout", "").strip():
        return []
    text = result["stdout"]
    command, os_name = result["command_id"], result["os"]
    facts = []

    def add(info, fields, summary, quote):
        facts.append({"information_id": info, "fields": fields,
                      "summary": summary, "quote": quote})

    if command == "uname":
        identity = re.match(r"^(Linux|Darwin)\s+\S+\s+(\d+\.\S+)\s+.+\b(x86_64|aarch64|arm64|i[3-6]86)\b", text)
        if identity:
            add("system", {"os": identity[1], "kernel": identity[2], "architecture": identity[3]},
                text.strip(), text.strip())

    if command == "top":
        cpu_line = _last(r"^(?:%Cpu\(s\):|CPU usage:).+$", text, re.M)
        load = _last(r"(?:load average:|Load Avg:)\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)", text)
        if cpu_line:
            quote = cpu_line[0]
            if os_name == "linux":
                values = [re.search(rf"([\d.]+)\s+{label}\b", quote) for label in ("us", "sy", "id")]
            else:
                values = [re.search(rf"([\d.]+)%\s+{label}\b", quote) for label in ("user", "sys", "idle")]
            if all(values) and load:
                user, system, idle = [float(v[1]) for v in values]
                if all(0 <= n <= 100 for n in (user, system, idle)) and user + system + idle <= 100.2:
                    fields = {"user_percent": user, "system_percent": system, "idle_percent": idle,
                              "usage_percent": round(100 - idle, 1),
                              "load_average": [float(v) for v in load.groups()]}
                    add("cpu", fields,
                        f"시스템 CPU 사용률 {fields['usage_percent']:.1f}% "
                        f"(100 - 유휴 {idle:.1f}%), 사용자 {user:.1f}%, 커널 {system:.1f}%.",
                        quote + "\n" + load[0])
        if os_name == "macos":
            mem = _last(r"PhysMem:\s*([\d.]+)([MG]) used.*?,\s*([\d.]+)([MG]) unused", text)
            if mem:
                used = float(mem[1]) * (1024 if mem[2] == "G" else 1)
                unused = float(mem[3]) * (1024 if mem[4] == "G" else 1)
                add("memory.summary", {"used_mib": used, "unused_mib": unused},
                    f"물리 메모리 사용 {used:g} MiB, 미사용 {unused:g} MiB. "
                    "미사용량은 Linux available과 다른 지표입니다.", mem[0])
        elif os_name == "linux":
            mem = _last(r"MiB Mem\s*:\s*([\d.]+) total,\s*([\d.]+) free,\s*([\d.]+) used", text)
            available = _last(r"([\d.]+) avail Mem", text)
            if mem and available:
                total, free, used = map(float, mem.groups())
                add("memory.summary", {"total_mib": total, "used_mib": used,
                                       "free_mib": free, "available_mib": float(available[1])},
                    f"메모리 전체 {total:g} MiB, 사용 {used:g} MiB, free {free:g} MiB, "
                    f"available {float(available[1]):g} MiB.", mem[0] + "\n" + available[0])

    if command == "free":
        mem = re.search(r"^Mem:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", text, re.M)
        swap = re.search(r"^Swap:\s+(\d+)\s+(\d+)\s+(\d+)\s*$", text, re.M)
        if mem:
            total, used, free, shared, cache, available = map(int, mem.groups())
            add("memory.summary", {"total_mib": total, "used_mib": used,
                                   "free_mib": free, "available_mib": available},
                f"메모리 전체 {total} MiB, 사용 {used} MiB, free {free} MiB, available {available} MiB.", mem[0])
        if swap:
            total, used, free = map(int, swap.groups())
            add("memory.detail", {"swap_total_mib": total, "swap_used_mib": used, "swap_free_mib": free},
                f"Swap 전체 {total} MiB, 사용 {used} MiB, 여유 {free} MiB.", swap[0])

    if command == "vm_stat":
        page = re.search(r"page size of (\d+) bytes", text)
        labels = {"free_pages": "Pages free", "active_pages": "Pages active",
                  "wired_pages": "Pages wired down", "compressor_pages": "Pages occupied by compressor"}
        values = {key: re.search(rf"^{label}:\s*(\d+)\.", text, re.M) for key, label in labels.items()}
        if page and all(values.values()):
            fields = {"page_size_bytes": int(page[1]), **{k: int(v[1]) for k, v in values.items()}}
            add("memory.detail", fields,
                f"페이지 크기 {fields['page_size_bytes']} bytes, free {fields['free_pages']} pages, "
                f"active {fields['active_pages']} pages, wired {fields['wired_pages']} pages, "
                f"compressor {fields['compressor_pages']} pages.", text.strip())

    if command == "ps":
        entries, lines = [], []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 6 and parts[0].isdigit() and parts[1].isdigit():
                try:
                    entries.append(dict(pid=int(parts[0]), ppid=int(parts[1]), name=parts[2],
                                        cpu_percent=float(parts[3]), memory_percent=float(parts[4]), rss_kib=int(parts[5])))
                    lines.append(line.strip())
                except ValueError:
                    continue
        if entries:
            summaries = [f"PID {p['pid']} {p['name']}: CPU {p['cpu_percent']:g}% "
                         f"(프로세스 코어 합산), 메모리 {p['memory_percent']:g}%, RSS {p['rss_kib']} KiB."
                         for p in entries]
            add("process", {"entries": entries}, "\n".join(summaries), "\n".join(lines))

    if command == "df":
        entries, lines = [], []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 6 and parts[4].endswith("%"):
                try:
                    entries.append(dict(filesystem=parts[0], total=parts[1], used=parts[2],
                                        available=parts[3], usage_percent=int(parts[4][:-1]), mount=parts[5]))
                    lines.append(line)
                except ValueError:
                    continue
        if entries:
            add("disk", {"entries": entries}, "\n".join(
                f"{p['mount']}: 전체 {p['total']}, 사용 {p['used']}, 여유 {p['available']}, 사용률 {p['usage_percent']}%."
                for p in entries), "\n".join(lines))

    if command in {"ss", "lsof"}:
        lines = text.strip().splitlines()
        if lines and (lines[0].startswith("Netid") or lines[0].startswith("COMMAND")):
            add("network", {"entries": lines[1:]},
                "제공된 소켓·포트 출력:\n" + "\n".join(lines[1:]) +
                "\n이 출력만으로 응답 시간·패킷 손실·외부 서비스 정상 여부는 확인할 수 없습니다.", text.strip())
    if command == "ls" and text.startswith("total "):
        lines = [line for line in text.splitlines() if re.match(r"^-[rwxstST-]{9}\s+\d+\s+\S+\s+\S+\s+\S+\s+\S+\s+\d+\s+\S+\s+.+$", line)]
        if lines or len(text.strip().splitlines()) == 1:
            add("log.files", {"entries": lines}, "등록된 로그 파일 목록:\n" + "\n".join(lines), text.strip())
    if command == "du":
        size = re.fullmatch(r"(\d+)\s+(/var/log/)\s*", text)
        if size:
            add("log.usage", {"size_kib": int(size[1]), "path": size[2]},
                f"{size[2]} 사용량 {size[1]} KiB.", size[0].strip())
    if command == "tail":
        lines = text.strip().splitlines()
        if len(lines) <= 100 and all(re.match(r"^\d{4}-\d{2}-\d{2}T\S+\s+(INFO|WARN|ERROR|DEBUG)\b", line) for line in lines):
            add("log.content", {"lines": lines}, "최근 애플리케이션 로그 발췌:\n" + "\n".join(lines), text.strip())
    return facts
