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


def causal_factors(evidence):
    """Observed contributors only; abnormality and applicability are separate.

    Reparse raw output so a stale/mutated facts dictionary cannot create factors.
    A log symptom without an explicit recorded reason is not a causal factor.
    """
    factors = []
    for e in evidence:
        if e.get("error") or e.get("exit_code") != 0:continue
        facts = extract_facts(e)
        eid = e["evidence_id"]
        def add(suffix, kind, text, quotes, **fields):
            factors.append(dict(factor_id=f"{eid}:{suffix}",kind=kind,text=text,
                server_id=e["server_id"],evidence_ids=[eid],quotes=quotes,**fields))
        for fact in facts:
            fields = fact["fields"]
            if fact["information_id"] == "process":
                rows = fact["quote"].splitlines()
                for i,entry in enumerate(fields["entries"]):
                    quote = rows[i]
                    if entry["cpu_percent"] > 0:
                        add(f"process:{i}:cpu", "process_cpu",
                            f"PID {entry['pid']} {entry['name']}의 CPU 사용률 {entry['cpu_percent']:g}%는 프로세스 코어 합산 값으로 시스템 전체 사용률과 기준이 다릅니다. 이 프로세스의 연산 작업은 관측된 CPU 사용에 기여할 수 있는 후보입니다.", [quote])
                    if entry["rss_kib"] > 0:
                        add(f"process:{i}:memory", "process_memory",
                            f"PID {entry['pid']} {entry['name']}의 메모리 점유율 {entry['memory_percent']:g}%, RSS {entry['rss_kib']} KiB가 관측됐습니다. 이 프로세스의 메모리 점유는 관측된 물리 메모리 사용에 기여할 수 있는 후보입니다.", [quote])
            elif fact["information_id"] == "log.usage" and fields["size_kib"] > 0:
                add("storage", "log_storage",
                    f"등록된 로그 디렉터리 {fields['path']}의 사용량 {fields['size_kib']} KiB가 관측됐습니다.",
                    [fact["quote"]], path=fields["path"])
            elif fact["information_id"] == "log.content":
                for i,line in enumerate(fields["lines"]):
                    # Explicit machine-written reason fields, not inferred keywords
                    # such as 'memory pressure' or 'timeout' standing alone.
                    if re.search(r"\b(?:WARN|ERROR)\b",line) and re.search(r"\breason=\S",line) and len(line) <= 220:
                        add(f"log:{i}:reason", "log_reason",
                            f"로그에 기록된 실패 이유는 ‘{line}’입니다. 이 기록은 관련 실패의 원인 조사 단서이며 독립적으로 확인한 근본 원인은 아닙니다.", [line])
    return factors


def verify_metric_values(text, evidence):
    """Check explicit metric/value phrases against reparsed native fields.

    This is a deterministic numeric-claim check, not a language intent router.
    Unlabelled qualitative prose still needs the final semantic validator.
    Calculated values are rendered separately from their checked operands and
    cannot authorize a different native CPU/memory/process metric here.
    """
    from decimal import Decimal
    sources=[]
    for source in evidence:
        if source.get('error') or source.get('exit_code') != 0:continue
        for fact in extract_facts(source):sources.append((source,fact))
    if not sources:return
    process_rows=[(s,row) for s,f in sources if f['information_id']=='process' for row in f['fields']['entries']]
    metric_names=re.compile(r'CPU|메모리|memory|RAM|RSS|디스크|disk|Swap',re.I)
    value_pattern=re.compile(r'(?<![\w.])(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|MiB|KiB|GiB)(?![A-Za-z])|(?<![\w.])(?P<percent>\d+(?:\.\d+)?)\s*%')
    descriptor=re.compile(r'(available|unused|free|total|used|idle|user|system|sys|전체|총|미사용|가용|여유|사용|유휴|사용자|커널|시스템)(?:\s*(?:용량|메모리|비율|사용률|점유율|수치|값|률|율|량))?\s*(?:은|는|이|가|:|=)?\s*$',re.I)
    scales={'MiB':Decimal(1),'KiB':Decimal(1)/1024,'GiB':Decimal(1024)}
    for match in value_pattern.finditer(text):
        prefix=text[:match.start()]
        # A new sentence has no implicit metric from an earlier sentence.
        prefix=re.split(r'(?<=[가-힣A-Za-z%])\.\s+|[\n;]',prefix)[-1]
        names=list(metric_names.finditer(prefix))
        if not names:continue
        label=names[-1];kind=label.group().lower();tail=prefix[label.end():]
        if len(tail)>100:continue
        value=Decimal(match['value'] or match['percent']);unit=match['unit'] or '%'
        desc=descriptor.search(prefix)
        desc=desc[1].lower() if desc else ''
        # A subfield can precede CPU ("사용자 CPU 사용률") as well as
        # follow it ("CPU 사용자 비율"). "시스템 CPU" still means total
        # host usage; only the explicit "CPU 시스템 비율" means kernel time.
        if kind == 'cpu' and desc in {'', '사용'}:
            before = re.search(r'(?<![A-Za-z가-힣])(사용자|커널|유휴|user|idle)\s*$', prefix[:label.start()], re.I)
            if before:desc=before[1].lower()
        named=[]
        for source,row in process_rows:
            positions=[m.start() for m in re.finditer(r'(?<![A-Za-z0-9_.-])'+re.escape(row['name'])+r'(?![A-Za-z0-9_.-])',prefix)]
            positions += [m.start() for m in re.finditer(r'\bPID\s*'+str(row['pid'])+r'(?!\d)',prefix,re.I)]
            if positions:named.append((max(positions),source,row))
        rows=[(s,r) for pos,s,r in named if pos==max((v[0] for v in named),default=-1)]
        # Explicit system CPU always means the host, even when a process was
        # mentioned elsewhere in the sentence.
        if kind=='cpu' and (re.search(r'(?:시스템|전체)\s*$',prefix[:label.start()]) or re.match(r'\s*(?:전체|시스템)\s*사용률',tail)):rows=[]
        iid=None;field=None;expected=[]
        if kind=='cpu' and unit=='%':
            if rows:
                iid='process';field='cpu_percent'
            else:
                iid='cpu';field={'유휴':'idle_percent','idle':'idle_percent','사용자':'user_percent','user':'user_percent','커널':'system_percent','시스템':'system_percent','system':'system_percent','sys':'system_percent'}.get(desc,'usage_percent')
        elif kind=='rss' and unit in scales:
            iid='process';field='rss_kib'
        elif kind in {'메모리','memory','ram'}:
            if unit=='%':
                iid='process';field='memory_percent'
                if not rows:
                    raise ValueError('시스템 메모리 비율은 원본 용량의 검증된 계산 결과로만 제공하세요. 프로세스 비율과 바꾸지 마세요.')
            elif unit in scales:
                iid='memory.summary';field={'전체':'total_mib','총':'total_mib','total':'total_mib','사용':'used_mib','used':'used_mib','free':'free_mib','available':'available_mib','가용':'available_mib','unused':'unused_mib','미사용':'unused_mib'}.get(desc)
                if field is None and re.search(r'사용\s*메모리',prefix,re.I):field='used_mib'
                if field is None:continue  # e.g. an unlabelled operand in a checked equation
        elif kind in {'디스크','disk'} and unit=='%':
            iid='disk';field='usage_percent'
        elif kind=='swap' and unit in scales:
            iid='memory.detail';field={'전체':'swap_total_mib','총':'swap_total_mib','total':'swap_total_mib','사용':'swap_used_mib','used':'swap_used_mib','여유':'swap_free_mib','free':'swap_free_mib'}.get(desc)
            if field is None:continue
        else:continue
        if iid=='process':
            if not rows:continue  # Process identity is required by the existing source/quote checker.
            expected=[Decimal(str(row[field])) for _,row in rows]
            if field=='rss_kib':expected=[v/1024/scales[unit] for v in expected]
        else:
            for source,fact in sources:
                if fact['information_id']!=iid:continue
                if iid=='disk':
                    entries=fact['fields']['entries']
                    mounts=[r for r in entries if re.search(r'(?<!\S)'+re.escape(r['mount'])+r'(?=\s|:|$)',prefix)]
                    expected += [Decimal(str(r[field])) for r in mounts or entries]
                elif field in fact['fields']:
                    v=Decimal(str(fact['fields'][field]))
                    expected.append(v/scales[unit] if unit in scales else v)
        if not expected or value not in expected:
            raise ValueError(f'지표와 수치가 일치하지 않습니다: {iid}.{field}={value} {unit}. 원문 값: '+', '.join(map(str,expected))+'. 해당 필드의 값만 인용하세요.')
