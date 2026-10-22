"""Reviewed Korean paraphrases, each anchored to an exact original-source span.

The model selects applicable steps; it cannot invent a shell command, numerical
threshold, or action. These are answer annotations, never indexed translations.
"""

ACTIONS = [
    ("linux-cpu-limit", "kernel-cpu-control", "The maximum bandwidth limit.", "특정 작업의 CPU 독점을 제한해야 한다면 cgroup v2의 cpu.max로 해당 그룹의 CPU 시간 상한 설정을 검토합니다.", "cgroup v2·CPU 컨트롤러와 적용 가능한 스케줄러, 루트가 아닌 대상 그룹·작업 소유자·권한을 먼저 확인; 처리량 감소와 지연 증가를 검토하고 운영자가 설정; 수치·경로를 임의 지정하거나 실행하지 않음", "configuration_change"),
    ("linux-cpu-idle", "kernel-cpu-control", "the cgroup itself will be treated as very low priority relative to its peers.", "다른 작업보다 우선순위가 낮아도 되는 비필수 작업은 cgroup v2의 cpu.idle 적용을 검토합니다.", "루트가 아닌 대상 그룹·CPU 컨트롤러·적용 스케줄러와 비필수 작업 여부를 확인; 지연에 민감한 서비스에는 적용하지 않으며 실행 지연 영향을 확인한 운영자만 설정", "configuration_change"),
    ("linux-memory-high", "kernel-memory-control", "the processes of the cgroup are throttled and put under heavy reclaim pressure.", "특정 작업의 메모리 사용을 제어해야 한다면 cgroup v2의 memory.high 설정을 검토합니다. 초과 시 작업을 제한하고 회수를 유도하는 방식이며 메모리 부족의 원인을 해결했다고 볼 수는 없습니다.", "cgroup v2·메모리 컨트롤러·루트가 아닌 대상 그룹·권한 확인; 외부 감시로 회수 압박을 완화할 운영 체계가 있어야 함; 지연과 과도한 회수 가능성, 극단 상황에서 한도 초과 가능성 확인; 임계값을 임의 지정하거나 실행하지 않음", "configuration_change"),
    ("mac-unused-quit", "apple-slow", "Quit any apps you’re not using.", "사용하지 않는 앱을 종료해 불필요한 자원 점유를 줄이는 방안을 검토합니다.", "업무에 필요하지 않은 앱인지, 저장하지 않은 작업·다른 서비스 의존성과 종료 영향을 확인한 뒤 운영자가 일반 종료; 자동 실행하지 않음", "process_stop"),
    ("linux-rotation", "logrotate", "It allows automatic rotation, compression,", "로그가 큰 경우 logrotate의 순환·압축 정책 적용을 검토합니다.", "설치 여부, 애플리케이션 로그 재열기 방식, 보존 정책과 영향 확인 후 운영자가 적용", "configuration_change"),
    ("linux-rotation-period", "logrotate", "Each log file may be handled daily,", "로그 생성량에 맞게 일별·주별·월별 또는 크기 기준의 순환 조건을 검토합니다.", "현재 설정·보존 요건을 먼저 확인; 설정 변경은 자동 실행하지 않음", "configuration_change"),
    ("mac-quit", "apple-quit", "The process quits when it’s safe to do so.", "원인이 의심되는 앱을 식별했다면 일반 종료를 먼저 검토합니다.", "앱 소유자·서비스 의존성·저장하지 않은 작업과 종료 영향 확인; 자동 종료하지 않음", "process_stop"),
    ("mac-force-quit", "apple-quit", "If the process has files open, you may lose data.", "응답하지 않는 앱의 강제 종료는 데이터 손실과 의존 프로세스 영향을 확인한 뒤 검토합니다.", "일반 종료가 어려운 경우에 한하며 운영자 판단 필요; 자동 실행하지 않음", "process_stop"),
    ("mac-move", "apple-storage", "you can move files to that device.", "다른 저장 장치로 옮길 수 있는 파일을 검토해 공간을 확보합니다.", "대상 파일의 소유자·업무 의존성·백업·이동 대상 여유 공간 확인 후 수동 수행", "file_change"),
    ("mac-compress", "apple-storage", "save space by compressing them", "삭제하면 안 되는 파일은 압축으로 공간을 줄일 수 있는지 검토합니다.", "애플리케이션이 사용하는 파일은 영향 확인; 자동 압축하지 않음", "file_change"),
    ("mac-trash", "apple-storage", "its storage space doesn't become available until you empty the Trash", "휴지통으로 이동한 파일은 휴지통을 비워야 공간이 반환되는지 확인합니다.", "보존·복구 필요성과 백업을 확인한 뒤 수동 처리; 자동 삭제하지 않음", "file_delete"),
]


def candidates(matches):
    result, seen = [], set()
    for match in matches:
        for action_id, doc_id, quote, text, condition, risk in ACTIONS:
            if doc_id != match["doc_id"] or quote not in match["text"] or action_id in seen:
                continue
            seen.add(action_id)
            if action_id.startswith("linux-cpu"):
                types=["cpu-spike"]
            elif action_id == "linux-memory-high":
                types=["memory-pressure"]
            elif action_id in {"mac-quit", "mac-force-quit", "mac-unused-quit"}:
                types=["cpu-spike", "memory-pressure"]
            else:
                types=["disk-full"]
            result.append({"action_id":action_id, "text":text, "precondition":condition, "risk":risk,"incident_types":types,
                           "chunk_id":match["chunk_id"], "quote":quote, "doc_id":doc_id})
    return result
