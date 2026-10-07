"""앱 시작 시 이전 실행에서 남았을 수 있는 llama-server.exe 고아 프로세스를 정리한다.

정상 종료(창 닫기)는 gui/main_window.py의 closeEvent가 처리하지만, 강제 종료나 크래시가
나면 llama-server.exe 자식 프로세스가 남을 수 있다 (Phase 2 테스트 중 실제로 관찰됨).
포트가 아니라 "우리 tools/llama.cpp 폴더에서 실행된 llama-server.exe인가"로 판별해서,
사용자가 별도로 띄워둔 다른 llama-server와 혼동하지 않게 한다.
"""
from __future__ import annotations

import logging
import subprocess

from core.procutil import NO_WINDOW_FLAGS

logger = logging.getLogger(__name__)


def kill_orphan_llama_server(config) -> int:
    """우리 exe 경로와 일치하는 llama-server.exe 프로세스를 찾아 종료한다.

    Windows 전용(PowerShell). 실패해도 앱 시작을 막지 않도록 예외를 모두 삼킨다.
    반환값: 종료시킨 프로세스 수.
    """
    try:
        server_exe = str(config.resolve_path("llm.server_exe")).lower()
    except Exception:
        return 0

    ps_script = (
        "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" "
        "| Select-Object ProcessId, ExecutablePath | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=NO_WINDOW_FLAGS,
        )
    except Exception:
        logger.debug("잔여 llama-server 조회 실패 (무시하고 계속 진행)", exc_info=True)
        return 0

    if result.returncode != 0 or not result.stdout.strip():
        return 0

    import json

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return 0
    if isinstance(data, dict):
        data = [data]

    killed = 0
    for proc in data:
        exe_path = (proc.get("ExecutablePath") or "").lower()
        pid = proc.get("ProcessId")
        if not exe_path or not pid or exe_path != server_exe:
            continue
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", f"Stop-Process -Id {pid} -Force"],
                capture_output=True,
                timeout=10,
                creationflags=NO_WINDOW_FLAGS,
            )
            logger.info("이전 실행에서 남은 llama-server.exe(pid=%s) 정리함", pid)
            killed += 1
        except Exception:
            logger.debug("llama-server 잔여 프로세스(pid=%s) 종료 실패", pid, exc_info=True)
    return killed
