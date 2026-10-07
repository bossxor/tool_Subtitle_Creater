"""Windows에서 자식 프로세스(ffmpeg, llama-server, powershell 등)를 띄울 때
검은 콘솔 창이 깜빡이지 않게 하는 공용 플래그.

우리 앱 자체는 --windowed(콘솔 없음)로 빌드하지만, ffmpeg.exe/llama-server.exe/
powershell.exe는 콘솔 서브시스템 프로그램이라 CREATE_NO_WINDOW 없이 그냥 실행하면
Windows가 매번 새 콘솔 창을 띄운다.
"""
from __future__ import annotations

import subprocess

NO_WINDOW_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)
