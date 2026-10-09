"""llama.cpp 서버(llama-server.exe)를 자식 프로세스로 띄우고 chat completion을 호출한다.

Qwen3 계열은 기본적으로 내부 사고(thinking) 토큰을 생성해 느려지므로,
항상 enable_thinking=False로 호출한다 (Phase 0 벤치마크에서 3.6배 단축 확인, 품질 차이 없음).
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

from core.procutil import NO_WINDOW_FLAGS


class LlamaServerError(RuntimeError):
    pass


class LlamaServer:
    def __init__(
        self,
        server_exe: Path,
        model_path: Path,
        port: int = 8090,
        n_gpu_layers: int = 999,
        ctx_size: int = 8192,
        log_path: Optional[Path] = None,
    ):
        self.server_exe = Path(server_exe)
        self.model_path = Path(model_path)
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.log_path = Path(log_path) if log_path else None
        self.proc: Optional[subprocess.Popen] = None
        self._log_file = None  # 서버 로그를 받는 파일 핸들. stop()에서 닫지 않으면 파일이 계속 잠긴다

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def is_alive(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def start(self, timeout: float = 180.0) -> None:
        if self.is_alive():
            return  # 이미 떠 있는 서버 재사용 (다른 곳에서 띄웠거나 이전 실행이 안 죽은 경우)

        if not self.server_exe.exists():
            raise LlamaServerError(f"llama-server 실행파일을 찾을 수 없음: {self.server_exe}")
        if not self.model_path.exists():
            raise LlamaServerError(f"LLM 모델 파일을 찾을 수 없음: {self.model_path}")

        cmd = [
            str(self.server_exe),
            "-m",
            str(self.model_path),
            "-ngl",
            str(self.n_gpu_layers),
            "-c",
            str(self.ctx_size),
            "--port",
            str(self.port),
        ]
        self._log_file = open(self.log_path, "w", encoding="utf-8") if self.log_path else None
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._log_file or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            creationflags=NO_WINDOW_FLAGS,
        )

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                code = self.proc.returncode
                self.proc = None
                self._close_log()
                raise LlamaServerError(f"llama-server가 시작 중 종료됨 (exit={code}). 로그: {self.log_path}")
            if self.is_alive():
                return
            time.sleep(0.5)
        self.stop()
        raise LlamaServerError(f"llama-server가 {timeout}초 안에 준비되지 않음. 로그: {self.log_path}")

    def stop(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
            self.proc = None
        self._close_log()

    def _close_log(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        timeout: float = 300.0,
    ) -> str:
        r = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
