"""자막 저장 폴더에 남기는 작업 로그: 어떤 파일을 어떤 작업으로 처리했는지 기록한다.

대사 내용은 쓰지 않고 파일 이름, 작업 종류, 결과, 줄 수 같은 요약만 적는다.
실행할 때마다 같은 파일 끝에 이어서 쓰고, 한 줄씩 바로 저장하므로 도중에 멈추거나
크래시가 나도 그때까지의 기록이 남는다. 로그를 쓰다 실패해도 본 작업은 멈추지 않는다.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_FILENAME = "Subtitle_Tool_작업로그.txt"


class WorkLog:
    def __init__(self, out_dir: str | Path):
        self.path = Path(out_dir) / LOG_FILENAME
        self._t0 = time.monotonic()
        self._counts: dict[str, int] = {}

    def _write(self, text: str) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fresh = not self.path.exists() or self.path.stat().st_size == 0
            # BOM은 파일 맨 앞에 한 번만 넣는다 (이어 쓸 때마다 넣으면 중간에 깨진 글자가 생김)
            with open(self.path, "a", encoding="utf-8-sig" if fresh else "utf-8", newline="\n") as f:
                f.write(text if text.endswith("\n") else text + "\n")
        except OSError:
            logger.warning("작업 로그를 쓰지 못함: %s", self.path, exc_info=True)

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def begin(self, title: str, details: list[str] | None = None) -> None:
        self._t0 = time.monotonic()
        self._counts = {}
        lines = ["", "=" * 70, f"[{self._now()}] {title}"]
        lines += [f"    {d}" for d in details or []]
        lines.append("=" * 70)
        self._write("\n".join(lines))

    def event(self, kind: str, name: str, status: str, detail: str = "") -> None:
        """kind: 작업 종류(음성 인식, 번역, 자막 저장, smi 변환 ...), name: 파일 이름, status: 완료/실패/..."""
        self._counts[status] = self._counts.get(status, 0) + 1
        tail = f" - {detail}" if detail else ""
        self._write(f"[{self._now()}] [{kind}] {name} : {status}{tail}")

    def note(self, text: str) -> None:
        self._write(f"[{self._now()}] {text}")

    def end(self, title: str = "작업 끝") -> None:
        sec = int(time.monotonic() - self._t0)
        counts = ", ".join(f"{k} {v}건" for k, v in self._counts.items()) or "기록된 작업 없음"
        self._write(f"[{self._now()}] {title} (소요 {sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}) - {counts}")
