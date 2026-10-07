"""파이프라인을 GUI와 분리된 스레드에서 돌리는 QThread 래퍼."""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from core.pipeline import PipelineCancelled, run_batch


class QtLogHandler(logging.Handler):
    """logging 레코드를 Qt 시그널로 중계한다 (다른 모듈의 logger.info 등도 GUI 로그창에 보이게)."""

    def __init__(self, emit_fn):
        super().__init__()
        self._emit_fn = emit_fn
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._emit_fn(self.format(record))
        except Exception:
            pass


class PipelineWorker(QThread):
    progress = Signal(str)
    progress_fraction = Signal(float)  # 0.0 ~ 1.0, 전체 진행률
    log = Signal(str)
    finished_ok = Signal(dict)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, videos, out_dir, config, glossary=None, parent=None):
        super().__init__(parent)
        self.videos = videos
        self.out_dir = out_dir
        self.config = config
        self.glossary = glossary
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True
        self.progress.emit("취소 요청됨. 현재 영상 처리가 끝나는 대로 멈춥니다...")

    def run(self) -> None:
        handler = QtLogHandler(self.log.emit)
        root_logger = logging.getLogger("core")
        root_logger.addHandler(handler)
        try:
            outputs = run_batch(
                self.videos,
                self.out_dir,
                self.config,
                glossary=self.glossary,
                progress_cb=self.progress.emit,
                cancel_check=lambda: self._cancel_requested,
                progress_fraction_cb=self.progress_fraction.emit,
            )
            self.finished_ok.emit({str(k): [str(p) for p in v] for k, v in outputs.items()})
        except PipelineCancelled:
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001 - GUI에 원인을 그대로 보여주기 위해 넓게 잡음
            logging.getLogger(__name__).exception("파이프라인 실행 중 오류")
            self.failed.emit(str(e))
        finally:
            root_logger.removeHandler(handler)
