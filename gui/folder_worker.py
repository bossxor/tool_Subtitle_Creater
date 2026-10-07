"""'폴더 추가' 정리 작업(smi 변환+AI 검수, 인코딩 수정)을 GUI와 분리된 스레드에서 돌린다."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from core.errors import PipelineCancelled
from core.folder_scan import FolderScanPlan, run_conversions


class FolderCleanupWorker(QThread):
    progress = Signal(str)
    finished_ok = Signal(int, int, int)  # converted, fixed, deleted
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, plan: FolderScanPlan, formats: list[str], config, delete_old_smi: bool, parent=None):
        super().__init__(parent)
        self.plan = plan
        self.formats = formats
        self.config = config
        self.delete_old_smi = delete_old_smi
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            converted, fixed, deleted = run_conversions(
                self.plan,
                self.formats,
                self.config,
                self.delete_old_smi,
                progress_cb=self.progress.emit,
                cancel_check=lambda: self._cancel_requested,
            )
            self.finished_ok.emit(converted, fixed, deleted)
        except PipelineCancelled:
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
