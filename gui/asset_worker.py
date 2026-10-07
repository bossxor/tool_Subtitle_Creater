"""첫 실행 때 필요한 모델/런타임을 내려받는 QThread 래퍼."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from core.errors import PipelineCancelled
from core.setup_assets import AssetTask, download_all


class AssetDownloadWorker(QThread):
    progress = Signal(str, int, int)  # label, downloaded_bytes, total_bytes(0=알 수 없음)
    finished_ok = Signal()
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, tasks: list[AssetTask], parent=None):
        super().__init__(parent)
        self.tasks = tasks
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            download_all(self.tasks, progress_cb=self.progress.emit, cancel_check=lambda: self._cancel_requested)
            self.finished_ok.emit()
        except PipelineCancelled:
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
