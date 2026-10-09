"""'폴더 추가' 정리 작업(smi 변환+AI 검수, 인코딩 수정)을 GUI와 분리된 스레드에서 돌린다."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from core.errors import PipelineCancelled
from core.folder_scan import FolderScanPlan, run_conversions
from core.worklog import WorkLog


class FolderCleanupWorker(QThread):
    progress = Signal(str)
    finished_ok = Signal(int, int, int)  # converted, fixed, deleted
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self, plan: FolderScanPlan, formats: list[str], config, delete_old_smi: bool, out_dir: str = "", parent=None
    ):
        super().__init__(parent)
        self.out_dir = out_dir
        self.plan = plan
        self.formats = formats
        self.config = config
        self.delete_old_smi = delete_old_smi
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        worklog = WorkLog(self.out_dir) if self.out_dir and self.config.get("subtitle.worklog", True) else None
        if worklog:
            worklog.begin(
                "폴더 정리 시작",
                [
                    f"smi 변환 {len(self.plan.convert_jobs)}개, 인코딩 수정 {len(self.plan.encoding_fix_jobs)}개, "
                    f"자막 없어서 새로 만들 영상 {len(self.plan.generate)}개",
                    f"자막 형식: {'/'.join(self.formats)}, 변환 후 원본 smi 삭제: {'예' if self.delete_old_smi else '아니오'}",
                    *[f"  새로 만들 영상: {v}" for v in self.plan.generate[:50]],
                ],
            )
        try:
            converted, fixed, deleted = run_conversions(
                self.plan,
                self.formats,
                self.config,
                self.delete_old_smi,
                progress_cb=self.progress.emit,
                cancel_check=lambda: self._cancel_requested,
                worklog=worklog,
            )
            if worklog:
                worklog.end(f"폴더 정리 끝 - 변환 {converted}개, 인코딩 수정 {fixed}개, 원본 smi 삭제 {deleted}개")
            self.finished_ok.emit(converted, fixed, deleted)
        except PipelineCancelled:
            if worklog:
                worklog.end("폴더 정리 취소로 끝남")
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001
            if worklog:
                worklog.note(f"오류로 중단됨: {e}")
                worklog.end("폴더 정리 오류로 끝남")
            self.failed.emit(str(e))
