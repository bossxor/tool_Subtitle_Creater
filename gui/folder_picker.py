"""Read-only folder discovery followed by explicit per-file task selection."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QProgressBar, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)

from core.errors import PipelineCancelled
from core.folder_scan import FolderScanPlan, plan_folder

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".ts", ".webm", ".m4v", ".flv"}


class FolderScanWorker(QThread):
    progress = Signal(str)
    found = Signal(object)
    failed = Signal(str)

    def __init__(self, folder, out_dir, config, recursive=False, parent=None):
        super().__init__(parent)
        self.folder = Path(folder)
        self.out_dir = out_dir
        self.config = config
        self.recursive = recursive

    def request_cancel(self):
        self.requestInterruption()

    def run(self):
        last_progress = 0.0

        def report(message):
            nonlocal last_progress
            now = time.monotonic()
            if now - last_progress >= 0.1:
                self.progress.emit(message)
                last_progress = now

        try:
            videos = []
            entries = self.folder.rglob("*") if self.recursive else self.folder.iterdir()
            for entry in entries:
                if self.isInterruptionRequested():
                    raise PipelineCancelled()
                if entry.suffix.lower() in VIDEO_EXTENSIONS and entry.is_file():
                    videos.append(entry)
                report(f"영상 검색 중 · {len(videos)}개 발견")
            plan = plan_folder(sorted(videos), self.out_dir, self.config,
                               cancel_check=self.isInterruptionRequested, progress_cb=report)
            if not self.isInterruptionRequested():
                self.found.emit((len(videos), plan))
        except PipelineCancelled:
            pass
        except Exception as e:
            self.failed.emit(str(e))


class FolderPickerDialog(QDialog):
    def __init__(self, folder, out_dir, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("폴더에서 작업 선택")
        self.resize(920, 650)
        self.setMinimumSize(760, 520)
        self.folder, self.out_dir, self.config = folder, out_dir, config
        self.worker = None
        self._reject_pending = False
        self.entries = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        title = QLabel("이번에 처리할 파일만 골라 주세요")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(title)
        hint = QLabel("체크한 작업만 적용합니다. 새 자막 생성은 대기열에 담고, 시작 버튼을 눌러 실행합니다.")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        layout.addWidget(hint)
        path = QLabel(str(folder))
        path.setTextFormat(Qt.PlainText)
        path.setWordWrap(True)
        layout.addWidget(path)

        search_row = QHBoxLayout()
        self.recursive_chk = QCheckBox("하위 폴더도 검색")
        self.rescan_btn = QPushButton("다시 검색")
        self.rescan_btn.clicked.connect(self.start_scan)
        search_row.addWidget(self.recursive_chk)
        search_row.addStretch()
        search_row.addWidget(self.rescan_btn)
        layout.addLayout(search_row)
        self.status_label = QLabel("검색 중…")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)
        layout.addWidget(self.busy_bar)

        filters = QHBoxLayout()
        self.kind_combo = QComboBox()
        for label, key in (("모든 작업", ""), ("새 자막 생성", "generate"),
                           ("SMI 형식 변환", "convert"), ("인코딩 수정", "encoding")):
            self.kind_combo.addItem(label, key)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("파일 이름 또는 경로 검색")
        self.kind_combo.currentIndexChanged.connect(self.filter_rows)
        self.search_edit.textChanged.connect(self.filter_rows)
        filters.addWidget(self.kind_combo)
        filters.addWidget(self.search_edit, 1)
        layout.addLayout(filters)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["선택", "작업", "파일", "위치"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(1, 115)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setWordWrap(False)
        self.table.itemChanged.connect(self.update_selection)
        layout.addWidget(self.table, 1)

        selection = QHBoxLayout()
        self.select_btn = QPushButton("현재 목록 모두 선택")
        self.clear_btn = QPushButton("전체 선택 해제")
        self.select_btn.clicked.connect(self.select_visible)
        self.clear_btn.clicked.connect(self.clear_selection)
        selection.addWidget(self.select_btn)
        selection.addWidget(self.clear_btn)
        selection.addStretch()
        self.count_label = QLabel("선택 0개")
        selection.addWidget(self.count_label)
        layout.addLayout(selection)
        note = QLabel("필터를 바꿔도 체크한 파일은 유지됩니다. 다시 검색하면 선택이 초기화됩니다.")
        note.setObjectName("muted")
        layout.addWidget(note)
        self.ai_chk = QCheckBox("SMI 변환에 AI 검수 추가 · 깨진 글자 복원, 시간이 더 걸립니다")
        self.ai_chk.setToolTip("끄면 모델을 실행하지 않고 형식만 변환합니다. 일반적인 인코딩은 자동 감지합니다.")
        layout.addWidget(self.ai_chk)
        self.delete_chk = QCheckBox("변환에 성공한 원본 SMI 삭제")
        layout.addWidget(self.delete_chk)
        output_hint = QLabel("변환 결과는 원본 자막 옆에 SRT/ASS로 저장합니다. 원본 SMI 삭제는 기본으로 꺼져 있습니다.")
        output_hint.setObjectName("muted")
        output_hint.setWordWrap(True)
        layout.addWidget(output_hint)
        actions = QHBoxLayout()
        cancel = QPushButton("닫기")
        cancel.clicked.connect(self.reject)
        self.apply_btn = QPushButton("선택한 작업 적용")
        self.apply_btn.setObjectName("primaryButton")
        self.apply_btn.clicked.connect(self.accept)
        actions.addStretch()
        actions.addWidget(cancel)
        actions.addWidget(self.apply_btn)
        layout.addLayout(actions)
        self.start_scan()

    def start_scan(self):
        if self.worker is not None:
            return
        self.entries = []
        self.table.setRowCount(0)
        self.status_label.setText("영상과 자막을 검색하는 중… 닫기를 누르면 검색을 취소합니다.")
        self.busy_bar.show()
        self.worker = FolderScanWorker(self.folder, self.out_dir, self.config,
                                       self.recursive_chk.isChecked(), self)
        self.worker.progress.connect(self.status_label.setText)
        self.worker.found.connect(self.show_plan)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.scan_stopped)
        self.rescan_btn.setEnabled(False)
        self.recursive_chk.setEnabled(False)
        self.update_selection()
        self.worker.start()

    def show_error(self, message):
        self.status_label.setText(f"검색 실패: {message}")

    def show_plan(self, result):
        count, plan = result
        self.entries = ([("generate", "새 자막 생성", video, video) for video in plan.generate]
                        + [("convert", "SMI 형식 변환", job.smi_path, job) for job in plan.convert_jobs]
                        + [("encoding", "인코딩 수정", job.path, job) for job in plan.encoding_fix_jobs])
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.entries))
        for row, (kind, label, path, job) in enumerate(self.entries):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            check.setCheckState(Qt.Unchecked)
            self.table.setItem(row, 0, check)
            for col, value in enumerate((label, path.name, str(path.parent)), 1):
                item = QTableWidgetItem(value)
                item.setToolTip(str(path) if col >= 2 else value)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)
        self.status_label.setText(f"영상 {count}개 확인 · 새 자막 {len(plan.generate)}개 / "
                                  f"SMI 변환 {len(plan.convert_jobs)}개 / 인코딩 수정 {len(plan.encoding_fix_jobs)}개"
                                  + (" · 선택할 작업이 없습니다." if not self.entries else ""))
        self.filter_rows()

    def scan_stopped(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        self.busy_bar.hide()
        self.rescan_btn.setEnabled(True)
        self.recursive_chk.setEnabled(True)
        self.update_selection()
        if self._reject_pending:
            super().reject()

    def filter_rows(self):
        kind = self.kind_combo.currentData()
        text = self.search_edit.text().strip().casefold()
        for row, (entry_kind, label, path, job) in enumerate(self.entries):
            self.table.setRowHidden(row, bool((kind and entry_kind != kind) or text not in str(path).casefold()))

    def select_visible(self):
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                self.table.item(row, 0).setCheckState(Qt.Checked)
        self.table.blockSignals(False)
        self.update_selection()

    def clear_selection(self):
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setCheckState(Qt.Unchecked)
        self.table.blockSignals(False)
        self.update_selection()

    def selected_plan(self):
        selected = {"generate": [], "convert": [], "encoding": []}
        for row, (kind, label, path, job) in enumerate(self.entries):
            if self.table.item(row, 0).checkState() == Qt.Checked:
                selected[kind].append(job)
        return FolderScanPlan(selected["generate"], selected["convert"], selected["encoding"])

    def update_selection(self):
        plan = self.selected_plan()
        total = len(plan.generate) + len(plan.convert_jobs) + len(plan.encoding_fix_jobs)
        self.count_label.setText(f"선택 {total}개")
        self.apply_btn.setText(f"선택한 작업 적용 · {total}개")
        self.apply_btn.setEnabled(total > 0 and self.worker is None)
        self.ai_chk.setEnabled(bool(plan.convert_jobs))
        self.delete_chk.setEnabled(bool(plan.convert_jobs))
        self.select_btn.setEnabled(self.worker is None)
        self.clear_btn.setEnabled(self.worker is None)

    def accept(self):
        if self.worker is None and self.apply_btn.isEnabled():
            super().accept()

    def reject(self):
        if self.worker is not None:
            self._reject_pending = True
            self.worker.request_cancel()
            self.status_label.setText("검색을 정리하는 중…")
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self.reject()
            event.ignore()
        else:
            event.accept()
