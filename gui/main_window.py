"""Subtitle Tool 메인 창.

영상 다중 선택 + 출력 폴더 지정 + 옵션 설정 후 실행하면 core.pipeline.run_batch가
백그라운드 스레드(worker.PipelineWorker)에서 돌아가고, 진행 상황은 로그창에 표시된다.
"""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.config import Config, DEFAULT_CONFIG_PATH
from core.folder_scan import plan_folder
from core.pipeline import predict_output_paths
from core.setup_assets import list_missing
from core.worklog import LOG_FILENAME
from gui.asset_worker import AssetDownloadWorker
from gui.folder_worker import FolderCleanupWorker
from gui.progress_model import StageTracker
from gui.worker import PipelineWorker

STYLE_SHEET = """
QMainWindow, QWidget#root {
    background: #f4f5f8;
}
QLabel {
    color: #2b2d33;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #e2e4ea;
    border-radius: 10px;
    margin-top: 22px;
    padding: 14px 12px 12px 12px;
    font-weight: normal;
    color: #2b2d33;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: -4px;
    padding: 0 6px;
    font-weight: 600;
    color: #3457d5;
}
QListWidget, QPlainTextEdit, QTableWidget, QLineEdit {
    background: #fbfbfd;
    border: 1px solid #dfe2e8;
    border-radius: 6px;
    padding: 4px;
    selection-background-color: #3457d5;
    selection-color: white;
}
QTableWidget QHeaderView::section {
    background: #eef0f5;
    border: none;
    padding: 4px;
    font-weight: 600;
}
QTabWidget::pane {
    border: 1px solid #e2e4ea;
    border-radius: 8px;
    background: #ffffff;
}
QTabBar::tab {
    background: #eef0f5;
    border: 1px solid #e2e4ea;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 6px 16px;
    margin-right: 2px;
    color: #4b5160;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #3457d5;
    font-weight: 600;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #d7dae1;
    border-radius: 6px;
    padding: 7px 16px;
    color: #2b2d33;
}
QPushButton:hover {
    background: #f0f2f7;
    border-color: #b9c0cf;
}
QPushButton:disabled {
    color: #a7abb6;
    background: #f4f5f8;
}
QPushButton#primaryButton {
    background: #3457d5;
    border: 1px solid #3457d5;
    color: white;
    font-weight: 600;
    padding: 9px 18px;
}
QPushButton#primaryButton:hover {
    background: #2c49b8;
}
QPushButton#primaryButton:disabled {
    background: #aab7ea;
    border-color: #aab7ea;
    color: #f0f2f7;
}
QPushButton#dangerButton {
    color: #c23b3b;
    border-color: #eac6c6;
}
QPushButton#dangerButton:hover {
    background: #fbebeb;
    border-color: #e39a9a;
}
QComboBox {
    background: #fbfbfd;
    border: 1px solid #dfe2e8;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 22px;
}
QCheckBox {
    spacing: 8px;
}
QProgressBar {
    background: #eef0f5;
    border: none;
    border-radius: 7px;
    height: 14px;
    text-align: center;
    color: #2b2d33;
}
QProgressBar::chunk {
    background: #3457d5;
    border-radius: 7px;
}
"""

# 진행 메시지 형식: "[단계 k/N] 파일명" (core/pipeline.py, core/folder_scan.py 에서 만든다)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".ts", ".webm", ".m4v", ".flv"}

LANGUAGES = [
    ("일본어", "ja"),
    ("한국어", "ko"),
    ("영어", "en"),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Subtitle Tool - 영상 자막 자동 생성")
        self.resize(900, 1000)
        # 내용물이 실제로 필요로 하는 최소 높이(minimumSizeHint, 보통 ~860px)보다 작게 잡으면
        # Qt가 각 영역을 강제로 쥐어짜서 글자가 겹쳐 보인다. 넉넉하게 잡는다.
        self.setMinimumSize(760, 880)
        self.setAcceptDrops(True)
        self.setStyleSheet(STYLE_SHEET)

        self.worker: PipelineWorker | None = None
        self.tracker = StageTracker()
        self._t0 = 0.0
        self._current_videos: list[str] = []

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addWidget(self._build_file_section())
        layout.addWidget(self._build_output_section())
        layout.addWidget(self._build_options_section())
        layout.addLayout(self._build_action_section())
        layout.addWidget(self._build_progress_section())

    # ---------- UI 구성 ----------

    def _build_file_section(self) -> QGroupBox:
        self.file_box = QGroupBox("1. 자막을 만들 영상 선택")
        box = self.file_box
        v = QVBoxLayout(box)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        v.addWidget(QLabel("영상 파일을 여기로 끌어다 놓거나 아래 버튼으로 추가하세요."))
        v.addWidget(self.file_list)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("영상 추가...")
        add_btn.clicked.connect(self._on_add_files)
        add_folder_btn = QPushButton("폴더 추가...")
        add_folder_btn.setToolTip(
            "폴더 안의 영상을 전부 훑어봅니다. 예전 smi 자막은 srt/ass로 변환하고,\n"
            "인코딩이 깨질 수 있는 자막은 고치고, 자막이 아예 없는 영상만 목록에 추가합니다."
        )
        add_folder_btn.clicked.connect(self._on_add_folder)
        remove_btn = QPushButton("선택 제거")
        remove_btn.clicked.connect(self._on_remove_selected)
        clear_btn = QPushButton("전체 지우기")
        clear_btn.clicked.connect(self._on_clear_all)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(add_folder_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch(1)
        v.addLayout(btn_row)

        self.delete_smi_chk = QCheckBox("폴더 추가 시 smi를 srt/ass로 변환한 뒤 원본 smi 삭제")
        self.delete_smi_chk.setChecked(False)
        self.delete_smi_chk.setToolTip("체크 안 하면 변환 후에도 원본 smi 파일을 그대로 남겨둡니다.")
        v.addWidget(self.delete_smi_chk)
        return box

    def _build_output_section(self) -> QGroupBox:
        box = QGroupBox("2. 자막 저장 폴더")
        h = QHBoxLayout(box)
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText("자막 파일이 저장될 폴더를 선택하세요")
        browse_btn = QPushButton("찾아보기...")
        browse_btn.clicked.connect(self._on_browse_out_dir)
        h.addWidget(self.out_dir_edit)
        h.addWidget(browse_btn)
        return box

    def _build_options_section(self) -> QGroupBox:
        box = QGroupBox("3. 옵션")
        outer = QVBoxLayout(box)
        tabs = QTabWidget()
        outer.addWidget(tabs)

        # --- 기본 옵션 탭 ---
        basic = QWidget()
        form = QVBoxLayout(basic)
        form.setSpacing(10)

        # 주의: QHBoxLayout을 form.addLayout()으로 바로 끼우면, QSS로 키운 QComboBox의
        # 실제 렌더링 높이가 레이아웃이 계산한 행 높이보다 커져서 다음 행과 겹치는 문제가 있었다
        # (Qt에서 스타일시트 입힌 콤보박스가 자주 겪는 문제). 그래서 각 행을 진짜 QWidget으로
        # 감싸서 자기 sizeHint를 제대로 갖게 만든다.
        def _row(*widgets) -> QWidget:
            w = QWidget()
            lay = QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            for item in widgets:
                lay.addWidget(item)
            lay.addStretch(1)
            return w

        self.source_lang_combo = QComboBox()
        for name, code in LANGUAGES:
            self.source_lang_combo.addItem(name, code)
        self.source_lang_combo.setCurrentIndex(0)  # 일본어

        self.target_lang_combo = QComboBox()
        for name, code in LANGUAGES:
            self.target_lang_combo.addItem(name, code)
        self.target_lang_combo.setCurrentIndex(1)  # 한국어

        form.addWidget(
            _row(QLabel("원어:"), self.source_lang_combo, QLabel("   번역 대상:"), self.target_lang_combo)
        )

        self.precision_combo = QComboBox()
        self.precision_combo.addItem("빠름 (large-v3-turbo, 권장)", "turbo")
        self.precision_combo.addItem("고정밀 (large-v3, 느림)", "large")
        form.addWidget(_row(QLabel("음성 인식 속도/정확도:"), self.precision_combo))

        fmt_label = QLabel("자막 형식:")
        fmt_label.setToolTip("체크한 형식마다 자막 파일이 하나씩 따로 생성됩니다. 보통은 하나만 선택하세요.")
        self.format_srt_chk = QCheckBox("srt (기본, 가장 호환성 좋음)")
        self.format_srt_chk.setChecked(True)
        self.format_ass_chk = QCheckBox("ass (스타일링 가능)")
        self.format_ass_chk.setChecked(False)
        self.format_smi_chk = QCheckBox("smi (곰플레이어 구버전 호환용)")
        self.format_smi_chk.setChecked(False)
        form.addWidget(_row(fmt_label, self.format_srt_chk, self.format_ass_chk, self.format_smi_chk))

        self.emit_source_chk = QCheckBox("원문(원어) 자막도 따로 생성")
        self.bilingual_chk = QCheckBox("이중 자막 생성 (번역 위 + 원문 아래, 같은 자막 파일에 함께 표시)")
        form.addWidget(self.emit_source_chk)
        form.addWidget(self.bilingual_chk)
        form.addStretch(1)
        tabs.addTab(basic, "기본")

        # --- 용어집 탭 ---
        glossary_tab = QWidget()
        gv = QVBoxLayout(glossary_tab)
        gv.addWidget(QLabel("등장인물 이름이나 고유명사의 번역을 고정하고 싶으면 추가하세요. (선택 사항)"))
        self.glossary_table = QTableWidget(0, 2)
        self.glossary_table.setHorizontalHeaderLabels(["원어 표기", "번역 고정"])
        self.glossary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.glossary_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        gv.addWidget(self.glossary_table)
        g_btn_row = QHBoxLayout()
        g_add_btn = QPushButton("행 추가")
        g_add_btn.clicked.connect(lambda: self.glossary_table.insertRow(self.glossary_table.rowCount()))
        g_del_btn = QPushButton("선택 행 삭제")
        g_del_btn.clicked.connect(self._on_remove_glossary_rows)
        g_btn_row.addWidget(g_add_btn)
        g_btn_row.addWidget(g_del_btn)
        g_btn_row.addStretch(1)
        gv.addLayout(g_btn_row)
        tabs.addTab(glossary_tab, "용어집")

        return box

    def _build_action_section(self) -> QHBoxLayout:
        h = QHBoxLayout()
        self.start_btn = QPushButton("자막 생성 시작")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.setMinimumHeight(38)
        self.start_btn.clicked.connect(self._on_start)
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.setObjectName("dangerButton")
        self.cancel_btn.setMinimumHeight(38)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        h.addWidget(self.start_btn, stretch=3)
        h.addWidget(self.cancel_btn, stretch=1)
        return h

    def _build_progress_section(self) -> QGroupBox:
        box = QGroupBox("진행 상황")
        v = QVBoxLayout(box)
        self.count_label = QLabel("")
        self.count_label.setStyleSheet("font-weight: 600; color: #3457d5;")
        self.status_label = QLabel("대기 중")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(False)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(["구분", "파일", "상태", "결과"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setVisible(False)
        self.result_title = QLabel("결과")
        self.result_title.setVisible(False)
        v.addWidget(self.count_label)
        v.addWidget(self.status_label)
        v.addWidget(self.progress_bar)
        v.addWidget(self.log_view)
        v.addWidget(self.result_title)
        v.addWidget(self.result_table)
        return box

    def _refresh_counts(self) -> None:
        self.count_label.setText("\n".join(self.tracker.lines()))

    def _folder_result_rows(self, plan) -> list[tuple[str, str, str, str]]:
        rows: list[tuple[str, str, str, str]] = []
        for j in plan.convert_jobs:
            if j.error:
                status, detail = "실패", j.error
            elif j.written:
                status, detail = "완료", ", ".join(p.name for p in j.written)
            else:
                status, detail = "대기", "아직 처리 안 됨(취소됨)"
            rows.append(("smi 변환", j.smi_path.name, status, detail))
        for j in plan.encoding_fix_jobs:
            if j.error:
                status, detail = "실패", j.error
            elif j.fixed:
                status, detail = "완료", "BOM UTF-8로 다시 저장"
            else:
                status, detail = "변경 없음", "이미 괜찮거나 인코딩 확인 불가"
            rows.append(("인코딩 수정", j.path.name, status, detail))
        for v in plan.generate:
            rows.append(("자막생성", v.name, "대기열 추가", "'자막 생성 시작'을 누르면 처리"))
        return rows

    def _show_results(self, rows: list[tuple[str, str, str, str]]) -> None:
        """rows: (구분, 파일, 상태, 결과) 목록을 결과 표에 채운다."""
        self.result_table.setRowCount(0)
        for kind, name, status, detail in rows:
            r = self.result_table.rowCount()
            self.result_table.insertRow(r)
            for c, text in enumerate((kind, name, status, detail)):
                self.result_table.setItem(r, c, QTableWidgetItem(text))
        self.result_table.resizeColumnsToContents()
        self.result_table.setVisible(bool(rows))
        self.result_title.setVisible(bool(rows))

    def _elapsed_text(self) -> str:
        sec = int(time.monotonic() - self._t0) if self._t0 else 0
        return f"{sec // 60:02d}:{sec % 60:02d}"

    # ---------- 이벤트 핸들러 ----------

    def _on_add_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(VIDEO_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(
            self, "영상 선택", "", f"영상 파일 ({exts});;모든 파일 (*.*)"
        )
        self._add_files(files)

    def _add_files(self, paths: list[str]) -> None:
        existing = {self.file_list.item(i).text() for i in range(self.file_list.count())}
        for p in paths:
            if p and p not in existing:
                self.file_list.addItem(p)
                existing.add(p)
        self._update_file_count()

    def _on_clear_all(self) -> None:
        self.file_list.clear()
        self._update_file_count()

    def _update_file_count(self) -> None:
        self.file_box.setTitle(f"1. 자막을 만들 영상 선택 (총 {self.file_list.count()}개)")

    def _on_remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
        self._update_file_count()

    def _on_remove_glossary_rows(self) -> None:
        rows = sorted({idx.row() for idx in self.glossary_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.glossary_table.removeRow(r)

    def _on_add_folder(self) -> None:
        out_dir = self.out_dir_edit.text().strip()
        if not out_dir:
            QMessageBox.information(
                self,
                "출력 폴더 먼저 지정",
                "폴더째로 추가하려면 어떤 영상에 이미 자막이 있는지 확인해야 해서,\n"
                "'2. 자막 저장 폴더'를 먼저 지정해 주세요.",
            )
            return

        folder = QFileDialog.getExistingDirectory(self, "영상이 들어있는 폴더 선택")
        if not folder:
            return

        found = sorted(
            p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not found:
            QMessageBox.information(self, "영상 없음", "선택한 폴더(하위 폴더 포함)에서 영상 파일을 찾지 못했습니다.")
            return

        config = self._build_config()
        formats = self._selected_formats()
        plan = plan_folder(found, out_dir, config)

        if not plan.convert_jobs and not plan.encoding_fix_jobs:
            self._add_files([str(v) for v in plan.generate])
            self._show_results(self._folder_result_rows(plan))
            QMessageBox.information(
                self, "폴더 추가 완료", f"영상 {len(found)}개 중 자막 없는 {len(plan.generate)}개를 목록에 추가했습니다."
            )
            return

        details = []
        if plan.convert_jobs:
            details.append(f"- 예전 smi 자막 {len(plan.convert_jobs)}개를 {'/'.join(formats)}로 변환")
        if plan.encoding_fix_jobs:
            details.append(f"- 인코딩이 깨질 수 있는 자막 {len(plan.encoding_fix_jobs)}개를 BOM UTF-8로 재저장")
        if plan.generate:
            details.append(f"- 자막이 아예 없는 영상 {len(plan.generate)}개는 목록에 추가(AI로 새로 생성)")
        note = ""
        if plan.convert_jobs:
            note = "\n\n(smi 변환은 AI로 글자를 검수하느라 번역만큼은 아니어도 시간이 걸립니다. " \
                   "기존 smi 파일은 '변환 후 smi 삭제' 체크 여부에 따라 지워집니다)"
        reply = QMessageBox.question(
            self, "폴더 정리 확인", "다음 작업을 진행할까요?\n\n" + "\n".join(details) + note,
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        if plan.convert_jobs:
            # smi 변환의 AI 검수는 범용 모델(Qwen)을 쓴다. 없으면 먼저 받을지 물어본다.
            verify_missing = [t for t in list_missing(config, include_verify_model=True) if "Qwen" in t.label]
            if verify_missing:
                ask = QMessageBox.question(
                    self,
                    "AI 검수 모델 필요",
                    "smi를 AI로 검수하며 변환하려면 범용 AI 모델(Qwen3-8B, 약 4.7GB)이 필요합니다. 지금 받을까요?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if ask != QMessageBox.Yes or not self._run_asset_download(verify_missing):
                    return

        converted, fixed, deleted = self._run_folder_cleanup(plan, formats, config, self.delete_smi_chk.isChecked())
        self._show_results(self._folder_result_rows(plan))
        self._refresh_counts()
        if converted is None:
            return  # 취소/실패 — 이미 안내 메시지 띄움

        failed_convert = [j for j in plan.convert_jobs if j.error]
        failed_fix = [j for j in plan.encoding_fix_jobs if j.error]

        self._add_files([str(v) for v in plan.generate])

        msg_lines = [f"영상 {len(found)}개를 확인했습니다."]
        if converted:
            msg_lines.append(f"smi {converted}개를 AI 검수 후 {'/'.join(formats)}로 변환했습니다.")
        if deleted:
            msg_lines.append(f"변환 후 원본 smi {deleted}개를 삭제했습니다.")
        if fixed:
            msg_lines.append(f"인코딩 {fixed}개를 BOM UTF-8로 고쳤습니다.")
        if plan.generate:
            msg_lines.append(f"자막 없는 {len(plan.generate)}개를 목록에 추가했습니다.")
        else:
            msg_lines.append("새로 생성할 영상은 없습니다.")
        if failed_convert or failed_fix:
            msg_lines.append(f"실패: 변환 {len(failed_convert)}개, 인코딩 수정 {len(failed_fix)}개 (로그 참고)")
        QMessageBox.information(self, "폴더 정리 완료", "\n".join(msg_lines))

    def _run_folder_cleanup(self, plan, formats, config, delete_old_smi: bool):
        """FolderCleanupWorker를 돌리며 진행 다이얼로그를 보여준다.
        성공하면 (converted, fixed, deleted), 취소/실패면 (None, None, None)."""
        total = len(plan.convert_jobs) + len(plan.encoding_fix_jobs)
        self.tracker.reset()
        self.tracker.set_expected("smi 변환+검수", len(plan.convert_jobs))
        self.tracker.set_expected("인코딩 수정", len(plan.encoding_fix_jobs))
        self._t0 = time.monotonic()
        dlg = QProgressDialog("폴더를 정리하는 중...", "취소", 0, max(total, 1), self)
        dlg.setWindowTitle("폴더 정리")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumWidth(420)

        worker = FolderCleanupWorker(plan, formats, config, delete_old_smi, self.out_dir_edit.text().strip())
        loop = QEventLoop()
        result = {"ok": None}

        stages = ("smi 변환+검수", "인코딩 수정")

        def on_progress(message: str) -> None:
            self.tracker.update(message)
            done, _all = self.tracker.overall_done(stages)
            dlg.setValue(done)
            dlg.setLabelText("\n".join(self.tracker.lines()))
            self.status_label.setText(message)
            self.log_view.appendPlainText(message)
            self._refresh_counts()

        def on_finished(converted: int, fixed: int, deleted: int) -> None:
            self.tracker.finish_all()
            dlg.setValue(max(total, 1))
            result["ok"] = (converted, fixed, deleted)
            loop.quit()

        def on_cancelled() -> None:
            loop.quit()

        def on_failed(msg: str) -> None:
            QMessageBox.critical(self, "폴더 정리 실패", f"정리 중 오류가 발생했습니다:\n{msg}")
            loop.quit()

        worker.progress.connect(on_progress)
        worker.finished_ok.connect(on_finished)
        worker.cancelled.connect(on_cancelled)
        worker.failed.connect(on_failed)
        dlg.canceled.connect(worker.request_cancel)

        worker.start()
        dlg.show()
        loop.exec()
        dlg.close()
        worker.wait(5000)
        return result["ok"] if result["ok"] is not None else (None, None, None)

    def _on_browse_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "자막 저장 폴더 선택")
        if d:
            self.out_dir_edit.setText(d)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = []
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if p and Path(p).suffix.lower() in VIDEO_EXTENSIONS:
                paths.append(p)
        self._add_files(paths)

    def _selected_formats(self) -> list[str]:
        formats = []
        if self.format_srt_chk.isChecked():
            formats.append("srt")
        if self.format_ass_chk.isChecked():
            formats.append("ass")
        if self.format_smi_chk.isChecked():
            formats.append("smi")
        return formats

    def _build_config(self) -> Config:
        config = Config.load(DEFAULT_CONFIG_PATH)
        config.set("source_language", self.source_lang_combo.currentData())
        config.set("target_language", self.target_lang_combo.currentData())
        config.set("stt.precision", self.precision_combo.currentData())
        config.set("subtitle.formats", self._selected_formats())
        config.set("subtitle.emit_source", self.emit_source_chk.isChecked())
        config.set("subtitle.bilingual", self.bilingual_chk.isChecked())
        return config

    def _collect_glossary(self) -> dict[str, str] | None:
        glossary = {}
        for r in range(self.glossary_table.rowCount()):
            src_item = self.glossary_table.item(r, 0)
            dst_item = self.glossary_table.item(r, 1)
            src = src_item.text().strip() if src_item else ""
            dst = dst_item.text().strip() if dst_item else ""
            if src and dst:
                glossary[src] = dst
        return glossary or None

    def _on_start(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return  # 이미 돌아가는 중이면 무시 (중복 실행 방지)
        videos = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not videos:
            QMessageBox.warning(self, "확인 필요", "영상을 하나 이상 추가해 주세요.")
            return
        out_dir = self.out_dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "확인 필요", "자막을 저장할 폴더를 선택해 주세요.")
            return
        missing = [v for v in videos if not Path(v).exists()]
        if missing:
            QMessageBox.warning(self, "파일 없음", "다음 파일을 찾을 수 없습니다:\n" + "\n".join(missing))
            return
        if not self._selected_formats():
            QMessageBox.warning(self, "확인 필요", "자막 형식을 하나 이상 선택해 주세요 (srt/ass/smi).")
            return

        config = self._build_config()
        glossary = self._collect_glossary()

        predicted = predict_output_paths(videos, out_dir, config)
        existing_files = [p for paths in predicted.values() for p in paths if p.exists()]
        if existing_files:
            names = "\n".join(f"- {p.name}" for p in existing_files)
            reply = QMessageBox.question(
                self,
                "덮어쓰기 확인",
                f"이미 같은 이름의 자막 파일이 있습니다. 덮어쓸까요?\n\n{names}",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        missing = list_missing(config)
        if missing:
            names = "\n".join(f"- {t.label}" for t in missing)
            reply = QMessageBox.question(
                self,
                "첫 실행 준비",
                f"처음 실행할 때 필요한 파일을 받아야 합니다 (용량이 클 수 있습니다):\n{names}\n\n지금 받을까요?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            if not self._run_asset_download(missing):
                return

        self.log_view.clear()
        self.tracker.reset()
        self.tracker.set_expected("오디오 추출", len(videos))
        self._current_videos = videos
        self._t0 = time.monotonic()
        self._show_results([])
        self._refresh_counts()
        self.status_label.setText("시작하는 중...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self.worker = PipelineWorker(videos, out_dir, config, glossary=glossary)
        self.worker.progress.connect(self._on_progress)
        self.worker.progress_fraction.connect(self._on_progress_fraction)
        self.worker.log.connect(self.log_view.appendPlainText)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _run_asset_download(self, tasks) -> bool:
        """첫 실행에 필요한 파일을 내려받는다. 완료하면 True, 취소/실패하면 False."""
        dlg = QProgressDialog("필요한 파일을 준비하는 중...", "취소", 0, 100, self)
        dlg.setWindowTitle("첫 실행 준비")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumWidth(420)

        worker = AssetDownloadWorker(tasks)
        loop = QEventLoop()
        result = {"ok": False}

        def on_progress(label: str, downloaded: int, total: int) -> None:
            mb = downloaded / 1_000_000
            if total > 0:
                dlg.setMaximum(100)
                dlg.setValue(int(downloaded / total * 100))
                dlg.setLabelText(f"{label}\n{mb:.0f}MB / {total / 1_000_000:.0f}MB")
            else:
                dlg.setMaximum(0)
                dlg.setLabelText(f"{label}\n{mb:.0f}MB")

        def on_finished() -> None:
            result["ok"] = True
            loop.quit()

        def on_cancelled() -> None:
            loop.quit()

        def on_failed(msg: str) -> None:
            QMessageBox.critical(self, "다운로드 실패", f"필요한 파일을 받는 중 오류가 발생했습니다:\n{msg}")
            loop.quit()

        worker.progress.connect(on_progress)
        worker.finished_ok.connect(on_finished)
        worker.cancelled.connect(on_cancelled)
        worker.failed.connect(on_failed)
        dlg.canceled.connect(worker.request_cancel)

        worker.start()
        dlg.show()
        loop.exec()
        dlg.close()
        worker.wait(5000)
        return result["ok"]

    def _on_cancel(self) -> None:
        if self.worker is not None:
            self.cancel_btn.setEnabled(False)
            self.worker.request_cancel()

    def _on_progress(self, message: str) -> None:
        if self.tracker.update(message):
            self._refresh_counts()
        self.status_label.setText(message)
        self.log_view.appendPlainText(message)

    def _on_progress_fraction(self, fraction: float) -> None:
        self.progress_bar.setValue(round(fraction * 100))

    def _reset_run_state(self) -> None:
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.worker = None

    def _on_finished_ok(self, outputs: dict) -> None:
        self.status_label.setText("완료")
        self.log_view.appendPlainText(f"작업 로그: {Path(self.out_dir_edit.text().strip()) / LOG_FILENAME}")
        self.tracker.finish_all()
        self._refresh_counts()
        self.count_label.setText(self.count_label.text() + f"\n소요 시간 {self._elapsed_text()}")
        self.progress_bar.setValue(100)
        rows = []
        lines = ["=== 완료 ==="]
        for video, paths in outputs.items():
            lines.append(video)
            for p in paths:
                lines.append(f"  - {p}")
            detail = ", ".join(Path(p).name for p in paths) or "결과 파일 없음"
            rows.append(("자막생성", Path(video).name, "완료", detail))
        self.log_view.appendPlainText("\n".join(lines))
        self._show_results(rows)
        self._reset_run_state()
        QMessageBox.information(self, "완료", "자막 생성이 끝났습니다.")

    def _on_cancelled(self) -> None:
        self.status_label.setText("취소됨")
        self._show_results([("자막생성", Path(v).name, "취소됨", "처리 전에 중단") for v in self._current_videos])
        self.log_view.appendPlainText("사용자 요청으로 취소되었습니다. 완료된 단계는 캐시에 남아있어 이어서 재개할 수 있습니다.")
        self._reset_run_state()

    def _on_failed(self, message: str) -> None:
        self.status_label.setText("오류 발생")
        self._show_results(
            [("자막생성", Path(v).name, "실패", message.splitlines()[0][:80]) for v in self._current_videos]
        )
        self.log_view.appendPlainText(f"오류: {message}")
        self._reset_run_state()
        QMessageBox.critical(self, "오류", f"자막 생성 중 오류가 발생했습니다:\n{message}")

    def closeEvent(self, event) -> None:
        if self.worker is not None and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "종료 확인",
                "작업이 진행 중입니다. 취소하고 종료할까요?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.request_cancel()
            self.worker.wait(15000)
        event.accept()
