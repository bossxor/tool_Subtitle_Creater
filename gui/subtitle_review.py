"""Existing subtitles: inspect, preview translation changes, then apply with backups."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QMessageBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from core.errors import PipelineCancelled
from core.subtitle_review import SUBTITLE_EXTENSIONS, VIDEO_EXTENSIONS, apply_changes, inspect_subtitle, glossary_for_row
from core.writers import wrap_lines


class ReviewWorker(QThread):
    progress = Signal(str)
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, job, parent=None):
        super().__init__(parent)
        self.job = job

    def run(self):
        try:
            self.result.emit(self.job(self.isInterruptionRequested, self.progress.emit))
        except PipelineCancelled:
            self.failed.emit("취소됨. 기존 자막은 변경하지 않았습니다.")
        except Exception as error:
            self.failed.emit(str(error))


class SubtitleReviewDialog(QDialog):
    regenerate = Signal(list)

    def __init__(self, config, parent=None, glossary=None):
        super().__init__(parent)
        self.config = config.copy()
        self.glossary = glossary
        self.entries = []
        self.previews = {}
        self.preview_glossaries = {}
        self.worker = None
        self._close_pending = False
        self.setWindowTitle("기존 자막 검사 · 선택 업데이트")
        self.resize(1180, 780)
        self.setMinimumSize(900, 600)
        layout = QVBoxLayout(self)
        heading = QLabel("기존 자막에서 고칠 부분을 확인하세요")
        heading.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(heading)
        hint = QLabel("검사는 파일을 바꾸지 않습니다. 원문이 있으면 번역 후보를 비교하고, 선택한 변경만 백업 후 적용합니다.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        toolbar = QHBoxLayout()
        self.add_btn = QPushButton("자막 파일 추가…")
        self.folder_btn = QPushButton("폴더 검사…")
        self.confirmed_btn = QPushButton("생성 확인된 파일만 선택")
        self.add_btn.clicked.connect(self._add_files)
        self.folder_btn.clicked.connect(self._add_folder)
        self.confirmed_btn.clicked.connect(self._select_confirmed)
        for button in (self.add_btn, self.folder_btn, self.confirmed_btn):
            toolbar.addWidget(button)
        self.recursive = QCheckBox("하위 폴더 포함")
        toolbar.addWidget(self.recursive)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        splitter = QSplitter(Qt.Vertical)
        self.files = QTableWidget(0, 5)
        self.files.setHorizontalHeaderLabels(["선택", "파일", "생성 이력", "원문", "검사 결과"])
        self.files.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.files.setSelectionMode(QAbstractItemView.SingleSelection)
        self.files.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.files.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.files.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.files.currentCellChanged.connect(lambda *_: self._show_preview())
        splitter.addWidget(self.files)
        detail = QWidget()
        details = QVBoxLayout(detail)
        self.detail_label = QLabel("파일을 선택하면 문제 줄과 변경 후보를 확인할 수 있습니다.")
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextFormat(Qt.PlainText)
        details.addWidget(self.detail_label)
        self.changes = QTableWidget(0, 6)
        self.changes.setHorizontalHeaderLabels(["적용", "시간", "일본어 원문", "현재 자막", "새 번역", "판정"])
        self.changes.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.changes.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.changes.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.changes.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.changes.itemChanged.connect(self._change_checked)
        details.addWidget(self.changes)
        splitter.addWidget(detail)
        splitter.setSizes([240, 400])
        layout.addWidget(splitter, 1)
        actions = QHBoxLayout()
        self.translate_btn = QPushButton("선택 파일의 번역 후보 만들기")
        self.link_btn = QPushButton("원본 영상 연결…")
        self.regenerate_btn = QPushButton("음성 인식부터 다시 만들기…")
        self.apply_btn = QPushButton("선택한 변경 적용")
        self.apply_btn.setEnabled(False)
        for button in (self.translate_btn, self.link_btn, self.regenerate_btn, self.apply_btn):
            actions.addWidget(button)
        self.translate_btn.clicked.connect(self._translate)
        self.link_btn.clicked.connect(self._link_video)
        self.regenerate_btn.clicked.connect(self._regenerate)
        self.apply_btn.clicked.connect(self._apply)
        layout.addLayout(actions)
        footer = QHBoxLayout()
        self.status = QLabel("자막을 추가해 주세요. 출처 불명·수정된 파일은 자동 선택하지 않습니다.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)
        footer.addWidget(self.status, 1)
        self.cancel_btn = QPushButton("닫기")
        self.cancel_btn.clicked.connect(self.reject)
        footer.addWidget(self.cancel_btn)
        layout.addLayout(footer)

    def _selected(self):
        return [entry for row, entry in enumerate(self.entries)
                if self.files.item(row, 0).checkState() == Qt.Checked]

    def _current(self):
        row = self.files.currentRow()
        return self.entries[row] if 0 <= row < len(self.entries) else None

    def _start(self, job, done):
        if self.worker is not None:
            return
        self.worker = ReviewWorker(job, self)
        for button in (self.add_btn, self.folder_btn, self.translate_btn, self.link_btn,
                       self.regenerate_btn, self.apply_btn, self.confirmed_btn):
            button.setEnabled(False)
        self.files.setEnabled(False)
        self.changes.setEnabled(False)
        self.cancel_btn.setText("취소")
        self.worker.progress.connect(self.status.setText)
        self.worker.result.connect(done)
        self.worker.failed.connect(self.status.setText)
        self.worker.finished.connect(self._stopped)
        self.worker.start()

    def _stopped(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        for button in (self.add_btn, self.folder_btn, self.translate_btn, self.link_btn,
                       self.regenerate_btn, self.confirmed_btn):
            button.setEnabled(True)
        self.files.setEnabled(True)
        self.changes.setEnabled(True)
        self.apply_btn.setEnabled(bool(self.previews))
        self.cancel_btn.setText("닫기")
        if self._close_pending:
            super().reject()

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "검사할 자막 선택", "", "자막 (*.srt *.ass *.smi)")
        if paths:
            self.scan([Path(p) for p in paths])

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "검사할 자막 폴더")
        if folder:
            recursive = self.recursive.isChecked()
            def job(cancel, progress):
                paths = []
                for path in Path(folder).rglob("*") if recursive else Path(folder).iterdir():
                    if cancel():
                        raise PipelineCancelled()
                    if path.is_file() and path.suffix.lower() in SUBTITLE_EXTENSIONS:
                        paths.append(path)
                return self._inspect(paths, cancel, progress)
            self._start(job, self._scanned)

    def _inspect(self, paths, cancel, progress):
        known = {e.path.resolve() for e in self.entries}
        entries, errors = [], []
        for i, path in enumerate(paths):
            if cancel():
                raise PipelineCancelled()
            if path.resolve() in known:
                continue
            progress(f"자막 검사 {i + 1}/{len(paths)} · {path.name}")
            try:
                entries.append(inspect_subtitle(path, self.config, cancel_check=cancel))
                known.add(path.resolve())
            except PipelineCancelled:
                raise
            except Exception as error:
                errors.append(f"{path.name}: {error}")
        return entries, errors

    def scan(self, paths):
        self._start(lambda cancel, progress: self._inspect(paths, cancel, progress), self._scanned)

    def _scanned(self, result):
        entries, errors = result
        self.entries.extend(entries)
        self._fill_files()
        self.status.setText(f"총 {len(self.entries)}개 · 생성 확인 {sum(e.confirmed for e in self.entries)}개 · 읽기 실패 {len(errors)}개"
                            + ("\n" + "\n".join(errors[:5]) if errors else ""))

    def _fill_files(self):
        checked = {self.files.item(row, 0).data(Qt.UserRole): self.files.item(row, 0).checkState()
                   for row in range(self.files.rowCount()) if self.files.item(row, 0) is not None}
        self.files.setRowCount(len(self.entries))
        for row, entry in enumerate(self.entries):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            key = str(entry.path.resolve())
            check.setData(Qt.UserRole, key)
            check.setCheckState(checked.get(key, Qt.Checked if entry.confirmed and entry.editable else Qt.Unchecked))
            self.files.setItem(row, 0, check)
            for col, text in enumerate((entry.path.name, entry.origin,
                                        "있음" if entry.source_cues else "없음",
                                        f"문자 검사 {len(entry.issues)}줄 · 의미 미검증" + (" · 읽기 전용" if not entry.editable else "")), 1):
                item = QTableWidgetItem(text)
                item.setToolTip(str(entry.path) if col == 1 else entry.note or text)
                self.files.setItem(row, col, item)
        self.files.resizeColumnToContents(0)
        self.files.resizeColumnToContents(2)
        self.files.resizeColumnToContents(3)
        if self.entries and self.files.currentRow() < 0:
            self.files.selectRow(0)
        self._show_preview()

    def _select_confirmed(self):
        for row, entry in enumerate(self.entries):
            self.files.item(row, 0).setCheckState(Qt.Checked if entry.confirmed and entry.editable else Qt.Unchecked)

    def _show_preview(self):
        self.changes.blockSignals(True)
        self.changes.setRowCount(0)
        entry = self._current()
        if entry is None:
            self.changes.blockSignals(False)
            return
        self.detail_label.setText(str(entry.path) + "\n" + (entry.note or
            ("원문을 사용해 번역을 다시 만들 수 있습니다." if entry.source_cues else "일본어 원문이 없어 문자·형식 검사만 가능합니다. 원본 영상을 연결하면 다시 인식할 수 있습니다.")))
        preview = self.previews.get(entry.path, {})
        issues = dict(entry.issues)
        self.changes.setRowCount(len(entry.cues))
        for row, cue in enumerate(entry.cues):
            candidate = preview.get(row)
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsEnabled | (Qt.ItemIsUserCheckable if candidate else Qt.NoItemFlags))
            check.setCheckState(Qt.Checked if candidate and candidate[1] else Qt.Unchecked)
            self.changes.setItem(row, 0, check)
            source = entry.source_cues[row].text if entry.source_cues else ""
            values = (f"{cue.start:.2f}–{cue.end:.2f}", source, cue.text,
                      candidate[0] if candidate else "", "변경 후보" if candidate else issues.get(row, "검사상 이상 없음 · 의미 미검증"))
            for col, text in enumerate(values, 1):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.changes.setItem(row, col, item)
        self.changes.resizeRowsToContents()
        self.changes.blockSignals(False)

    def _change_checked(self, item):
        entry = self._current()
        if entry and item.column() == 0 and item.row() in self.previews.get(entry.path, {}):
            text, _ = self.previews[entry.path][item.row()]
            self.previews[entry.path][item.row()] = text, item.checkState() == Qt.Checked

    def _translate(self):
        selected = self._selected()
        eligible = [e for e in selected if e.editable and e.source_cues and not e.metadata.get("bilingual")]
        if not eligible:
            self.status.setText("원문이 있고 편집 가능한 파일을 선택해 주세요. 이중 자막·스타일 보존 불가 파일은 전체 재생성을 이용하세요.")
            return
        def job(cancel, progress):
            from core.pipeline import _start_server_with_fallback
            from core.translate import translate_cues, TranslateReport
            from core.translate_gemma import translate_cues_gemma
            gemma = self.config.get("llm.translate_backend", "translategemma") == "translategemma"
            key = "translate_" if gemma else ""
            log = self.config.resolve_path("paths.cache_dir", "cache") / "subtitle-review.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            server = _start_server_with_fallback(self.config, self.config.resolve_path("llm." + key + "model"),
                self.config.get("llm." + key + "port", 8092 if gemma else 8090),
                self.config.get("llm." + key + "ctx_size", 3072 if gemma else 8192),
                self.config.get("llm.translate_extra_args", ["--no-jinja"]) if gemma else [], log,
                "llm." + key + "n_gpu_layers")
            results = {}
            try:
                for i, entry in enumerate(eligible):
                    if cancel():
                        raise PipelineCancelled()
                    progress(f"번역 후보 {i + 1}/{len(eligible)} · {entry.path.name}")
                    groups, row_glossaries = [], {}
                    for row, cue in enumerate(entry.source_cues):
                        terms = dict(self.glossary or glossary_for_row(entry, row))
                        row_glossaries[row] = terms
                        if not groups or groups[-1][0] != terms:
                            groups.append((terms, []))
                        groups[-1][1].append(cue)
                    out, report = {}, TranslateReport(total=len(entry.source_cues))
                    for terms, group in groups:
                        translated, partial = (translate_cues_gemma if gemma else translate_cues)(
                            group, server, self.config, glossary=terms,
                            source_lang=entry.metadata.get("source_language", "ja"),
                            target_lang=entry.metadata.get("target_language", "ko"), cancel_check=cancel,
                            on_batch_done=lambda: progress(f"번역 중 · {entry.path.name}"),
                            **({"context_cues": entry.source_cues} if gemma else {}))
                        out.update(translated)
                        report.failed_ids.extend(partial.failed_ids)
                        report.transliterated_ids.extend(partial.transliterated_ids)
                        report.issues.update(partial.issues)
                    flagged = set(report.failed_ids) | set(report.transliterated_ids)
                    replacements = {}
                    for row, source in enumerate(entry.source_cues):
                        text = wrap_lines(out[source.id], self.config.get("subtitle.max_chars_per_line", 20), self.config.get("subtitle.max_lines", 2))
                        if source.id not in flagged and text != entry.cues[row].text:
                            replacements[row] = (text, True)
                    results[entry.path] = replacements, report, row_glossaries
            finally:
                server.stop()
            return results
        self._start(job, self._translated)

    def _translated(self, result):
        for path, (changes, report, glossary) in result.items():
            self.previews[path] = changes
            self.preview_glossaries[path] = glossary
            entry = next(e for e in self.entries if e.path == path)
            entry.note = "후보 생성: " + report.summary() + " · 미번역·음역은 적용 후보에서 제외했습니다."
        self._show_preview()
        self.status.setText(f"번역 후보 준비 완료 · 변경 {sum(len(v[0]) for v in result.values())}줄. 비교 후 선택한 변경만 적용하세요.")

    def _apply(self):
        selected = [(entry, {row: value for row, (value, checked) in self.previews.get(entry.path, {}).items() if checked})
                    for entry in self._selected()]
        selected = [(entry, changes) for entry, changes in selected if changes]
        if not selected:
            self.status.setText("적용할 파일과 변경 줄을 선택해 주세요.")
            return
        if QMessageBox.question(self, "변경 적용", f"선택한 {len(selected)}개 파일의 {sum(len(c) for _, c in selected)}줄을 적용합니다.\n원본 백업을 같은 폴더에 남깁니다. 적용할까요?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        messages = []
        for entry, changes in selected:
            try:
                backup = apply_changes(entry, changes, glossary_by_row=self.preview_glossaries.get(entry.path))
                messages.append(f"적용: {entry.path.name} · 백업 {backup.name}")
                self.previews.pop(entry.path, None)
                self.preview_glossaries.pop(entry.path, None)
                self.entries[self.entries.index(entry)] = inspect_subtitle(entry.path, self.config, legacy=False)
            except Exception as error:
                messages.append(f"실패: {entry.path.name} · {error}")
        self._fill_files()
        self.apply_btn.setEnabled(bool(self.previews))
        self.status.setText("\n".join(messages))

    def _link_video(self):
        entry = self._current()
        if entry is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "이 자막의 원본 영상 선택", str(entry.path.parent), "영상 (*.mp4 *.mkv *.avi *.mov *.wmv *.ts *.webm *.m4v *.flv *.wav)")
        if path:
            entry.metadata["video"] = path
            self.status.setText(f"연결: {entry.path.name} → {Path(path).name}. '음성 인식부터 다시 만들기'로 대기열에 추가할 수 있습니다.")

    def _regenerate(self):
        videos = []
        for entry in self._selected():
            path = Path(entry.metadata.get("video", ""))
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(str(path))
        if not videos:
            self.status.setText("선택한 자막에 원본 영상이 연결되어 있지 않습니다. '원본 영상 연결'을 이용해 주세요.")
            return
        self.regenerate.emit(list(dict.fromkeys(videos)))
        self.accept()

    def reject(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.status.setText("취소 요청 중… 모델 요청이 끝나면 멈춥니다.")
            self._close_pending = True
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
