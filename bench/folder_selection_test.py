"""Folder discovery, explicit selection, and fast conversion regression tests."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QEventLoop, QTimer, Qt
from PySide6.QtWidgets import QApplication, QDialog

from core.config import Config
from core.errors import PipelineCancelled
from core.folder_scan import ConvertJob, EncodingFixJob, FolderScanPlan, plan_folder, run_conversions
from gui.folder_picker import FolderPickerDialog, FolderScanWorker
from gui.main_window import MainWindow
from gui.progress_model import StageTracker

SMI = "<SAMI><BODY><SYNC Start=0><P Class=KRCC>hello<SYNC Start=1500><P Class=KRCC>&nbsp;</BODY></SAMI>"


class FolderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = Config.load()
        self.config.set("paths.cache_dir", str(self.root / "cache"))
        self.out = self.root / "out"
        self.out.mkdir()
        self.videos = []
        for name in ("new", "old", "encoded"):
            video = self.root / f"{name}.mp4"
            video.touch()
            self.videos.append(video)
        self.smi = self.root / "old.ko.smi"
        self.smi.write_text(SMI, encoding="utf-8")
        self.srt = self.root / "encoded.srt"
        self.srt.write_text("1\n00:00:00,000 --> 00:00:01,000\ntext\n", encoding="utf-8")

    def wait_events(self, milliseconds=30):
        loop = QEventLoop()
        QTimer.singleShot(milliseconds, loop.quit)
        loop.exec()

    def picker(self):
        picker = FolderPickerDialog(self.root, self.out, self.config)
        self.addCleanup(picker.deleteLater)
        for _ in range(100):
            if picker.worker is None:
                break
            self.wait_events()
        self.assertIsNone(picker.worker, "Folder scan did not finish")
        return picker

    def test_scan_reads_each_directory_once_without_writing(self):
        before = {p: p.read_bytes() for p in self.root.iterdir() if p.is_file()}
        calls = []
        original = Path.iterdir
        def record(directory):
            calls.append(directory)
            return original(directory)
        with patch.object(Path, "iterdir", record):
            plan = plan_folder(self.videos, self.out, self.config)
        self.assertEqual(len(calls), 2)
        self.assertEqual(plan.generate, [self.videos[0]])
        self.assertEqual(len(plan.convert_jobs), 1)
        self.assertEqual(len(plan.encoding_fix_jobs), 1)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_scan_can_cancel_before_planning(self):
        with self.assertRaises(PipelineCancelled):
            plan_folder(self.videos, self.out, self.config, cancel_check=lambda: True)

    def test_subtitle_matching_and_deduplication(self):
        # A language suffix and dotted video name must match case-insensitively.
        dotted = self.root / "EP.01.MP4"
        dotted.touch()
        subtitle = self.root / "ep.01.KO.smi"
        subtitle.write_text(SMI, encoding="utf-8")
        plan = plan_folder([dotted, dotted], self.root, self.config)
        self.assertEqual(len(plan.convert_jobs), 1)
        self.assertEqual(plan.convert_jobs[0].smi_path, subtitle)

    def test_picker_starts_with_no_tasks_selected(self):
        picker = self.picker()
        self.assertEqual(len(picker.entries), 3)
        self.assertFalse(picker.apply_btn.isEnabled())
        self.assertEqual(picker.selected_plan(), FolderScanPlan([], [], []))
        self.assertFalse(picker.ai_chk.isChecked())
        self.assertFalse(picker.delete_chk.isChecked())
        self.assertFalse(picker.recursive_chk.isChecked())
        self.assertTrue(self.smi.exists())
        self.assertFalse(self.srt.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_select_filtered_files_and_preserve_selection(self):
        picker = self.picker()
        picker.kind_combo.setCurrentIndex(picker.kind_combo.findData("generate"))
        picker.select_visible()
        plan = picker.selected_plan()
        self.assertEqual(plan.generate, [self.videos[0]])
        self.assertEqual(plan.convert_jobs, [])
        self.assertEqual(plan.encoding_fix_jobs, [])
        picker.kind_combo.setCurrentIndex(picker.kind_combo.findData("convert"))
        picker.select_visible()
        self.assertEqual(len(picker.selected_plan().convert_jobs), 1)
        self.assertEqual(picker.selected_plan().generate, [self.videos[0]])
        self.assertTrue(picker.apply_btn.isEnabled())
        picker.clear_selection()
        self.assertFalse(picker.apply_btn.isEnabled())

    def test_subfolders_only_scanned_when_requested(self):
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "nested_video.mp4").touch()
        picker = self.picker()
        self.assertEqual(len(picker.entries), 3)
        picker.recursive_chk.setChecked(True)
        picker.start_scan()
        for _ in range(100):
            if picker.worker is None:
                break
            self.wait_events()
        self.assertEqual(len(picker.entries), 4)

    def test_fast_conversion_never_loads_ai_and_only_changes_selected_file(self):
        unselected = self.root / "other.ko.smi"
        unselected.write_text(SMI, encoding="utf-8")
        original = unselected.read_bytes()
        job = ConvertJob(self.videos[1], self.smi)
        plan = FolderScanPlan([], [job], [])
        with patch("core.folder_scan.LlamaServer") as server, patch("core.folder_scan.verify_cues") as verify:
            self.assertEqual(run_conversions(plan, ["srt"], self.config, False, verify_with_ai=False), (1, 0, 0))
        server.assert_not_called()
        verify.assert_not_called()
        self.assertTrue(self.smi.exists())
        self.assertIn("hello", job.written[0].read_text(encoding="utf-8-sig"))
        self.assertEqual(unselected.read_bytes(), original)
        self.assertFalse((self.root / "other.ko.srt").exists())

    def test_encoding_only_never_loads_ai(self):
        plan = FolderScanPlan([], [], [EncodingFixJob(self.srt)])
        with patch("core.folder_scan.LlamaServer") as server:
            self.assertEqual(run_conversions(plan, [], self.config, False, verify_with_ai=False), (0, 1, 0))
        server.assert_not_called()
        self.assertTrue(self.srt.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_smi_only_output_does_not_delete_source(self):
        plan = FolderScanPlan([], [ConvertJob(self.videos[1], self.smi)], [])
        with self.assertRaises(ValueError):
            run_conversions(plan, ["smi"], self.config, True, verify_with_ai=False)
        self.assertTrue(self.smi.exists())

    def test_cancelled_picker_never_runs_cleanup_or_adds_files(self):
        win = MainWindow()
        win.out_dir_edit.setText(str(self.out))
        with patch("gui.main_window.QFileDialog.getExistingDirectory", return_value=str(self.root)), \
             patch("gui.main_window.FolderPickerDialog") as picker_class, \
             patch.object(win, "_run_folder_cleanup") as cleanup:
            picker_class.return_value.exec.return_value = QDialog.Rejected
            win._on_add_folder()
        cleanup.assert_not_called()
        self.assertEqual(win.file_list.count(), 0)
        win.close()

    def test_generate_selection_only_adds_checked_video_without_starting(self):
        win = MainWindow()
        win.out_dir_edit.setText(str(self.out))
        with patch("gui.main_window.QFileDialog.getExistingDirectory", return_value=str(self.root)), \
             patch("gui.main_window.FolderPickerDialog") as picker_class, \
             patch.object(win, "_run_folder_cleanup") as cleanup, \
             patch("gui.main_window.list_missing") as assets:
            picker = picker_class.return_value
            picker.exec.return_value = QDialog.Accepted
            picker.selected_plan.return_value = FolderScanPlan([self.videos[0]], [], [])
            picker.ai_chk.isChecked.return_value = False
            win._on_add_folder()
        cleanup.assert_not_called()
        assets.assert_not_called()
        self.assertEqual(win.file_list.count(), 1)
        self.assertEqual(win.file_list.item(0).text(), str(self.videos[0]))
        self.assertIsNone(win.worker)
        win.close()

    def test_only_selected_cleanup_is_dispatched_without_ai_assets(self):
        win = MainWindow()
        win.out_dir_edit.setText(str(self.out))
        plan = FolderScanPlan([], [ConvertJob(self.videos[1], self.smi)], [])
        with patch("gui.main_window.QFileDialog.getExistingDirectory", return_value=str(self.root)), \
             patch("gui.main_window.FolderPickerDialog") as picker_class, \
             patch.object(win, "_run_folder_cleanup", return_value=(1, 0, 0)) as cleanup, \
             patch("gui.main_window.list_missing") as assets, \
             patch("gui.main_window.QMessageBox.information"):
            picker = picker_class.return_value
            picker.exec.return_value = QDialog.Accepted
            picker.selected_plan.return_value = plan
            picker.ai_chk.isChecked.return_value = False
            picker.delete_chk.isChecked.return_value = False
            win._on_add_folder()
        assets.assert_not_called()
        self.assertEqual(cleanup.call_args.args[0], plan)
        self.assertEqual(cleanup.call_args.kwargs, {"verify_with_ai": False})
        self.assertEqual(win.file_list.count(), 0)
        win.close()

    def test_closing_scan_waits_for_thread_before_dismissing(self):
        class SlowScanner(FolderScanWorker):
            def run(self):
                self.msleep(100)
        with patch("gui.folder_picker.FolderScanWorker", SlowScanner):
            picker = FolderPickerDialog(self.root, self.out, self.config)
            picker.show()
            worker = picker.worker
            picker.reject()
            self.assertIs(picker.worker, worker)
            self.assertTrue(picker.isVisible())
            self.wait_events(200)
            self.assertIsNone(picker.worker)
            self.assertFalse(picker.isVisible())
            picker.deleteLater()

    def test_fast_conversion_progress_counts(self):
        tracker = StageTracker()
        tracker.set_expected("smi 변환", 2)
        tracker.update("[smi 변환 2/2] old.mp4")
        self.assertIn("smi 변환: 2개 중 1개 완료", tracker.lines())


if __name__ == "__main__":
    unittest.main(verbosity=2)
