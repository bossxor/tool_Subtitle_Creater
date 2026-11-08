"""Exercise review/preview/apply and quality presentation without model inference."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox
from core.config import Config
from core.segmenter import Cue
from core.subtitle_review import record_generation, inspect_subtitle, apply_changes
from core.translate import TranslateReport
from core.writers import DisplayCue, write_srt
from gui.main_window import MainWindow
from gui.subtitle_review import SubtitleReviewDialog


class Server:
    generation = restart_count = 0
    def complete(self, *args, **kwargs):
        return "왔어?"
    def stop(self):
        pass


class ReviewGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_worker(self, dialog):
        deadline = time.monotonic() + 10
        while dialog.worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertIsNone(dialog.worker)

    def test_scan_preview_apply_preserves_original_until_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = Config.load()
            config.set("paths.cache_dir", str(root / "cache"))
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            path = root / "movie.ko.srt"
            write_srt([DisplayCue(0, 2, "돌아오셨군요.")], path)
            record_generation(path, video, [Cue(0, 0, 2, "おかえり。")], config)
            original = path.read_bytes()
            dialog = SubtitleReviewDialog(config)
            dialog.scan([path])
            self.wait_worker(dialog)
            self.assertEqual(dialog.files.item(0, 0).checkState(), Qt.Checked)
            self.assertEqual(path.read_bytes(), original)
            with patch("core.pipeline._start_server_with_fallback", return_value=Server()):
                dialog._translate()
                self.wait_worker(dialog)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(dialog.changes.item(0, 4).text(), "왔어?")
            with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
                dialog._apply()
            self.assertIn("왔어?", path.read_text(encoding="utf-8-sig"))
            self.assertEqual(next(root.glob("*.backup-*")).read_bytes(), original)
            dialog.close()

    def test_quality_failures_are_visible_in_main_results(self):
        window = MainWindow()
        window.worker = type("Worker", (), {"reports": {Path("v.mp4"): TranslateReport(total=1, failed_ids=[0])}})()
        with patch.object(QMessageBox, "information"):
            window._on_finished_ok({"v.mp4": ["v.ko.srt"]})
        self.assertEqual(window._result_rows[0][2], "일부 미번역")
        window.worker = None
        window.close()

    def test_reprocess_option_is_passed_to_pipeline(self):
        window = MainWindow()
        self.assertTrue(hasattr(window, "reprocess_combo"))
        window.reprocess_combo.setCurrentIndex(window.reprocess_combo.findData("all"))
        self.assertEqual(window._build_config().get("pipeline.reprocess"), "all")
        window.close()

    def test_quality_translation_choice_reaches_pipeline_with_one_worker(self):
        window = MainWindow()
        index = next(i for i in range(window.translation_combo.count())
                     if "27b" in window.translation_combo.itemData(i))
        window.translation_combo.setCurrentIndex(index)
        config = window._build_config()
        self.assertIn("27b", config.get("llm.translate_model"))
        self.assertEqual(config.get("llm.translate_parallel"), 1)
        self.assertIn("32GB", window.translation_combo.toolTip())
        window.close()

    def test_japanese_model_is_disabled_when_source_language_changes(self):
        window = MainWindow()
        index = window.precision_combo.findData("japanese")
        self.assertGreaterEqual(index, 0)
        window.precision_combo.setCurrentIndex(index)
        self.assertEqual(window._build_config().get("stt.precision"), "japanese")
        window.source_lang_combo.setCurrentIndex(window.source_lang_combo.findData("en"))
        self.assertEqual(window.precision_combo.currentData(), "large")
        self.assertFalse(window.precision_combo.model().item(index).isEnabled())
        window.close()

    def test_retranslation_preserves_selected_rows_glossary_and_scene_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = Config.load()
            config.set("paths.cache_dir", str(root / "cache"))
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            path = root / "movie.ko.srt"
            source = [Cue(0, 0, 2, "東京"), Cue(1, 2.1, 4, "東京")]
            write_srt([DisplayCue(0, 2, "Tokyo"), DisplayCue(2.1, 4, "Tokyo")], path)
            record_generation(path, video, source, config, glossary={"東京": "Tokyo"})
            apply_changes(inspect_subtitle(path, config), {0: "Tokio"}, glossary_by_row={0: {"東京": "Tokio"}})
            original = path.read_bytes()
            dialog = SubtitleReviewDialog(config)
            dialog.scan([path])
            self.wait_worker(dialog)
            calls = []
            def translate(cues, server, config, *, glossary, context_cues, **kwargs):
                calls.append((glossary, len(context_cues)))
                return {c.id: glossary["東京"] for c in cues}, TranslateReport(total=len(cues))
            with patch("core.pipeline._start_server_with_fallback", return_value=Server()), \
                 patch("core.translate_gemma.translate_cues_gemma", side_effect=translate):
                dialog._translate()
                self.wait_worker(dialog)
            self.assertEqual(calls, [({"東京": "Tokio"}, 2), ({"東京": "Tokyo"}, 2)])
            self.assertEqual(dialog.previews[path], {})
            self.assertEqual(path.read_bytes(), original)
            dialog.close()

    def test_excluded_file_stays_excluded_when_more_files_are_scanned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = Config.load()
            config.set("paths.cache_dir", str(root / "cache"))
            video = root / "movie.mp4"
            video.write_bytes(b"v")
            paths = []
            for name in ("a", "b"):
                path = root / f"{name}.ko.srt"
                write_srt([DisplayCue(0, 2, "안녕")], path)
                record_generation(path, video, [Cue(0, 0, 2, "やあ")], config)
                paths.append(path)
            dialog = SubtitleReviewDialog(config)
            dialog.scan([paths[0]])
            self.wait_worker(dialog)
            dialog.files.item(0, 0).setCheckState(Qt.Unchecked)
            dialog.scan([paths[1]])
            self.wait_worker(dialog)
            self.assertEqual(dialog.files.item(0, 0).checkState(), Qt.Unchecked)
            dialog.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
