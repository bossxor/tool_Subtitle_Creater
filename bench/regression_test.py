"""Regression checks without model downloads or GPU inference.

Run: .venv\Scripts\python.exe bench/regression_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QEventLoop, QThread, QTimer, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from core.audio import AudioExtractionError, extract_audio
from core.config import Config
from core.errors import PipelineCancelled
from core.pipeline import (
    ProgressTracker, _stt_cache_file, _translation_cache_file, _write_json,
    cache_dir_for, stage_audio, stage_translate, stage_write,
)
from core.segmenter import Cue
from core.setup_assets import WHISPER_FILES, _fetch_whisper_model, list_missing
from gui.main_window import MainWindow


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "video.mp4"
        self.video.write_bytes(b"video")
        self.config = Config.load()
        self.config.set("paths.cache_dir", str(self.root / "cache"))
        self.cues = [Cue(0, 0, 2, "Hello")]

    def test_replaced_video_does_not_reuse_audio(self):
        before = cache_dir_for(self.video, self.config)
        self.video.write_bytes(b"replacement video")
        self.assertNotEqual(before, cache_dir_for(self.video, self.config))

    def test_stt_cache_depends_on_language_and_beam(self):
        before = _stt_cache_file(self.video, self.config)
        self.config.set("source_language", "en")
        self.assertNotEqual(before, _stt_cache_file(self.video, self.config))
        before = _stt_cache_file(self.video, self.config)
        self.config.set("stt.beam_size", 5)
        self.assertNotEqual(before, _stt_cache_file(self.video, self.config))

    def test_translation_uses_cues_model_and_glossary(self):
        def key(cues=None, glossary=None):
            return _translation_cache_file(self.video, cues or self.cues, self.config, glossary)
        before = key()
        self.assertNotEqual(before, key([Cue(0, 0, 2, "Goodbye")]))
        self.assertNotEqual(before, key(glossary={"Alice": "A"}))
        self.config.set("llm.translate_backend", "json")
        self.assertNotEqual(before, key())

    def test_output_options_do_not_invalidate_translation(self):
        before = _translation_cache_file(self.video, self.cues, self.config, None)
        self.config.set("subtitle.bilingual", True)
        self.assertEqual(before, _translation_cache_file(self.video, self.cues, self.config, {}))

    def test_completed_stt_skips_audio_even_if_wav_is_missing(self):
        _write_json(_stt_cache_file(self.video, self.config), {"words": []})
        with patch("core.pipeline.extract_audio") as extract:
            stage_audio([self.video], self.config)
        extract.assert_not_called()

    def test_silent_video_needs_no_translation_server(self):
        fractions = []
        tracker = ProgressTracker(1, fractions.append)
        with patch("core.pipeline._start_server_with_fallback") as start:
            result = stage_translate([self.video], {self.video: []}, self.config, tracker=tracker)
        start.assert_not_called()
        self.assertEqual(result, {self.video: {}})
        self.assertEqual(fractions, [0.63])

    def test_translation_cache_hit_needs_no_server(self):
        cache = _translation_cache_file(self.video, self.cues, self.config, None)
        _write_json(cache, {"0": "translated"})
        with patch("core.pipeline._start_server_with_fallback") as start:
            result = stage_translate([self.video], {self.video: self.cues}, self.config)
        start.assert_not_called()
        self.assertEqual(result[self.video], {0: "translated"})

    def test_failed_ffmpeg_does_not_publish_partial_wav(self):
        out = self.root / "audio.wav"
        def fail(cmd, **kwargs):
            Path(cmd[-1]).write_bytes(b"partial WAV")
            return type("Result", (), {"returncode": 1, "stderr": "failed"})()
        with patch("core.audio.subprocess.run", side_effect=fail):
            with self.assertRaises(AudioExtractionError):
                extract_audio(self.video, out)
        self.assertFalse(out.exists())
        self.assertEqual(list(self.root.glob("*.wav")), [])

    def test_cancel_before_saving_creates_no_subtitles(self):
        with self.assertRaises(PipelineCancelled):
            stage_write([self.video], {self.video: self.cues}, {self.video: {0: "text"}},
                        self.root, self.config, cancel_check=lambda: True)
        self.assertEqual(list(self.root.glob("*.srt")), [])

    def test_cancel_is_honored_when_translation_is_skipped(self):
        self.config.set("target_language", self.config.get("source_language"))
        with self.assertRaises(PipelineCancelled):
            stage_translate([self.video], {self.video: self.cues}, self.config,
                            cancel_check=lambda: True)

    def test_missing_vocabulary_is_detected_and_repaired(self):
        model = self.root / "model"
        model.mkdir()
        for name in WHISPER_FILES:
            (model / name).write_bytes(b"data")
        (model / "vocabulary.json").write_bytes(b"")
        self.config.set("stt.models.large", str(model))
        self.config.set("stt.precision", "large")
        tasks = list_missing(self.config)
        self.assertTrue(any(t.check_path == model / "model.bin" for t in tasks))
        with patch("core.setup_assets._download_file") as download:
            _fetch_whisper_model("test/repo", model, "test")(None)
        self.assertEqual(download.call_count, 1)
        self.assertEqual(download.call_args.args[1], model / "vocabulary.json")

    def test_locked_executable_prevents_moving_user_data(self):
        import build_exe
        (self.root / "Subtitle_Tool.exe").touch()
        with patch.object(build_exe, "DIST_APP", self.root), \
             patch.object(Path, "open", side_effect=PermissionError("in use")), \
             patch.object(build_exe, "preserve_user_data") as preserve:
            with self.assertRaises(SystemExit):
                build_exe.main()
        preserve.assert_not_called()


class SlowFinishingWorker(QThread):
    finished_ok = Signal(dict)
    progress = Signal(str)
    progress_fraction = Signal(float)
    log = Signal(str)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, *args, parent=None, **kwargs):
        super().__init__(parent)
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True

    def run(self):
        self.finished_ok.emit({})
        self.msleep(150)  # A terminal result is not the QThread.finished signal.


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.addCleanup(self.window.close)

    def wait_events(self, milliseconds):
        loop = QEventLoop()
        QTimer.singleShot(milliseconds, loop.quit)
        loop.exec()

    def test_small_window_and_config_defaults(self):
        self.window.resize(980, 640)
        self.window.show()
        self.app.processEvents()
        self.assertEqual(self.window.height(), 640)
        self.assertTrue(self.window.empty_state.isVisible())
        self.assertEqual(self.window.precision_combo.currentData(), Config.load().get("stt.precision"))

    def test_worker_retained_until_thread_finished(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "a.mp4"
            video.touch()
            self.window._add_files([str(video)])
            self.window.out_dir_edit.setText(td)
            with patch("gui.main_window.PipelineWorker", SlowFinishingWorker), \
                 patch("gui.main_window.list_missing", return_value=[]), \
                 patch.object(QMessageBox, "information"):
                self.window._on_start()
                self.wait_events(40)
                self.assertIsNotNone(self.window.worker)
                self.assertFalse(self.window.start_btn.isEnabled())
                self.assertFalse(self.window.settings_content.isEnabled())
                self.wait_events(200)
                self.assertIsNone(self.window.worker)
                self.assertTrue(self.window.start_btn.isEnabled())

    def test_close_waits_asynchronously_for_worker(self):
        self.window.show()
        worker = SlowFinishingWorker(parent=self.window)
        self.window.worker = worker
        worker.finished.connect(self.window._on_worker_stopped)
        worker.start()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.window.close()
        self.assertTrue(self.window.isVisible())
        self.assertTrue(worker.cancel_requested)
        self.wait_events(250)
        self.assertFalse(self.window.isVisible())
        self.assertIsNone(self.window.worker)


if __name__ == "__main__":
    unittest.main(verbosity=2)
