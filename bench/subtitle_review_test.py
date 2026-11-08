"""Provenance, format preservation and safe updates using temporary files."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import Config
from core.segmenter import Cue
from core.writers import DisplayCue, write_srt, write_ass, write_smi
from core.pipeline import cache_dir_for, _write_json
from core.segmenter import cues_to_json


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("core.subtitle_review"), "review module not implemented")
        from core import subtitle_review
        self.api = subtitle_review
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = Config.load()
        self.config.set("paths.cache_dir", str(self.root / "cache"))
        self.video = self.root / "movie.mp4"
        self.video.write_bytes(b"video")
        self.path = self.root / "movie.ko.srt"
        self.source = [Cue(7, 0, 2, "おかえり。")]
        write_srt([DisplayCue(0, 2, "돌아오셨군요.")], self.path)

    def record(self):
        self.api.record_generation(self.path, self.video, self.source, self.config)

    def test_external_file_is_never_auto_selected(self):
        entry = self.api.inspect_subtitle(self.path, self.config)
        self.assertFalse(entry.confirmed)
        self.assertEqual(entry.origin, "출처 불명")

    def test_manifest_detects_manual_edits(self):
        self.record()
        self.assertTrue(self.api.inspect_subtitle(self.path, self.config).confirmed)
        self.path.write_text(self.path.read_text(encoding="utf-8-sig").replace("돌아오셨군요.", "왔어?"), encoding="utf-8-sig")
        entry = self.api.inspect_subtitle(self.path, self.config)
        self.assertFalse(entry.confirmed)
        self.assertEqual(entry.origin, "수정됨")

    def test_update_preserves_timestamps_and_backups_original(self):
        self.record()
        entry = self.api.inspect_subtitle(self.path, self.config)
        original = self.path.read_bytes()
        backup = self.api.apply_changes(entry, {0: "왔어?"})
        self.assertEqual(backup.read_bytes(), original)
        text = self.path.read_text(encoding="utf-8-sig")
        self.assertIn("00:00:00,000 --> 00:00:02,000", text)
        self.assertIn("왔어?", text)
        self.assertTrue(self.api.inspect_subtitle(self.path, self.config).confirmed)

    def test_selected_glossary_changes_preserve_untouched_rows(self):
        source = [Cue(0, 0, 2, "東京"), Cue(1, 3, 5, "東京")]
        write_srt([DisplayCue(0, 2, "Tokyo"), DisplayCue(3, 5, "Tokyo")], self.path)
        self.api.record_generation(self.path, self.video, source, self.config, glossary={"東京": "Tokyo"})
        entry = self.api.inspect_subtitle(self.path, self.config)
        self.api.apply_changes(entry, {0: "Tokio"}, glossary_by_row={0: {"東京": "Tokio"}})
        updated = self.api.inspect_subtitle(self.path, self.config)
        self.assertTrue(updated.confirmed)
        self.assertEqual(updated.issues, [])
        self.assertEqual([c.text for c in updated.cues], ["Tokio", "Tokyo"])

    def test_concurrent_edits_prevent_apply(self):
        entry = self.api.inspect_subtitle(self.path, self.config)
        self.path.write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.api.apply_changes(entry, {0: "왔어?"})
        self.assertEqual(self.path.read_text(), "changed")

    def test_changed_timing_does_not_reuse_unaligned_source(self):
        self.record()
        self.path.write_text(self.path.read_text(encoding="utf-8-sig").replace("00:00:02,000", "00:00:03,000"), encoding="utf-8-sig")
        self.assertEqual(self.api.inspect_subtitle(self.path, self.config).source_cues, [])

    def test_all_generated_formats_can_be_inspected_and_patched(self):
        for writer, ext in [(write_srt, "srt"), (write_ass, "ass"), (write_smi, "smi")]:
            path = self.root / f"movie.ko.{ext}"
            writer([DisplayCue(0, 2, "돌아오셨군요.")], path)
            self.api.record_generation(path, self.video, self.source, self.config)
            entry = self.api.inspect_subtitle(path, self.config)
            self.assertEqual(len(entry.cues), 1)
            self.api.apply_changes(entry, {0: "왔어?"})
            self.assertIn("왔어?", path.read_text(encoding="utf-8-sig"))
            if ext == "ass":
                self.assertIn("[V4+ Styles]", path.read_text(encoding="utf-8-sig"))

    def test_unsupported_styled_text_is_read_only(self):
        self.path.write_text("1\n00:00:00,000 --> 00:00:02,000\n<i>안녕</i>\n", encoding="utf-8")
        entry = self.api.inspect_subtitle(self.path, self.config)
        self.assertFalse(entry.editable)
        with self.assertRaises(ValueError):
            self.api.apply_changes(entry, {0: "왔어?"})

    def test_legacy_content_match_survives_first_update(self):
        cache = cache_dir_for(self.video, self.config)
        _write_json(cache / "cues_source.large.json", cues_to_json(self.source))
        _write_json(cache / "cues_translated.ko.json", {"7": "돌아오셨군요."})
        entry = self.api.inspect_subtitle(self.path, self.config)
        self.assertEqual(entry.origin, "캐시 일치")
        self.api.apply_changes(entry, {0: "왔어?"})
        updated = self.api.inspect_subtitle(self.path, self.config)
        self.assertEqual(len(updated.source_cues), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
