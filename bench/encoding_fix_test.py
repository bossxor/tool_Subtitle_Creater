"""Encoding repair must work on hidden Windows subtitles and be idempotent."""
from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.encoding_fix import fix_encoding_file, needs_fix
from core.config import Config
from core.folder_scan import plan_folder


class EncodingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_repair_preserves_crlf_and_korean_and_is_idempotent(self):
        text = "1\r\n00:00:00,000 --> 00:00:01,000\r\n자막 테스트\r\n"
        for encoding in ("utf-8", "cp949"):
            with self.subTest(encoding=encoding):
                path = self.root / f"{encoding}.srt"
                path.write_bytes(text.encode(encoding))
                self.assertTrue(needs_fix(path))
                self.assertTrue(fix_encoding_file(path))
                self.assertEqual(path.read_bytes(), text.encode("utf-8-sig"))
                timestamp = path.stat().st_mtime_ns
                self.assertFalse(needs_fix(path))
                self.assertFalse(fix_encoding_file(path))
                self.assertEqual(timestamp, path.stat().st_mtime_ns)

    @unittest.skipUnless(os.name == "nt", "Windows file attributes")
    def test_hidden_file_is_repaired_without_changing_attributes(self):
        path = self.root / "hidden.ass"
        text = "[Script Info]\r\nTitle: 인코딩 확인\r\n"
        path.write_bytes(text.encode("cp949"))
        self.assertTrue(ctypes.windll.kernel32.SetFileAttributesW(str(path), 2))
        self.assertTrue(fix_encoding_file(path))
        self.assertEqual(path.read_bytes(), text.encode("utf-8-sig"))
        self.assertTrue(path.stat().st_file_attributes & 2)
        self.assertFalse(needs_fix(path))

    @unittest.skipUnless(os.name == "nt", "Windows file attributes")
    def test_hidden_repaired_file_does_not_reappear_in_scan(self):
        video = self.root / "episode.mp4"
        video.touch()
        path = self.root / "episode.srt"
        path.write_bytes(b"1\r\n00:00:00,000 --> 00:00:01,000\r\nhello\r\n")
        self.assertTrue(ctypes.windll.kernel32.SetFileAttributesW(str(path), 2))
        config = Config.load()
        self.assertEqual(len(plan_folder([video], self.root, config).encoding_fix_jobs), 1)
        self.assertTrue(fix_encoding_file(path))
        self.assertEqual(plan_folder([video], self.root, config).encoding_fix_jobs, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
