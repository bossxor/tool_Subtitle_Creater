"""Source-language audio selection, including a real multitrack extraction."""
import json
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.audio import AudioExtractionError, _audio_stream, extract_audio


class AudioLanguageTests(unittest.TestCase):
    def test_commentary_title_is_respected_without_disposition_flag(self):
        streams = [
            {"index": 1, "tags": {"language": "jpn", "title": "Japanese Commentary"}, "disposition": {"default": 1}},
            {"index": 2, "tags": {"language": "jpn", "title": "Japanese"}, "disposition": {}},
        ]
        with patch("core.audio.subprocess.run", return_value=subprocess.CompletedProcess(
                [], 0, json.dumps({"streams": streams}), "")):
            self.assertEqual(_audio_stream(Path("video.mkv"), "ja"), 2)

    def test_japanese_commentary_loses_to_main_dialogue(self):
        streams = [
            {"index": 1, "tags": {"language": "ja"}, "disposition": {"default": 1, "comment": 1}},
            {"index": 2, "tags": {"language": "jpn"}, "disposition": {}},
        ]
        with patch("core.audio.subprocess.run", return_value=subprocess.CompletedProcess(
                [], 0, json.dumps({"streams": streams}), "")):
            self.assertEqual(_audio_stream(Path("video.mkv"), "ja"), 2)

    def test_untagged_audio_uses_default_and_empty_audio_fails(self):
        for streams, expected in [([{"index": 1}, {"index": 2, "disposition": {"default": 1}}], 2), ([], None)]:
            with self.subTest(streams=streams), patch("core.audio.subprocess.run", return_value=subprocess.CompletedProcess(
                    [], 0, json.dumps({"streams": streams}), "")):
                if expected is None:
                    with self.assertRaises(AudioExtractionError):
                        _audio_stream(Path("video.mkv"), "ja")
                else:
                    self.assertEqual(_audio_stream(Path("video.mkv"), "ja"), expected)

    def test_japanese_track_wins_over_english_default(self):
        streams = [
            {"index": 1, "tags": {"language": "eng"}, "disposition": {"default": 1}},
            {"index": 2, "tags": {"language": "jpn"}, "disposition": {}},
        ]
        calls = []
        def run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[0] == "ffprobe":
                return subprocess.CompletedProcess(cmd, 0, json.dumps({"streams": streams}), "")
            Path(cmd[-1]).write_bytes(b"complete")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        with tempfile.TemporaryDirectory() as folder, patch("core.audio.subprocess.run", side_effect=run):
            extract_audio(Path(folder) / "input.mkv", Path(folder) / "out.wav", language="ja")
        self.assertEqual(calls[-1][calls[-1].index("-map") + 1], "0:2")

    def test_probe_failure_never_publishes_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "out.wav"
            with patch("core.audio.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "bad input")):
                with self.assertRaises(AudioExtractionError):
                    extract_audio(Path(folder) / "input.mkv", output, language="ja")
            self.assertFalse(output.exists())

    def test_real_multitrack_audio_uses_japanese(self):
        with tempfile.TemporaryDirectory() as folder:
            video, output = Path(folder) / "two_tracks.mkv", Path(folder) / "selected.wav"
            subprocess.run([
                "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-f", "lavfi", "-i", "sine=frequency=880:duration=1", "-map", "0:a", "-map", "1:a",
                "-c:a", "pcm_s16le", "-metadata:s:a:0", "language=eng", "-metadata:s:a:1", "language=jpn",
                "-disposition:a:0", "default", "-disposition:a:1", "0", str(video),
            ], check=True, capture_output=True)
            extract_audio(video, output, language="ja")
            with wave.open(str(output)) as audio:
                samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
                peak = np.argmax(abs(np.fft.rfft(samples))) * audio.getframerate() / len(samples)
            self.assertAlmostEqual(peak, 880, delta=2)


if __name__ == "__main__":
    unittest.main()
