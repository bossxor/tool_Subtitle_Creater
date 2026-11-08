"""ffmpeg으로 영상에서 음성 인식용 오디오(16kHz mono wav)를 추출한다."""
from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

from core.procutil import NO_WINDOW_FLAGS

logger = logging.getLogger(__name__)


def _audio_stream(video_path: Path, language: str) -> int:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "a", "-show_streams",
        "-of", "json", str(video_path),
    ], capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW_FLAGS)
    if result.returncode != 0:
        raise AudioExtractionError(f"ffprobe 실패: {video_path}\n{result.stderr.strip()}")
    try:
        streams = json.loads(result.stdout)["streams"]
        aliases = {"ja": {"ja", "jpn"}, "ko": {"ko", "kor"}, "en": {"en", "eng"}}
        preferred = aliases.get(language.lower(), {language.lower()})
        matching = [s for s in streams if s.get("tags", {}).get("language", "").lower() in preferred]
        if not matching:
            logger.warning("%s 음성 트랙 표시가 없음: 기본 트랙 사용 (%s)", language, video_path.name)
        candidates = matching or streams
        # Commentary should not win merely because it is marked default.
        selected = min(candidates, key=lambda s: (
            bool(s.get("disposition", {}).get("comment", 0)) or bool(re.search(
                r"\bcommentary\b|コメンタリー|音声解説|코멘터리|해설",
                s.get("tags", {}).get("title", ""), re.IGNORECASE)),
            -int(s.get("disposition", {}).get("default", 0)), int(s["index"]),
        ))
        index = int(selected["index"])
        if index < 0:
            raise ValueError("negative stream index")
        return index
    except (ValueError, KeyError, TypeError) as exc:
        raise AudioExtractionError(f"음성 트랙을 찾을 수 없음: {video_path}") from exc


class AudioExtractionError(RuntimeError):
    pass


def extract_audio(video_path: Path, out_wav: Path, sample_rate: int = 16000, *, language: str | None = None) -> Path:
    """video_path의 오디오 트랙을 16kHz mono wav로 뽑아 out_wav에 저장한다.

    out_wav가 이미 있으면 다시 만들지 않는다 (캐시 재사용).
    """
    video_path = Path(video_path)
    out_wav = Path(out_wav)
    if out_wav.exists() and out_wav.stat().st_size > 0:
        return out_wav

    stream = _audio_stream(video_path, language) if language else None
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete WAV: ffmpeg can leave a nonempty partial file on failure.
    with tempfile.NamedTemporaryFile(dir=out_wav.parent, suffix=".wav", delete=False) as f:
        temporary = Path(f.name)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        *(["-map", f"0:{stream}"] if stream is not None else []),
        "-vn",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(temporary),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW_FLAGS)
        if result.returncode != 0:
            raise AudioExtractionError(
                f"ffmpeg 오디오 추출 실패: {video_path}\n{result.stderr.strip()}"
            )
        temporary.replace(out_wav)
    finally:
        temporary.unlink(missing_ok=True)
    return out_wav


def probe_duration(path: Path) -> float:
    """ffprobe로 길이를 초 단위로 반환한다."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW_FLAGS)
    if result.returncode != 0:
        raise AudioExtractionError(f"ffprobe 실패: {path}\n{result.stderr.strip()}")
    return float(result.stdout.strip())
