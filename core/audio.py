"""ffmpeg으로 영상에서 음성 인식용 오디오(16kHz mono wav)를 추출한다."""
from __future__ import annotations

import subprocess
from pathlib import Path

from core.procutil import NO_WINDOW_FLAGS


class AudioExtractionError(RuntimeError):
    pass


def extract_audio(video_path: Path, out_wav: Path, sample_rate: int = 16000) -> Path:
    """video_path의 오디오 트랙을 16kHz mono wav로 뽑아 out_wav에 저장한다.

    out_wav가 이미 있으면 다시 만들지 않는다 (캐시 재사용).
    """
    video_path = Path(video_path)
    out_wav = Path(out_wav)
    if out_wav.exists() and out_wav.stat().st_size > 0:
        return out_wav

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW_FLAGS)
    if result.returncode != 0:
        raise AudioExtractionError(
            f"ffmpeg 오디오 추출 실패: {video_path}\n{result.stderr.strip()}"
        )
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
