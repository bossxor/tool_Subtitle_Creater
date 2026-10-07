"""faster-whisper 래퍼. 단어 단위 타임스탬프를 뽑아 segmenter가 자막 길이로 재구성할 수 있게 한다."""
from __future__ import annotations

import os
import site
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _candidate_search_dirs() -> tuple[list[Path], list[Path]]:
    """nvidia-*-cu12 DLL이 있을 만한 폴더들을 반환한다.

    (nvidia 하위 폴더를 뒤질 곳, PyInstaller가 DLL을 평평하게 모아뒀을 때를 대비해
    통째로 검색 경로에 추가할 곳) 두 그룹으로 나눈다 — 개발 환경(venv site-packages)과
    PyInstaller로 묶인 exe(내부 _internal 폴더) 둘 다 커버한다.
    """
    nested_bases: list[Path] = []
    try:
        nested_bases.extend(Path(sp) for sp in site.getsitepackages())
    except Exception:
        pass

    flat_bases: list[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        flat_bases.append(exe_dir)
        flat_bases.append(exe_dir / "_internal")
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        flat_bases.append(Path(meipass))
    return nested_bases, flat_bases


def _setup_cuda_dll_path() -> None:
    """nvidia-*-cu12 패키지의 DLL을 PATH/DLL 검색 경로에 올린다 (Windows)."""
    nested_bases, flat_bases = _candidate_search_dirs()
    for base in nested_bases:
        nvidia_dir = base / "nvidia"
        if not nvidia_dir.exists():
            continue
        for bin_dir in nvidia_dir.glob("*/bin"):
            try:
                os.add_dll_directory(str(bin_dir))
            except (AttributeError, OSError):
                pass
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    # PyInstaller로 묶이면 DLL이 하위 폴더 없이 exe 폴더/_internal 바로 아래 평평하게 모일 수 있다
    for base in flat_bases:
        if not base.exists():
            continue
        try:
            os.add_dll_directory(str(base))
        except (AttributeError, OSError):
            pass


_setup_cuda_dll_path()

from faster_whisper import BatchedInferencePipeline, WhisperModel  # noqa: E402


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    words: list[Word]
    language: str
    language_probability: float
    duration: float


class STTEngine:
    """모델을 한 번 로드해 여러 영상을 순서대로 처리하기 위한 클래스."""

    def __init__(self, model_dir: Path, device: str = "cuda", compute_type: str = "float16"):
        self.model = WhisperModel(str(model_dir), device=device, compute_type=compute_type)
        self.pipeline = BatchedInferencePipeline(self.model)

    def transcribe(
        self,
        wav_path: Path,
        language: Optional[str] = None,
        batch_size: int = 16,
        beam_size: int = 1,
    ) -> TranscriptionResult:
        segments, info = self.pipeline.transcribe(
            str(wav_path),
            language=language,
            beam_size=beam_size,
            batch_size=batch_size,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        words: list[Word] = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    words.append(Word(start=w.start, end=w.end, text=w.word))
            elif seg.text.strip():
                # word_timestamps가 비어 있는 드문 경우를 대비한 폴백
                words.append(Word(start=seg.start, end=seg.end, text=seg.text.strip()))

        return TranscriptionResult(
            words=words,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )


def words_to_json(result: TranscriptionResult) -> dict:
    return {
        "language": result.language,
        "language_probability": result.language_probability,
        "duration": result.duration,
        "words": [{"start": w.start, "end": w.end, "text": w.text} for w in result.words],
    }


def words_from_json(data: dict) -> TranscriptionResult:
    return TranscriptionResult(
        words=[Word(**w) for w in data["words"]],
        language=data["language"],
        language_probability=data["language_probability"],
        duration=data["duration"],
    )
