"""전체 흐름을 오케스트레이션한다.

핵심 원칙(DESIGN.md 4절): 여러 영상을 처리할 때 STT 모델과 LLM 서버를 영상마다 새로
띄우지 않고, 단계별로 한 번만 로드해 전체 영상에 적용한다. 각 단계 결과는 cache/에
저장해 중단 후 재개하거나 번역 설정만 바꿔 재실행할 수 있게 한다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from core.audio import extract_audio
from core.errors import PipelineCancelled
from core.llm import LlamaServer
from core.segmenter import Cue, build_cues, cues_from_json, cues_to_json
from core.stt import STTEngine, words_from_json, words_to_json
from core.translate import translate_cues
from core.writers import build_display_cues, write_ass, write_smi, write_srt

__all__ = ["run_batch", "PipelineCancelled", "cache_dir_for", "predict_output_paths"]

logger = logging.getLogger(__name__)

ProgressCB = Optional[Callable[[str], None]]
CancelCheck = Optional[Callable[[], bool]]
FractionCB = Optional[Callable[[float], None]]

# 영상 1개를 100으로 봤을 때 각 단계가 차지하는 비중. 번역이 필요 없는 경우(원어==번역 대상)에도
# translate 단계는 그대로 실행되어(캐시처럼 즉시) 63을 한 번에 더하므로 합이 항상 100을 유지한다.
STAGE_WEIGHTS = {"audio": 5, "stt": 25, "segment": 2, "translate": 63, "write": 5}


@dataclass
class ProgressTracker:
    """영상 여러 개를 한 번에 처리할 때 전체 진행률(0~1)을 계산해 콜백으로 알려준다."""

    total_videos: int
    on_fraction: FractionCB = None
    done: float = field(default=0.0, init=False)

    def add(self, amount: float) -> None:
        if amount <= 0:
            return
        self.done += amount
        if self.on_fraction:
            total = max(1, self.total_videos) * 100
            self.on_fraction(min(1.0, self.done / total))


def _report(progress_cb: ProgressCB, message: str) -> None:
    logger.info(message)
    if progress_cb:
        progress_cb(message)


def _check_cancel(cancel_check: CancelCheck) -> None:
    if cancel_check is not None and cancel_check():
        raise PipelineCancelled()


def cache_dir_for(video_path: Path, config) -> Path:
    root = config.resolve_path("paths.cache_dir", "cache")
    video_path = Path(video_path)
    h = hashlib.sha1(str(video_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return root / f"{video_path.stem}_{h}"


#  지원하는 자막 포맷. 키가 config의 subtitle.formats에 쓰는 이름이자 그대로 파일 확장자가 된다.
SUBTITLE_FORMATS = ("srt", "ass", "smi")


def predict_output_paths(videos: list[str | Path], out_dir: str | Path, config) -> dict[Path, list[Path]]:
    """실제로 실행하지 않고, 이 설정으로 만들어질 출력 파일 경로만 미리 계산한다.
    (기존 자막 파일을 덮어쓸지 미리 물어보는 용도, '폴더 추가' 때 이미 자막 있는지 확인하는 용도)
    """
    out_dir = Path(out_dir)
    formats = [f for f in config.get("subtitle.formats", ["srt", "ass"]) if f in SUBTITLE_FORMATS]
    emit_source = config.get("subtitle.emit_source", False)
    target_lang = config.get("target_language", "ko")
    source_lang = config.get("source_language", "ja")

    result: dict[Path, list[Path]] = {}
    for v in videos:
        v = Path(v)
        stem = v.stem
        paths = [out_dir / f"{stem}.{target_lang}.{ext}" for ext in formats]
        if emit_source and source_lang != target_lang:
            paths += [out_dir / f"{stem}.{source_lang}.{ext}" for ext in formats]
        result[v] = paths
    return result


def stage_audio(
    videos: list[Path],
    config,
    progress_cb: ProgressCB = None,
    cancel_check: CancelCheck = None,
    tracker: Optional[ProgressTracker] = None,
) -> dict[Path, Path]:
    wav_paths: dict[Path, Path] = {}
    for i, v in enumerate(videos, 1):
        _check_cancel(cancel_check)
        _report(progress_cb, f"[오디오 추출 {i}/{len(videos)}] {v.name}")
        cd = cache_dir_for(v, config)
        wav = cd / "audio.wav"
        extract_audio(v, wav)
        wav_paths[v] = wav
        if tracker:
            tracker.add(STAGE_WEIGHTS["audio"])
    return wav_paths


def stage_stt(
    videos: list[Path],
    wav_paths: dict[Path, Path],
    config,
    progress_cb: ProgressCB = None,
    cancel_check: CancelCheck = None,
    tracker: Optional[ProgressTracker] = None,
) -> dict[Path, "TranscriptionResult"]:
    precision = config.get("stt.precision", "turbo")
    model_dir = config.resolve_path(f"stt.models.{precision}")
    device = config.get("stt.device", "cuda")
    compute_type = config.get("stt.compute_type", "float16")
    batch_size = config.get("stt.batch_size", 16)
    beam_size = config.get("stt.beam_size", 1)
    source_lang = config.get("source_language")

    results = {}
    engine: STTEngine | None = None
    for i, v in enumerate(videos, 1):
        _check_cancel(cancel_check)
        cd = cache_dir_for(v, config)
        cache_file = cd / f"raw_words.{precision}.json"
        if cache_file.exists():
            results[v] = words_from_json(json.loads(cache_file.read_text(encoding="utf-8")))
            if tracker:
                tracker.add(STAGE_WEIGHTS["stt"])
            continue
        if engine is None:
            _report(progress_cb, f"STT 모델 로드 중 ({precision})")
            engine = STTEngine(model_dir, device=device, compute_type=compute_type)
        _report(progress_cb, f"[음성 인식 {i}/{len(videos)}] {v.name}")
        res = engine.transcribe(wav_paths[v], language=source_lang, batch_size=batch_size, beam_size=beam_size)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(words_to_json(res), ensure_ascii=False), encoding="utf-8")
        results[v] = res
        if tracker:
            tracker.add(STAGE_WEIGHTS["stt"])
    return results


def stage_segment(
    videos: list[Path],
    stt_results: dict,
    config,
    progress_cb: ProgressCB = None,
    cancel_check: CancelCheck = None,
    tracker: Optional[ProgressTracker] = None,
) -> dict[Path, list[Cue]]:
    precision = config.get("stt.precision", "turbo")
    results = {}
    for i, v in enumerate(videos, 1):
        _check_cancel(cancel_check)
        cd = cache_dir_for(v, config)
        # STT 정밀도(모델)가 바뀌면 단어 타임스탬프가 달라지므로 세그먼트도 다시 만들어야 한다
        cache_file = cd / f"cues_source.{precision}.json"
        if cache_file.exists():
            results[v] = cues_from_json(json.loads(cache_file.read_text(encoding="utf-8")))
            if tracker:
                tracker.add(STAGE_WEIGHTS["segment"])
            continue
        _report(progress_cb, f"[세그먼트 구성 {i}/{len(videos)}] {v.name}")
        cues = build_cues(stt_results[v].words, config)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cues_to_json(cues), ensure_ascii=False, indent=1), encoding="utf-8")
        results[v] = cues
        if tracker:
            tracker.add(STAGE_WEIGHTS["segment"])
    return results


def stage_translate(
    videos: list[Path],
    cues_map: dict[Path, list[Cue]],
    config,
    glossary: dict[str, str] | None = None,
    progress_cb: ProgressCB = None,
    cancel_check: CancelCheck = None,
    tracker: Optional[ProgressTracker] = None,
) -> dict[Path, dict[int, str]]:
    source_lang = config.get("source_language")
    target_lang = config.get("target_language")
    results: dict[Path, dict[int, str]] = {}
    translate_weight = STAGE_WEIGHTS["translate"]

    if source_lang == target_lang:
        for v in videos:
            results[v] = {c.id: c.text for c in cues_map[v]}
            if tracker:
                tracker.add(translate_weight)
        return results

    pending = []
    for v in videos:
        cache_file = cache_dir_for(v, config) / f"cues_translated.{target_lang}.json"
        if cache_file.exists():
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            results[v] = {int(k): val for k, val in raw.items()}
            if tracker:
                tracker.add(translate_weight)
        else:
            pending.append(v)

    if not pending:
        return results

    _check_cancel(cancel_check)
    log_dir = config.resolve_path("paths.cache_dir", "cache")
    log_dir.mkdir(parents=True, exist_ok=True)
    server = LlamaServer(
        server_exe=config.resolve_path("llm.server_exe"),
        model_path=config.resolve_path("llm.model"),
        port=config.get("llm.port", 8090),
        n_gpu_layers=config.get("llm.n_gpu_layers", 999),
        ctx_size=config.get("llm.ctx_size", 8192),
        log_path=log_dir / "llama-server.log",
    )
    _report(progress_cb, f"LLM 서버 시작 중 ({config.resolve_path('llm.model').name})")
    server.start()
    try:
        batch_size = config.get("llm.batch_sentences", 25)
        for i, v in enumerate(pending, 1):
            _check_cancel(cancel_check)
            _report(progress_cb, f"[번역 {i}/{len(pending)}] {v.name}")

            n_cues = len(cues_map[v])
            n_batches = max(1, math.ceil(n_cues / batch_size)) if n_cues else 1
            per_batch_weight = translate_weight / n_batches
            on_batch_done = (lambda w=per_batch_weight: tracker.add(w)) if tracker else None

            ko_map, report = translate_cues(
                cues_map[v],
                server,
                config,
                glossary=glossary,
                cancel_check=cancel_check,
                on_batch_done=on_batch_done,
            )
            if report.failed_ids:
                logger.warning(
                    "%s: 번역 실패로 원문이 유지된 항목 %d개: %s", v, len(report.failed_ids), report.failed_ids
                )
            if tracker and n_cues == 0:
                # translate_cues가 빈 목록은 배치를 만들지 않고 바로 반환해 on_batch_done이
                # 한 번도 안 불리므로, 이 영상 몫은 여기서 대신 채워준다.
                tracker.add(per_batch_weight)
            cache_file = cache_dir_for(v, config) / f"cues_translated.{target_lang}.json"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps({str(k): val for k, val in ko_map.items()}, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            results[v] = ko_map
    finally:
        server.stop()
    return results


def stage_write(
    videos: list[Path],
    cues_map: dict[Path, list[Cue]],
    translations: dict[Path, dict[int, str]],
    out_dir: Path,
    config,
    progress_cb: ProgressCB = None,
    tracker: Optional[ProgressTracker] = None,
) -> dict[Path, list[Path]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [f for f in config.get("subtitle.formats", ["srt", "ass"]) if f in SUBTITLE_FORMATS]
    emit_source = config.get("subtitle.emit_source", False)
    bilingual = config.get("subtitle.bilingual", False)
    target_lang = config.get("target_language", "ko")
    source_lang = config.get("source_language", "ja")

    writer_for = {
        "srt": (write_srt, config.get("subtitle.srt_encoding", "utf-8-sig")),
        "ass": (write_ass, config.get("subtitle.ass_encoding", "utf-8-sig")),
        "smi": (write_smi, config.get("subtitle.smi_encoding", "utf-8-sig")),
    }

    def _write_all(display, stem: str, lang: str, paths: list[Path]) -> None:
        for fmt in formats:
            writer, encoding = writer_for[fmt]
            p = out_dir / f"{stem}.{lang}.{fmt}"
            writer(display, p, encoding=encoding)
            paths.append(p)

    outputs: dict[Path, list[Path]] = {}
    for i, v in enumerate(videos, 1):
        _report(progress_cb, f"[자막 저장 {i}/{len(videos)}] {v.name}")
        stem = Path(v).stem
        cues = cues_map[v]
        src_map = {c.id: c.text for c in cues}
        bilingual_src = src_map if (bilingual and source_lang != target_lang) else None
        display = build_display_cues(cues, translations[v], config, source_text_by_id=bilingual_src)

        paths: list[Path] = []
        _write_all(display, stem, target_lang, paths)
        if emit_source and source_lang != target_lang:
            src_display = build_display_cues(cues, src_map, config)
            _write_all(src_display, stem, source_lang, paths)

        outputs[v] = paths
        logger.info("완료: %s -> %s", v, [str(p) for p in paths])
        if tracker:
            tracker.add(STAGE_WEIGHTS["write"])
    return outputs


def run_batch(
    videos: list[str | Path],
    out_dir: str | Path,
    config,
    glossary: dict[str, str] | None = None,
    progress_cb: ProgressCB = None,
    cancel_check: CancelCheck = None,
    progress_fraction_cb: FractionCB = None,
) -> dict[Path, list[Path]]:
    """videos를 일괄 처리한다. progress_cb(message)는 사람이 읽을 진행 상황 문자열을 받고,
    progress_fraction_cb(0.0~1.0)는 전체 진행률을, cancel_check()가 True를 반환하면
    다음 안전 지점(영상 단위, 번역은 배치 단위)에서 PipelineCancelled를 던진다.
    """
    videos = [Path(v) for v in videos]
    tracker = ProgressTracker(total_videos=len(videos), on_fraction=progress_fraction_cb)

    wav_paths = stage_audio(videos, config, progress_cb, cancel_check, tracker)
    stt_results = stage_stt(videos, wav_paths, config, progress_cb, cancel_check, tracker)
    cues_map = stage_segment(videos, stt_results, config, progress_cb, cancel_check, tracker)
    translations = stage_translate(videos, cues_map, config, glossary, progress_cb, cancel_check, tracker)
    return stage_write(videos, cues_map, translations, out_dir, config, progress_cb, tracker)
