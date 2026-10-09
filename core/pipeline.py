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
from core.stt import STTEngine, confidence_summary, words_from_json, words_to_json
from core.translate import translate_cues
from core.translate_gemma import translate_cues_gemma
from core.worklog import WorkLog
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


def _confidence_text(result) -> str:
    """작업 로그용: 음성 인식이 얼마나 불안했는지. 번역이 이상할 때 원인이 인식인지 번역인지 가리는 단서."""
    summary = confidence_summary(result)
    if summary is None:
        return ""
    avg, low, label = summary
    text = f", 인식 확신도 평균 {avg:.2f}, 낮은 단어 {low:.1f}% ({label})"
    if label == "불안정":
        text += " - 배경음·겹치는 말소리 때문에 인식이 불안정함. 번역이 이상하면 번역보다 음성 인식이 원인일 가능성이 큼"
    return text


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
    worklog: Optional[WorkLog] = None,
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
            if worklog:
                worklog.event(
                    "음성 인식", v.name, "캐시 사용",
                    f"단어 {len(results[v].words)}개 (이전 결과 재사용){_confidence_text(results[v])}",
                )
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
        if worklog:
            worklog.event(
                "음성 인식", v.name, "완료",
                f"모델 {precision}, 감지 언어 {res.language}, 단어 {len(res.words)}개, 길이 {res.duration / 60:.1f}분"
                f"{_confidence_text(res)}",
            )
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
    worklog: Optional[WorkLog] = None,
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
            if worklog:
                worklog.event("자막 구간 나누기", v.name, "캐시 사용", f"{len(results[v])}줄")
            if tracker:
                tracker.add(STAGE_WEIGHTS["segment"])
            continue
        _report(progress_cb, f"[세그먼트 구성 {i}/{len(videos)}] {v.name}")
        cues = build_cues(stt_results[v].words, config)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cues_to_json(cues), ensure_ascii=False, indent=1), encoding="utf-8")
        results[v] = cues
        if worklog:
            worklog.event("자막 구간 나누기", v.name, "완료", f"{len(cues)}줄")
        if tracker:
            tracker.add(STAGE_WEIGHTS["segment"])
    return results


def _start_server_with_fallback(
    config, model_path: Path, port: int, ctx: int, extra_args, log_path: Path, ngl_key: str = "llm.n_gpu_layers"
) -> LlamaServer:
    """GPU 메모리가 모자라 서버가 못 뜨면 GPU에 올리는 층 수를 줄여서 다시 시도한다 (느려지지만 동작).

    설정값이 "auto"면 -ngl을 넘기지 않아 llama.cpp가 남은 GPU 메모리에 맞춰 층 수를 정한다.
    """
    first = config.get(ngl_key, 999)
    if first == "auto":
        first = None
    last_error: Exception | None = None
    for ngl in dict.fromkeys([first, 24, 0] if first is None else [first, 36, 24, 0]):
        server = LlamaServer(
            server_exe=config.resolve_path("llm.server_exe"),
            model_path=model_path,
            port=port,
            n_gpu_layers=ngl,
            ctx_size=ctx,
            log_path=log_path,
            extra_args=extra_args,
        )
        try:
            server.start()
            if ngl != first:
                logger.warning("GPU 메모리 부족으로 GPU 층 수를 %s로 낮춰 번역 서버를 시작함 (느려질 수 있음)", ngl)
            return server
        except Exception as e:  # noqa: BLE001
            last_error = e
            server.stop()
            logger.warning("번역 서버 시작 실패(GPU 층 %s): %s", ngl, str(e).splitlines()[0])
    raise last_error  # type: ignore[misc]


def stage_translate(
    videos: list[Path],
    cues_map: dict[Path, list[Cue]],
    config,
    glossary: dict[str, str] | None = None,
    progress_cb: ProgressCB = None,
    cancel_check: CancelCheck = None,
    tracker: Optional[ProgressTracker] = None,
    worklog: Optional[WorkLog] = None,
) -> dict[Path, dict[int, str]]:
    source_lang = config.get("source_language")
    target_lang = config.get("target_language")
    results: dict[Path, dict[int, str]] = {}
    translate_weight = STAGE_WEIGHTS["translate"]

    if source_lang == target_lang:
        for v in videos:
            results[v] = {c.id: c.text for c in cues_map[v]}
            if worklog:
                worklog.event("번역", v.name, "건너뜀", "원어와 번역 대상 언어가 같아 번역하지 않음")
            if tracker:
                tracker.add(translate_weight)
        return results

    pending = []
    for v in videos:
        cache_file = cache_dir_for(v, config) / f"cues_translated.{target_lang}.json"
        if cache_file.exists():
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            results[v] = {int(k): val for k, val in raw.items()}
            if worklog:
                worklog.event("번역", v.name, "캐시 사용", f"{len(results[v])}줄 (이전 번역 결과 재사용)")
            if tracker:
                tracker.add(translate_weight)
        else:
            pending.append(v)

    if not pending:
        return results

    _check_cancel(cancel_check)
    log_dir = config.resolve_path("paths.cache_dir", "cache")
    log_dir.mkdir(parents=True, exist_ok=True)
    use_gemma = config.get("llm.translate_backend", "translategemma") == "translategemma"
    if use_gemma:
        model_path = config.resolve_path("llm.translate_model")
        port = config.get("llm.translate_port", 8092)  # Qwen 서버(8090)와 포트를 나눠서 서로 안 섞이게 한다
        ctx = config.get("llm.translate_ctx_size", 4096)
        extra_args = config.get("llm.translate_extra_args", ["--no-jinja"])
        translate_fn = translate_cues_gemma
        ngl_key = "llm.translate_n_gpu_layers"
    else:
        model_path = config.resolve_path("llm.model")
        port = config.get("llm.port", 8090)
        ctx = config.get("llm.ctx_size", 8192)
        extra_args = []
        translate_fn = translate_cues
        ngl_key = "llm.n_gpu_layers"
    _report(progress_cb, f"번역 모델 시작 중 ({model_path.name})")
    server = _start_server_with_fallback(config, model_path, port, ctx, extra_args, log_dir / "llama-server.log", ngl_key)
    try:
        batch_size = config.get("llm.batch_sentences", 25)
        for i, v in enumerate(pending, 1):
            _check_cancel(cancel_check)
            _report(progress_cb, f"[번역 {i}/{len(pending)}] {v.name}")

            n_cues = len(cues_map[v])
            n_batches = max(1, math.ceil(n_cues / batch_size)) if n_cues else 1
            per_batch_weight = translate_weight / n_batches
            on_batch_done = (lambda w=per_batch_weight: tracker.add(w)) if tracker else None
            restarts_before = server.restart_count

            ko_map, report = translate_fn(
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
            if worklog:
                detail = f"총 {len(ko_map)}줄"
                if report.transliterated_ids:
                    detail += f", 소리 나는 대로 한글로 옮긴 줄 {len(report.transliterated_ids)}개"
                if report.failed_ids:
                    detail += f", 일본어 원문이 남은 줄 {len(report.failed_ids)}개(줄 번호 {report.failed_ids[:30]})"
                restarts = server.restart_count - restarts_before
                if restarts:
                    detail += f", 번역 서버가 응답하지 않아 {restarts}번 다시 시작함(GPU 메모리 부족 의심)"
                worklog.event("번역", v.name, "완료" if not report.failed_ids else "일부 미번역", detail)
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
    worklog: Optional[WorkLog] = None,
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

    current_video = {"name": ""}  # 로그에 영상 전체 이름을 쓰려고 _write_all이 읽는다

    def _write_all(display, stem: str, lang: str, paths: list[Path]) -> None:
        for fmt in formats:
            writer, encoding = writer_for[fmt]
            p = out_dir / f"{stem}.{lang}.{fmt}"
            existed = p.exists()
            writer(display, p, encoding=encoding)
            paths.append(p)
            if worklog:
                worklog.event("자막 저장", current_video["name"], "덮어씀" if existed else "새로 만듦", f"{p.name} ({len(display)}줄)")

    outputs: dict[Path, list[Path]] = {}
    for i, v in enumerate(videos, 1):
        _report(progress_cb, f"[자막 저장 {i}/{len(videos)}] {v.name}")
        stem = Path(v).stem
        current_video["name"] = Path(v).name
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

    worklog = WorkLog(out_dir) if config.get("subtitle.worklog", True) else None
    if worklog:
        fmts = "/".join(config.get("subtitle.formats", ["srt"]))
        worklog.begin(
            f"자막 생성 시작 - 영상 {len(videos)}개",
            [
                f"언어: {config.get('source_language')} -> {config.get('target_language')}",
                f"음성 인식 정밀도: {config.get('stt.precision')}, 자막 형식: {fmts}",
                "대상 파일:",
                *[f"  {i}. {v}" for i, v in enumerate(videos, 1)],
            ],
        )
    try:
        wav_paths = stage_audio(videos, config, progress_cb, cancel_check, tracker)
        stt_results = stage_stt(videos, wav_paths, config, progress_cb, cancel_check, tracker, worklog)
        cues_map = stage_segment(videos, stt_results, config, progress_cb, cancel_check, tracker, worklog)
        translations = stage_translate(
            videos, cues_map, config, glossary, progress_cb, cancel_check, tracker, worklog
        )
        outputs = stage_write(videos, cues_map, translations, out_dir, config, progress_cb, tracker, worklog)
    except PipelineCancelled:
        if worklog:
            worklog.note("사용자가 취소함. 이미 끝난 단계는 캐시에 남아 있어 이어서 할 수 있음")
            worklog.end("취소로 끝남")
        raise
    except Exception as e:
        if worklog:
            worklog.note(f"오류로 중단됨: {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
            worklog.end("오류로 끝남")
        raise
    if worklog:
        worklog.end(f"자막 생성 끝 - 영상 {len(outputs)}개")
    return outputs
