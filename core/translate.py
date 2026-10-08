"""cue 목록을 배치로 묶어 LLM에 번역+다듬기를 요청한다.

- 타임스탬프는 절대 건드리지 않는다 (LLM에는 텍스트만 전달).
- 문맥 유지를 위해 이전 배치의 마지막 몇 문장(원문+번역)을 함께 전달한다.
- 검증(validate.py)에서 실패한 id만 골라 재시도한다. 끝까지 실패하면 원문을 그대로 둔다.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from core.errors import PipelineCancelled
from core.llm import LlamaServer
from core.segmenter import Cue
from core.textfix import collapse_repeats, is_kana_only, kana_to_hangul
from core.validate import validate_batch

logger = logging.getLogger(__name__)

_LANG_NAMES = {"ja": "일본어", "ko": "한국어", "en": "영어"}


@dataclass
class TranslateReport:
    total: int = 0
    failed_ids: list[int] | None = None  # 끝내 번역 못 해서 일본어 원문이 남은 id
    transliterated_ids: list[int] | None = None  # 가나뿐이라 소리 나는 대로 한글로 옮긴 id

    def __post_init__(self):
        if self.failed_ids is None:
            self.failed_ids = []
        if self.transliterated_ids is None:
            self.transliterated_ids = []


def _lang_name(code: str) -> str:
    return _LANG_NAMES.get(code, code)


def _build_system_prompt(source_lang: str, target_lang: str, glossary: dict[str, str] | None) -> str:
    src = _lang_name(source_lang)
    tgt = _lang_name(target_lang)
    parts = [
        f"당신은 {src} 영상 자막을 {tgt}로 옮기는 전문 번역가입니다.",
        "입력은 JSON 객체이며, 선택적으로 'context'(이미 번역된 이전 문장들, 참고용, 번역하지 말 것)와 "
        "'items'(번역할 문장 목록, 각 항목은 {id, ja}) 필드를 가집니다.",
        f"'items'의 각 문장을 자연스러운 {tgt} 구어체 대사로 번역하세요.",
        "음성인식 과정에서 생긴 것으로 보이는 어색한 오탈자(동음이의어 오인식 등)는 문맥에 맞게 자연스럽게 교정하세요.",
        "존댓말/반말 등 어조는 문맥과 인물 관계를 보고 판단하세요.",
        f"중요: 출력 'ko' 필드에는 히라가나, 가타카나, 한자를 단 한 글자도 남기지 말고 전부 {tgt}(한글)로만 표기하세요.",
        "고유명사(인명, 지명, 작품명)도 한글 발음으로 표기하세요.",
    ]
    if glossary:
        pairs = "; ".join(f"{k} -> {v}" for k, v in glossary.items())
        parts.append(f"다음 용어/인명은 반드시 이렇게 번역하세요: {pairs}")
    parts.append(
        "항목에 'avoid' 필드가 있으면, 이전 번역에 그 글자가 그대로 남아서 거절됐다는 뜻입니다. "
        "그 글자를 쓰지 말고 뜻을 한글로 풀어서 쓰세요."
    )
    parts.append(
        "한숨, 신음, 감탄사처럼 뜻이 없는 소리는 소리 나는 대로 한글로 적으세요 (예: あっ -> 앗, はぁ -> 하아). "
        "같은 소리가 길게 반복되면 3번까지만 적으세요."
    )
    parts.append(
        "출력은 반드시 JSON 배열만 반환하세요. 각 항목은 {id, ko} 형식이어야 하고, "
        "'items'와 같은 개수, 같은 id를 모두 포함해야 합니다. 설명이나 코드블록 없이 순수 JSON 배열만 출력하세요."
    )
    return "\n".join(parts) + "\n/no_think"


def _build_user_payload(
    cues: list[Cue], context_pairs: list[tuple[str, str]], avoid: dict[int, str] | None = None
) -> str:
    items = []
    for c in cues:
        item = {"id": c.id, "ja": c.text}
        if avoid and avoid.get(c.id):
            item["avoid"] = avoid[c.id]
        items.append(item)
    payload: dict = {"items": items}
    if context_pairs:
        payload["context"] = [{"ja": ja, "ko": ko} for ja, ko in context_pairs]
    return json.dumps(payload, ensure_ascii=False)


def _build_verify_system_prompt(target_lang: str) -> str:
    tgt = _lang_name(target_lang)
    parts = [
        f"당신은 {tgt} 자막 품질을 검수하는 전문가입니다.",
        "입력은 JSON 객체이며 'items'(검수할 문장 목록, 각 항목은 {id, text}) 필드를 가집니다.",
        "각 text는 기존 자막 파일에서 그대로 가져온 것으로, 인코딩 오류 때문에 깨진 글자나 이상한 기호가 "
        "섞여 있을 수 있습니다.",
        "깨진 글자/기호가 있으면 문맥에 맞게 자연스러운 한글로 복원하거나 제거하세요. "
        "내용, 어조, 의미는 절대 바꾸지 마세요.",
        "이미 정상인 문장은 토씨 하나도 바꾸지 말고 그대로 두세요.",
        "출력은 반드시 JSON 배열만 반환하세요. 각 항목은 {id, ko} 형식이어야 하고, "
        "'items'와 같은 개수, 같은 id를 모두 포함해야 합니다. 설명이나 코드블록 없이 순수 JSON 배열만 출력하세요.",
    ]
    return "\n".join(parts) + "\n/no_think"


def _build_verify_payload(cues: list[Cue]) -> str:
    return json.dumps({"items": [{"id": c.id, "text": c.text} for c in cues]}, ensure_ascii=False)


def verify_cues(
    cues: list[Cue],
    llm: LlamaServer,
    config,
    target_lang: str | None = None,
    cancel_check=None,
    on_batch_done=None,
) -> tuple[dict[int, str], TranslateReport]:
    """번역이 아니라 검수용. 기존 자막에서 가져온 텍스트를 그대로 두되, 인코딩 오류로 깨진
    글자만 AI로 확인해서 복원한다 (smi -> srt/ass 변환 시 사용, DESIGN.md 참고)."""
    if not cues:
        return {}, TranslateReport(total=0)

    target_lang = target_lang or config.get("target_language", "ko")
    batch_size = config.get("llm.batch_sentences", 25)
    max_retries = config.get("llm.max_retries", 3)
    max_tokens = config.get("llm.max_tokens", 2000)

    system = _build_verify_system_prompt(target_lang)
    results: dict[int, str] = {}
    report = TranslateReport(total=len(cues))

    for start in range(0, len(cues), batch_size):
        if cancel_check is not None and cancel_check():
            raise PipelineCancelled()
        batch = cues[start : start + batch_size]
        by_id: dict[int, str] = {}
        pending = list(batch)

        for attempt in range(max_retries):
            expected_ids = [c.id for c in pending]
            user = _build_verify_payload(pending)
            try:
                # 검수는 번역보다 보수적으로(temperature 낮게) — 불필요한 변형을 줄인다
                raw = llm.chat(system, user, temperature=0.1, max_tokens=max_tokens)
            except Exception:
                logger.exception("LLM 검수 호출 실패 (batch start=%s, attempt=%s)", start, attempt)
                if attempt == max_retries - 1:
                    break
                continue

            vr = validate_batch(expected_ids, raw)
            by_id.update(vr.by_id)
            if vr.error:
                logger.warning("검수 배치 검증 문제(attempt %s): %s", attempt, vr.error)
            if vr.ok:
                break
            pending = [c for c in pending if c.id in vr.bad_ids]
            if not pending:
                break

        for c in batch:
            if c.id not in by_id:
                by_id[c.id] = c.text  # 검수 실패하면 원문 그대로 둔다(안전)
                report.failed_ids.append(c.id)

        results.update(by_id)
        if on_batch_done is not None:
            on_batch_done()

    return results, report


def translate_cues(
    cues: list[Cue],
    llm: LlamaServer,
    config,
    glossary: dict[str, str] | None = None,
    source_lang: str | None = None,
    target_lang: str | None = None,
    cancel_check=None,
    on_batch_done=None,
) -> tuple[dict[int, str], TranslateReport]:
    if not cues:
        return {}, TranslateReport(total=0)

    source_lang = source_lang or config.get("source_language", "ja")
    target_lang = target_lang or config.get("target_language", "ko")
    batch_size = config.get("llm.batch_sentences", 25)
    context_n = config.get("llm.context_sentences", 2)
    max_retries = config.get("llm.max_retries", 3)
    temperature = config.get("llm.temperature", 0.3)
    max_tokens = config.get("llm.max_tokens", 2000)

    system = _build_system_prompt(source_lang, target_lang, glossary)
    results: dict[int, str] = {}
    report = TranslateReport(total=len(cues))
    prev_context: list[tuple[str, str]] = []

    # 같은 소리가 수십~수백 글자 반복되는 줄(음성 인식 반복 루프, 신음 등)은 LLM이 반복 출력으로
    # 토큰을 다 써서 JSON이 잘리는 원인이었다. LLM에 보내기 전에 3번 반복으로 줄인다.
    work_cues = [Cue(id=c.id, start=c.start, end=c.end, text=collapse_repeats(c.text)) for c in cues]

    for start in range(0, len(work_cues), batch_size):
        if cancel_check is not None and cancel_check():
            raise PipelineCancelled()
        batch = work_cues[start : start + batch_size]
        by_id: dict[int, str] = {}
        pending = list(batch)
        ctx = prev_context
        avoid: dict[int, str] = {}  # 이전 시도에서 번역문에 남은 일본 문자 (재시도 힌트)
        # 출력은 입력 글자 수에 비례해 길어지므로, 긴 줄이 있는 배치는 한도를 넉넉히 준다
        batch_max_tokens = max(max_tokens, min(4000, 400 + 3 * sum(len(c.text) for c in batch)))

        for attempt in range(max_retries):
            expected_ids = [c.id for c in pending]
            user = _build_user_payload(pending, ctx, avoid)
            try:
                # 같은 입력에 같은 답이 반복되는 걸 피하려고 재시도마다 온도를 조금씩 올린다
                raw = llm.chat(
                    system, user, temperature=temperature + 0.25 * attempt, max_tokens=batch_max_tokens
                )
            except Exception:
                logger.exception("LLM 호출 실패 (batch start=%s, attempt=%s)", start, attempt)
                if attempt == max_retries - 1:
                    break
                continue

            vr = validate_batch(expected_ids, raw)
            by_id.update(vr.by_id)
            if vr.error:
                logger.warning("번역 배치 검증 문제(attempt %s): %s", attempt, vr.error)
            if vr.ok:
                break
            avoid.update(vr.leaked_chars or {})
            pending = [c for c in pending if c.id in vr.bad_ids]
            ctx = []  # 재시도에서는 문맥 없이 문제 항목만 다시 시도해 프롬프트를 단순화한다
            if not pending:
                break

        # 끝까지 실패한 항목: 가나뿐이면 소리 나는 대로 한글로 옮기고, 한자가 섞여 있으면 일본어
        # 원문을 그대로 둔다 (파이프라인이 멈추지 않게 하고, 어느 줄인지 보고서에 남긴다)
        for c in batch:
            if c.id not in by_id:
                if is_kana_only(c.text):
                    by_id[c.id] = kana_to_hangul(c.text)
                    report.transliterated_ids.append(c.id)
                else:
                    by_id[c.id] = c.text
                    report.failed_ids.append(c.id)

        results.update(by_id)
        if on_batch_done is not None:
            on_batch_done()
        tail = batch[-context_n:] if context_n > 0 else []
        prev_context = [(c.text, by_id[c.id]) for c in tail]

    return results, report
