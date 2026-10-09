"""TranslateGemma(번역 전용 모델) 기반 번역 백엔드.

범용 채팅 모델(Qwen3-8B)로 JSON 배치 번역을 했을 때 뜻이 틀린 줄이 많아서(평가용 40줄 중 약 10줄),
번역 전용으로 학습된 TranslateGemma 12B로 바꿨다. 같은 40줄에서 뜻이 틀린 줄이 1~4줄로 줄었다.

- 모델이 정해진 번역 프롬프트 형식을 쓰므로 JSON 배치 대신 줄 단위로 번역한다. 여러 줄을 한 번에 넣어 봤더니
  출력 줄이 어긋나는 경우가 있었다(줄 단위가 안전).
- 문맥이 없어서 반말 대사도 존댓말로 옮기는 경향이 있어, 일본어 문장 끝으로 반말을 판별해(core.textfix.japanese_register)
  반말일 때만 지시를 덧붙인다. 판별이 애매하거나 존댓말이면 지시를 넣지 않는다.
- 줄마다 독립 요청이라 llama-server 슬롯 수만큼 병렬로 보낸다.
"""
from __future__ import annotations

import logging
import re
import requests
from concurrent.futures import ThreadPoolExecutor

from core.errors import PipelineCancelled
from core.llm import LlamaServer
from core.segmenter import Cue
from core.textfix import collapse_repeats, is_kana_only, japanese_register, kana_to_hangul
from core.translate import TranslateReport
from core.validate import _JA_CHAR_RE

logger = logging.getLogger(__name__)

_LANG_NAMES = {"ja": "Japanese", "ko": "Korean", "en": "English"}
# 반말일 때만 지시를 넣는다. 모델의 기본값이 이미 존댓말이라 존댓말 지시는 득이 없고 오히려 문장이 딱딱해지거나
# 길어졌다. 지시 문구는 길게 풀어 쓸수록 엉뚱한 번역("뭐, 갑자기 왜?")과 없는 말 추가("네,")가 늘어서 짧게 썼다.
_REGISTER_HINT = {"casual": "Translate into informal Korean (반말)."}
_LATIN_RE = re.compile(r"[A-Za-z]{2,}")
_HAS_LATIN_RE = re.compile(r"[A-Za-z]")


def build_prompt(text: str, source_lang: str, target_lang: str, hint: str = "") -> str:
    src = _LANG_NAMES.get(source_lang, source_lang)
    tgt = _LANG_NAMES.get(target_lang, target_lang)
    body = (
        f"You are a professional {src} ({source_lang}) to {tgt} ({target_lang}) translator. Your goal is to accurately "
        f"convey the meaning and nuances of the original {src} text while adhering to {tgt} grammar, vocabulary, "
        f"and cultural sensitivities.\n"
        f"Produce only the {tgt} translation, without any additional explanations or commentary. "
        + (hint + " " if hint else "")
        + f"Please translate the following {src} text into {tgt}:\n\n\n{text}"
    )
    return f"<start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n"


def _clean_output(out: str, source: str) -> str:
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    text = " ".join(lines)
    # 원문에 따옴표가 없는데 모델이 통째로 따옴표로 감싼 경우만 벗긴다
    if len(text) >= 2 and text[0] in "\"“" and text[-1] in "\"”" and not any(q in source for q in "\"“「"):
        text = text[1:-1].strip()
    return text


def translate_cues_gemma(
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
    max_retries = config.get("llm.max_retries", 3)
    temperature = config.get("llm.translate_temperature", 0.1)
    workers = max(1, int(config.get("llm.translate_parallel", 4)))
    request_timeout = float(config.get("llm.translate_timeout", 120))

    glossary_hint = ""
    if glossary:
        pairs = "; ".join(f"{k} -> {v}" for k, v in glossary.items())
        glossary_hint = f"Always translate these terms/names as follows: {pairs}."

    work = [Cue(id=c.id, start=c.start, end=c.end, text=collapse_repeats(c.text)) for c in cues]
    report = TranslateReport(total=len(cues))

    def translate_one(c: Cue) -> tuple[int, str | None]:
        if cancel_check is not None and cancel_check():
            raise PipelineCancelled()
        base_hint = ""
        if target_lang == "ko" and source_lang == "ja":
            reg = japanese_register(c.text)
            base_hint = _REGISTER_HINT.get(reg, "") if reg else ""
        base_hint = (base_hint + " " + glossary_hint).strip()
        n_predict = min(400, 60 + 4 * len(c.text))
        source_has_latin = bool(_HAS_LATIN_RE.search(c.text))
        bad = ""
        latin_fallback: str | None = None
        for attempt in range(max_retries):
            if cancel_check is not None and cancel_check():
                raise PipelineCancelled()
            hint = base_hint
            if bad:
                hint = (hint + f" Do not use these characters: {bad}. Write everything in Hangul.").strip()
            generation = llm.generation
            try:
                raw = llm.complete(
                    build_prompt(c.text, source_lang, target_lang, hint),
                    temperature=temperature + 0.3 * attempt,
                    n_predict=n_predict,
                    timeout=request_timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                # 평소 한 줄은 몇 초면 끝난다. 응답이 없으면 서버가 GPU 메모리 부족 등으로 극단적으로 느려진
                # 것이므로 같은 서버에 계속 기다리지 않고 다시 띄운 뒤 재시도한다.
                logger.warning("번역 서버 응답 없음 (cue %s, %s초 초과): %s", c.id, request_timeout, type(e).__name__)
                if cancel_check is not None and cancel_check():
                    raise PipelineCancelled() from e
                try:
                    if llm.restart_if_generation(generation):
                        logger.warning("번역 서버를 다시 시작함 (%s번째)", llm.restart_count)
                except Exception:  # noqa: BLE001
                    logger.exception("번역 서버 재시작 실패")
                continue
            except Exception as e:  # noqa: BLE001
                logger.warning("TranslateGemma 요청 실패 (cue %s, attempt %s): %s", c.id, attempt, e)
                continue
            out = _clean_output(raw, c.text)
            if not out:
                continue
            leaked = "".join(dict.fromkeys(_JA_CHAR_RE.findall(out)))
            if leaked:
                bad = leaked
                continue
            if not source_has_latin and _LATIN_RE.search(out):
                # 원문에 없는 영어 단어가 섞임. 한 번 더 시켜 보되, 끝내 안 되면 마지막 결과를 쓴다
                latin_fallback = out
                bad = " ".join(_LATIN_RE.findall(out))
                continue
            return c.id, out
        return c.id, latin_fallback

    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(work), batch_size):
            if cancel_check is not None and cancel_check():
                raise PipelineCancelled()
            batch = work[start : start + batch_size]
            for cid, out in pool.map(translate_one, batch):
                if out is not None:
                    results[cid] = out
            # 끝내 못 옮긴 줄: 가나뿐이면 소리 나는 대로 한글로, 아니면 일본어 원문을 남기고 보고서에 적는다
            for c in batch:
                if c.id not in results:
                    if is_kana_only(c.text):
                        results[c.id] = kana_to_hangul(c.text)
                        report.transliterated_ids.append(c.id)
                    else:
                        results[c.id] = c.text
                        report.failed_ids.append(c.id)
            if on_batch_done is not None:
                on_batch_done()
    return results, report
