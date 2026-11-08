"""번역 결과 검증. Phase 0 벤치마크에서 확인된 문제(한자/가나 잔존, 개수/순서 불일치)를 걸러낸다."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

# 히라가나/가타카나 + 한자(간체/번체 포함) 범위. 한글(가-힣, U+AC00-D7A3)은 겹치지 않는다.
_JA_CHAR_RE = re.compile(r"[぀-ヿ一-鿿]")
_LATIN_RE = re.compile(r"[A-Za-z]+[0-9]*")
_META_RE = re.compile(
    r"(?:no (?:Japanese )?text (?:provided|to translate)|please provide (?:the |some )?(?:Japanese )?text|"
    r"no translation needed|번역할 (?:내용|텍스트|문장|원문)[이가은는 ]*(?:없|제공되지)|"
    r"(?:이 |해당 |제공된 )?(?:텍스트|문구|문장|내용)[은는을를 ]*번역할 수 없|"
    r"(?:죄송|미안|유감)[^\n]{0,80}번역할 수 없|"
    r"(?:일본어|원문|텍스트)를? (?:입력|제공)해|^\s*[\(（]?(?:빈칸|번역 불가|번역할 내용 없음)[\)）]?\s*$)",
    re.IGNORECASE,
)


def translation_issue(text, source: str = "", target_lang: str = "ko", *,
                      glossary: dict[str, str] | None = None) -> str | None:
    """Check output language and known non-translation responses, not semantic accuracy."""
    if not isinstance(text, str) or not text.strip():
        return "빈 번역 또는 잘못된 자료형"
    if _META_RE.search(text) and not _META_RE.search(source):
        # Dialogue can itself discuss translation; a keyword alone is not an exemption.
        refusal = (re.search(r"번역할 수 없", text) and
                   re.search(r"(?:翻訳|訳)(?:せ(?:ない|ません)|でき(?:ない|ません)|すこと[がは]でき(?:ない|ません))", source))
        missing = (re.search(r"(?:번역할|제공되지|빈칸)", text) and
                   re.search(r"(?:翻訳する(?:文章|内容|テキスト)|原文|テキスト)[^。！？\n]{0,8}(?:ない|ありません)", source))
        request = (re.search(r"(?:입력|제공)해", text) and
                   re.search(r"(?:原文|テキスト|日本語)[^。！？\n]{0,8}(?:入力|提供)", source))
        if not (refusal or missing or request):
            return "번역 대신 안내문 출력"
    if target_lang == "en" and re.search(r"[぀-ヿ一-鿿가-힣]", text):
        return "영어 이외 문자 잔존"
    if target_lang == "ko":
        if _JA_CHAR_RE.search(text):
            return "일본 문자 잔존"
        terms = {key: value for key, value in (glossary or {}).items() if key and key in source}
        allowed = {t.casefold() for t in _LATIN_RE.findall(source + " " + " ".join(terms.values()))}
        if any(t.casefold() not in allowed for t in _LATIN_RE.findall(text)):
            return "원문에 없는 영문 출력"
        if re.search(r"[^\W\d_]", text, re.UNICODE) and not re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", text):
            if not _LATIN_RE.search(text) or not allowed:
                return "한국어 번역 없음"
            remaining = source
            for term in sorted(terms, key=len, reverse=True):
                remaining = remaining.replace(term, "")
            if _JA_CHAR_RE.search(remaining):
                return "약어만 남고 한국어 대사 누락"
    return None


class TranslationParseError(ValueError):
    pass


def parse_translation_json(raw: str) -> list[dict]:
    """LLM 출력에서 JSON 배열을 뽑는다. 코드블록이나 잡설이 섞여 나오는 경우를 방어한다."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass  # 아래 복구 경로로 넘어간다
    # 출력이 max_tokens에서 잘려 닫는 ']'가 없거나 JSON이 깨진 경우: 이미 끝까지 나온 항목만이라도
    # 건진다. 전부 버리고 25줄을 통째로 다시 시키면 같은 곳에서 또 잘리기 때문이다.
    recovered = _recover_items(text)
    if recovered:
        return recovered
    raise TranslationParseError(f"JSON 배열을 찾지 못함: {raw[:200]!r}")


_ITEM_RE = re.compile(r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"ko"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}')


def _recover_items(text: str) -> list[dict]:
    items = []
    for m in _ITEM_RE.finditer(text):
        try:
            ko = json.loads('"' + m.group(2) + '"')
        except json.JSONDecodeError:
            continue
        items.append({"id": int(m.group(1)), "ko": ko})
    return items


def has_leftover_source_script(text: str) -> bool:
    """한국어 번역문에 히라가나/가타카나/한자가 남아있는지 확인한다."""
    return bool(_JA_CHAR_RE.search(text))


@dataclass
class BatchValidationResult:
    ok: bool
    by_id: dict[int, str]  # id -> ko text (성공한 것만)
    bad_ids: list[int]  # 잔존 문자 등으로 재시도 필요한 id
    error: str | None = None
    leaked_chars: dict[int, str] | None = None  # id -> 번역문에 남은 일본 문자들 (재시도 힌트용)


def validate_batch(expected_ids: list[int], raw_response: str, *,
                   source_by_id: dict[int, str] | None = None,
                   target_lang: str | None = "ko",
                   glossary: dict[str, str] | None = None) -> BatchValidationResult:
    try:
        parsed = parse_translation_json(raw_response)
    except TranslationParseError as e:
        return BatchValidationResult(ok=False, by_id={}, bad_ids=list(expected_ids), error=str(e))

    by_id: dict[int, str] = {}
    for item in parsed:
        if not isinstance(item, dict) or "id" not in item or "ko" not in item:
            continue
        try:
            iid = int(item["id"])
        except (TypeError, ValueError):
            continue
        if iid in expected_ids and isinstance(item["ko"], str) and item["ko"].strip():
            by_id[iid] = item["ko"].strip()

    missing = [i for i in expected_ids if i not in by_id]
    leaked = [i for i in expected_ids if i in by_id and target_lang == "ko" and has_leftover_source_script(by_id[i])]
    invalid = [i for i in expected_ids if i in by_id and source_by_id is not None
               and translation_issue(by_id[i], source_by_id.get(i, ""), target_lang or "ko", glossary=glossary)]
    bad_ids = sorted(set(missing) | set(leaked) | set(invalid))

    leaked_chars = {i: "".join(dict.fromkeys(_JA_CHAR_RE.findall(by_id[i]))) for i in leaked}
    for i in set(leaked) | set(invalid):
        by_id.pop(i, None)

    ok = len(bad_ids) == 0
    error = None
    if missing:
        error = f"누락된 id: {missing}"
    if leaked:
        error = (error + "; " if error else "") + f"원어 문자 잔존 id: {leaked}"
    if invalid:
        error = (error + "; " if error else "") + f"번역 품질 검증 실패 id: {invalid}"

    return BatchValidationResult(ok=ok, by_id=by_id, bad_ids=bad_ids, error=error, leaked_chars=leaked_chars)
