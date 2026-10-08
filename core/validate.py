"""번역 결과 검증. Phase 0 벤치마크에서 확인된 문제(한자/가나 잔존, 개수/순서 불일치)를 걸러낸다."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

# 히라가나/가타카나 + 한자(간체/번체 포함) 범위. 한글(가-힣, U+AC00-D7A3)은 겹치지 않는다.
_JA_CHAR_RE = re.compile(r"[぀-ヿ一-鿿]")


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


def validate_batch(expected_ids: list[int], raw_response: str) -> BatchValidationResult:
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
        by_id[iid] = str(item["ko"])

    missing = [i for i in expected_ids if i not in by_id]
    leaked = [i for i in expected_ids if i in by_id and has_leftover_source_script(by_id[i])]
    bad_ids = sorted(set(missing) | set(leaked))

    leaked_chars = {i: "".join(dict.fromkeys(_JA_CHAR_RE.findall(by_id[i]))) for i in leaked}
    for i in leaked:
        del by_id[i]

    ok = len(bad_ids) == 0
    error = None
    if missing:
        error = f"누락된 id: {missing}"
    if leaked:
        error = (error + "; " if error else "") + f"원어 문자 잔존 id: {leaked}"

    return BatchValidationResult(ok=ok, by_id=by_id, bad_ids=bad_ids, error=error, leaked_chars=leaked_chars)
