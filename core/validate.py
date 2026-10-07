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
    if start == -1 or end == -1 or end < start:
        raise TranslationParseError(f"JSON 배열을 찾지 못함: {raw[:200]!r}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise TranslationParseError(f"JSON 파싱 실패: {e}") from e
    if not isinstance(data, list):
        raise TranslationParseError("최상위가 배열이 아님")
    return data


def has_leftover_source_script(text: str) -> bool:
    """한국어 번역문에 히라가나/가타카나/한자가 남아있는지 확인한다."""
    return bool(_JA_CHAR_RE.search(text))


@dataclass
class BatchValidationResult:
    ok: bool
    by_id: dict[int, str]  # id -> ko text (성공한 것만)
    bad_ids: list[int]  # 잔존 문자 등으로 재시도 필요한 id
    error: str | None = None


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

    for i in leaked:
        del by_id[i]

    ok = len(bad_ids) == 0
    error = None
    if missing:
        error = f"누락된 id: {missing}"
    if leaked:
        error = (error + "; " if error else "") + f"원어 문자 잔존 id: {leaked}"

    return BatchValidationResult(ok=ok, by_id=by_id, bad_ids=bad_ids, error=error)
