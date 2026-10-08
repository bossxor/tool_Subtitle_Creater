"""번역 전후에 쓰는 작은 텍스트 보정 함수들.

- collapse_repeats: 같은 소리가 길게 반복되는 줄(예: 음성 인식 반복 루프, 신음 같은 소리)을 줄여서
  LLM이 반복 출력으로 토큰을 다 써버려 JSON이 잘리는 문제를 막는다.
- kana_to_hangul: 가나만으로 된 줄(감탄사 등)을 LLM이 끝내 한글로 못 바꿨을 때, 일본어 그대로 남기지
  않고 소리 나는 대로 한글로 옮기는 마지막 수단.
"""
from __future__ import annotations

import re

# 같은 1~4글자 덩어리가 4번 이상 연속되면 3번으로 줄인다 ("あっあっあっあっあっ" -> "あっあっあっ")
_REPEAT_RE = re.compile(r"(.{1,4}?)\1{3,}", re.DOTALL)
_KANA_ONLY_RE = re.compile(r"^[぀-ヿー、。！？!?…・〜~\s]+$")
_HAS_KANA_RE = re.compile(r"[぀-ヿー]")


def collapse_repeats(text: str, keep: int = 3) -> str:
    return _REPEAT_RE.sub(lambda m: m.group(1) * keep, text)


def is_kana_only(text: str) -> bool:
    """한자 없이 가나(와 기호)로만 이뤄졌고 가나가 하나 이상 있으면 True."""
    return bool(_KANA_ONLY_RE.match(text)) and bool(_HAS_KANA_RE.search(text))


# ---- 가나 -> 한글 (소리 나는 대로, 단순 규칙) ----
_L = {"ㄱ": 0, "ㄴ": 2, "ㄷ": 3, "ㄹ": 5, "ㅁ": 6, "ㅂ": 7, "ㅅ": 9, "ㅇ": 11, "ㅈ": 12, "ㅊ": 14,
      "ㅋ": 15, "ㅌ": 16, "ㅍ": 17, "ㅎ": 18}
_V = {"ㅏ": 0, "ㅑ": 2, "ㅓ": 4, "ㅔ": 5, "ㅗ": 8, "ㅘ": 9, "ㅛ": 12, "ㅜ": 13, "ㅠ": 17, "ㅡ": 18, "ㅣ": 20}


def _syl(cons: str, vowel: str, final: int = 0) -> str:
    return chr(0xAC00 + (_L[cons] * 21 + _V[vowel]) * 28 + final)


# (자음 목록, 모음 목록) 행 단위 정의. 모음 순서는 a i u e o.
_ROWS = [
    ("あいうえお", ["ㅇ"] * 5, "ㅏㅣㅜㅔㅗ"),
    ("かきくけこ", ["ㅋ"] * 5, "ㅏㅣㅜㅔㅗ"),
    ("さしすせそ", ["ㅅ"] * 5, "ㅏㅣㅡㅔㅗ"),
    ("たちつてと", ["ㅌ", "ㅊ", "ㅊ", "ㅌ", "ㅌ"], "ㅏㅣㅡㅔㅗ"),
    ("なにぬねの", ["ㄴ"] * 5, "ㅏㅣㅜㅔㅗ"),
    ("はひふへほ", ["ㅎ"] * 5, "ㅏㅣㅜㅔㅗ"),
    ("まみむめも", ["ㅁ"] * 5, "ㅏㅣㅜㅔㅗ"),
    ("やゆよ", ["ㅇ"] * 3, "ㅑㅠㅛ"),
    ("らりるれろ", ["ㄹ"] * 5, "ㅏㅣㅜㅔㅗ"),
    ("わを", ["ㅇ"] * 2, "ㅘㅗ"),
    ("がぎぐげご", ["ㄱ"] * 5, "ㅏㅣㅜㅔㅗ"),
    ("ざじずぜぞ", ["ㅈ"] * 5, "ㅏㅣㅡㅔㅗ"),
    ("だぢづでど", ["ㄷ", "ㅈ", "ㅈ", "ㄷ", "ㄷ"], "ㅏㅣㅡㅔㅗ"),
    ("ばびぶべぼ", ["ㅂ"] * 5, "ㅏㅣㅜㅔㅗ"),
    ("ぱぴぷぺぽ", ["ㅍ"] * 5, "ㅏㅣㅜㅔㅗ"),
]
_KANA: dict[str, str] = {}
for _chars, _cons, _vows in _ROWS:
    for _c, _k, _v in zip(_chars, _cons, _vows):
        _KANA[_c] = _syl(_k, _v)
_SMALL_VOWEL = {"ぁ": "아", "ぃ": "이", "ぅ": "우", "ぇ": "에", "ぉ": "오"}
# 요음: き+ゃ -> 캬 등
_YOON_BASE = {"き": "ㅋ", "ぎ": "ㄱ", "し": "ㅅ", "じ": "ㅈ", "ち": "ㅊ", "に": "ㄴ", "ひ": "ㅎ",
              "び": "ㅂ", "ぴ": "ㅍ", "み": "ㅁ", "り": "ㄹ"}
_YOON_VOWEL = {"ゃ": "ㅑ", "ゅ": "ㅠ", "ょ": "ㅛ"}


def _to_hiragana(ch: str) -> str:
    o = ord(ch)
    return chr(o - 0x60) if 0x30A1 <= o <= 0x30F6 else ch


def _add_final(prev: str, final: int) -> str | None:
    """직전 글자가 받침 없는 한글 음절이면 받침을 붙여서 돌려준다."""
    o = ord(prev) - 0xAC00
    if 0 <= o < 11172 and o % 28 == 0:
        return chr(0xAC00 + o + final)
    return None


def kana_to_hangul(text: str) -> str:
    out: list[str] = []
    chars = [_to_hiragana(c) for c in text]
    i = 0
    while i < len(chars):
        c = chars[i]
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if c in _YOON_BASE and nxt in _YOON_VOWEL:
            out.append(_syl(_YOON_BASE[c], _YOON_VOWEL[nxt]))
            i += 2
            continue
        if c in _KANA:
            out.append(_KANA[c])
        elif c in _SMALL_VOWEL:
            out.append(_SMALL_VOWEL[c])
        elif c == "ん":
            merged = _add_final(out[-1], 21) if out else None  # ㅇ 받침
            if merged:
                out[-1] = merged
            else:
                out.append("응")
        elif c == "っ":
            merged = _add_final(out[-1], 19) if out else None  # ㅅ 받침
            if merged:
                out[-1] = merged
        elif c == "ー":
            out.append("~")
        elif c in "、。":
            out.append("," if c == "、" else ".")
        elif c == "！":
            out.append("!")
        elif c == "？":
            out.append("?")
        elif c == "・":
            out.append(" ")
        else:
            out.append(c)
        i += 1
    return "".join(out)
