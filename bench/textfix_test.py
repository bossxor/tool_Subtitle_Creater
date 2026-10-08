"""core.textfix / core.validate 의 핵심 동작을 확인하는 가벼운 테스트 (pytest 없이 실행).

    .venv\Scripts\python.exe bench/textfix_test.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.textfix import collapse_repeats, is_kana_only, kana_to_hangul  # noqa: E402
from core.validate import parse_translation_json, validate_batch  # noqa: E402

JA = re.compile(r"[぀-ヿ一-鿿]")


def check(name, got, want):
    assert got == want, f"{name}: got {got!r}, want {want!r}"


# 반복 압축: 긴 반복은 3번으로, 정상 문장/3번 이하는 그대로
check("collapse loop", collapse_repeats("あっあっあっあっあっあっ"), "あっあっあっ")
check("collapse long vowel run", collapse_repeats("ああああああああ"), "あああ")
check("keep normal", collapse_repeats("はい、わかりました"), "はい、わかりました")
check("keep 3x", collapse_repeats("そうそうそう"), "そうそうそう")

# 가나 -> 한글
check("small tsu", kana_to_hangul("あっ"), "앗")
check("small vowel", kana_to_hangul("あぁ"), "아아")
check("n alone", kana_to_hangul("ん"), "응")
check("yoon + long vowel", kana_to_hangul("きゃー"), "캬~")
check("katakana", kana_to_hangul("ハァ"), "하아")
every_kana = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽきゃしゅちょっー"
assert not JA.findall(kana_to_hangul(every_kana)), "변환 결과에 일본 문자가 남음 (검증기에서 또 걸림)"

check("kana only", is_kana_only("あっ…"), True)
check("kanji is not kana only", is_kana_only("帰り"), False)
check("punct is not kana only", is_kana_only("..."), False)

# 잘린 JSON: 끝까지 나온 항목은 살리고 잘린 줄만 재시도 대상이 된다
trunc = '[{"id": 150, "ko": "안녕"}, {"id": 151, "ko": "따옴표 \\"테스트\\" 입니다"}, {"id": 152, "ko": "잘린 줄'
check("recover", [i["id"] for i in parse_translation_json(trunc)], [150, 151])
r = validate_batch([150, 151, 152, 153], trunc)
check("retry ids", r.bad_ids, [152, 153])
check("kept text with quote", r.by_id[151], '따옴표 "테스트" 입니다')

# 남은 일본 문자는 재시도 힌트로 돌려준다
r = validate_batch([1], '[{"id": 1, "ko": "그帰り 갔다"}]')
check("leaked chars", r.leaked_chars, {1: "帰り"})  # 한자와 가나 둘 다 남은 글자로 센다
check("leaked is retried", r.bad_ids, [1])

print("모든 테스트 통과")
