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

# ---- 작업 로그 / 폴더 정리 안전성 ----
import tempfile  # noqa: E402

from core.worklog import LOG_FILENAME, WorkLog  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    log = WorkLog(tmp)
    log.begin("테스트 시작", ["상세 한 줄"])
    log.event("번역", "a.mp4", "완료", "총 3줄")
    log.end("테스트 끝")
    log2 = WorkLog(tmp)  # 같은 파일에 이어 쓰기
    log2.event("자막 저장", "b.mp4", "새로 만듦", "b.ko.srt (2줄)")
    raw = (Path(tmp) / LOG_FILENAME).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "로그 맨 앞에 BOM이 없음"
    assert raw.count(b"\xef\xbb\xbf") == 1, "이어 쓸 때 BOM이 중간에 또 들어감"
    text = raw.decode("utf-8-sig")
    assert "[번역] a.mp4 : 완료 - 총 3줄" in text and "[자막 저장] b.mp4 : 새로 만듦" in text

    # 자막이 0줄인 smi는 실패로 처리하고 원본을 지우지 않는다
    from core.config import Config, DEFAULT_CONFIG_PATH  # noqa: E402
    from core.folder_scan import ConvertJob, FolderScanPlan, run_conversions  # noqa: E402

    smi = Path(tmp) / "c.ko.smi"
    smi.write_text("<SAMI><BODY></BODY></SAMI>", encoding="utf-8")
    plan = FolderScanPlan(generate=[], convert_jobs=[ConvertJob(video=Path(tmp) / "c.mp4", smi_path=smi)], encoding_fix_jobs=[])
    cfg = Config.load(DEFAULT_CONFIG_PATH)
    cfg.set("paths.cache_dir", str(Path(tmp) / "cache"))
    conv, _fixed, deleted = run_conversions(plan, ["srt"], cfg, delete_old_smi=True, worklog=WorkLog(tmp))
    assert conv == 0 and deleted == 0, (conv, deleted)
    assert smi.exists(), "빈 smi인데 원본이 지워짐"
    assert plan.convert_jobs[0].error, "실패로 기록되지 않음"

print("작업 로그/안전성 테스트 통과")

# ---- 말투 판별 / TranslateGemma 백엔드 (가짜 LLM, GPU 불필요) ----
from core.segmenter import Cue  # noqa: E402
from core.textfix import japanese_register  # noqa: E402
from core.translate_gemma import build_prompt, translate_cues_gemma  # noqa: E402

check("register polite", japanese_register("課長はもう来てますか？"), "polite")
check("register casual", japanese_register("悪い悪い。電車が止まっちゃってさ。"), "casual")
check("register multi-sentence", japanese_register("ご飯できてるよ。先に食べる？それともお風呂？"), "casual")
check("register greeting is undecided", japanese_register("ただいま。"), None)
check("register ええ is polite", japanese_register("ええ、朝から会議があるので。"), "polite")

pr = build_prompt("やあ", "ja", "ko", "Translate into informal Korean (반말).")
assert pr.startswith("<start_of_turn>user\n") and pr.endswith("<start_of_turn>model\n"), "Gemma 턴 형식이 아님"
assert "Japanese (ja) to Korean (ko) translator" in pr and pr.rstrip().endswith("やあ<end_of_turn>\n<start_of_turn>model".rstrip())


class FakeLLM:
    """complete()가 프롬프트에 따라 정해진 답을 돌려주는 가짜 서버. 호출 기록도 남긴다."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def complete(self, prompt, temperature=0.1, n_predict=300, stop=None, timeout=300):
        self.calls.append((prompt, temperature))
        text = prompt.split("\n\n\n", 1)[1].rsplit("<end_of_turn>", 1)[0]
        ans = self.answers[text]
        return ans.pop(0) if isinstance(ans, list) else ans


cfg = Config.load(DEFAULT_CONFIG_PATH)
cues = [
    Cue(id=0, start=0, end=1, text="おはよう"),          # 정상
    Cue(id=1, start=1, end=2, text="あっ"),              # 가나만: 계속 가나로 답하면 음역 폴백
    Cue(id=2, start=2, end=3, text="帰り道"),            # 한자 섞임: 끝내 실패하면 원문 유지
    Cue(id=3, start=3, end=4, text="行こう"),            # 첫 시도엔 일본 문자가 남고 재시도에서 성공
    Cue(id=4, start=4, end=5, text="いいね"),            # 원문에 없는 영어가 섞이면 재시도
]
fake = FakeLLM({
    "おはよう": "안녕",
    "あっ": "あっ",
    "帰り道": "帰り道",
    "行こう": ["行こう", "가자"],
    "いいね": ["good 좋다", "좋다"],
})
out, rep = translate_cues_gemma(cues, fake, cfg, cancel_check=None)
check("normal", out[0], "안녕")
check("kana-only falls back to hangul", out[1], "앗")
check("kanji line keeps source", out[2], "帰り道")
check("retry fixes leaked kana", out[3], "가자")
check("retry fixes stray english", out[4], "좋다")
check("report transliterated", rep.transliterated_ids, [1])
check("report failed", rep.failed_ids, [2])
temps = [t for p, t in fake.calls if p.endswith("行こう<end_of_turn>\n<start_of_turn>model\n")]
assert len(temps) == 2 and temps[1] > temps[0], f"재시도에서 온도가 안 올라감: {temps}"
assert any("Do not use these characters" in p for p, _ in fake.calls), "재시도에 avoid 힌트가 없음"

print("말투/TranslateGemma 백엔드 테스트 통과")

# ---- 음성 인식 확신도 요약 / 예전 캐시 호환 ----
from core.stt import TranscriptionResult, Word, confidence_summary, words_from_json, words_to_json  # noqa: E402


def _res(probs):
    return TranscriptionResult(
        words=[Word(start=i, end=i + 0.5, text="あ", prob=p) for i, p in enumerate(probs)],
        language="ja", language_probability=1.0, duration=10.0,
    )


good = confidence_summary(_res([0.95] * 98 + [0.2, 0.3]))
assert good and good[2] == "양호" and abs(good[1] - 2.0) < 1e-6, good
mid = confidence_summary(_res([0.95] * 93 + [0.2] * 7))
assert mid and mid[2] == "보통", mid
bad = confidence_summary(_res([0.95] * 85 + [0.2] * 15))
assert bad and bad[2] == "불안정", bad
assert confidence_summary(_res([None, None])) is None, "확신도가 없으면 None이어야 함"
# 확신도가 저장되고, 예전 캐시(prob 키 없음)도 그대로 읽힌다
assert words_from_json(words_to_json(_res([0.9, 0.4]))).words[1].prob == 0.4
old_cache = {"language": "ja", "language_probability": 1.0, "duration": 5.0, "words": [{"start": 0, "end": 1, "text": "あ"}]}
assert words_from_json(old_cache).words[0].prob is None

print("음성 인식 확신도 테스트 통과")
