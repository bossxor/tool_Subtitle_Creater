"""faster-whisper의 단어(토큰) 단위 타임스탬프를 자막 한 줄 분량의 cue로 재구성한다.

일본어는 띄어쓰기가 없어서 "단어"가 사실상 서브캐릭터 토큰 단위로 나온다.
문장부호(。！？)는 토큰 끝에 붙어 있으므로 이를 1차 분할 기준으로 쓰고,
문장부호 없이 말이 길게 이어질 때는 토큰 사이의 침묵 구간(pause)을 2차 기준으로 쓴다.
그래도 너무 길면 글자/시간 상한으로 강제 분할한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.stt import Word

SENTENCE_END = ("。", "！", "？", "…", "!", "?")


@dataclass
class Cue:
    id: int
    start: float
    end: float
    text: str
    meta: dict = field(default_factory=dict)


def build_cues(words: list[Word], config) -> list[Cue]:
    max_dur = config.get("segmenter.max_cue_duration", 8.0)
    min_dur = config.get("segmenter.min_cue_duration", 0.8)
    pause_gap = config.get("segmenter.pause_split_gap", 0.6)
    max_words = config.get("segmenter.max_words_per_cue", 40)

    raw_cues: list[Cue] = []
    buf: list[Word] = []

    def flush() -> None:
        if not buf:
            return
        text = "".join(w.text for w in buf).strip()
        if text:
            raw_cues.append(Cue(id=len(raw_cues), start=buf[0].start, end=buf[-1].end, text=text))
        buf.clear()

    n = len(words)
    for i, w in enumerate(words):
        buf.append(w)
        cur_dur = buf[-1].end - buf[0].start
        ends_sentence = w.text.rstrip().endswith(SENTENCE_END)
        next_gap = (words[i + 1].start - w.end) if i + 1 < n else None
        long_pause = next_gap is not None and next_gap >= pause_gap
        overflow = cur_dur >= max_dur or len(buf) >= max_words

        if ends_sentence or long_pause or overflow:
            flush()
    flush()

    cues = _filter_hallucinations(raw_cues, config)
    cues = _merge_tiny_fragments(cues, min_dur)

    for i, c in enumerate(cues):
        c.id = i
    return cues


def _filter_hallucinations(cues: list[Cue], config) -> list[Cue]:
    blocklist = config.get("segmenter.hallucination_blocklist", []) or []
    out: list[Cue] = []
    prev_text = None
    for c in cues:
        if any(b in c.text for b in blocklist):
            continue
        if c.text == prev_text:
            # 동일 문장이 연속으로 반복되면(2회차부터) whisper의 반복 환각으로 보고 버린다
            continue
        prev_text = c.text
        out.append(c)
    return out


def _merge_tiny_fragments(cues: list[Cue], min_dur: float) -> list[Cue]:
    """너무 짧은(<0.3s) 잔여 조각을 다음 cue에 붙인다. 정상적인 짧은 감탄사(예: min_dur 근처)는 그대로 둔다."""
    if not cues:
        return cues
    merged: list[Cue] = []
    carry: Cue | None = None
    tiny_threshold = min(0.3, min_dur)
    for c in cues:
        if carry is not None:
            c = Cue(id=c.id, start=carry.start, end=c.end, text=carry.text + c.text)
            carry = None
        if (c.end - c.start) < tiny_threshold and c is not cues[-1]:
            carry = c
            continue
        merged.append(c)
    if carry is not None:
        merged.append(carry)
    return merged


def cues_to_json(cues: list[Cue]) -> list[dict]:
    return [{"id": c.id, "start": c.start, "end": c.end, "text": c.text, "meta": c.meta} for c in cues]


def cues_from_json(data: list[dict]) -> list[Cue]:
    return [Cue(id=d["id"], start=d["start"], end=d["end"], text=d["text"], meta=d.get("meta", {})) for d in data]
