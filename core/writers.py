"""번역된 cue를 실제 자막 표시 규격(줄바꿈, 표시 시간)에 맞춰 srt/smi로 저장한다."""
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

from core.segmenter import Cue


@dataclass
class DisplayCue:
    start: float
    end: float
    text: str


def wrap_lines(text: str, max_chars_per_line: int, max_lines: int) -> str:
    text = " ".join(text.split())  # 개행/중복 공백 정리
    if not text:
        return text
    lines = textwrap.wrap(text, width=max_chars_per_line, break_long_words=True, break_on_hyphens=False)
    if not lines:
        return text
    if len(lines) <= max_lines:
        return "\n".join(lines)
    head = lines[: max_lines - 1]
    tail = " ".join(lines[max_lines - 1 :])
    return "\n".join(head + [tail])


def _visible_char_count(text: str) -> int:
    return len(text.replace("\n", "").replace(" ", ""))


def _compose_bilingual(ko_text: str, source_text: str, max_chars_per_line: int, max_lines: int) -> str:
    """번역문(위, 줄바꿈 적용)과 원문(아래, 한 줄로 요약)을 한 cue에 같이 담는다."""
    ko_wrapped = wrap_lines(ko_text, max_chars_per_line, max_lines)
    src_line = " ".join(source_text.split())
    limit = max_chars_per_line * 2
    if len(src_line) > limit:
        src_line = src_line[: limit - 1] + "…"
    return f"{ko_wrapped}\n{src_line}" if src_line else ko_wrapped


def build_display_cues(
    cues: list[Cue],
    text_by_id: dict[int, str],
    config,
    source_text_by_id: dict[int, str] | None = None,
) -> list[DisplayCue]:
    """cue의 원 타임스탬프(start/end)는 유지하되, 표시 시간을 자막 규격(CPS, 최소/최대)에 맞춰 조정한다.

    source_text_by_id를 주면 이중 자막(번역 위 + 원문 아래)으로 합성한다.
    """
    max_lines = config.get("subtitle.max_lines", 2)
    max_chars = config.get("subtitle.max_chars_per_line", 20)
    min_sec = config.get("subtitle.min_display_sec", 1.0)
    max_sec = config.get("subtitle.max_display_sec", 7.0)
    max_cps = config.get("subtitle.max_cps", 15)
    min_gap = 0.05

    ordered = sorted(cues, key=lambda c: c.start)
    display: list[DisplayCue] = []
    for i, c in enumerate(ordered):
        raw_text = text_by_id.get(c.id, c.text)
        if source_text_by_id is not None:
            wrapped = _compose_bilingual(raw_text, source_text_by_id.get(c.id, ""), max_chars, max_lines)
        else:
            wrapped = wrap_lines(raw_text, max_chars, max_lines)
        if not wrapped:
            continue

        start = c.start
        end = c.end
        dur = end - start

        char_count = _visible_char_count(wrapped)
        ideal_dur = char_count / max_cps if max_cps > 0 else dur
        needed = max(dur, min_sec, ideal_dur)
        needed = min(needed, max_sec)

        next_start = ordered[i + 1].start if i + 1 < len(ordered) else None
        limit = (next_start - min_gap) if next_start is not None else start + needed
        end = min(start + needed, limit) if limit > start else start + min(needed, dur if dur > 0 else min_sec)
        if end <= start:
            end = start + min(0.5, max(min_sec, dur))

        display.append(DisplayCue(start=start, end=end, text=wrapped))
    return display


def _srt_timestamp(t: float) -> str:
    t = max(0.0, t)
    total_ms = round(t * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues: list[DisplayCue], path: Path, encoding: str = "utf-8-sig") -> None:
    """encoding 기본값은 BOM이 붙는 utf-8-sig다. BOM이 없으면 곰플레이어 등 일부 플레이어가
    한글을 시스템 기본 코드페이지(CP949)로 잘못 해석해 깨진 글자로 보여준다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, c in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(c.start)} --> {_srt_timestamp(c.end)}")
        lines.append(c.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding=encoding)


def _smi_ms(t: float) -> int:
    return max(0, round(t * 1000))


def write_smi(cues: list[DisplayCue], path: Path, encoding: str = "utf-8-sig", lang_class: str = "KRCC") -> None:
    """encoding 기본값은 BOM이 붙는 utf-8-sig. SAMI(.smi)는 원래 EUC-KR/CP949 시대의 포맷이라
    BOM 없이 UTF-8로만 저장하면 곰플레이어 등에서 한글이 깨져 보이는 경우가 많다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<SAMI>",
        "<HEAD>",
        "<STYLE TYPE=\"text/css\">",
        "<!--",
        "P { font-family: 맑은 고딕, sans-serif; font-size: 20pt; text-align: center;"
        " color: white; background-color: black; }",
        f".{lang_class} {{ Name: Korean; lang: ko-KR; SAMIType: CC; }}",
        "-->",
        "</STYLE>",
        "</HEAD>",
        "<BODY>",
    ]
    for c in cues:
        text_html = c.text.replace("\n", "<br>")
        parts.append(f"<SYNC Start={_smi_ms(c.start)}><P Class={lang_class}>{text_html}")
        parts.append(f"<SYNC Start={_smi_ms(c.end)}><P Class={lang_class}>&nbsp;")
    parts.append("</BODY></SAMI>")
    path.write_text("\n".join(parts), encoding=encoding, errors="replace")


def _ass_timestamp(t: float) -> str:
    """ASS는 centisecond(1/100초) 단위, H:MM:SS.cc 형식(시간 앞에 0 안 붙임)."""
    t = max(0.0, t)
    total_cs = round(t * 100)
    h, rem = divmod(total_cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    # { } 는 ASS에서 스타일 오버라이드 태그로 해석되므로 전각 문자로 바꿔 깨지지 않게 한다
    text = text.replace("{", "｛").replace("}", "｝")
    return text.replace("\n", "\\N")


def write_ass(
    cues: list[DisplayCue],
    path: Path,
    encoding: str = "utf-8-sig",
    font_name: str = "맑은 고딕",
    font_size: int = 20,
) -> None:
    """encoding 기본값은 BOM이 붙는 utf-8-sig (srt/smi와 동일한 이유)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"""[Script Info]
Title: Subtitle Tool
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: None
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size * 2},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,1,2,30,30,36,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""

    lines = [header]
    for c in cues:
        text = _ass_escape(c.text)
        lines.append(
            f"Dialogue: 0,{_ass_timestamp(c.start)},{_ass_timestamp(c.end)},Default,,0,0,0,,{text}"
        )
    path.write_text("\n".join(lines) + "\n", encoding=encoding)
