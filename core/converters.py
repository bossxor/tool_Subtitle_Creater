"""기존 .smi 자막을 파싱해 srt/ass로 변환한다. AI 재처리 없이 포맷만 바꾼다."""
from __future__ import annotations

import html
import re
from pathlib import Path

from core.encoding_fix import detect_and_decode
from core.writers import DisplayCue, write_ass, write_smi, write_srt

_SYNC_RE = re.compile(r"<SYNC[^>]*Start\s*=\s*\"?(\d+)\"?[^>]*>", re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(raw: str) -> str:
    text = _BR_RE.sub("\n", raw)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def parse_smi(path: Path) -> list[DisplayCue]:
    """SMI 파일을 읽어 DisplayCue 목록으로 돌려준다. 빈 싱크(&nbsp;)는 이전 cue의 종료 시각으로 쓴다.

    옛날 한국 smi 자막은 CP949로 저장된 경우가 흔해서, BOM/UTF-8/CP949/UTF-16 순으로
    인코딩을 자동 감지해서 읽는다(core.encoding_fix).
    """
    raw_bytes = Path(path).read_bytes()
    raw = detect_and_decode(raw_bytes)
    if raw is None:
        raw = raw_bytes.decode("utf-8", errors="replace")
    marks = list(_SYNC_RE.finditer(raw))
    entries: list[tuple[int, str]] = []
    for i, m in enumerate(marks):
        start = int(m.group(1))
        seg_end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        entries.append((start, raw[m.end():seg_end]))

    cues: list[DisplayCue] = []
    for i, (start_ms, body) in enumerate(entries):
        text = _clean_text(body)
        if not text:
            continue
        end_ms = entries[i + 1][0] if i + 1 < len(entries) else start_ms + 4000
        if end_ms <= start_ms:
            end_ms = start_ms + 1000
        cues.append(DisplayCue(start=start_ms / 1000, end=end_ms / 1000, text=text))
    return cues


def extract_lang_suffix(path: Path, stem: str) -> str:
    """'ep01.ko.smi' + stem='ep01' -> 'ko'.  'ep01.smi' + stem='ep01' -> ''."""
    name = Path(path).stem  # 확장자 하나만 떼어낸 이름, 예: 'ep01.ko'
    if name == stem:
        return ""
    prefix = stem + "."
    return name[len(prefix):] if name.startswith(prefix) else ""


_WRITERS = {"srt": write_srt, "ass": write_ass, "smi": write_smi}


def write_cues_as(
    cues: list[DisplayCue],
    out_dir: Path,
    formats: list[str],
    stem: str,
    lang: str = "",
    encodings: dict[str, str] | None = None,
) -> list[Path]:
    """이미 만들어진 cue 목록을 formats(예: ["srt","ass"])로 저장한다. parse_smi 결과를 그대로
    쓸 수도 있고, AI 검수(core.translate.verify_cues)로 글자를 고친 뒤에 쓸 수도 있다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encodings = encodings or {}

    written: list[Path] = []
    for fmt in formats:
        writer = _WRITERS.get(fmt)
        if writer is None:
            continue
        name = f"{stem}.{lang}.{fmt}" if lang else f"{stem}.{fmt}"
        p = out_dir / name
        writer(cues, p, encoding=encodings.get(fmt, "utf-8-sig"))
        written.append(p)
    return written


def convert_smi_file(
    smi_path: Path,
    out_dir: Path,
    formats: list[str],
    stem: str | None = None,
    lang: str = "",
    encodings: dict[str, str] | None = None,
) -> list[Path]:
    """smi_path를 파싱해서 formats로 바로 저장한다 (AI 검수 없이). AI로 깨진 글자까지 확인하려면
    parse_smi() + core.translate.verify_cues() + write_cues_as()를 직접 조합해서 쓴다
    (core.folder_scan.run_conversions가 그렇게 한다)."""
    smi_path = Path(smi_path)
    cues = parse_smi(smi_path)
    if not cues:
        raise ValueError(f"SMI에서 자막을 하나도 못 읽음: {smi_path}")
    return write_cues_as(cues, out_dir, formats, stem or smi_path.stem, lang, encodings)
