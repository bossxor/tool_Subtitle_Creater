"""Read-only subtitle inspection, provenance and backed-up text-only updates."""
from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from core.encoding_fix import detect_and_decode
from core.errors import PipelineCancelled
from core.segmenter import Cue, cues_from_json, cues_to_json
from core.validate import translation_issue
from core.writers import DisplayCue, build_display_cues

logger = logging.getLogger(__name__)
_TIME = r"\d{1,3}:\d{2}:\d{2}[,.]\d{2,3}"
_SRT = re.compile(rf"^\d+[ \t]*\r?\n(?P<start>{_TIME})[ \t]+-->[ \t]+(?P<end>{_TIME})[^\r\n]*\r?\n(?P<text>.*?)(?=\r?\n[ \t]*\r?\n|\Z)", re.M | re.S)
_SYNC = re.compile(r'<SYNC\b[^>]*\bStart\s*=\s*["\x27]?(\d+)[^>]*>', re.I)
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".smi"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".ts", ".webm", ".m4v", ".flv", ".wav"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_path(path: Path) -> Path:
    return Path(str(path) + ".subtitle-tool.json")


def _save_manifest(path: Path, data: dict) -> None:
    target = manifest_path(path)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".json.tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def record_generation(path: Path, video: Path, cues: list[Cue], config, *, report=None, glossary=None) -> None:
    """Sidecar is deliberately independent of optional human-readable work logs."""
    video = Path(video)
    stat = video.stat() if video.exists() else None
    _save_manifest(path, {
        "version": 1, "output_sha256": file_hash(path), "video": str(video.resolve()),
        "video_size": stat.st_size if stat else None,
        "video_mtime_ns": stat.st_mtime_ns if stat else None,
        "source_language": config.get("source_language", "ja"),
        "target_language": config.get("target_language", "ko"),
        "bilingual": config.get("subtitle.bilingual", False),
        "glossary": glossary or {},
        "model": config.get("llm.translate_model") if config.get("llm.translate_backend") == "translategemma" else config.get("llm.model"),
        "source_cues": cues_to_json(cues),
        "output_times": [[c.start, c.end] for c in _parse(Path(path)).cues],
        "quality": {"status": report.status, "summary": report.summary()} if report else {},
    })


@dataclass
class SubtitleEntry:
    path: Path
    raw: str
    digest: str
    cues: list[DisplayCue]
    spans: list[tuple[int, int]]
    editable: bool = True
    origin: str = "출처 불명"
    confirmed: bool = False
    source_cues: list[Cue] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    issues: list[tuple[int, str]] = field(default_factory=list)
    note: str = ""


def _seconds(value: str) -> float:
    h, m, s = value.replace(",", ".").split(":")
    if not 0 <= int(m) < 60 or not 0 <= float(s) < 60:
        raise ValueError("자막 시간 형식 오류")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _parse(path: Path) -> SubtitleEntry:
    data = path.read_bytes()
    raw = detect_and_decode(data)
    if raw is None:
        raise ValueError("인코딩을 확인할 수 없습니다.")
    entry = SubtitleEntry(path, raw, hashlib.sha256(data).hexdigest(), [], [])

    def add(start, end, text_start, text_end, text):
        if end <= start:
            raise ValueError("종료 시간이 시작 시간보다 빠르거나 같습니다.")
        entry.cues.append(DisplayCue(start, end, text))
        entry.spans.append((text_start, text_end))

    if path.suffix.lower() == ".srt":
        for match in _SRT.finditer(raw):
            text = match["text"].rstrip("\r\n")
            add(_seconds(match["start"]), _seconds(match["end"]), match.start("text"), match.start("text") + len(text), text)
            if re.search(r"<[^>]+>|\{[^}]+\}", text):
                entry.editable = False
    elif path.suffix.lower() == ".ass":
        fields = []
        in_events = False
        for line in re.finditer(r"[^\r\n]+", raw):
            value = line.group()
            if value.startswith("["):
                in_events = value.strip().lower() == "[events]"
            if not in_events:
                continue
            if value.lower().startswith("format:"):
                fields = [s.strip().lower() for s in value.split(":", 1)[1].split(",")]
            if not value.lower().startswith("dialogue:"):
                continue
            if not fields or fields[-1] != "text" or "start" not in fields or "end" not in fields:
                raise ValueError("지원하지 않는 ASS 이벤트 필드입니다.")
            body = value.split(":", 1)[1]
            parts = body.split(",", len(fields) - 1)
            if len(parts) != len(fields):
                raise ValueError("손상된 ASS 대사입니다.")
            text = parts[-1]
            offset = line.start() + value.index(":") + 1 + sum(len(s) + 1 for s in parts[:-1])
            add(_seconds(parts[fields.index("start")].strip()), _seconds(parts[fields.index("end")].strip()),
                offset, offset + len(text), text.replace(r"\N", "\n").replace(r"\n", "\n"))
            if "{" in text or re.search(r"\\(?![Nn])", text):
                entry.editable = False
    elif path.suffix.lower() == ".smi":
        marks = list(_SYNC.finditer(raw))
        classes = set()
        for i, mark in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
            body = raw[mark.end():end]
            paragraphs = list(re.finditer(r"<P\b[^>]*>", body, re.I))
            if len(paragraphs) != 1:
                entry.editable = False
            if not paragraphs:
                continue
            p = paragraphs[0]
            cls = re.search(r'\bclass\s*=\s*["\x27]?([^\s>"\x27]+)', p.group(), re.I)
            if cls:
                classes.add(cls[1].lower())
            content = body[p.end():]
            close = re.search(r"</(?:P|BODY|SAMI)\b", content, re.I)
            if close:
                content = content[:close.start()]
            payload = content.strip()
            clean = html.unescape(re.sub(r"<br\s*/?>", "\n", payload, flags=re.I)).strip()
            if not clean:
                continue
            offset = mark.end() + p.end() + len(content) - len(content.lstrip())
            finish = int(marks[i + 1][1]) / 1000 if i + 1 < len(marks) else int(mark[1]) / 1000 + 4
            add(int(mark[1]) / 1000, finish, offset, offset + len(payload), clean)
            if re.search(r"<(?!br\b)[^>]+>", payload, re.I):
                entry.editable = False
        if len(classes) > 1:
            entry.editable = False
    else:
        raise ValueError("지원 형식은 SRT, ASS, SMI입니다.")
    if not entry.cues:
        raise ValueError("읽을 수 있는 자막 대사가 없습니다.")
    if not entry.editable:
        entry.note = "스타일·다국어 구조 보존을 위해 읽기 전용입니다."
    return entry


def _same_cues(left, right) -> bool:
    return len(left) == len(right) and all(
        abs(a.start - b.start) < 0.011 and abs(a.end - b.end) < 0.011
        and " ".join(a.text.split()) == " ".join(b.text.split()) for a, b in zip(left, right))


def _legacy_source(entry: SubtitleEntry, config, cancel_check=None) -> None:
    """Content match only: a familiar filename alone never proves provenance."""
    stem = re.sub(r"\.(?:ko|ja|en)$", "", entry.path.stem, flags=re.I)
    roots = {config.resolve_path("paths.cache_dir", "cache")}
    from core.config import ROOT
    roots.add(ROOT / "dist" / "Subtitle_Tool" / "cache")
    for root in roots:
        if not root.exists():
            continue
        for folder in root.iterdir():
            if cancel_check and cancel_check():
                raise PipelineCancelled()
            if not folder.is_dir() or not re.fullmatch(re.escape(stem) + r"_[0-9a-f]{8,12}", folder.name):
                continue
            for translated in folder.glob("cues_translated.ko*.json"):
                if cancel_check and cancel_check():
                    raise PipelineCancelled()
                try:
                    saved = json.loads(translated.read_text(encoding="utf-8"))
                    mapping = {int(k): v for k, v in saved.get("translations", saved).items()}
                    for src in folder.glob("cues_source*.json"):
                        cues = cues_from_json(json.loads(src.read_text(encoding="utf-8")))
                        if set(mapping) != {c.id for c in cues}:
                            continue
                        display = build_display_cues(cues, mapping, config)
                        if _same_cues(entry.cues, display):
                            entry.origin = "캐시 일치"
                            entry.confirmed = True
                            entry.source_cues = cues
                            entry.metadata = {"source_language": "ja", "target_language": "ko", "source_cues": cues_to_json(cues)}
                            return
                except (OSError, ValueError, TypeError, KeyError):
                    continue


def glossary_for_row(entry: SubtitleEntry, row: int) -> dict[str, str]:
    rows = entry.metadata.get("glossary_by_row", {})
    terms = rows.get(str(row), entry.metadata.get("glossary", {})) if isinstance(rows, dict) else entry.metadata.get("glossary", {})
    return {k: v for k, v in terms.items() if isinstance(k, str) and isinstance(v, str)} if isinstance(terms, dict) else {}


def inspect_subtitle(path: Path, config, *, legacy: bool = True, cancel_check=None) -> SubtitleEntry:
    entry = _parse(Path(path))
    sidecar = manifest_path(entry.path)
    if sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if metadata.get("version") != 1:
                raise ValueError("생성 이력 버전 오류")
            entry.source_cues = cues_from_json(metadata["source_cues"])
            entry.metadata = metadata
            entry.confirmed = metadata.get("output_sha256") == entry.digest
            entry.origin = "생성 확인" if entry.confirmed else "수정됨"
            output_times = metadata.get("output_times", [])
            if len(output_times) != len(entry.cues) or any(
                len(times) != 2 or abs(c.start - times[0]) > .011 or abs(c.end - times[1]) > .011
                for c, times in zip(entry.cues, output_times)):
                entry.source_cues = []
                entry.note = "자막 구간이 생성 이력과 달라 원문을 줄별로 연결할 수 없습니다. 전체 재생성을 이용하세요."
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            entry.confirmed = False
            entry.origin = "출처 불명"
            entry.source_cues = []
            entry.metadata = {}
            entry.note = "생성 이력을 읽을 수 없습니다. 출처 불명으로 처리합니다."
    elif legacy:
        _legacy_source(entry, config, cancel_check)
    if len(entry.source_cues) != len(entry.cues):
        entry.source_cues = []
    source_by_row = {i: c.text for i, c in enumerate(entry.source_cues)}
    for i, cue in enumerate(entry.cues):
        issue = translation_issue(cue.text, source_by_row.get(i, ""), entry.metadata.get("target_language", "ko"), glossary=glossary_for_row(entry, i))
        if issue:
            entry.issues.append((i, issue))
    return entry


def apply_changes(entry: SubtitleEntry, replacements: dict[int, str], *,
                  glossary_by_row: dict[int, dict[str, str]] | None = None) -> Path:
    if not entry.editable or not replacements:
        raise ValueError("적용할 변경이 없거나 읽기 전용 자막입니다.")
    if file_hash(entry.path) != entry.digest:
        raise ValueError("검사 후 파일이 바뀌었습니다. 다시 검사해 주세요.")
    raw = entry.raw
    newline = "\r\n" if "\r\n" in raw else "\n"
    for index in sorted(replacements, reverse=True):
        if not 0 <= index < len(entry.spans) or not isinstance(replacements[index], str) or not replacements[index].strip():
            raise ValueError("유효하지 않은 변경 항목입니다.")
        text = "\n".join(line.strip() for line in replacements[index].splitlines() if line.strip())
        if entry.path.suffix.lower() == ".ass":
            text = text.replace("{", "｛").replace("}", "｝").replace("\n", r"\N")
        elif entry.path.suffix.lower() == ".smi":
            text = html.escape(text).replace("\n", "<br>")
        else:
            text = text.replace("\n", newline)
        start, end = entry.spans[index]
        raw = raw[:start] + text + raw[end:]
    backup = entry.path.with_name(entry.path.name + ".backup-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
    shutil.copy2(entry.path, backup)
    with tempfile.NamedTemporaryFile(dir=entry.path.parent, suffix=".subtitle.tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_bytes(raw.encode("utf-8-sig"))
        if file_hash(entry.path) != entry.digest:
            raise ValueError("적용 중 파일이 바뀌었습니다. 다시 검사해 주세요.")
        os.replace(temporary, entry.path)
    finally:
        temporary.unlink(missing_ok=True)
    if entry.confirmed and entry.metadata:
        metadata = dict(entry.metadata, output_sha256=file_hash(entry.path), version=1,
                        output_times=[[c.start, c.end] for c in entry.cues])
        if glossary_by_row is not None and entry.source_cues:
            previous = metadata.get("glossary_by_row", {})
            row_glossaries = dict(previous) if isinstance(previous, dict) else {}
            for row in replacements:
                source = entry.source_cues[row].text
                glossary = glossary_by_row.get(row, glossary_for_row(entry, row))
                row_glossaries[str(row)] = {key: value for key, value in glossary.items() if key and key in source}
            metadata["glossary_by_row"] = row_glossaries
        try:
            _save_manifest(entry.path, metadata)
        except OSError:
            logger.exception("자막은 적용됐지만 생성 이력을 갱신하지 못했습니다: %s", entry.path)
    return backup
