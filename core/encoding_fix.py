"""텍스트 자막(srt/ass, 그리고 smi 파싱 전 디코딩)의 인코딩을 감지하고 BOM 있는 UTF-8로 통일한다.

BOM 없는 UTF-8이나 옛날 한국 자막에 흔한 CP949(EUC-KR 계열)로 저장된 파일은 곰플레이어 등에서
글자가 깨져 보인다. 감지 순서: UTF-8 BOM(이미 정상) → BOM 없는 UTF-8 → CP949 → UTF-16.
"""
from __future__ import annotations

from pathlib import Path

_BOM = b"\xef\xbb\xbf"
_CANDIDATE_ENCODINGS = ("utf-8", "cp949", "utf-16")


def detect_and_decode(raw: bytes) -> str | None:
    """raw 바이트를 디코딩한다. 어떤 인코딩인지 확신이 안 서면 None을 돌려준다(함부로 안 건드림)."""
    if raw.startswith(_BOM):
        return raw.decode("utf-8-sig")
    for enc in _CANDIDATE_ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def needs_fix(path: Path) -> bool:
    """BOM 없는 UTF-8/CP949/UTF-16으로 저장돼 있어 손봐야 하면 True."""
    raw = Path(path).read_bytes()
    if raw.startswith(_BOM):
        return False
    return detect_and_decode(raw) is not None


def fix_encoding_file(path: Path) -> bool:
    """내용은 그대로 두고 utf-8-sig(BOM 포함)로 다시 저장한다. 바꿨으면 True, 이미 괜찮거나
    인코딩을 확신할 수 없어 손대지 않았으면 False."""
    path = Path(path)
    raw = path.read_bytes()
    if raw.startswith(_BOM):
        return False
    text = detect_and_decode(raw)
    if text is None:
        return False
    # Windows denies CREATE_ALWAYS (write_text's "w" mode) for hidden files.
    # Open the existing file for update instead, preserving hidden attributes
    # and the decoded text's original line endings.
    encoded = text.encode("utf-8-sig")
    with path.open("r+b") as stream:
        stream.write(encoded)
        stream.truncate()
    return True
