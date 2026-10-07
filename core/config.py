"""config.yaml 로더. 중첩 dict를 dot-path로 편하게 읽게 해준다."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import yaml


def _detect_root() -> Path:
    """개발 중에는 저장소 루트, PyInstaller로 묶인 exe에서는 exe가 있는 폴더를 기준으로 삼는다.

    PyInstaller onedir 빌드는 `Subtitle_Tool.exe` 옆에 `_internal/`을 두고 그 안에서 모듈을
    로드하므로, 이 파일의 `__file__`은 `_internal/core/config.py`를 가리켜 기존 방식
    (parent.parent)으로는 exe 폴더가 아니라 `_internal`이 나온다. 그러면 config.yaml,
    models/, tools/, cache/를 exe 옆이 아니라 엉뚱한 곳에서 찾게 되므로 frozen 여부를 봐서
    분기한다.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


ROOT = _detect_root()
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


class Config:
    def __init__(self, data: dict):
        self._data = data

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "Config":
        path = Path(path)
        if not path.exists() and getattr(sys, "frozen", False):
            # exe 옆에 사용자가 아직 config.yaml을 안 뒀다면, PyInstaller가 함께 묶은
            # 기본 config.yaml(_internal 안)을 대신 쓴다.
            bundled_dir = Path(getattr(sys, "_MEIPASS", "")) if getattr(sys, "_MEIPASS", "") else (
                Path(sys.executable).resolve().parent / "_internal"
            )
            bundled = bundled_dir / "config.yaml"
            if bundled.exists():
                path = bundled
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def resolve_path(self, dotted_key: str, default: str | None = None) -> Path:
        raw = self.get(dotted_key, default)
        p = Path(raw)
        return p if p.is_absolute() else (ROOT / p)

    def copy(self) -> "Config":
        return Config(copy.deepcopy(self._data))

    @property
    def data(self) -> dict:
        return self._data
