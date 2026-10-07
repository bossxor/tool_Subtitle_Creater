"""PyInstaller로 Subtitle Tool을 onedir exe로 묶는다.

models/, tools/llama.cpp는 용량이 커서(수 GB~11GB) 여기 포함하지 않는다.
exe 실행 폴더에 두면 core.config의 ROOT(=exe가 있는 폴더) 기준으로 자동 인식되고,
없으면 GUI가 첫 실행 때 core/setup_assets.py로 내려받는다.

사용법:
    .venv\\Scripts\\python.exe build_exe.py
"""
from __future__ import annotations

from pathlib import Path

import PyInstaller.__main__
from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(__file__).resolve().parent
VENV_SITE = ROOT / ".venv" / "Lib" / "site-packages"


def collect_nvidia_binaries() -> list[str]:
    args = []
    nvidia_dir = VENV_SITE / "nvidia"
    if not nvidia_dir.exists():
        return args
    for bin_dir in nvidia_dir.glob("*/bin"):
        for dll in bin_dir.glob("*.dll"):
            args += ["--add-binary", f"{dll};."]
    return args


def main() -> None:
    args = [
        str(ROOT / "main.py"),
        "--name",
        "Subtitle_Tool",
        "--onedir",
        "--windowed",
        "--noconfirm",
        "--clean",
        "--add-data",
        f"{ROOT / 'config.yaml'};.",
    ]
    args += collect_nvidia_binaries()

    # faster_whisper는 VAD 모델(assets/silero_vad_v6.onnx)을 패키지 데이터로 갖고 있는데,
    # PyInstaller의 임포트 분석만으로는 이런 비-코드 리소스 파일을 못 찾는다. 실제로
    # 이걸 빠뜨렸다가 실행 시 "silero_vad_v6.onnx File doesn't exist" 오류로 처음 빌드가 깨졌음.
    for src, dest in collect_data_files("faster_whisper"):
        args += ["--add-data", f"{src};{dest}"]

    print(f"nvidia DLL {len(collect_nvidia_binaries()) // 2}개 포함, 빌드 시작...")
    PyInstaller.__main__.run(args)


if __name__ == "__main__":
    main()
