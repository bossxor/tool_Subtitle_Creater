"""PyInstaller로 Subtitle Tool을 onedir exe로 묶는다.

models/, tools/llama.cpp는 용량이 커서(수 GB~11GB) 여기 포함하지 않는다.
exe 실행 폴더에 두면 core.config의 ROOT(=exe가 있는 폴더) 기준으로 자동 인식되고,
없으면 GUI가 첫 실행 때 core/setup_assets.py로 내려받는다.

사용법:
    .venv\\Scripts\\python.exe build_exe.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import PyInstaller.__main__
from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(__file__).resolve().parent
VENV_SITE = ROOT / ".venv" / "Lib" / "site-packages"
DIST_APP = ROOT / "dist" / "Subtitle_Tool"
PRESERVE_DIR = ROOT / "dist_preserve"
# exe 폴더 안에 쌓이는 사용자 데이터. PyInstaller는 빌드 때 dist/Subtitle_Tool을 통째로 지우므로
# (캐시에는 몇 시간짜리 음성 인식/번역 결과가 들어 있을 수 있다) 빌드 전에 빼 뒀다가 되돌린다.
USER_DATA = ("cache", "models", "tools")


def preserve_user_data() -> dict[str, tuple[str, str]]:
    """{이름: ("junction", 대상경로) | ("dir", 보관경로)}. 아무것도 지우지 않고 옮기기만 한다."""
    saved: dict[str, tuple[str, str]] = {}
    for name in USER_DATA:
        path = DIST_APP / name
        if not path.exists():
            continue
        try:
            target = os.readlink(path)  # junction/심볼릭 링크일 때만 성공
        except OSError:
            target = None
        if target is not None:
            os.rmdir(path)  # 링크만 제거(가리키는 실제 폴더는 그대로)
            saved[name] = ("junction", target.removeprefix("\\\\?\\"))
        else:
            PRESERVE_DIR.mkdir(exist_ok=True)
            dest = PRESERVE_DIR / name
            if dest.exists():
                raise SystemExit(f"{dest} 가 이미 있어 덮어쓸 수 없습니다. 내용을 확인하고 직접 옮겨 주세요.")
            shutil.move(str(path), str(dest))
            saved[name] = ("dir", str(dest))
    return saved


def restore_user_data(saved: dict[str, tuple[str, str]]) -> None:
    for name, (kind, where) in saved.items():
        path = DIST_APP / name
        if path.exists():
            print(f"경고: {path} 가 이미 있어 {name} 복원을 건너뜁니다 (보관본: {where})")
            continue
        if kind == "junction":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(path), where], check=True, capture_output=True)
        else:
            shutil.move(where, str(path))
        print(f"사용자 데이터 복원: {name}")
    if PRESERVE_DIR.exists() and not any(PRESERVE_DIR.iterdir()):
        PRESERVE_DIR.rmdir()


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
    executable = DIST_APP / "Subtitle_Tool.exe"
    if executable.exists():
        try:
            # Windows denies write access to a running executable. Probe before
            # moving model junctions, which a running inference still needs.
            with executable.open("r+b"):
                pass
        except PermissionError as e:
            raise SystemExit("Subtitle Tool이 실행 중이거나 실행 파일이 잠겨 있습니다. 종료 후 다시 빌드하세요.") from e
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
    saved = preserve_user_data()
    original_path = os.environ.get("PATH", "")
    # Dependency discovery must not pick an unrelated ICU/Qt DLL from a tool
    # injected into PATH (e.g. Poppler's icuuc.dll exports versioned symbols,
    # while Qt uses the Windows ICU API). Keep the build search path explicit.
    windows_dir = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    os.environ["PATH"] = os.pathsep.join(map(str, [
        Path(sys.executable).parent,
        windows_dir / "System32",
        windows_dir,
        VENV_SITE / "PySide6",
    ]))
    try:
        PyInstaller.__main__.run(args)
    finally:
        os.environ["PATH"] = original_path
        # 빌드가 실패해도 사용자 데이터는 반드시 되돌린다
        DIST_APP.mkdir(parents=True, exist_ok=True)
        restore_user_data(saved)

    # PyInstaller가 nvidia 패키지 폴더(_internal/nvidia)를 통째로 또 복사해서, 위에서 평평하게 넣은
    # DLL(_internal 바로 아래)과 중복된다(약 0.9GB). core/stt.py가 평평한 위치도 DLL 검색 경로에 올리므로
    # 중복 폴더는 지워도 되고, 지운 빌드로 GPU 음성 인식+번역이 끝까지 도는 것을 확인했다.
    dup = ROOT / "dist" / "Subtitle_Tool" / "_internal" / "nvidia"
    if dup.exists():
        shutil.rmtree(dup)
        print(f"중복 CUDA DLL 폴더 제거: {dup}")


if __name__ == "__main__":
    main()
