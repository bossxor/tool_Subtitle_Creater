"""Subtitle Tool GUI 실행 진입점.

    python main.py

Phase 3에서 이 파일을 PyInstaller로 묶어 exe로 만든다.
"""
from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from core.cleanup import kill_orphan_llama_server
from core.config import Config, DEFAULT_CONFIG_PATH
from gui.main_window import MainWindow


def _selftest() -> int:
    """exe로 묶인 뒤 CUDA/ctranslate2 등이 실제로 로드되는지 확인하는 숨은 모드.
    `Subtitle_Tool.exe --selftest`로 실행. Qt 창은 띄우지 않는다."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    import core.config as cfgmod
    from core.config import Config, DEFAULT_CONFIG_PATH

    config = Config.load(DEFAULT_CONFIG_PATH)
    precision = config.get("stt.precision")
    print(f"[selftest] ROOT={cfgmod.ROOT}, precision={precision}")

    model_dir = config.resolve_path(f"stt.models.{precision}")
    print(f"[selftest] STT 모델 경로: {model_dir} (존재: {model_dir.exists()})")
    if model_dir.exists():
        from core.stt import STTEngine

        STTEngine(model_dir)
        print("[selftest] STTEngine 로드 성공 (CUDA/ctranslate2 정상 동작)")
    else:
        print("[selftest] 모델이 없어 STT 로드는 건너뜀 (경로 해석만 확인)")

    server_exe = config.resolve_path("llm.server_exe")
    print(f"[selftest] llama-server 경로: {server_exe} (존재: {server_exe.exists()})")

    if "--full-pipeline" in sys.argv:
        idx = sys.argv.index("--full-pipeline")
        video, out_dir = sys.argv[idx + 1], sys.argv[idx + 2]
        from core.pipeline import run_batch

        print(f"[selftest] 전체 파이프라인 실행: {video} -> {out_dir}")
        outputs = run_batch([video], out_dir, config, progress_cb=lambda m: print(f"[selftest] {m}"))
        print(f"[selftest] 파이프라인 결과: {outputs}")

    print("[selftest] OK")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    if "--selftest" in sys.argv:
        return _selftest()

    try:
        kill_orphan_llama_server(Config.load(DEFAULT_CONFIG_PATH))
    except Exception:
        logging.getLogger(__name__).debug("잔여 프로세스 정리 단계에서 오류 (무시)", exc_info=True)

    app = QApplication(sys.argv)
    app.setApplicationName("Subtitle Tool")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
