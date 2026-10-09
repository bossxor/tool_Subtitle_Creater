"""실행 중 '취소' 버튼을 누르면 실제로 파이프라인이 중단되는지 확인."""
import tempfile
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gui.main_window import MainWindow  # noqa: E402

QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.critical = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "bench" / "sample" / "sample_10min.wav"
OUT_DIR = ROOT / "bench" / "out" / "gui_cancel_e2e"

# 기존 샘플/사용자 캐시를 지우지 않고 독립적인 빈 캐시에서 시작한다.
from core.config import Config
from core.pipeline import cache_dir_for

temporary_cache = tempfile.TemporaryDirectory(prefix="subtitle_cancel_test_")
test_config = Config.load()
test_config.set("paths.cache_dir", temporary_cache.name)
cache_dir = cache_dir_for(VIDEO, test_config)

app = QApplication(sys.argv)
win = MainWindow()
original_config = win._build_config


def build_test_config():
    config = original_config()
    config.set("paths.cache_dir", temporary_cache.name)
    return config


win._build_config = build_test_config
win._add_files([str(VIDEO)])
win.out_dir_edit.setText(str(OUT_DIR))

result = {"state": None}

orig_finished = win._on_finished_ok
orig_failed = win._on_failed
orig_cancelled = win._on_cancelled
orig_stopped = win._on_worker_stopped


def wrapped_finished(outputs):
    orig_finished(outputs)
    result["state"] = "finished"


def wrapped_failed(msg):
    orig_failed(msg)
    result["state"] = f"failed: {msg}"


def wrapped_cancelled():
    orig_cancelled()
    result["state"] = "cancelled"


def wrapped_stopped():
    orig_stopped()
    app.quit()


win._on_finished_ok = wrapped_finished
win._on_failed = wrapped_failed
win._on_cancelled = wrapped_cancelled
win._on_worker_stopped = wrapped_stopped

win._on_start()
print("worker running:", win.worker.isRunning())

# 오디오 추출/STT가 시작되자마자(1초 뒤) 취소 요청
def _watchdog():
    if win.worker is not None and win.worker.isRunning():
        win.worker.wait(15000)
    app.quit()


QTimer.singleShot(1000, win._on_cancel)
QTimer.singleShot(60_000, _watchdog)  # 안전장치
app.exec()

print("RESULT:", result["state"])
assert result["state"] == "cancelled", f"취소가 반영되지 않음: {result['state']}"

assert not list(cache_dir.glob("cues_translated.ko.*.json")), "취소했는데도 번역까지 끝까지 진행됨 (취소가 안 먹힘)"
print("취소 확인 완료: 번역 단계 전에 멈춤")
temporary_cache.cleanup()
