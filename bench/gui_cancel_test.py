"""실행 중 '취소' 버튼을 누르면 실제로 파이프라인이 중단되는지 확인."""
import shutil
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

# 이 영상의 캐시를 완전히 지워서 STT부터 다시 돌게 만든다
import hashlib

h = hashlib.sha1(str(VIDEO.resolve()).encode("utf-8")).hexdigest()[:8]
cache_dir = ROOT / "cache" / f"{VIDEO.stem}_{h}"
shutil.rmtree(cache_dir, ignore_errors=True)

app = QApplication(sys.argv)
win = MainWindow()
win._add_files([str(VIDEO)])
win.out_dir_edit.setText(str(OUT_DIR))

result = {"state": None}

orig_finished = win._on_finished_ok
orig_failed = win._on_failed
orig_cancelled = win._on_cancelled


def wrapped_finished(outputs):
    orig_finished(outputs)
    result["state"] = "finished"
    app.quit()


def wrapped_failed(msg):
    orig_failed(msg)
    result["state"] = f"failed: {msg}"
    app.quit()


def wrapped_cancelled():
    orig_cancelled()
    result["state"] = "cancelled"
    app.quit()


win._on_finished_ok = wrapped_finished
win._on_failed = wrapped_failed
win._on_cancelled = wrapped_cancelled

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

translated_cache = cache_dir / "cues_translated.ko.json"
assert not translated_cache.exists(), "취소했는데도 번역까지 끝까지 진행됨 (취소가 안 먹힘)"
print("취소 확인 완료: 번역 단계 전에 멈춤")
