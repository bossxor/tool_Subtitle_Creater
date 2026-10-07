"""GUI의 '시작' 버튼 -> PipelineWorker 스레드 -> run_batch 연결이 실제로 동작하는지 확인하는 스모크 테스트.
캐시가 있는 샘플을 써서 몇 초 안에 끝나야 정상이다.

주의: MainWindow._on_start()는 시그널 연결 직후 바로 worker.start()를 호출하므로,
바깥에서 시그널을 나중에 connect하면(캐시 히트로 매우 빨리 끝나는 경우) 경쟁 상태로
emit을 놓칠 수 있다. 그래서 여기서는 인스턴스의 콜백 메서드 자체를 start() 호출 전에
감싼다.
"""
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gui.main_window import MainWindow  # noqa: E402

QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.critical = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)  # 덮어쓰기/첫실행 확인을 항상 예로

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "bench" / "sample" / "sample_10min.wav"
OUT_DIR = ROOT / "bench" / "out" / "gui_e2e"

app = QApplication(sys.argv)
win = MainWindow()
win._add_files([str(VIDEO)])
win.out_dir_edit.setText(str(OUT_DIR))

result = {"ok": False, "outputs": None, "error": None}

orig_finished = win._on_finished_ok
orig_failed = win._on_failed
orig_cancelled = win._on_cancelled


def wrapped_finished(outputs):
    orig_finished(outputs)
    result["ok"] = True
    result["outputs"] = outputs
    app.quit()


def wrapped_failed(msg):
    orig_failed(msg)
    result["error"] = msg
    app.quit()


def wrapped_cancelled():
    orig_cancelled()
    result["error"] = "unexpected cancel"
    app.quit()


win._on_finished_ok = wrapped_finished
win._on_failed = wrapped_failed
win._on_cancelled = wrapped_cancelled

win._on_start()  # 내부적으로 (재정의된) 콜백들을 worker.start() 전에 connect한다
print("worker running:", win.worker.isRunning() if win.worker else None)
win.worker.progress.connect(lambda m: print("PROGRESS:", m, flush=True))
win.worker.log.connect(lambda m: print("LOG:", m, flush=True))
win.worker.progress_fraction.connect(lambda f: print(f"PCT: {f*100:.1f}%", flush=True))

def _watchdog():
    # 그냥 app.quit()만 하면 워커 스레드(및 그 안의 llama-server.exe 자식 프로세스)가
    # 정리되지 않고 좀비로 남을 수 있다. 취소 요청 후 스레드 종료를 기다린다.
    if win.worker is not None and win.worker.isRunning():
        win.worker.request_cancel()
        win.worker.wait(15000)
    app.quit()


QTimer.singleShot(280_000, _watchdog)  # 안전장치
app.exec()

if result["error"]:
    print("FAIL:", result["error"])
    sys.exit(1)
if not result["ok"]:
    print("FAIL: timeout, finished_ok 시그널을 못 받음")
    sys.exit(1)

print("OK:", result["outputs"])
for video, paths in result["outputs"].items():
    for p in paths:
        assert Path(p).exists(), f"출력 파일 없음: {p}"
print("모든 출력 파일 존재 확인 완료")
