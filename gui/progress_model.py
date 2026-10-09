"""진행 메시지("[단계 k/N] 파일명")를 받아서 화면에 보여줄 개수 요약을 만든다.

GUI와 분리해 둬서 Qt 없이도 테스트할 수 있다.
- smi 변환:   "smi 변환+검수" 단계
- 인코딩 수정: "인코딩 수정" 단계
- 자막생성:   파이프라인 단계 전체. 완료 개수는 "자막 저장" 단계로 센다(영상 하나가 저장까지 끝나야 완료).
"""
from __future__ import annotations

import re

PROGRESS_RE = re.compile(r"^\[(?P<stage>.*?)\s*(?P<k>\d+)/(?P<n>\d+)\]\s*(?P<name>.*)$")

SMI_STAGE = "smi 변환+검수"
ENCODING_STAGE = "인코딩 수정"
PIPELINE_STAGES = ("오디오 추출", "음성 인식", "세그먼트 구성", "번역", "자막 저장")
SAVE_STAGE = "자막 저장"


class StageTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.totals: dict[str, int] = {}
        self.done: dict[str, int] = {}
        self.order: list[str] = []
        self.current: tuple[str, int, int, str] | None = None
        self.finished = False

    def set_expected(self, stage: str, total: int) -> None:
        """메시지가 오기 전에 미리 알고 있는 총 개수를 넣는다 (예: smi 파일 3개)."""
        if stage not in self.order:
            self.order.append(stage)
        self.totals[stage] = total
        self.done.setdefault(stage, 0)

    def update(self, message: str) -> bool:
        m = PROGRESS_RE.match(message)
        if not m:
            return False
        stage = (m["stage"] or "").strip() or "처리"
        k, n = int(m["k"]), int(m["n"])
        if stage not in self.order:
            self.order.append(stage)
        # 이 단계보다 앞선 단계는 이미 끝난 것으로 본다 (다음 단계 메시지가 왔으니)
        for s in self.order:
            if s == stage:
                break
            self.done[s] = self.totals.get(s, self.done.get(s, 0))
        self.totals[stage] = n
        self.done[stage] = k - 1  # k번째 항목 처리 중 = 앞의 k-1개 완료
        self.current = (stage, k, n, m["name"])
        return True

    def finish_all(self) -> None:
        for s in self.order:
            self.done[s] = self.totals.get(s, self.done.get(s, 0))
        self.current = None
        self.finished = True

    def overall_done(self, stages: tuple[str, ...]) -> tuple[int, int]:
        """여러 단계를 합친 (완료 수, 전체 수)."""
        total = sum(self.totals.get(s, 0) for s in stages)
        done = sum(self.done.get(s, 0) for s in stages)
        return done, total

    def lines(self) -> list[str]:
        out: list[str] = []
        smi_stages = [s for s in (SMI_STAGE, "smi 변환") if s in self.order]
        if smi_stages:
            d = sum(self.done.get(s, 0) for s in smi_stages)
            t = sum(self.totals.get(s, 0) for s in smi_stages)
            out.append(f"smi 변환: {t}개 중 {d}개 완료")
        if ENCODING_STAGE in self.order:
            d, t = self.done.get(ENCODING_STAGE, 0), self.totals.get(ENCODING_STAGE, 0)
            out.append(f"인코딩 수정: {t}개 중 {d}개 완료")

        pipe_seen = [s for s in PIPELINE_STAGES if s in self.order]
        if pipe_seen:
            total = self.totals.get("오디오 추출", self.totals.get(SAVE_STAGE, 0))
            done = self.done.get(SAVE_STAGE, 0)
            out.append(f"자막생성: {total}개 중 {done}개 완료")

        if self.finished:
            out.append("모든 작업 끝")
        elif self.current:
            stage, k, n, name = self.current
            out.append(f"현재: {stage} {k}/{n} · {name}")
        return out


def format_duration(sec: float) -> str:
    """경과 시간 표시용: 1:02:03 / 02:03."""
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_remaining(sec: float) -> str:
    """남은 시간 표시용: 약 1시간 5분 / 약 12분 / 1분 미만."""
    minutes = int(round(sec / 60))
    if minutes < 1:
        return "1분 미만"
    h, m = divmod(minutes, 60)
    return f"약 {h}시간 {m}분" if h else f"약 {m}분"


class EtaEstimator:
    """진행률(0~1) 변화로 남은 시간을 추정한다.

    - 캐시를 재사용한 부분은 순식간에 진행률이 뛰므로, 직전 갱신에서 BURST_GAP초 안에 들어온 증가분은
      속도 계산에서 뺀다 (안 빼면 처음에 속도를 크게 잡아 남은 시간이 너무 짧게 나온다).
    - 단계마다 실제 속도가 진행률 가중치와 다르므로 최근 WINDOW초 동안의 속도로 계산한다.
    - 갱신이 없는 동안에도 시간은 흐르므로, 오래 진행이 없으면 속도가 줄어 남은 시간이 늘어난다.
    """

    BURST_GAP = 1.0
    WINDOW = 20 * 60
    MIN_ELAPSED = 30.0

    def __init__(self) -> None:
        self.reset()

    def reset(self, now: float | None = None) -> None:
        self.t_start = now
        self.last_t: float | None = None
        self.last_f = 0.0
        self.steps: list[tuple[float, float]] = []  # (시각, 실제 처리로 늘어난 진행률)

    def update(self, now: float, fraction: float) -> None:
        if self.t_start is None:
            self.t_start = now
        prev_t = self.last_t if self.last_t is not None else self.t_start
        delta = fraction - self.last_f
        if delta > 0 and now - prev_t >= self.BURST_GAP:
            self.steps.append((now, delta))
        self.last_t, self.last_f = now, max(self.last_f, fraction)
        cutoff = now - self.WINDOW
        while self.steps and self.steps[0][0] < cutoff:
            self.steps.pop(0)

    def remaining(self, now: float) -> float | None:
        """남은 초. 아직 추정할 자료가 부족하면 None."""
        if self.t_start is None or now - self.t_start < self.MIN_ELAPSED or self.last_f >= 1.0:
            return None
        window_start = max(self.t_start, now - self.WINDOW)
        progressed = sum(d for t, d in self.steps if t >= window_start)
        span = now - window_start
        if progressed <= 0 or span <= 0:
            return None
        return (1.0 - self.last_f) / (progressed / span)
