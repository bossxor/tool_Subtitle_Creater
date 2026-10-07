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
        if SMI_STAGE in self.order:
            d, t = self.done.get(SMI_STAGE, 0), self.totals.get(SMI_STAGE, 0)
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
