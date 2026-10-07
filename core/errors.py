"""여러 모듈에서 공유하는 예외. pipeline.py와 translate.py가 순환 임포트 없이 같이 쓸 수 있게 분리."""


class PipelineCancelled(Exception):
    """사용자가 취소를 요청해 처리를 중단했을 때 발생한다."""
