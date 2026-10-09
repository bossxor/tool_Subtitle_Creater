"""폴더를 지정하면 영상마다 자막 상태를 보고 할 일을 자동으로 정한다.

- 자막이 아예 없다 -> AI로 새로 만들어야 할 영상 목록에 넣는다.
- .smi가 있다 -> srt/ass로 변환한다(AI 재처리 없음). 원본 smi는 호출부 설정에 따라 지울 수 있다.
- .srt/.ass가 있는데 인코딩이 이상하다(BOM 없는 UTF-8이나 CP949 등) -> BOM 있는 UTF-8로 다시 쓴다.
- 이미 멀쩡하다 -> 손대지 않는다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from core.converters import extract_lang_suffix, parse_smi, write_cues_as
from core.encoding_fix import fix_encoding_file, needs_fix
from core.errors import PipelineCancelled
from core.llm import LlamaServer
from core.segmenter import Cue
from core.translate import verify_cues
from core.worklog import WorkLog
from core.writers import DisplayCue

SUB_EXTS = ("srt", "ass", "smi")


def find_subtitle_files(stem: str, dirs: list[Path]) -> list[Path]:
    """영상 파일명(확장자 제외)에 대응하는 자막 파일을 주어진 폴더들에서 찾는다.
    'stem.srt', 'stem.ko.srt' 둘 다 매칭하고, 'stem2.srt' 같은 건 안 걸리게 한다."""
    pattern = re.compile(rf"^{re.escape(stem)}(?:\.[^.\\/]+)?\.(?:{'|'.join(SUB_EXTS)})$", re.IGNORECASE)
    seen: set[Path] = set()
    found: list[Path] = []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.is_file() and pattern.match(f.name):
                resolved = f.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(f)
    return found


@dataclass
class ConvertJob:
    video: Path
    smi_path: Path
    written: list[Path] = field(default_factory=list)
    error: str | None = None


@dataclass
class EncodingFixJob:
    path: Path
    fixed: bool = False
    error: str | None = None


@dataclass
class FolderScanPlan:
    generate: list[Path]  # 자막이 아예 없어서 AI로 새로 만들어야 하는 영상
    convert_jobs: list[ConvertJob]  # smi -> srt/ass 변환 대상
    encoding_fix_jobs: list[EncodingFixJob]  # srt/ass 인코딩만 고치면 되는 것


def plan_folder(videos: list[Path], out_dir: Path, config, cancel_check=None, progress_cb=None) -> FolderScanPlan:
    out_dir = Path(out_dir)
    target_lang = config.get("target_language", "ko")

    generate: list[Path] = []
    convert_jobs: list[ConvertJob] = []
    encoding_fix_jobs: list[EncodingFixJob] = []

    # Index each directory once rather than reading it again for every video.
    indexes = {}
    seen_convert = set()
    seen_fix = set()

    def check_cancel():
        if cancel_check is not None and cancel_check():
            raise PipelineCancelled()

    def indexed_subtitles(directory):
        directory = Path(directory)
        if directory not in indexes:
            index = {}
            if directory.is_dir():
                for path in directory.iterdir():
                    check_cancel()
                    if path.suffix.lower().lstrip(".") not in SUB_EXTS or not path.is_file():
                        continue
                    name = path.stem.casefold()
                    for key in {name, name.rsplit(".", 1)[0]}:
                        index.setdefault(key, []).append(path)
            indexes[directory] = index
        return indexes[directory]

    for i, v in enumerate(videos, 1):
        check_cancel()
        if progress_cb:
            progress_cb(f"자막 상태 확인 {i}/{len(videos)} · {Path(v).name}")
        v = Path(v)
        dirs = [out_dir, v.parent]
        subs = {}
        for directory in dirs:
            for path in indexed_subtitles(directory).get(v.stem.casefold(), []):
                subs.setdefault(path.resolve(), path)
        if not subs:
            generate.append(v)
            continue

        for resolved, s in subs.items():
            ext = s.suffix.lower().lstrip(".")
            if ext == "smi" and resolved not in seen_convert:
                convert_jobs.append(ConvertJob(video=v, smi_path=s))
                seen_convert.add(resolved)
            elif ext != "smi" and resolved not in seen_fix:
                seen_fix.add(resolved)
                if needs_fix(s):
                    encoding_fix_jobs.append(EncodingFixJob(path=s))

    return FolderScanPlan(generate=generate, convert_jobs=convert_jobs, encoding_fix_jobs=encoding_fix_jobs)


def run_conversions(
    plan: FolderScanPlan,
    formats: list[str],
    config,
    delete_old_smi: bool,
    progress_cb=None,
    cancel_check=None,
    worklog: WorkLog | None = None,
    verify_with_ai: bool = True,
) -> tuple[int, int, int]:
    """convert_jobs와 encoding_fix_jobs를 실제로 실행한다.

    verify_with_ai=True이면 LLM 서버를 띄워 각 문장을 AI로
    검수한다(core.translate.verify_cues) — 인코딩 오류로 깨진 글자를 문맥 보고 복원하되
    번역/의역은 하지 않는다. False이면 모델을 실행하지 않고 형식만 빠르게 변환한다.
    인코딩만 고치는 encoding_fix_jobs는 AI 없이 바로 처리한다.

    반환값: (변환 성공 수, 인코딩 수정 수, 삭제한 smi 수)
    """
    encodings = {
        "srt": config.get("subtitle.srt_encoding", "utf-8-sig"),
        "ass": config.get("subtitle.ass_encoding", "utf-8-sig"),
        "smi": config.get("subtitle.smi_encoding", "utf-8-sig"),
    }
    conversion_formats = [fmt for fmt in formats if fmt in ("srt", "ass")]
    if plan.convert_jobs and not conversion_formats:
        raise ValueError("SMI 변환에는 SRT 또는 ASS 출력 형식이 필요합니다.")
    target_lang = config.get("target_language", "ko")
    converted = 0
    deleted = 0

    # 진행 메시지는 단계별로 k/N 을 보낸다 (smi 변환은 smi 개수 기준, 인코딩 수정은 그 기준)
    n_convert = len(plan.convert_jobs)
    n_fix = len(plan.encoding_fix_jobs)

    if plan.convert_jobs:
        if cancel_check is not None and cancel_check():
            raise PipelineCancelled()
        log_dir = config.resolve_path("paths.cache_dir", "cache")
        log_dir.mkdir(parents=True, exist_ok=True)
        server = LlamaServer(
            server_exe=config.resolve_path("llm.server_exe"),
            model_path=config.resolve_path("llm.model"),
            port=config.get("llm.port", 8090),
            n_gpu_layers=config.get("llm.n_gpu_layers", 999),
            ctx_size=config.get("llm.ctx_size", 8192),
            log_path=log_dir / "llama-server.log",
        ) if verify_with_ai else None
        if progress_cb and server is not None:
            progress_cb("LLM 서버 시작 중 (기존 자막 검수용)")
        if server is not None:
            server.start()
        try:
            for k, job in enumerate(plan.convert_jobs, 1):
                if cancel_check is not None and cancel_check():
                    raise PipelineCancelled()
                if progress_cb:
                    stage = "smi 변환+검수" if verify_with_ai else "smi 변환"
                    progress_cb(f"[{stage} {k}/{n_convert}] {job.video.name}")
                try:
                    parsed = parse_smi(job.smi_path)
                    if not parsed:
                        # 파서가 못 읽은 형식일 수 있다. 빈 자막을 만들고 원본까지 지우면 안 된다.
                        raise ValueError("SMI에서 자막을 하나도 못 읽음 (원본은 그대로 둠)")
                    cue_objs = [
                        Cue(id=idx, start=c.start, end=c.end, text=c.text) for idx, c in enumerate(parsed)
                    ]
                    corrected = {}
                    _vreport = None
                    if server is not None:
                        corrected, _vreport = verify_cues(
                            cue_objs, server, config, target_lang=target_lang, cancel_check=cancel_check
                        )
                    final_cues = [
                        DisplayCue(start=c.start, end=c.end, text=corrected.get(c.id, c.text)) for c in cue_objs
                    ]
                    lang = extract_lang_suffix(job.smi_path, job.video.stem) or target_lang
                    job.written = write_cues_as(
                        final_cues, job.smi_path.parent, conversion_formats, job.video.stem, lang, encodings
                    )
                    converted += 1
                    if worklog:
                        worklog.event(
                            "smi 변환", job.smi_path.name, "완료",
                            f"{', '.join(w.name for w in job.written)} 생성, {len(final_cues)}줄, "
                            f"AI 검수 후 원문 유지 {len(_vreport.failed_ids)}줄" if _vreport else
                            f"{', '.join(w.name for w in job.written)} 생성, {len(final_cues)}줄, 형식만 변환(AI 검수 없음)",
                        )
                    written_ok = bool(job.written) and all(w.exists() and w.stat().st_size > 0 for w in job.written)
                    if delete_old_smi and not written_ok:
                        if worklog:
                            worklog.event("원본 smi 삭제", job.smi_path.name, "보존", "결과 파일을 확인하지 못해 원본을 지우지 않음")
                    elif delete_old_smi:
                        job.smi_path.unlink(missing_ok=True)
                        deleted += 1
                        if worklog:
                            worklog.event("원본 smi 삭제", job.smi_path.name, "삭제함", "변환 성공 후 삭제 옵션에 따라")
                    elif worklog:
                        worklog.event("원본 smi 삭제", job.smi_path.name, "보존", "삭제 옵션을 켜지 않아 원본을 그대로 둠")
                except PipelineCancelled:
                    raise
                except Exception as e:  # noqa: BLE001
                    job.error = str(e)
                    if worklog:
                        worklog.event("smi 변환", job.smi_path.name, "실패", job.error.splitlines()[0][:200] if job.error else "")
        finally:
            if server is not None:
                server.stop()

    fixed = 0
    for k, job in enumerate(plan.encoding_fix_jobs, 1):
        if cancel_check is not None and cancel_check():
            raise PipelineCancelled()
        if progress_cb:
            progress_cb(f"[인코딩 수정 {k}/{n_fix}] {job.path.name}")
        try:
            job.fixed = fix_encoding_file(job.path)
            if job.fixed:
                fixed += 1
            if worklog:
                worklog.event(
                    "인코딩 수정", job.path.name, "수정함" if job.fixed else "변경 없음",
                    "BOM 있는 UTF-8로 다시 저장" if job.fixed else "이미 정상이거나 인코딩을 확신할 수 없어 그대로 둠",
                )
        except Exception as e:  # noqa: BLE001
            job.error = str(e)
            if worklog:
                worklog.event("인코딩 수정", job.path.name, "실패", job.error.splitlines()[0][:200] if job.error else "")

    return converted, fixed, deleted
