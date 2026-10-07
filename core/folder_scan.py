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


def plan_folder(videos: list[Path], out_dir: Path, config) -> FolderScanPlan:
    out_dir = Path(out_dir)
    target_lang = config.get("target_language", "ko")

    generate: list[Path] = []
    convert_jobs: list[ConvertJob] = []
    encoding_fix_jobs: list[EncodingFixJob] = []

    for v in videos:
        v = Path(v)
        dirs = [out_dir, v.parent]
        subs = find_subtitle_files(v.stem, dirs)
        if not subs:
            generate.append(v)
            continue

        for s in subs:
            ext = s.suffix.lower().lstrip(".")
            if ext == "smi":
                convert_jobs.append(ConvertJob(video=v, smi_path=s))
            elif needs_fix(s):
                encoding_fix_jobs.append(EncodingFixJob(path=s))

    return FolderScanPlan(generate=generate, convert_jobs=convert_jobs, encoding_fix_jobs=encoding_fix_jobs)


def run_conversions(
    plan: FolderScanPlan,
    formats: list[str],
    config,
    delete_old_smi: bool,
    progress_cb=None,
    cancel_check=None,
) -> tuple[int, int, int]:
    """convert_jobs와 encoding_fix_jobs를 실제로 실행한다.

    smi -> srt/ass 변환은 그냥 포맷만 바꾸는 게 아니라, LLM 서버를 띄워 각 문장을 AI로
    검수한다(core.translate.verify_cues) — 인코딩 오류로 깨진 글자를 문맥 보고 복원하되
    번역/의역은 하지 않는다. 그래서 변환 단계가 번역만큼은 아니어도 시간이 좀 걸린다.
    인코딩만 고치는 encoding_fix_jobs는 AI 없이 바로 처리한다.

    반환값: (변환 성공 수, 인코딩 수정 수, 삭제한 smi 수)
    """
    encodings = {
        "srt": config.get("subtitle.srt_encoding", "utf-8-sig"),
        "ass": config.get("subtitle.ass_encoding", "utf-8-sig"),
        "smi": config.get("subtitle.smi_encoding", "utf-8-sig"),
    }
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
        )
        if progress_cb:
            progress_cb("LLM 서버 시작 중 (기존 자막 검수용)")
        server.start()
        try:
            for k, job in enumerate(plan.convert_jobs, 1):
                if cancel_check is not None and cancel_check():
                    raise PipelineCancelled()
                if progress_cb:
                    progress_cb(f"[smi 변환+검수 {k}/{n_convert}] {job.video.name}")
                try:
                    parsed = parse_smi(job.smi_path)
                    cue_objs = [
                        Cue(id=idx, start=c.start, end=c.end, text=c.text) for idx, c in enumerate(parsed)
                    ]
                    corrected, _vreport = verify_cues(
                        cue_objs, server, config, target_lang=target_lang, cancel_check=cancel_check
                    )
                    final_cues = [
                        DisplayCue(start=c.start, end=c.end, text=corrected.get(c.id, c.text)) for c in cue_objs
                    ]
                    lang = extract_lang_suffix(job.smi_path, job.video.stem) or target_lang
                    job.written = write_cues_as(
                        final_cues, job.smi_path.parent, formats, job.video.stem, lang, encodings
                    )
                    converted += 1
                    if delete_old_smi:
                        job.smi_path.unlink(missing_ok=True)
                        deleted += 1
                except PipelineCancelled:
                    raise
                except Exception as e:  # noqa: BLE001
                    job.error = str(e)
        finally:
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
        except Exception as e:  # noqa: BLE001
            job.error = str(e)

    return converted, fixed, deleted
