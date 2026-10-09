"""모델/런타임을 exe에 동봉하지 않고 첫 실행 때 내려받는다.

전체 모델(약 11GB)을 exe 배포판에 동봉하면 배포 용량이 너무 커지므로,
config.yaml이 실제로 필요로 하는 것만(기본은 large-v3-turbo + Qwen3-8B) 첫 실행 시
받는다. huggingface_hub 같은 무거운 패키지를 exe에 넣지 않으려고 requests로 직접
다운로드한다.
"""
from __future__ import annotations

import logging
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)

ProgressCB = Optional[Callable[[str, int, int], None]]  # (label, downloaded_bytes, total_bytes)

HF_WHISPER_REPO = {
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "large": "Systran/faster-whisper-large-v3",
    "kotoba": "kotoba-tech/kotoba-whisper-v2.0-faster",
}
WHISPER_FILES = ["config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"]

LLAMA_CPP_VERSION = "b11070"
LLAMA_CPP_BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_VERSION}"
LLAMA_CPP_ASSETS = [
    f"llama-{LLAMA_CPP_VERSION}-bin-win-cuda-12.4-x64.zip",
    f"cudart-llama-bin-win-cuda-12.4-x64.zip",
]

LLM_REPO = "Qwen/Qwen3-8B-GGUF"
LLM_FILE = "Qwen3-8B-Q4_K_M.gguf"

# 번역 전용 모델 (기본 번역 백엔드). 6.6GB, GPU 8GB에 올라가는 4비트 양자화본
TRANSLATE_REPO = "mradermacher/translategemma-12b-it-GGUF"
TRANSLATE_FILE = "translategemma-12b-it.IQ4_XS.gguf"


@dataclass
class AssetTask:
    label: str
    check_path: Path  # 이 파일/폴더가 있으면 이미 받은 것으로 본다
    fetch: Callable[[ProgressCB], None]


def _hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _download_file(url: str, dest: Path, label: str, progress_cb: ProgressCB) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if progress_cb:
                    progress_cb(label, downloaded, total)
    tmp.replace(dest)


def _fetch_whisper_model(repo: str, out_dir: Path, label: str) -> Callable[[ProgressCB], None]:
    def _fetch(progress_cb: ProgressCB) -> None:
        for fname in WHISPER_FILES:
            dest = out_dir / fname
            if dest.exists():
                continue
            _download_file(_hf_url(repo, fname), dest, f"{label}: {fname}", progress_cb)

    return _fetch


def _fetch_llama_cpp(out_dir: Path) -> Callable[[ProgressCB], None]:
    def _fetch(progress_cb: ProgressCB) -> None:
        tmp_dir = out_dir / "_download_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            for asset in LLAMA_CPP_ASSETS:
                zip_path = tmp_dir / asset
                _download_file(f"{LLAMA_CPP_BASE}/{asset}", zip_path, f"llama.cpp 런타임: {asset}", progress_cb)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(out_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return _fetch


def _fetch_llm(out_dir: Path) -> Callable[[ProgressCB], None]:
    def _fetch(progress_cb: ProgressCB) -> None:
        dest = out_dir / LLM_FILE
        _download_file(_hf_url(LLM_REPO, LLM_FILE), dest, f"번역 모델: {LLM_FILE}", progress_cb)

    return _fetch


def _fetch_translate_model(out_dir: Path) -> Callable[[ProgressCB], None]:
    def _fetch(progress_cb: ProgressCB) -> None:
        dest = out_dir / TRANSLATE_FILE
        _download_file(_hf_url(TRANSLATE_REPO, TRANSLATE_FILE), dest, f"번역 모델: {TRANSLATE_FILE}", progress_cb)

    return _fetch


def list_missing(config, include_verify_model: bool = False) -> list[AssetTask]:
    """현재 설정(config)이 실제로 필요로 하는 것 중 아직 없는 항목만 반환한다.

    include_verify_model: smi 변환 때 AI 검수에 쓰는 Qwen 모델까지 확인할지(기본은 번역에 필요한 것만).
    """
    tasks: list[AssetTask] = []

    precision = config.get("stt.precision", "turbo")
    model_dir = config.resolve_path(f"stt.models.{precision}")
    if not (model_dir / "model.bin").exists():
        repo = HF_WHISPER_REPO.get(precision, HF_WHISPER_REPO["turbo"])
        tasks.append(
            AssetTask(
                label=f"음성 인식 모델 ({precision})",
                check_path=model_dir / "model.bin",
                fetch=_fetch_whisper_model(repo, model_dir, f"음성 인식 모델({precision})"),
            )
        )

    server_exe = config.resolve_path("llm.server_exe")
    if not server_exe.exists():
        tasks.append(
            AssetTask(
                label="번역 실행 엔진 (llama.cpp)",
                check_path=server_exe,
                fetch=_fetch_llama_cpp(server_exe.parent),
            )
        )

    use_gemma = config.get("llm.translate_backend", "translategemma") == "translategemma"
    if use_gemma:
        translate_model = config.resolve_path("llm.translate_model")
        if not translate_model.exists():
            tasks.append(
                AssetTask(
                    label="번역 모델 (TranslateGemma 12B, 6.6GB)",
                    check_path=translate_model,
                    fetch=_fetch_translate_model(translate_model.parent),
                )
            )

    # Qwen: json 백엔드의 번역 모델이거나, smi 변환의 AI 검수 모델
    llm_model = config.resolve_path("llm.model")
    if (not use_gemma or include_verify_model) and not llm_model.exists():
        tasks.append(
            AssetTask(
                label="범용 AI 모델 (Qwen3-8B, 4.7GB)",
                check_path=llm_model,
                fetch=_fetch_llm(llm_model.parent),
            )
        )

    return tasks


def download_all(tasks: list[AssetTask], progress_cb: ProgressCB = None, cancel_check=None) -> None:
    from core.errors import PipelineCancelled

    for task in tasks:
        if cancel_check is not None and cancel_check():
            raise PipelineCancelled()
        logger.info("다운로드 중: %s", task.label)
        task.fetch(progress_cb)
