# Subtitle Creater (Subtitle_Tool)

영상 여러 개를 고르고 자막 저장 폴더를 지정하면, **음성 인식 → 번역 → 다듬기**까지 자동으로 처리해 자막 파일(`.srt`)을 만들어 주는 Windows 데스크톱 도구입니다.
모든 AI가 **내 PC에서 로컬로** 돌아가므로 API 비용이나 토큰 사용이 없습니다. 주 용도는 **일본어 영화·드라마 → 한국어 자막**입니다.

> 내 작업물 대시보드: **https://bossxor.github.io/works-dashboard/**

## 주요 기능

- **영상 다중 선택 / 폴더째 추가**: 폴더(하위 폴더 포함)를 지정하면 영상마다 자막 상태를 보고 할 일을 알아서 정합니다.
  - 자막이 아예 없으면 → AI로 새로 생성
  - 예전 `.smi` 자막이 있으면 → `.srt`로 변환하고 AI로 깨진 글자 검수 (원본 smi 삭제 여부는 체크박스로 선택, 기본은 보존)
  - `.srt`/`.ass`인데 인코딩이 이상하면(BOM 없는 UTF-8, CP949 등) → BOM 있는 UTF-8로 고침
  - 이미 정상이면 그대로 둠
- **일본어 → 한국어 번역 + 다듬기**: 앞뒤 문맥과 용어집(인명·고유명사 고정)을 반영합니다.
- **자막 형식 선택**: 기본은 `srt` 하나. `ass`, `smi`도 체크박스로 추가할 수 있습니다. 모든 형식을 BOM 포함 UTF-8로 저장해 곰플레이어 등에서 한글이 깨지지 않게 했습니다.
- **진행 상황 표시**: `smi 변환 N개 중 k개`, `자막생성 N개 중 k개`처럼 단계별 개수와 전체 진행률(%)을 보여주고, 끝나면 결과 표(파일별 완료/실패)와 소요 시간을 보여줍니다.
- **취소 / 이어하기**: 단계별 결과를 캐시에 저장하므로 중간에 멈춰도 완료된 부분은 다시 하지 않습니다.
- **덮어쓰기 확인**: 같은 이름의 자막이 이미 있으면 덮어쓸지 물어봅니다.

## 동작 방식

```
영상 ─ ffmpeg(오디오 추출) ─ faster-whisper large-v3-turbo(GPU) ─ 문장 단위로 재분할
     ─ TranslateGemma 12B (llama.cpp CUDA)로 줄 단위 번역 ─ 검증/재시도 ─ srt(+ass/smi) 저장
```

- **음성 인식**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3-turbo`, 배치 추론 + VAD
- **번역**: 번역 전용 모델 [TranslateGemma 12B](https://huggingface.co/mradermacher/translategemma-12b-it-GGUF) (IQ4_XS, 6.6GB)를 [llama.cpp](https://github.com/ggml-org/llama.cpp) 서버로 실행. 일본어 문장 끝으로 반말/존댓말을 판별해 반말일 때 지시를 줍니다. 범용 모델(Qwen3-8B)은 smi 변환의 AI 검수에 씁니다.
- **타임스탬프는 코드가 관리**하고 LLM에는 텍스트만 전달해서 싱크가 어긋나지 않게 했습니다.
- 번역 결과에 한자·가나가 한글로 안 바뀌고 남거나 원문에 없는 영어가 섞이면 해당 줄만 다시 번역하고, 끝내 안 되면 한글 음역 또는 원문 유지로 처리하며 작업 로그에 남깁니다.

자세한 설계 근거와 실측 수치는 [`DESIGN.md`](DESIGN.md)에 있습니다.

## 실측 성능 (RTX 3070 Ti 8GB 기준)

| 단계 | 결과 |
|---|---|
| 음성 인식 (large-v3-turbo) | 약 91배속 (10분 오디오 → 약 7초) |
| 번역 (TranslateGemma 12B IQ4_XS, 4개 병렬) | 40줄에 약 20~24초 (1시간 영상 약 7분 예상) |

> 음성 인식 수치는 CC0 일본어 **낭독** 샘플로 잰 값입니다. 배경음악·겹치는 대사가 있는 실제 드라마에서는 달라질 수 있습니다.
>
> **번역 정확도**: 일상 대사 40줄(제가 기준 번역을 쓰고 채점한 소규모 평가)에서 뜻이 틀린 줄이 Qwen3-8B는 약 10줄, TranslateGemma 12B는 1~4줄이었습니다. 음성 인식 오류는 이 평가에 포함되지 않았습니다. 자세한 내용은 [`DESIGN.md`](DESIGN.md)를 참고하세요.

## 요구 사항

- Windows 10/11 (현재 Windows 전용)
- NVIDIA GPU (VRAM 8GB 이상 권장) + 최신 드라이버
- **ffmpeg**가 PATH에 있어야 합니다 (`winget install Gyan.FFmpeg`)
- Python 3.11 (소스로 실행/빌드할 때)
- 첫 실행 때 필요한 모델·런타임을 자동으로 내려받습니다 (기본 설정 기준 약 8~9GB: 음성 인식 모델 + 번역 모델 6.6GB + llama.cpp). smi를 AI로 검수하며 변환할 때만 Qwen3-8B(4.7GB)를 추가로 받습니다.
- 번역 모델은 Gemma 이용 약관(license: gemma)의 적용을 받습니다.

## 실행 방법

### 소스로 실행

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

처음 "자막 생성 시작"을 누르면 필요한 파일이 없을 때 내려받을지 물어봅니다.

### CLI로 실행

```bash
.venv\Scripts\python.exe -m core.cli --videos a.mp4 b.mp4 --out-dir D:\subs
```

### exe로 빌드

```bash
.venv\Scripts\python.exe build_exe.py
```

`dist/Subtitle_Tool/Subtitle_Tool.exe`가 만들어집니다 (PyInstaller onedir). 모델(`models/`)과 llama.cpp(`tools/llama.cpp/`)는 용량 때문에 exe에 포함하지 않고, exe 옆 폴더에서 찾거나 첫 실행 때 내려받습니다.

## 설정

`config.yaml`에서 STT 정밀도, 번역 배치 크기, 자막 줄 길이/표시 시간, 출력 형식과 인코딩 등을 바꿀 수 있습니다. 대부분은 GUI 옵션으로도 조절됩니다.

## 프로젝트 구조

```
core/   음성 인식·번역·검증·자막 쓰기·폴더 정리 등 핵심 로직 (GUI와 분리, CLI로도 실행 가능)
gui/    PySide6 GUI (메인 창, 백그라운드 워커, 진행 개수 계산)
bench/  모델 벤치마크 및 GUI 스모크 테스트 스크립트
main.py        GUI 진입점
build_exe.py   PyInstaller 빌드 스크립트
config.yaml    기본 설정
DESIGN.md      설계서 (실측 결과, 트러블슈팅 기록 포함)
```

## 알려진 한계

- 이 도구를 만든 PC(개발 환경)에서만 검증했습니다. 모델이 하나도 없는 깨끗한 PC에서 첫 실행 다운로드 전체 흐름은 아직 검증하지 못했습니다.
- 실제 드라마(배경음악, 다중 화자, 겹치는 대사)로는 아직 충분히 검증하지 못했습니다.
- 고유명사·고어체 등은 번역 검증을 통과하지 못해 원문이 그대로 남는 문장이 약 3% 정도 있습니다 (파이프라인은 멈추지 않고 로그로 알려 줍니다).
- smi → srt 변환 시 AI 검수는 깨진 글자를 복원하지만, 정말로 복구 불가능한 손상은 그럴듯한 다른 내용을 만들어 낼 수 있습니다. (일반적인 인코딩 문제는 자동 인코딩 감지가 먼저 결정론적으로 고칩니다.)
- 문장이 너무 길면 시간을 나누지 않고 2줄 줄바꿈만 합니다.
