# 한글 TTS CLI

한글 문장을 입력하면 `.wav` 음성 파일을 생성하는 CLI 도구입니다.

기본 엔진은 `edge`입니다. Microsoft Edge 온라인 TTS를 사용하므로 모델 다운로드 없이 한글 음성을 빠르게 만들 수 있습니다. 오프라인 모델이나 voice cloning이 필요하면 `cosyvoice` 엔진을 선택해서 사용할 수 있습니다.

지원 기능:

- 한글 텍스트를 WAV 파일로 변환
- 기본 고품질 한국어 음성: `--engine edge`
- 로컬 CosyVoice 모델 실행: `--engine cosyvoice`
- `--prompt-audio`를 사용한 CosyVoice 참고 음성 기반 voice cloning
- 테스트용 `fake` 엔진

## Windows 기준 준비 사항

권장 환경:

- Windows 10 또는 Windows 11
- Python 3.10 이상 권장
- Git for Windows
- PowerShell
- ffmpeg
- `edge` 엔진 사용 시 인터넷 연결
- `cosyvoice` 엔진 사용 시 모델 파일을 받을 충분한 디스크 공간
- CUDA 사용 시 NVIDIA GPU, NVIDIA 드라이버, CUDA 지원 PyTorch

주의 사항:

- 기본 `edge` 엔진은 온라인 TTS입니다. 인터넷이 필요하지만 품질과 속도가 CosyVoice 기본 한국어 화자보다 안정적입니다.
- `edge` 엔진은 MP3 스트림을 받은 뒤 ffmpeg로 WAV로 변환합니다. ffmpeg가 설치되어 있어야 합니다.
- `cosyvoice` 엔진은 로컬 모델을 사용하지만, 기본 한국어 SFT 화자의 긴 기술 설명문 품질은 낮을 수 있습니다.
- 모델 파일은 큽니다. `vendor/`, `pretrained_models/`, 새로 생성한 `.wav` 파일은 기본적으로 git에 올리지 마세요.

## 포함된 샘플

- `samples/quality_check_text.txt`: 99% 이상 검증용 기준 문장입니다.
- `samples/quality_check.wav`: `edge` 엔진으로 생성한 검증용 WAV입니다. Whisper base 전사 후 정규화 문자 기준 100.0% 일치를 확인했습니다.
- `samples/quality_check_whisper_base.txt`: `samples/quality_check.wav`를 Whisper base로 전사한 결과입니다.
- `samples/quality_check_score.txt`: 원문과 전사문을 정규화해서 비교한 문자 유사도 결과입니다.
- `samples/previous_answer_text.txt`: 이전 긴 기술 설명 답변을 음성 합성용 한글 발음 기준으로 정리한 원문입니다.
- `samples/previous_answer_full.wav`: `edge` 엔진으로 다시 생성한 긴 기술 설명 WAV입니다. 약 176.0초, 24kHz mono WAV입니다. 기술 용어가 많아 Whisper base 기준 정규화 문자 유사도는 93.1%였으므로 성공 기준 샘플로 보지 않습니다.
- `samples/reference.wav`: CosyVoice voice cloning 명령의 `--prompt-audio` 경로 확인용 짧은 테스트 샘플입니다. 실제 사람 음성이 아니므로 품질 확인용은 아닙니다.
- `samples/tts_korean.wav`: CosyVoice로 실제 한글 텍스트 `반갑습니다.`를 합성한 결과 파일입니다.
- `samples/tts_korean_long.wav`: CosyVoice로 실제 한글 텍스트 `반갑습니다. 반갑습니다. 반갑습니다. 반갑습니다.`를 합성한 샘플입니다.

## 1. 저장소 받기

PowerShell에서 실행합니다.

```powershell
git clone https://github.com/nowcika/tts_test.git
cd tts_test
```

## 2. Python 가상환경 만들기

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

`py -3.10`이 동작하지 않으면 설치된 Python 버전에 맞게 아래처럼 실행하세요.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

PowerShell 실행 정책 때문에 가상환경 활성화가 막히면 현재 사용자 범위에서 다음을 먼저 실행하세요.

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 3. ffmpeg 설치

`edge` 엔진은 WAV 변환에 ffmpeg를 사용합니다.

winget이 있으면 다음 명령으로 설치합니다.

```powershell
winget install Gyan.FFmpeg
```

설치 후 새 PowerShell을 열고 확인합니다.

```powershell
ffmpeg -version
```

## 4. 기본 사용법

기본 엔진은 `edge`입니다.

```powershell
python tts.py "안녕하세요. 반갑습니다." --out hello.wav
```

음성을 바꾸려면 `--voice`를 사용합니다.

```powershell
python tts.py "안녕하세요. 남성 음성 테스트입니다." --voice ko-KR-InJoonNeural --out male.wav
```

속도를 조절하려면 `--rate`를 사용합니다. 음수 값은 `--rate=-5%`처럼 등호를 붙여 입력합니다.

```powershell
python tts.py "조금 천천히 읽습니다." --rate=-5% --out slow.wav
```

사용 가능한 한국어 Edge 음성:

- `ko-KR-SunHiNeural`
- `ko-KR-InJoonNeural`
- `ko-KR-HyunsuMultilingualNeural`

## 5. 품질 검증 샘플 생성

저장소에 포함된 기준 문장을 다시 생성하려면 다음을 실행합니다.

```powershell
python -c "from pathlib import Path; from korean_tts.cli import main; text=Path('samples/quality_check_text.txt').read_text(encoding='utf-8'); raise SystemExit(main([text, '--engine', 'edge', '--out', 'quality_check.wav', '--voice', 'ko-KR-SunHiNeural', '--rate=-5%']))"
```

현재 포함된 `samples/quality_check.wav`는 Whisper base 전사 결과와 원문을 비교했을 때 정규화 문자 기준 100.0% 일치했습니다.

포함된 전사 결과로 점수를 다시 계산하려면 다음을 실행합니다.

```powershell
python tools/score_transcript.py samples/quality_check_text.txt samples/quality_check_whisper_base.txt
```

## 6. 개발 테스트 실행

```powershell
pytest -v
```

모델 없이 CLI 흐름만 확인하려면 `fake` 엔진을 사용합니다.

```powershell
python tts.py "안녕하세요" --engine fake --out test.wav
```

`test.wav`에는 실제 음성이 아니라 테스트용 텍스트가 저장됩니다.

## 7. CosyVoice 선택 설치

CosyVoice는 오프라인 모델 실행이나 참고 음성 기반 voice cloning이 필요할 때만 설치합니다.

```powershell
mkdir vendor
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git vendor/CosyVoice
```

이미 `--recursive` 없이 clone했다면 submodule을 받습니다.

```powershell
git -C vendor/CosyVoice submodule update --init --recursive
```

CosyVoice 의존성을 설치합니다.

```powershell
pip install -r vendor/CosyVoice/requirements.txt
```

CPU만 사용할 경우에는 CPU용 PyTorch를 설치합니다.

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

CUDA를 사용할 경우에는 본인 CUDA 환경에 맞는 PyTorch 설치 명령을 사용해야 합니다.

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

모델을 다운로드합니다.

```powershell
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT'); snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"
```

CosyVoice 실행 전에 `PYTHONPATH`를 설정합니다.

```powershell
$env:PYTHONPATH = "vendor/CosyVoice"
```

CosyVoice로 기본 한글 TTS를 실행합니다.

```powershell
python tts.py "안녕하세요. 반갑습니다." --engine cosyvoice --out cosyvoice.wav
```

참고 음성 기반 voice cloning 예시입니다.

```powershell
python tts.py "이 목소리로 말합니다." `
  --engine cosyvoice `
  --model-dir pretrained_models/CosyVoice2-0.5B `
  --prompt-audio samples/reference.wav `
  --prompt-text "테스트 샘플입니다." `
  --out cloned.wav
```

## 문제 해결

### `ffmpeg is required`

ffmpeg가 설치되지 않았거나 PATH에 없습니다. 새 PowerShell을 열고 확인하세요.

```powershell
ffmpeg -version
```

### `No module named 'edge_tts'`

의존성이 설치되지 않았습니다.

```powershell
pip install -r requirements-dev.txt
```

### `No module named 'cosyvoice'`

CosyVoice 엔진을 사용할 때 `PYTHONPATH`가 설정되지 않은 상태입니다.

```powershell
$env:PYTHONPATH = "vendor/CosyVoice"
```

### `Model directory does not exist`

CosyVoice 모델 다운로드가 끝났는지 확인하세요.

```powershell
dir pretrained_models
```

### CUDA를 요청했는데 사용할 수 없다는 오류

CUDA 지원 PyTorch가 설치되지 않았거나 NVIDIA 드라이버/CUDA 환경이 맞지 않는 상태입니다. 먼저 CPU로 실행하세요.

```powershell
python tts.py "테스트입니다." --engine cosyvoice --device cpu --out cpu.wav
```
