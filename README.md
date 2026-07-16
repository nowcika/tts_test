# 한글 TTS CLI

한글 문장을 입력하면 `.wav` 음성 파일을 생성하는 로컬 CLI 도구입니다. 실제 음성 합성은 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)를 사용합니다.

지원 기능:

- 한글 텍스트를 WAV 파일로 변환
- `--device auto`, `--device cpu`, `--device cuda` 선택
- 기본 한국어 음성 생성
- `--prompt-audio`를 사용한 참고 음성 기반 voice cloning

## Windows 기준 준비 사항

권장 환경:

- Windows 10 또는 Windows 11
- Python 3.10 이상 권장
- Git for Windows
- PowerShell
- 실제 음성 생성용 모델 파일을 받을 충분한 디스크 공간
- CUDA 사용 시 NVIDIA GPU, NVIDIA 드라이버, CUDA 지원 PyTorch

주의 사항:

- CPU만으로도 실행할 수 있지만 음성 생성이 느릴 수 있습니다. 긴 설명문 샘플 `samples/previous_answer_full.wav`는 CPU 환경에서 약 1시간 이상 걸렸습니다.
- 모델 파일은 큽니다. `vendor/`, `pretrained_models/`, 생성된 `.wav` 파일은 git에 올리지 마세요. `.gitignore`에 이미 제외되어 있습니다.
- 테스트용 `fake` 엔진은 CosyVoice와 모델 파일 없이 동작합니다.
- 실제 음성 생성은 CosyVoice 저장소와 모델 다운로드가 필요합니다.
- voice cloning 품질은 `--prompt-audio`와 `--prompt-text`가 정확히 맞을수록 좋아집니다.


## 포함된 샘플 WAV

저장소에는 두 가지 샘플 WAV가 포함되어 있습니다.

- `samples/reference.wav`: voice cloning 명령의 `--prompt-audio` 경로 확인용 짧은 테스트 샘플입니다. 실제 사람 음성이 아니므로 품질 확인용은 아닙니다.
- `samples/tts_korean.wav`: CosyVoice로 실제 한글 텍스트 `반갑습니다.`를 합성한 결과 파일입니다. 22.05kHz mono WAV이며, Whisper base 전사에서 `반갑습니다.`로 확인했습니다.
- `samples/tts_korean_long.wav`: CosyVoice로 실제 한글 텍스트 `반갑습니다. 반갑습니다. 반갑습니다. 반갑습니다.`를 합성한 긴 샘플입니다. 약 5.4초 길이이며, Whisper base에서 한국어 `반갑습니다` 반복 음성으로 확인했습니다.
- `samples/previous_answer_text.txt`: 이전 답변 전체를 음성 합성용으로 정리한 원문입니다.
- `samples/previous_answer_full.wav`: `samples/previous_answer_text.txt`를 CosyVoice로 실제 합성한 긴 설명 샘플입니다. 약 149.5초, 22.05kHz mono WAV입니다. Whisper base 전사에서 한국어 음성으로 인식되는 것은 확인했지만, `CosyVoice`, `AutoModel`, `SFT` 같은 기술 용어와 긴 문장 때문에 원문과 정확히 일치하지 않는 오인식이 있습니다.

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

## 3. 개발 테스트 실행

CosyVoice 없이 기본 테스트를 실행할 수 있습니다.

```powershell
pytest -v
```

모델 없이 CLI 흐름만 확인하려면 `fake` 엔진을 사용합니다.

```powershell
python tts.py "안녕하세요" --engine fake --out test.wav
```

`test.wav`에는 실제 음성이 아니라 테스트용 텍스트가 저장됩니다.

## 4. CosyVoice 설치

실제 음성을 생성하려면 공식 CosyVoice 저장소를 `vendor/CosyVoice`에 받습니다.

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

## 5. PyTorch 설치

CPU만 사용할 경우에는 CPU용 PyTorch를 설치합니다.

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

CUDA를 사용할 경우에는 본인 CUDA 환경에 맞는 PyTorch 설치 명령을 사용해야 합니다. 예를 들어 CUDA 12.1용 wheel은 다음과 같습니다.

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CUDA 버전이 다르면 PyTorch 공식 설치 페이지에서 맞는 명령을 확인하세요.

## 6. 모델 다운로드

기본 텍스트 음성 생성용 모델과 voice cloning용 모델을 모두 받습니다.

```powershell
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT'); snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"
```

다운로드 후 아래 폴더가 있어야 합니다.

```text
pretrained_models/CosyVoice-300M-SFT
pretrained_models/CosyVoice2-0.5B
```

## 7. CosyVoice import 경로 설정

Windows PowerShell에서는 실행 전에 `PYTHONPATH`를 다음처럼 설정합니다.

```powershell
$env:PYTHONPATH = "vendor/CosyVoice"
```

매번 입력하기 싫다면 실행하는 PowerShell 세션에서 한 번만 설정하면 됩니다.

## 8. 사용법

### 기본 한글 TTS

기본 모델은 `pretrained_models/CosyVoice-300M-SFT`입니다.

```powershell
$env:PYTHONPATH = "vendor/CosyVoice"
python tts.py "안녕하세요. 반갑습니다." --out hello.wav
```

### CPU 강제 실행

```powershell
$env:PYTHONPATH = "vendor/CosyVoice"
python tts.py "CPU로 음성을 생성합니다." --device cpu --out cpu.wav
```

### CUDA 강제 실행

```powershell
$env:PYTHONPATH = "vendor/CosyVoice"
python tts.py "CUDA로 음성을 생성합니다." --device cuda --out cuda.wav
```

CUDA가 제대로 잡히지 않으면 `--device cpu`로 먼저 확인하세요.

### 참고 음성 기반 voice cloning

예제 파일 `samples/reference.wav`를 포함했습니다. 이 파일은 경로와 WAV 형식 확인용 짧은 테스트 샘플이며, 실제 voice cloning 품질 확인에는 사람이 말한 참고 음성을 사용하는 것이 좋습니다. 실제 참고 음성을 쓸 때는 `--prompt-text`에 그 음성에서 실제로 말한 문장을 넣습니다.

```powershell
$env:PYTHONPATH = "vendor/CosyVoice"
python tts.py "이 목소리로 말합니다." `
  --model-dir pretrained_models/CosyVoice2-0.5B `
  --prompt-audio samples/reference.wav `
  --prompt-text "테스트 샘플입니다." `
  --out cloned.wav
```

`--prompt-text`는 생략할 수 있지만, 정확히 입력하는 편이 voice cloning 품질에 유리합니다.

## 9. 수동 검증 명령

실제 모델 설치 후 아래 명령으로 확인합니다.

```powershell
$env:PYTHONPATH = "vendor/CosyVoice"
python tts.py "CPU 검증 문장입니다." --device cpu --out cpu-check.wav
```

```powershell
$env:PYTHONPATH = "vendor/CosyVoice"
python tts.py "CUDA 검증 문장입니다." --device cuda --out cuda-check.wav
```

```powershell
$env:PYTHONPATH = "vendor/CosyVoice"
python tts.py "참고 음성과 비슷하게 말합니다." `
  --model-dir pretrained_models/CosyVoice2-0.5B `
  --prompt-audio samples/reference.wav `
  --prompt-text "테스트 샘플입니다." `
  --out cloned-check.wav
```

## 문제 해결

### `No module named 'cosyvoice'`

`PYTHONPATH`가 설정되지 않은 상태입니다.

```powershell
$env:PYTHONPATH = "vendor/CosyVoice"
```

### `Model directory does not exist`

모델 다운로드가 끝났는지 확인하세요.

```powershell
dir pretrained_models
```

### CUDA를 요청했는데 사용할 수 없다는 오류

CUDA 지원 PyTorch가 설치되지 않았거나 NVIDIA 드라이버/CUDA 환경이 맞지 않는 상태입니다. 먼저 CPU로 실행하세요.

```powershell
python tts.py "테스트입니다." --device cpu --out cpu.wav
```

### 생성된 음성이 너무 느림

CPU 실행은 느릴 수 있습니다. NVIDIA GPU가 있다면 CUDA용 PyTorch를 설치하고 `--device cuda`를 사용하세요.
