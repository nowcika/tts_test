# Korean TTS CLI

Local command-line Korean text-to-speech that writes WAV files using
CosyVoice. The CLI supports Korean text input, CPU or CUDA device selection,
automatic device selection, and prompt-audio voice cloning.

## Development Setup

Install the test dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the development test suite:

```bash
pytest -v
```

## CosyVoice Setup

Clone the official CosyVoice repository into `vendor/CosyVoice`:

```bash
mkdir -p vendor
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git vendor/CosyVoice
```

If you already cloned without submodules, run:

```bash
git -C vendor/CosyVoice submodule update --init --recursive
```

Install CosyVoice dependencies from the official repository:

```bash
pip install -r vendor/CosyVoice/requirements.txt
```

Download the default text-only SFT model and the CosyVoice2 cloning model:

```text
pretrained_models/CosyVoice-300M-SFT
pretrained_models/CosyVoice2-0.5B
```

Download both models:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT'); snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"
```

You can also use the official CosyVoice model download instructions for your
environment. Make sure the final local directories exist at:

```text
pretrained_models/CosyVoice-300M-SFT
pretrained_models/CosyVoice2-0.5B
```

If Python cannot import CosyVoice from your environment, run the CLI with the
official repository on `PYTHONPATH`:

```bash
PYTHONPATH=vendor/CosyVoice python tts.py "안녕하세요" --out hello.wav
```

## Usage

Text-only synthesis, using `--device auto` by default:

```bash
PYTHONPATH=vendor/CosyVoice python tts.py "안녕하세요. 반갑습니다." --out hello.wav
```

Force CPU:

```bash
PYTHONPATH=vendor/CosyVoice python tts.py "CPU로 음성을 생성합니다." --device cpu --out cpu.wav
```

Force CUDA:

```bash
PYTHONPATH=vendor/CosyVoice python tts.py "CUDA로 음성을 생성합니다." --device cuda --out cuda.wav
```

Prompt-audio voice cloning:

```bash
PYTHONPATH=vendor/CosyVoice python tts.py "이 목소리로 말합니다." \
  --model-dir pretrained_models/CosyVoice2-0.5B \
  --prompt-audio reference.wav \
  --prompt-text "참고 음성의 실제 문장입니다." \
  --out cloned.wav
```

`--prompt-text` is optional when using `--prompt-audio`, but providing an
accurate transcript usually improves cloning quality.

By default, the CLI loads `pretrained_models/CosyVoice-300M-SFT` for text-only synthesis. Use `--model-dir pretrained_models/CosyVoice2-0.5B` with `--prompt-audio` for voice cloning, or point `--model-dir` at another compatible local model directory.

## Manual Real-Model Verification

After CosyVoice is installed and the model is available locally, run these
commands to verify real synthesis:

```bash
PYTHONPATH=vendor/CosyVoice python tts.py "CPU 검증 문장입니다." \
  --device cpu \
  --out /tmp/korean-tts-cpu.wav
```

```bash
PYTHONPATH=vendor/CosyVoice python tts.py "CUDA 검증 문장입니다." \
  --device cuda \
  --out /tmp/korean-tts-cuda.wav
```

```bash
PYTHONPATH=vendor/CosyVoice python tts.py "참고 음성과 비슷하게 말합니다." \
  --model-dir pretrained_models/CosyVoice2-0.5B \
  --prompt-audio reference.wav \
  --prompt-text "참고 음성의 실제 문장입니다." \
  --out /tmp/korean-tts-cloned.wav
```

