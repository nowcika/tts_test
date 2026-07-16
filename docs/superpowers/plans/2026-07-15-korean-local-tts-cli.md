# Korean Local TTS CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Python CLI that converts Korean text to `.wav` using CosyVoice, with CPU/CUDA device selection and optional prompt-audio voice cloning.

**Architecture:** The CLI stays small and delegates behavior to focused modules. `tts.py` handles command-line entry, `korean_tts` validates inputs and routes to an engine, and `korean_tts/engines/cosyvoice.py` isolates all CosyVoice-specific imports and API calls. Tests use fake engines and monkeypatching so normal verification does not require model downloads.

**Tech Stack:** Python 3.10+, argparse, pytest, torch when installed, soundfile/torchaudio through CosyVoice dependencies, CosyVoice local repository/model files for real synthesis.

---

## File Structure

- Create `tts.py`: console entry point that calls `korean_tts.cli.main`.
- Create `korean_tts/__init__.py`: package marker and version.
- Create `korean_tts/errors.py`: user-facing exception type.
- Create `korean_tts/device.py`: resolve `auto`, `cpu`, and `cuda`.
- Create `korean_tts/audio.py`: validate output paths and write generated audio.
- Create `korean_tts/engines/base.py`: engine protocol and synthesis request dataclass.
- Create `korean_tts/engines/fake.py`: test-only/simple fake engine used by CLI dispatch tests.
- Create `korean_tts/engines/cosyvoice.py`: CosyVoice adapter.
- Create `korean_tts/cli.py`: argument parsing, validation, engine dispatch, and error formatting.
- Create `tests/test_device.py`: device selection tests.
- Create `tests/test_cli.py`: CLI parsing and validation tests.
- Create `tests/test_audio.py`: output path/audio writing tests.
- Create `tests/test_cosyvoice_adapter.py`: CosyVoice adapter tests using monkeypatched fake dependencies.
- Create `requirements-dev.txt`: pytest dependency for local tests.
- Create `README.md`: setup and usage instructions.

---

### Task 1: Device Selection

**Files:**
- Create: `korean_tts/__init__.py`
- Create: `korean_tts/errors.py`
- Create: `korean_tts/device.py`
- Test: `tests/test_device.py`

- [ ] **Step 1: Write failing device tests**

Create `tests/test_device.py`:

```python
import pytest

from korean_tts.device import resolve_device
from korean_tts.errors import UserFacingError


def test_resolve_cpu():
    assert resolve_device("cpu", cuda_available=lambda: True) == "cpu"


def test_resolve_cuda_when_available():
    assert resolve_device("cuda", cuda_available=lambda: True) == "cuda"


def test_resolve_cuda_when_unavailable_raises_clear_error():
    with pytest.raises(UserFacingError, match="CUDA was requested but is not available"):
        resolve_device("cuda", cuda_available=lambda: False)


def test_resolve_auto_prefers_cuda():
    assert resolve_device("auto", cuda_available=lambda: True) == "cuda"


def test_resolve_auto_falls_back_to_cpu():
    assert resolve_device("auto", cuda_available=lambda: False) == "cpu"


def test_resolve_rejects_unknown_device():
    with pytest.raises(UserFacingError, match="Unsupported device"):
        resolve_device("mps", cuda_available=lambda: False)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_device.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'korean_tts'`.

- [ ] **Step 3: Implement device module**

Create `korean_tts/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `korean_tts/errors.py`:

```python
class UserFacingError(Exception):
    """An error that should be printed without a Python traceback."""
```

Create `korean_tts/device.py`:

```python
from __future__ import annotations

from collections.abc import Callable

from .errors import UserFacingError


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_device(
    requested: str,
    cuda_available: Callable[[], bool] | None = None,
) -> str:
    check_cuda = cuda_available or _torch_cuda_available
    normalized = requested.strip().lower()

    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        if not check_cuda():
            raise UserFacingError(
                "CUDA was requested but is not available. Use --device cpu or install a CUDA-enabled PyTorch build."
            )
        return "cuda"
    if normalized == "auto":
        return "cuda" if check_cuda() else "cpu"

    raise UserFacingError("Unsupported device. Use one of: auto, cpu, cuda.")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_device.py -v
```

Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add korean_tts/__init__.py korean_tts/errors.py korean_tts/device.py tests/test_device.py
git commit -m "Add device selection"
```

---

### Task 2: Engine Interface And Audio Output

**Files:**
- Create: `korean_tts/engines/base.py`
- Create: `korean_tts/engines/__init__.py`
- Create: `korean_tts/audio.py`
- Test: `tests/test_audio.py`

- [ ] **Step 1: Write failing audio tests**

Create `tests/test_audio.py`:

```python
from pathlib import Path

import pytest

from korean_tts.audio import ensure_output_path, write_wav
from korean_tts.engines.base import AudioResult
from korean_tts.errors import UserFacingError


def test_ensure_output_path_requires_wav_suffix(tmp_path):
    with pytest.raises(UserFacingError, match="Output path must end with .wav"):
        ensure_output_path(tmp_path / "voice.mp3")


def test_ensure_output_path_creates_parent_directory(tmp_path):
    output = tmp_path / "nested" / "voice.wav"
    assert ensure_output_path(output) == output
    assert output.parent.is_dir()


def test_write_wav_delegates_to_soundfile(monkeypatch, tmp_path):
    calls = []

    def fake_write(path, waveform, sample_rate):
        calls.append((path, waveform, sample_rate))

    monkeypatch.setattr("korean_tts.audio.sf.write", fake_write)
    output = tmp_path / "voice.wav"
    result = AudioResult(waveform=[0.0, 0.1, -0.1], sample_rate=24000)

    write_wav(output, result)

    assert calls == [(str(output), [0.0, 0.1, -0.1], 24000)]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_audio.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `korean_tts.audio` or `korean_tts.engines`.

- [ ] **Step 3: Implement engine interface and audio helpers**

Create `korean_tts/engines/__init__.py`:

```python
from .base import AudioResult, SynthesisRequest, TTSEngine

__all__ = ["AudioResult", "SynthesisRequest", "TTSEngine"]
```

Create `korean_tts/engines/base.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class SynthesisRequest:
    text: str
    device: str
    model_dir: Path
    prompt_audio: Path | None = None
    prompt_text: str = ""


@dataclass(frozen=True)
class AudioResult:
    waveform: object
    sample_rate: int


class TTSEngine(Protocol):
    def synthesize(self, request: SynthesisRequest) -> AudioResult:
        raise NotImplementedError
```

Create `korean_tts/audio.py`:

```python
from __future__ import annotations

from pathlib import Path

import soundfile as sf

from .engines.base import AudioResult
from .errors import UserFacingError


def ensure_output_path(path: Path) -> Path:
    if path.suffix.lower() != ".wav":
        raise UserFacingError("Output path must end with .wav.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_wav(path: Path, result: AudioResult) -> None:
    output_path = ensure_output_path(path)
    sf.write(str(output_path), result.waveform, result.sample_rate)
```

- [ ] **Step 4: Add test dependency**

Create `requirements-dev.txt`:

```text
pytest>=8.0
soundfile>=0.12
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
pytest tests/test_audio.py -v
```

Expected: PASS, 3 tests.

- [ ] **Step 6: Commit**

```bash
git add korean_tts/engines/__init__.py korean_tts/engines/base.py korean_tts/audio.py tests/test_audio.py requirements-dev.txt
git commit -m "Add audio output helpers"
```

---

### Task 3: CLI Parsing And Validation

**Files:**
- Create: `tts.py`
- Create: `korean_tts/cli.py`
- Create: `korean_tts/engines/fake.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_cli.py`:

```python
from pathlib import Path

import pytest

from korean_tts.cli import main


def test_cli_rejects_empty_text(capsys):
    exit_code = main(["   ", "--engine", "fake"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Text input cannot be empty" in captured.err


def test_cli_rejects_missing_prompt_audio(tmp_path, capsys):
    missing = tmp_path / "missing.wav"

    exit_code = main(["안녕하세요", "--engine", "fake", "--prompt-audio", str(missing)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Prompt audio does not exist" in captured.err


def test_cli_rejects_missing_model_dir(tmp_path, capsys):
    missing_model = tmp_path / "model"

    exit_code = main(["안녕하세요", "--engine", "fake", "--model-dir", str(missing_model)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Model directory does not exist" in captured.err


def test_cli_generates_wav_with_fake_engine(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output = tmp_path / "hello.wav"

    exit_code = main(
        [
            "안녕하세요",
            "--engine",
            "fake",
            "--model-dir",
            str(model_dir),
            "--out",
            str(output),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert output.read_text(encoding="utf-8") == "fake wav for: 안녕하세요\n"


def test_cli_passes_prompt_audio_and_text_to_fake_engine(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    prompt_audio = tmp_path / "sample.wav"
    prompt_audio.write_bytes(b"sample")
    output = tmp_path / "cloned.wav"

    exit_code = main(
        [
            "따라 말합니다",
            "--engine",
            "fake",
            "--model-dir",
            str(model_dir),
            "--prompt-audio",
            str(prompt_audio),
            "--prompt-text",
            "샘플입니다",
            "--out",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.read_text(encoding="utf-8") == "fake wav for: 따라 말합니다 prompt=샘플입니다\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'korean_tts.cli'`.

- [ ] **Step 3: Implement CLI and fake engine**

Create `korean_tts/engines/fake.py`:

```python
from __future__ import annotations

from pathlib import Path

from .base import AudioResult, SynthesisRequest


class FakeEngine:
    def synthesize(self, request: SynthesisRequest) -> AudioResult:
        prompt = f" prompt={request.prompt_text}" if request.prompt_audio else ""
        return AudioResult(waveform=f"fake wav for: {request.text}{prompt}\n", sample_rate=0)


def write_fake_output(path: Path, result: AudioResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(result.waveform), encoding="utf-8")
```

Create `korean_tts/cli.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .audio import write_wav
from .device import resolve_device
from .engines.base import SynthesisRequest, TTSEngine
from .engines.fake import FakeEngine, write_fake_output
from .errors import UserFacingError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Korean text to a local WAV file.")
    parser.add_argument("text", help="Korean text to synthesize.")
    parser.add_argument("--out", default="output.wav", help="Output WAV path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--prompt-audio", help="Reference WAV file for voice cloning.")
    parser.add_argument("--prompt-text", default="", help="Transcript for --prompt-audio.")
    parser.add_argument("--model-dir", default="pretrained_models/CosyVoice-300M-SFT")
    parser.add_argument("--engine", default="cosyvoice", choices=["cosyvoice", "fake"])
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.text.strip():
        raise UserFacingError("Text input cannot be empty.")

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise UserFacingError(f"Model directory does not exist: {model_dir}")

    if args.prompt_audio:
        prompt_audio = Path(args.prompt_audio)
        if not prompt_audio.is_file():
            raise UserFacingError(f"Prompt audio does not exist: {prompt_audio}")


def _create_engine(name: str) -> TTSEngine:
    if name == "fake":
        return FakeEngine()
    if name == "cosyvoice":
        from .engines.cosyvoice import CosyVoiceEngine

        return CosyVoiceEngine()
    raise UserFacingError("Unsupported engine. Use one of: cosyvoice, fake.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
        device = resolve_device(args.device)
        request = SynthesisRequest(
            text=args.text.strip(),
            device=device,
            model_dir=Path(args.model_dir),
            prompt_audio=Path(args.prompt_audio) if args.prompt_audio else None,
            prompt_text=args.prompt_text,
        )
        engine = _create_engine(args.engine)
        result = engine.synthesize(request)

        output = Path(args.out)
        if args.engine == "fake":
            write_fake_output(output, result)
        else:
            write_wav(output, result)
    except UserFacingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0
```

Create `tts.py`:

```python
from korean_tts.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add tts.py korean_tts/cli.py korean_tts/engines/fake.py tests/test_cli.py
git commit -m "Add Korean TTS CLI"
```

---

### Task 4: CosyVoice Adapter

**Files:**
- Create: `korean_tts/engines/cosyvoice.py`
- Test: `tests/test_cosyvoice_adapter.py`

- [ ] **Step 1: Write failing CosyVoice adapter tests**

Create `tests/test_cosyvoice_adapter.py`:

```python
from pathlib import Path

import pytest

from korean_tts.engines.base import SynthesisRequest
from korean_tts.engines.cosyvoice import CosyVoiceEngine
from korean_tts.errors import UserFacingError


class FakeCosyVoiceModel:
    def __init__(self):
        self.calls = []

    def inference_sft(self, text, speaker):
        self.calls.append(("sft", text, speaker))
        yield {"tts_speech": [0.1, -0.1]}

    def inference_zero_shot(self, text, prompt_text, prompt_speech_16k):
        self.calls.append(("zero_shot", text, prompt_text, prompt_speech_16k))
        yield {"tts_speech": [0.2, -0.2]}


def test_text_only_uses_sft_path(monkeypatch, tmp_path):
    fake_model = FakeCosyVoiceModel()
    monkeypatch.setattr(
        "korean_tts.engines.cosyvoice._load_cosyvoice_model",
        lambda model_dir, device: fake_model,
    )

    result = CosyVoiceEngine().synthesize(
        SynthesisRequest(text="안녕하세요", device="cpu", model_dir=tmp_path)
    )

    assert result.waveform == [0.1, -0.1]
    assert result.sample_rate == 22050
    assert fake_model.calls == [("sft", "안녕하세요", "中文女")]


def test_prompt_audio_uses_zero_shot_path(monkeypatch, tmp_path):
    fake_model = FakeCosyVoiceModel()
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"wav")
    monkeypatch.setattr(
        "korean_tts.engines.cosyvoice._load_cosyvoice_model",
        lambda model_dir, device: fake_model,
    )
    monkeypatch.setattr(
        "korean_tts.engines.cosyvoice._load_prompt_audio_16k",
        lambda path: "prompt-waveform",
    )

    result = CosyVoiceEngine().synthesize(
        SynthesisRequest(
            text="따라 말합니다",
            device="cuda",
            model_dir=tmp_path,
            prompt_audio=prompt_audio,
            prompt_text="샘플입니다",
        )
    )

    assert result.waveform == [0.2, -0.2]
    assert result.sample_rate == 22050
    assert fake_model.calls == [("zero_shot", "따라 말합니다", "샘플입니다", "prompt-waveform")]


def test_dependency_import_error_becomes_user_facing_error(monkeypatch, tmp_path):
    def fail_import(model_dir: Path, device: str):
        raise ImportError("No module named 'cosyvoice'")

    monkeypatch.setattr("korean_tts.engines.cosyvoice._load_cosyvoice_model", fail_import)

    with pytest.raises(UserFacingError, match="CosyVoice dependencies are not installed"):
        CosyVoiceEngine().synthesize(SynthesisRequest(text="안녕하세요", device="cpu", model_dir=tmp_path))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_cosyvoice_adapter.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'korean_tts.engines.cosyvoice'`.

- [ ] **Step 3: Implement CosyVoice adapter**

Create `korean_tts/engines/cosyvoice.py`:

```python
from __future__ import annotations

from pathlib import Path

from .base import AudioResult, SynthesisRequest
from ..errors import UserFacingError

DEFAULT_SPEAKER = "中文女"
DEFAULT_SAMPLE_RATE = 22050


def _load_cosyvoice_model(model_dir: Path, device: str):
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2
    except ImportError as exc:
        raise ImportError("CosyVoice is not importable") from exc

    # CosyVoice uses PyTorch device state internally. Keep this adapter thin and
    # pass the local model path through the official constructor.
    return CosyVoice2(str(model_dir))


def _load_prompt_audio_16k(path: Path):
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError("torchaudio is not importable") from exc

    waveform, sample_rate = torchaudio.load(str(path))
    if sample_rate == 16000:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    return resampler(waveform)


def _first_audio_chunk(chunks):
    for chunk in chunks:
        if "tts_speech" not in chunk:
            continue
        return chunk["tts_speech"]
    raise UserFacingError("CosyVoice did not return any audio.")


class CosyVoiceEngine:
    def synthesize(self, request: SynthesisRequest) -> AudioResult:
        try:
            model = _load_cosyvoice_model(request.model_dir, request.device)
            if request.prompt_audio:
                prompt_speech = _load_prompt_audio_16k(request.prompt_audio)
                waveform = _first_audio_chunk(
                    model.inference_zero_shot(request.text, request.prompt_text, prompt_speech)
                )
            else:
                waveform = _first_audio_chunk(model.inference_sft(request.text, DEFAULT_SPEAKER))
        except ImportError as exc:
            raise UserFacingError(
                "CosyVoice dependencies are not installed. Install the official CosyVoice repository dependencies first."
            ) from exc

        return AudioResult(waveform=waveform, sample_rate=DEFAULT_SAMPLE_RATE)
```

- [ ] **Step 4: Run adapter tests**

Run:

```bash
pytest tests/test_cosyvoice_adapter.py -v
```

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add korean_tts/engines/cosyvoice.py tests/test_cosyvoice_adapter.py
git commit -m "Add CosyVoice engine adapter"
```

---

### Task 5: Documentation And Full Verification

**Files:**
- Create: `README.md`
- Modify: `requirements-dev.txt`

- [ ] **Step 1: Write README**

Create `README.md`:

```markdown
# Korean Local TTS CLI

Local command-line Korean text-to-speech using CosyVoice.

## What It Supports

- Korean text to WAV.
- CPU or NVIDIA CUDA device selection.
- Automatic device selection with `--device auto`.
- Prompt-audio voice cloning with `--prompt-audio`.

## Install Test Dependencies

```bash
pip install -r requirements-dev.txt
```

## Install CosyVoice

Clone the official CosyVoice repository with submodules, install its dependencies, and download a model into this project:

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git vendor/CosyVoice
cd vendor/CosyVoice
pip install -r requirements.txt
cd ../..
python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT'); snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"
```

If the CosyVoice import path is not visible, run the CLI with:

```bash
PYTHONPATH=vendor/CosyVoice python tts.py "안녕하세요" --out hello.wav
```

## Usage

Text-only synthesis:

```bash
python tts.py "안녕하세요. 반갑습니다." --out hello.wav
```

CPU:

```bash
python tts.py "CPU로 생성합니다." --device cpu --out cpu.wav
```

CUDA:

```bash
python tts.py "GPU로 생성합니다." --device cuda --out gpu.wav
```

Prompt-audio voice cloning:

```bash
python tts.py "이 목소리로 말합니다." --prompt-audio sample.wav --prompt-text "샘플 음성의 원문입니다." --out cloned.wav
```

`--prompt-text` is optional, but voice cloning quality is usually better when the transcript matches the prompt audio.

## Development Tests

```bash
pytest -v
```
```

- [ ] **Step 2: Run all tests**

Run:

```bash
pytest -v
```

Expected: PASS for all tests.

- [ ] **Step 3: Run fake CLI smoke test**

Run:

```bash
mkdir -p /tmp/korean-tts-model
python tts.py "안녕하세요" --engine fake --model-dir /tmp/korean-tts-model --out /tmp/korean-tts.wav
```

Expected: exit code 0 and `/tmp/korean-tts.wav` contains `fake wav for: 안녕하세요`.

- [ ] **Step 4: Inspect git status**

Run:

```bash
git status --short
```

Expected: only `README.md` and any intentionally modified project files are shown before commit.

- [ ] **Step 5: Commit**

```bash
git add README.md requirements-dev.txt
git commit -m "Document Korean TTS CLI usage"
```

---

## Manual Real-Model Verification

Run these only after CosyVoice dependencies and model files are installed:

```bash
PYTHONPATH=vendor/CosyVoice python tts.py "안녕하세요. 반갑습니다." --device cpu --out hello-cpu.wav
PYTHONPATH=vendor/CosyVoice python tts.py "안녕하세요. 반갑습니다." --device cuda --out hello-cuda.wav
PYTHONPATH=vendor/CosyVoice python tts.py "이 목소리로 말합니다." --prompt-audio sample.wav --prompt-text "샘플 음성의 원문입니다." --device cuda --out cloned.wav
```

Expected:

- Each command exits with code 0.
- Each output file is a playable WAV.
- CUDA command fails with a clear user-facing error if CUDA is not available.

---

## Self-Review

- Spec coverage: CLI shape, `auto|cpu|cuda`, default voice path, prompt-audio cloning path, model directory validation, user-facing errors, unit tests, and manual CPU/CUDA verification are covered.
- Placeholder scan: no red-flag placeholder language or unfilled implementation instructions remain.
- Type consistency: `SynthesisRequest`, `AudioResult`, `resolve_device`, `CosyVoiceEngine.synthesize`, and CLI option names are used consistently across tasks.
