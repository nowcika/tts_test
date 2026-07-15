# Korean Local TTS CLI Design

## Goal

Build a local command-line tool that converts Korean text into an audio file. The tool must support both CPU and NVIDIA CUDA execution, and it must support two speech modes:

- Default Korean voice generation from text only.
- Prompt-audio voice cloning, where a short reference audio file guides the generated voice.

The first implemented engine will be CosyVoice. The CLI will keep a small engine boundary so another local TTS backend can be added later without changing the user-facing command shape.

## User Interface

Primary command examples:

```bash
python tts.py "안녕하세요. 반갑습니다." --out hello.wav
python tts.py "이 목소리로 말합니다." --prompt-audio sample.wav --out cloned.wav
python tts.py "CPU로 생성합니다." --device cpu --out cpu.wav
python tts.py "GPU로 생성합니다." --device cuda --out gpu.wav
```

Initial CLI options:

- Positional `text`: Korean text to synthesize.
- `--out`: output audio path, defaulting to `output.wav`.
- `--device`: `auto`, `cpu`, or `cuda`; default `auto`.
- `--prompt-audio`: optional reference audio file for voice cloning.
- `--prompt-text`: optional transcript of `--prompt-audio`; when omitted, the first implementation will pass an empty prompt transcript and document that quality is better when a transcript is supplied.
- `--model-dir`: local model directory, defaulting to `pretrained_models/CosyVoice2-0.5B`.
- `--engine`: defaults to `cosyvoice`.

## Architecture

The project will be a small Python CLI with these units:

- `tts.py`: command entry point and argument parsing.
- `korean_tts/device.py`: device selection and validation.
- `korean_tts/engines/base.py`: small engine interface.
- `korean_tts/engines/cosyvoice.py`: CosyVoice adapter.
- `korean_tts/audio.py`: output path validation and audio writing helpers.

The CLI parses arguments, resolves the device, constructs the selected engine, runs synthesis, and writes a `.wav` file. The CosyVoice adapter owns all CosyVoice-specific imports and call shapes so the rest of the project does not depend directly on CosyVoice internals.

## CosyVoice Scope

CosyVoice is the first target because the official project describes multilingual TTS, Korean support, and zero-shot voice cloning. The implementation will use the official CosyVoice Python APIs rather than reimplementing model logic.

The implementation will assume the CosyVoice repository and model files are available locally, or it will provide setup scripts/documentation to fetch them. Model downloads are deliberately separate from normal CLI execution because the model files are large and network-dependent.

## Data Flow

For text-only synthesis:

1. Read Korean text from the CLI argument.
2. Resolve `auto` device to `cuda` if available, otherwise `cpu`.
3. Load the configured CosyVoice model.
4. Generate speech through the CosyVoice text-to-speech path using the engine configured default speaker behavior.
5. Write the generated waveform to the requested output path.

For prompt-audio synthesis:

1. Validate `--prompt-audio` exists and is readable.
2. Read optional `--prompt-text`.
3. Resolve device and load the model.
4. Run the CosyVoice zero-shot inference path with the reference audio.
5. Write the generated waveform to the requested output path.

## Error Handling

The CLI should fail with clear messages for:

- Missing or unreadable prompt audio.
- Missing model directory.
- Requested `--device cuda` when CUDA is unavailable.
- Empty text input.
- Unsupported engine name.
- CosyVoice dependency import failures.

Dependency and model errors should tell the user what setup command or path to check instead of exposing only a Python traceback.

## Testing

Initial tests will avoid loading real TTS models. They will cover:

- CLI argument parsing.
- Device selection behavior with mocked CUDA availability.
- Validation for missing prompt audio and empty text.
- Engine dispatch using a fake engine.

Manual verification will cover real CosyVoice synthesis on CPU and CUDA after dependencies and models are installed.

## Sources

- Official CosyVoice repository: https://github.com/FunAudioLLM/CosyVoice
- CosyVoice paper: https://arxiv.org/abs/2407.05407
- CosyVoice 2 paper: https://arxiv.org/abs/2412.10117
- XTTS reference considered as an alternative: https://arxiv.org/abs/2406.04904
