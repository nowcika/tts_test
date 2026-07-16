from pathlib import Path

import pytest

from korean_tts.engines.base import AudioResult
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

    exit_code = main(["안녕하세요", "--engine", "cosyvoice", "--model-dir", str(missing_model)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Model directory does not exist" in captured.err


def test_cli_edge_engine_does_not_require_model_dir(monkeypatch, tmp_path):
    class StubEdgeEngine:
        def synthesize(self, request):
            assert request.text == "안녕하세요"
            assert request.voice == "ko-KR-SunHiNeural"
            assert request.rate == "+0%"
            return AudioResult(waveform=[0.0, 0.1], sample_rate=24000)

    monkeypatch.setattr("korean_tts.cli._create_engine", lambda name: StubEdgeEngine())
    output = tmp_path / "edge.wav"

    exit_code = main(["안녕하세요", "--engine", "edge", "--out", str(output)])

    assert exit_code == 0
    assert output.is_file()


def test_cli_default_engine_is_edge(monkeypatch, tmp_path):
    seen = []

    class StubEdgeEngine:
        def synthesize(self, request):
            seen.append(request)
            return AudioResult(waveform=[0.0, 0.1], sample_rate=24000)

    def create_engine(name):
        assert name == "edge"
        return StubEdgeEngine()

    monkeypatch.setattr("korean_tts.cli._create_engine", create_engine)
    output = tmp_path / "default.wav"

    exit_code = main(["안녕하세요", "--out", str(output)])

    assert exit_code == 0
    assert output.is_file()
    assert seen[0].voice == "ko-KR-SunHiNeural"


def test_cli_edge_engine_rejects_prompt_audio(capsys):
    prompt_audio = Path("samples/reference.wav")
    assert prompt_audio.is_file()

    exit_code = main(["안녕하세요", "--engine", "edge", "--prompt-audio", str(prompt_audio)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--prompt-audio is only supported" in captured.err


def test_cli_generates_wav_with_fake_engine(tmp_path):
    output = tmp_path / "hello.wav"

    exit_code = main(
        [
            "안녕하세요",
            "--engine",
            "fake",
            "--out",
            str(output),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert output.read_text(encoding="utf-8") == "fake wav for: 안녕하세요\n"


def test_cli_rejects_non_wav_output_with_fake_engine(tmp_path, capsys):
    exit_code = main(
        [
            "안녕하세요",
            "--engine",
            "fake",
            "--out",
            str(tmp_path / "bad.txt"),
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Output path must end with .wav" in captured.err


def test_cli_passes_prompt_audio_and_text_to_fake_engine(tmp_path):
    prompt_audio = Path("samples/reference.wav")
    assert prompt_audio.is_file()
    output = tmp_path / "cloned.wav"

    exit_code = main(
        [
            "따라 말합니다",
            "--engine",
            "fake",
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
