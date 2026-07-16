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

    exit_code = main(["안녕하세요", "--engine", "cosyvoice", "--model-dir", str(missing_model)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Model directory does not exist" in captured.err


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
