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
