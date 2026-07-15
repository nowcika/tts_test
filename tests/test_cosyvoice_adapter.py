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
