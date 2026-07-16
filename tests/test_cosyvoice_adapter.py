import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from korean_tts.engines.base import SynthesisRequest
from korean_tts.engines.cosyvoice import CosyVoiceEngine, _load_cosyvoice_model
from korean_tts.errors import UserFacingError


class FakeCosyVoiceModel:
    sample_rate = 24000

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
    assert result.sample_rate == 24000
    assert fake_model.calls == [("sft", "안녕하세요", "韩语女")]


def test_text_only_concatenates_multiple_audio_chunks(monkeypatch, tmp_path):
    class MultiChunkModel(FakeCosyVoiceModel):
        def inference_sft(self, text, speaker):
            self.calls.append(("sft", text, speaker))
            yield {"tts_speech": [0.1, -0.1]}
            yield {"tts_speech": [0.2, -0.2]}

    fake_model = MultiChunkModel()
    monkeypatch.setattr(
        "korean_tts.engines.cosyvoice._load_cosyvoice_model",
        lambda model_dir, device: fake_model,
    )

    result = CosyVoiceEngine().synthesize(
        SynthesisRequest(text="긴 문장입니다", device="cpu", model_dir=tmp_path)
    )

    assert result.waveform == [0.1, -0.1, 0.2, -0.2]
    assert result.sample_rate == 24000
    assert fake_model.calls == [("sft", "긴 문장입니다", "韩语女")]


def test_prompt_audio_uses_zero_shot_path(monkeypatch, tmp_path):
    fake_model = FakeCosyVoiceModel()
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"wav")
    monkeypatch.setattr(
        "korean_tts.engines.cosyvoice._load_cosyvoice_model",
        lambda model_dir, device: fake_model,
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
    assert result.sample_rate == 24000
    assert fake_model.calls == [("zero_shot", "따라 말합니다", "샘플입니다", str(prompt_audio))]


def test_dependency_import_error_becomes_user_facing_error(monkeypatch, tmp_path):
    def fail_import(model_dir: Path, device: str):
        raise ImportError("No module named 'cosyvoice'")

    monkeypatch.setattr("korean_tts.engines.cosyvoice._load_cosyvoice_model", fail_import)

    with pytest.raises(UserFacingError, match="CosyVoice dependencies are not installed"):
        CosyVoiceEngine().synthesize(SynthesisRequest(text="안녕하세요", device="cpu", model_dir=tmp_path))


def install_fake_cosyvoice_modules(monkeypatch, cuda_is_available):
    construction_cuda_states = []
    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(is_available=cuda_is_available)

    class FakeCosyVoice2:
        def __init__(self, model_dir):
            self.model_dir = model_dir
            construction_cuda_states.append(torch_module.cuda.is_available())

    cosyvoice_module = ModuleType("cosyvoice")
    cli_module = ModuleType("cosyvoice.cli")
    cosyvoice_cli_module = ModuleType("cosyvoice.cli.cosyvoice")
    cosyvoice_cli_module.AutoModel = FakeCosyVoice2

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "cosyvoice", cosyvoice_module)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli", cli_module)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", cosyvoice_cli_module)

    return torch_module, construction_cuda_states


def test_load_cosyvoice_model_forces_cpu_during_construction(monkeypatch, tmp_path):
    def cuda_is_available():
        return True

    torch_module, construction_cuda_states = install_fake_cosyvoice_modules(
        monkeypatch, cuda_is_available
    )

    model = _load_cosyvoice_model(tmp_path, "cpu")

    assert model.model_dir == str(tmp_path)
    assert construction_cuda_states == [False]
    assert torch_module.cuda.is_available is cuda_is_available
    assert torch_module.cuda.is_available() is True


def test_load_cosyvoice_model_leaves_cuda_available_for_cuda_device(monkeypatch, tmp_path):
    def cuda_is_available():
        return True

    torch_module, construction_cuda_states = install_fake_cosyvoice_modules(
        monkeypatch, cuda_is_available
    )

    _load_cosyvoice_model(tmp_path, "cuda")

    assert construction_cuda_states == [True]
    assert torch_module.cuda.is_available is cuda_is_available

