import sys
from types import ModuleType

import numpy as np
import pytest

from korean_tts.engines.base import SynthesisRequest
from korean_tts.engines.edge import EdgeTTSEngine
from korean_tts.errors import UserFacingError


def test_edge_engine_uses_requested_voice_and_rate(monkeypatch, tmp_path):
    calls = []

    class FakeCommunicate:
        def __init__(self, text, voice, rate):
            calls.append((text, voice, rate))

        async def save(self, path):
            calls.append(("save", path))

    edge_module = ModuleType("edge_tts")
    edge_module.Communicate = FakeCommunicate
    monkeypatch.setitem(sys.modules, "edge_tts", edge_module)
    monkeypatch.setattr("korean_tts.engines.edge._convert_mp3_to_wav", lambda mp3, wav: None)
    monkeypatch.setattr(
        "korean_tts.engines.edge.sf.read",
        lambda path, dtype: (np.array([0.0, 0.1], dtype=np.float32), 24000),
    )

    result = EdgeTTSEngine().synthesize(
        SynthesisRequest(
            text="안녕하세요",
            device="cpu",
            model_dir=tmp_path,
            voice="ko-KR-InJoonNeural",
            rate="-10%",
        )
    )

    assert calls[0] == ("안녕하세요", "ko-KR-InJoonNeural", "-10%")
    assert calls[1][0] == "save"
    assert result.sample_rate == 24000
    assert result.waveform.tolist() == pytest.approx([0.0, 0.1])


def test_edge_engine_import_error_is_user_facing(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "edge_tts", None)

    with pytest.raises(UserFacingError, match="edge-tts is not installed"):
        EdgeTTSEngine().synthesize(
            SynthesisRequest(text="안녕하세요", device="cpu", model_dir=tmp_path)
        )
