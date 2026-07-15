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
