from __future__ import annotations

from pathlib import Path

from ..audio import ensure_output_path
from .base import AudioResult, SynthesisRequest


class FakeEngine:
    def synthesize(self, request: SynthesisRequest) -> AudioResult:
        prompt = f" prompt={request.prompt_text}" if request.prompt_audio else ""
        return AudioResult(waveform=f"fake wav for: {request.text}{prompt}\n", sample_rate=0)


def write_fake_output(path: Path, result: AudioResult) -> None:
    output_path = ensure_output_path(path)
    output_path.write_text(str(result.waveform), encoding="utf-8")
