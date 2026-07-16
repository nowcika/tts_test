from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SynthesisRequest:
    text: str
    device: str
    model_dir: Path
    prompt_audio: Path | None = None
    prompt_text: str = ""
    voice: str = "ko-KR-SunHiNeural"
    rate: str = "+0%"


@dataclass(frozen=True)
class AudioResult:
    waveform: object
    sample_rate: int


class TTSEngine(Protocol):
    def synthesize(self, request: SynthesisRequest) -> AudioResult:
        raise NotImplementedError
