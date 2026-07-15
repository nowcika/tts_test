from __future__ import annotations

from pathlib import Path

import soundfile as sf

from .engines.base import AudioResult
from .errors import UserFacingError


def ensure_output_path(path: Path) -> Path:
    if path.suffix.lower() != ".wav":
        raise UserFacingError("Output path must end with .wav.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_wav(path: Path, result: AudioResult) -> None:
    output_path = ensure_output_path(path)
    sf.write(str(output_path), result.waveform, result.sample_rate)
