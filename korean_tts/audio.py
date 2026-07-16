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


def _normalize_waveform(waveform: object) -> object:
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    if hasattr(waveform, "ndim") and hasattr(waveform, "shape"):
        if waveform.ndim == 2 and waveform.shape[0] == 1:
            return waveform[0]
    return waveform


def write_wav(path: Path, result: AudioResult) -> None:
    output_path = ensure_output_path(path)
    sf.write(str(output_path), _normalize_waveform(result.waveform), result.sample_rate)
