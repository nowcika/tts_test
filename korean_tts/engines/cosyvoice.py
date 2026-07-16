from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .base import AudioResult, SynthesisRequest
from ..errors import UserFacingError

DEFAULT_SPEAKER = "韩语女"


@contextmanager
def _force_cpu_if_requested(device: str) -> Iterator[None]:
    if device != "cpu":
        yield
        return

    try:
        import torch
    except ImportError:
        yield
        return

    original_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        yield
    finally:
        torch.cuda.is_available = original_is_available


def _load_cosyvoice_model(model_dir: Path, device: str):
    try:
        from cosyvoice.cli.cosyvoice import AutoModel
    except ImportError as exc:
        raise ImportError("CosyVoice is not importable") from exc

    # CosyVoice chooses device from torch.cuda.is_available() during model
    # construction, so CPU requests must mask CUDA while the model is created.
    with _force_cpu_if_requested(device):
        return AutoModel(model_dir=str(model_dir))


def _concatenate_audio_chunks(audio_chunks: list[object]) -> object:
    first_chunk = audio_chunks[0]
    if hasattr(first_chunk, "detach"):
        import torch

        return torch.cat(audio_chunks, dim=-1)
    if hasattr(first_chunk, "ndim"):
        import numpy as np

        return np.concatenate(audio_chunks, axis=-1)

    waveform = []
    for audio_chunk in audio_chunks:
        waveform.extend(audio_chunk)
    return waveform


def _audio_from_chunks(chunks):
    audio_chunks = []
    for chunk in chunks:
        if "tts_speech" not in chunk:
            continue
        audio_chunks.append(chunk["tts_speech"])
    if audio_chunks:
        return _concatenate_audio_chunks(audio_chunks)
    raise UserFacingError("CosyVoice did not return any audio.")


class CosyVoiceEngine:
    def synthesize(self, request: SynthesisRequest) -> AudioResult:
        try:
            model = _load_cosyvoice_model(request.model_dir, request.device)
            if request.prompt_audio:
                waveform = _audio_from_chunks(
                    model.inference_zero_shot(
                        request.text, request.prompt_text, str(request.prompt_audio)
                    )
                )
            else:
                waveform = _audio_from_chunks(model.inference_sft(request.text, DEFAULT_SPEAKER))
        except ImportError as exc:
            raise UserFacingError(
                "CosyVoice dependencies are not installed. Install the official CosyVoice repository dependencies first."
            ) from exc

        return AudioResult(waveform=waveform, sample_rate=int(model.sample_rate))
