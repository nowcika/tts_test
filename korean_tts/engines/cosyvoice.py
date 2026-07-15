from __future__ import annotations

from pathlib import Path

from .base import AudioResult, SynthesisRequest
from ..errors import UserFacingError

DEFAULT_SPEAKER = "中文女"


def _load_cosyvoice_model(model_dir: Path, device: str):
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2
    except ImportError as exc:
        raise ImportError("CosyVoice is not importable") from exc

    # CosyVoice uses PyTorch device state internally. Keep this adapter thin and
    # pass the local model path through the official constructor.
    return CosyVoice2(str(model_dir))


def _first_audio_chunk(chunks):
    for chunk in chunks:
        if "tts_speech" not in chunk:
            continue
        return chunk["tts_speech"]
    raise UserFacingError("CosyVoice did not return any audio.")


class CosyVoiceEngine:
    def synthesize(self, request: SynthesisRequest) -> AudioResult:
        try:
            model = _load_cosyvoice_model(request.model_dir, request.device)
            if request.prompt_audio:
                waveform = _first_audio_chunk(
                    model.inference_zero_shot(
                        request.text, request.prompt_text, str(request.prompt_audio)
                    )
                )
            else:
                waveform = _first_audio_chunk(model.inference_sft(request.text, DEFAULT_SPEAKER))
        except ImportError as exc:
            raise UserFacingError(
                "CosyVoice dependencies are not installed. Install the official CosyVoice repository dependencies first."
            ) from exc

        return AudioResult(waveform=waveform, sample_rate=int(model.sample_rate))
