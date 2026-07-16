from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf

from .base import AudioResult, SynthesisRequest
from ..errors import UserFacingError

DEFAULT_EDGE_VOICE = "ko-KR-SunHiNeural"


class EdgeTTSEngine:
    def synthesize(self, request: SynthesisRequest) -> AudioResult:
        try:
            import edge_tts
        except ImportError as exc:
            raise UserFacingError(
                "edge-tts is not installed. Install it with: pip install edge-tts"
            ) from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mp3_path = temp_path / "speech.mp3"
            wav_path = temp_path / "speech.wav"

            communicate = edge_tts.Communicate(
                request.text,
                voice=request.voice,
                rate=request.rate,
            )
            try:
                asyncio.run(communicate.save(str(mp3_path)))
                _convert_mp3_to_wav(mp3_path, wav_path)
                waveform, sample_rate = sf.read(str(wav_path), dtype="float32")
            except Exception as exc:
                raise UserFacingError(f"Edge TTS synthesis failed: {exc}") from exc

        return AudioResult(waveform=waveform, sample_rate=int(sample_rate))


def _convert_mp3_to_wav(mp3_path: Path, wav_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mp3_path),
        "-ac",
        "1",
        "-ar",
        "24000",
        str(wav_path),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise UserFacingError(
            "ffmpeg is required to convert Edge TTS output to WAV."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise UserFacingError("ffmpeg failed to convert Edge TTS output to WAV.") from exc
