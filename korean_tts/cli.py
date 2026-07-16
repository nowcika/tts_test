from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .audio import write_wav
from .device import resolve_device
from .engines.base import SynthesisRequest, TTSEngine
from .engines.fake import FakeEngine, write_fake_output
from .errors import UserFacingError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Korean text to a local WAV file.")
    parser.add_argument("text", help="Korean text to synthesize.")
    parser.add_argument("--out", default="output.wav", help="Output WAV path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--prompt-audio", help="Reference WAV file for voice cloning.")
    parser.add_argument("--prompt-text", default="", help="Transcript for --prompt-audio.")
    parser.add_argument("--model-dir", default="pretrained_models/CosyVoice-300M-SFT")
    parser.add_argument("--engine", default="cosyvoice", choices=["cosyvoice", "fake"])
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.text.strip():
        raise UserFacingError("Text input cannot be empty.")

    if args.prompt_audio:
        prompt_audio = Path(args.prompt_audio)
        if not prompt_audio.is_file():
            raise UserFacingError(f"Prompt audio does not exist: {prompt_audio}")

    if args.engine != "fake":
        model_dir = Path(args.model_dir)
        if not model_dir.is_dir():
            raise UserFacingError(f"Model directory does not exist: {model_dir}")


def _create_engine(name: str) -> TTSEngine:
    if name == "fake":
        return FakeEngine()
    if name == "cosyvoice":
        from .engines.cosyvoice import CosyVoiceEngine

        return CosyVoiceEngine()
    raise UserFacingError("Unsupported engine. Use one of: cosyvoice, fake.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
        device = resolve_device(args.device)
        request = SynthesisRequest(
            text=args.text.strip(),
            device=device,
            model_dir=Path(args.model_dir),
            prompt_audio=Path(args.prompt_audio) if args.prompt_audio else None,
            prompt_text=args.prompt_text,
        )
        engine = _create_engine(args.engine)
        result = engine.synthesize(request)

        output = Path(args.out)
        if args.engine == "fake":
            write_fake_output(output, result)
        else:
            write_wav(output, result)
    except UserFacingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0
