from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path


def normalize(text: str) -> str:
    return "".join(re.findall(r"[0-9A-Za-z가-힣]+", text.lower()))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare source text and ASR transcript by normalized character similarity."
    )
    parser.add_argument("source")
    parser.add_argument("transcript")
    args = parser.parse_args()

    source = normalize(Path(args.source).read_text(encoding="utf-8"))
    transcript = normalize(Path(args.transcript).read_text(encoding="utf-8"))
    similarity = SequenceMatcher(None, source, transcript).ratio() * 100

    print(f"normalized_source_chars={len(source)}")
    print(f"normalized_transcript_chars={len(transcript)}")
    print(f"character_similarity_percent={similarity:.1f}")
    print(f"exact_match={source == transcript}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
