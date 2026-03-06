from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def train_sentencepiece(vocab_size: int, model_prefix: str, input_path: Path) -> None:
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    args = {
        "input": str(input_path),
        "model_prefix": model_prefix,
        "vocab_size": vocab_size,
        "model_type": "bpe",
        "character_coverage": 1.0,
        # The starter corpus is intentionally tiny; allow training to produce
        # fewer pieces than the requested vocab_size instead of erroring.
        "hard_vocab_limit": "false",
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 3,
    }
    spm.SentencePieceTrainer.Train(" ".join(f"--{k}={v}" for k, v in args.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SentencePiece BPE tokenizer on the cybersecurity corpus.")
    parser.add_argument("--vocab_size", type=int, default=16_000, help="Vocabulary size (default: 16000).")
    args = parser.parse_args()

    root = get_project_root()
    cleaned_dir = root / "data" / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    input_path = cleaned_dir / "train.txt"
    model_prefix = str(cleaned_dir / "spm")

    train_sentencepiece(args.vocab_size, model_prefix, input_path)

    print(f"Trained SentencePiece model at: {model_prefix}.model / .vocab")


if __name__ == "__main__":
    main()

