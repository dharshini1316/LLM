from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sentencepiece as spm


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def encode_split(sp: spm.SentencePieceProcessor, path: Path) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"Text file not found: {path}")
    text = path.read_text(encoding="utf-8")
    ids = sp.encode(text, out_type=int)
    # Use uint16 which is safe up to vocab_size < 65535 (16k / 32k are fine).
    return np.array(ids, dtype=np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode train/valid text into token ID arrays.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to SentencePiece model (default: data/cleaned/spm.model).",
    )
    args = parser.parse_args()

    root = get_project_root()
    cleaned_dir = root / "data" / "cleaned"
    model_path = Path(args.model) if args.model else cleaned_dir / "spm.model"

    if not model_path.exists():
        raise SystemExit(f"SentencePiece model not found: {model_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    train_txt = cleaned_dir / "train.txt"
    valid_txt = cleaned_dir / "valid.txt"

    train_ids = encode_split(sp, train_txt)
    valid_ids = encode_split(sp, valid_txt)

    np.save(cleaned_dir / "train.npy", train_ids)
    np.save(cleaned_dir / "valid.npy", valid_ids)

    print(f"Encoded train tokens: {train_ids.shape[0]}")
    print(f"Encoded valid tokens: {valid_ids.shape[0]}")
    print(f"Saved to {cleaned_dir / 'train.npy'} and {cleaned_dir / 'valid.npy'}")


if __name__ == "__main__":
    main()

