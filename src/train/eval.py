from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.model.gpt import GPT, GPTConfig


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_data() -> np.ndarray:
    root = get_project_root()
    cleaned = root / "data" / "cleaned"
    return np.load(cleaned / "valid.npy", mmap_mode="r")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate validation perplexity for a checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: checkpoints/latest.pt).",
    )
    args = parser.parse_args()

    device = get_device()
    root = Path(__file__).resolve().parents[2]
    ckpt_path = Path(args.checkpoint) if args.checkpoint else root / "checkpoints" / "latest.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig.from_dict(ckpt["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data = load_data()
    block_size = cfg.block_size

    losses = []
    with torch.no_grad():
        # Evaluate on a few batches for speed.
        num_batches = min(64, max(1, len(data) // (block_size + 1)))
        for i in range(num_batches):
            # simple sequential chunks
            start = i * (block_size + 1)
            end = start + block_size + 1
            if end >= len(data):
                break
            chunk = torch.from_numpy(data[start:end].astype("int64")).unsqueeze(0).to(device)
            x = chunk[:, :-1]
            y = chunk[:, 1:]
            _, loss = model(x, y)
            losses.append(loss.item())

    if not losses:
        raise SystemExit("Validation set too small for evaluation.")

    avg_loss = sum(losses) / len(losses)
    ppl = float(torch.exp(torch.tensor(avg_loss)))
    print(f"Validation loss: {avg_loss:.4f}")
    print(f"Validation perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()

