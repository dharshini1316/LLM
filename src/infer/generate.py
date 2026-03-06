from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from src.model.gpt import GPT, GPTConfig


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(
    checkpoint_path: Path,
    spm_path: Path,
    device: torch.device,
) -> tuple[GPT, spm.SentencePieceProcessor]:
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    if not spm_path.exists():
        raise SystemExit(f"SentencePiece model not found: {spm_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GPTConfig.from_dict(ckpt["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_path))

    return model, sp


def generate_text(
    model: GPT,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.9,
    top_k: int = 40,
) -> str:
    ids = sp.encode(prompt, out_type=int)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    out_ids = y[0].tolist()
    text = sp.decode(out_ids)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained cybersec GPT model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: checkpoints/latest.pt).",
    )
    parser.add_argument(
        "--spm_model",
        type=str,
        default=None,
        help="Path to SentencePiece model (default: data/cleaned/spm.model).",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text. If omitted, read from stdin.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    root = get_project_root()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else root / "checkpoints" / "latest.pt"
    spm_path = Path(args.spm_model) if args.spm_model else root / "data" / "cleaned" / "spm.model"

    device = get_device()
    model, sp = load_model_and_tokenizer(ckpt_path, spm_path, device)

    if args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = input("Enter prompt: ")

    text = generate_text(
        model,
        sp,
        prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)


if __name__ == "__main__":
    main()

