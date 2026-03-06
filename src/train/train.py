from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.config import get_model_config
from src.model.gpt import GPT, GPTConfig


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    root = get_project_root()
    cleaned = root / "data" / "cleaned"
    train = np.load(cleaned / "train.npy", mmap_mode="r")
    valid = np.load(cleaned / "valid.npy", mmap_mode="r")
    return train, valid


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_spm(spm_path: Path) -> spm.SentencePieceProcessor | None:
    if not spm_path.exists():
        return None
    proc = spm.SentencePieceProcessor()
    proc.load(str(spm_path))
    return proc


def infer_vocab_size(sp_proc: spm.SentencePieceProcessor | None, fallback: int) -> int:
    if sp_proc is None:
        return fallback
    try:
        return int(sp_proc.get_piece_size())
    except Exception:  # noqa: BLE001
        return fallback


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Data is 1D array of token IDs.
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    x = torch.from_numpy(x.astype(np.int64)).to(device)
    y = torch.from_numpy(y.astype(np.int64)).to(device)
    return x, y


def save_checkpoint(
    model: GPT,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler | None,
    step: int,
    out_dir: Path,
    train_meta: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "model_config": model.config.to_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "train_meta": train_meta,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    path = out_dir / f"ckpt_step{step}.pt"
    latest = out_dir / "latest.pt"
    torch.save(ckpt, path)
    torch.save(ckpt, latest)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> Tuple[GPT, AdamW, LambdaLR, torch.cuda.amp.GradScaler | None, int, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    config = GPTConfig.from_dict(ckpt["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])

    train_meta = ckpt.get("train_meta", {})
    lr = float(train_meta.get("lr", 3e-4))
    betas = tuple(train_meta.get("betas", (0.9, 0.95)))
    weight_decay = float(train_meta.get("weight_decay", 0.1))
    total_steps = int(train_meta.get("total_steps", 1000))
    warmup_steps = int(train_meta.get("warmup_steps", max(10, int(0.05 * total_steps))))

    optimizer = AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])

    scaler_state = ckpt.get("scaler_state")
    scaler = torch.cuda.amp.GradScaler() if scaler_state is not None else None
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    step = ckpt.get("step", 0)
    return model, optimizer, scheduler, scaler, step, train_meta


def create_model_and_optim(
    preset: str,
    lr: float,
    device: torch.device,
    total_steps: int,
    vocab_size: int,
) -> Tuple[GPT, AdamW, LambdaLR, torch.cuda.amp.GradScaler | None]:
    model_cfg = get_model_config(preset)
    gpt_cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=model_cfg.block_size,
        n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head,
        n_embd=model_cfg.n_embd,
        dropout=model_cfg.dropout,
    )
    model = GPT(gpt_cfg).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    warmup_steps = max(10, int(0.05 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    return model, optimizer, scheduler, scaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small GPT-style model on cybersecurity text.")
    parser.add_argument(
        "--preset",
        type=str,
        default="CPU_TINY",
        choices=["CPU_TINY", "GPU_SMALL", "GPU_MED"],
        help="Model size preset.",
    )
    parser.add_argument("--max_steps", type=int, default=500, help="Number of optimization steps.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Base learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints/latest.pt if available.",
    )
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--spm_model",
        type=str,
        default=None,
        help="Optional SentencePiece model path for vocab size + decoding samples (default: data/cleaned/spm.model).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_data, valid_data = load_data()
    model_cfg = get_model_config(args.preset)

    root = get_project_root()
    ckpt_dir = root / "checkpoints"
    spm_path = Path(args.spm_model) if args.spm_model else root / "data" / "cleaned" / "spm.model"
    sp_proc = load_spm(spm_path)
    vocab_size = infer_vocab_size(sp_proc, model_cfg.vocab_size)

    train_meta = {
        "preset": args.preset,
        "lr": args.lr,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "total_steps": args.max_steps,
        "warmup_steps": max(10, int(0.05 * args.max_steps)),
        "spm_model": str(spm_path),
    }

    start_step = 0
    if args.resume:
        latest = ckpt_dir / "latest.pt"
        if latest.exists():
            print(f"Resuming from {latest}")
            model, optimizer, scheduler, scaler, start_step, loaded_meta = load_checkpoint(latest, device)
            if loaded_meta.get("spm_model"):
                spm_path = Path(loaded_meta["spm_model"])
                sp_proc = load_spm(spm_path)
        else:
            print(f"No checkpoint found at {latest}, starting from scratch.")
            model, optimizer, scheduler, scaler = create_model_and_optim(
                args.preset,
                args.lr,
                device,
                args.max_steps,
                vocab_size=vocab_size,
            )
    else:
        model, optimizer, scheduler, scaler = create_model_and_optim(
            args.preset,
            args.lr,
            device,
            args.max_steps,
            vocab_size=vocab_size,
        )

    model.train()

    pbar = tqdm(range(start_step, args.max_steps), initial=start_step, total=args.max_steps)
    for step in pbar:
        x, y = get_batch(train_data, model.config.block_size, model_cfg.batch_size, device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        scheduler.step()

        if step % args.log_interval == 0:
            pbar.set_description(f"step {step} loss {loss.item():.4f}")

        if (step + 1) % args.eval_interval == 0 or step == args.max_steps - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(valid_data, model.config.block_size, model_cfg.batch_size, device)
                _, vloss = model(vx, vy)
            print(f"\nEval step {step}: val_loss={vloss.item():.4f}")

            # Sample a short generation for sanity-check.
            if sp_proc is not None:
                sample_prompt = "SOC note: "
                idx = torch.tensor([sp_proc.encode(sample_prompt, out_type=int)], dtype=torch.long, device=device)
            else:
                idx = torch.zeros((1, 1), dtype=torch.long, device=device)
            sample = model.generate(idx, max_new_tokens=64, temperature=0.9, top_k=40)
            print("Sample generation:")
            if sp_proc is not None:
                print(sp_proc.decode(sample[0].tolist()))
            else:
                print(sample[0].tolist())
            model.train()

        if (step + 1) % args.save_interval == 0 or step == args.max_steps - 1:
            save_checkpoint(model, optimizer, scheduler, scaler, step + 1, ckpt_dir, train_meta=train_meta)


if __name__ == "__main__":
    main()

