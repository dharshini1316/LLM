from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelConfig:
    vocab_size: int = 16_000
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    batch_size: int = 16


PRESETS: Dict[str, ModelConfig] = {
    # Very small model and batches for CPU / laptops.
    "CPU_TINY": ModelConfig(
        vocab_size=16_000,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=128,
        dropout=0.1,
        batch_size=8,
    ),
    # Small GPU-friendly config (~4–6 GB VRAM).
    "GPU_SMALL": ModelConfig(
        vocab_size=16_000,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.1,
        batch_size=16,
    ),
    # Medium config for 8–12 GB+ VRAM.
    "GPU_MED": ModelConfig(
        vocab_size=16_000,
        block_size=256,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.1,
        batch_size=16,
    ),
}


def get_model_config(name: str) -> ModelConfig:
    name = name.upper()
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Valid options: {list(PRESETS.keys())}")
    return PRESETS[name]

