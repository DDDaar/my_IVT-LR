from __future__ import annotations

import os
from typing import Optional

import torch


def _cache_path(cache_dir: str, expert: str, idx: int) -> str:
    # Store each expert separately to keep formats simple and allow partial caching.
    return os.path.join(cache_dir, expert, f"{int(idx)}.pt")


def load_teacher_vector(cache_dir: str, expert: str, idx: int) -> Optional[torch.Tensor]:
    path = _cache_path(cache_dir, expert, idx)
    if not os.path.exists(path):
        return None
    vec = torch.load(path, map_location="cpu")
    if not torch.is_tensor(vec):
        raise ValueError(f"Teacher cache must store a torch.Tensor. Got type={type(vec)} at {path}")
    if vec.ndim != 1:
        vec = vec.view(-1)
    return vec


def save_teacher_vector(cache_dir: str, expert: str, idx: int, vec: torch.Tensor) -> str:
    path = _cache_path(cache_dir, expert, idx)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vec_cpu = vec.detach().to("cpu")
    if vec_cpu.ndim != 1:
        vec_cpu = vec_cpu.view(-1)
    torch.save(vec_cpu, path)
    return path

