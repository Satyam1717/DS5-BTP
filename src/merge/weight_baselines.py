from __future__ import annotations

from typing import Dict, Sequence

import torch


def _shared_keys(state_dicts: Sequence[Dict[str, torch.Tensor]]) -> list[str]:
    if not state_dicts:
        raise ValueError("Need at least one state_dict")
    keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        keys &= set(sd.keys())
    if not keys:
        raise ValueError("No overlapping parameter names to merge")
    return sorted(keys)


def weight_lerp(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    lam: float,
) -> Dict[str, torch.Tensor]:
    """Linear interpolation of matching tensors in weight space."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    out = {k: v.clone() for k, v in a.items()}
    for key in _shared_keys([a, b]):
        if a[key].shape != b[key].shape:
            continue
        out[key] = (1.0 - lam) * a[key].float() + lam * b[key].float()
        out[key] = out[key].to(dtype=a[key].dtype)
    return out


def weight_uniform_soup(state_dicts: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Average overlapping tensors. Non-overlapping keys come from the first model."""
    if len(state_dicts) < 2:
        raise ValueError("Need at least two state_dicts")
    keys = _shared_keys(state_dicts)
    out = {k: v.clone() for k, v in state_dicts[0].items()}
    n = float(len(state_dicts))
    for key in keys:
        shapes = {tuple(sd[key].shape) for sd in state_dicts}
        if len(shapes) != 1:
            continue
        acc = None
        for sd in state_dicts:
            val = sd[key].float()
            acc = val if acc is None else acc + val
        out[key] = (acc / n).to(dtype=state_dicts[0][key].dtype)
    return out


def task_vector_merge(
    base: Dict[str, torch.Tensor],
    experts: Sequence[Dict[str, torch.Tensor]],
    *,
    scaling: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Task arithmetic in weight space:
      W' = W_base + scaling * sum_i (W_i - W_base)
    """
    if not experts:
        raise ValueError("Need at least one expert")
    out = {k: v.clone() for k, v in base.items()}
    for key in _shared_keys([base, *experts]):
        if any(sd[key].shape != base[key].shape for sd in experts):
            continue
        delta = None
        base_f = base[key].float()
        for sd in experts:
            d = sd[key].float() - base_f
            delta = d if delta is None else delta + d
        out[key] = (base_f + scaling * delta).to(dtype=base[key].dtype)
    return out
