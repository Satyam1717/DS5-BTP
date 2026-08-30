from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F

from .types import MergeStrategy


def _as_list(tensors: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    if len(tensors) < 2:
        raise ValueError("Need at least two latent tensors to merge.")
    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) != 1:
        raise ValueError(f"Latent shape mismatch: {shapes}")
    return list(tensors)


def lerp(a: torch.Tensor, b: torch.Tensor, lam: float) -> torch.Tensor:
    """Linear interpolation: (1 - lam) * a + lam * b."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    return (1.0 - lam) * a + lam * b


def uniform_soup(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Average multiple latent tensors with equal weights."""
    items = _as_list(tensors)
    return torch.stack(items, dim=0).mean(dim=0)


def weighted_soup(tensors: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    """Convex combination of latent tensors."""
    items = _as_list(tensors)
    if len(weights) != len(items):
        raise ValueError("weights must match number of tensors")
    weight_tensor = torch.tensor(weights, dtype=items[0].dtype, device=items[0].device)
    if torch.any(weight_tensor < 0):
        raise ValueError("weights must be non-negative")
    total = float(weight_tensor.sum().item())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    weight_tensor = weight_tensor / total
    stacked = torch.stack(items, dim=0)
    return torch.sum(stacked * weight_tensor.view(-1, *([1] * (stacked.ndim - 1))), dim=0)


def slerp(a: torch.Tensor, b: torch.Tensor, lam: float, eps: float = 1e-8) -> torch.Tensor:
    """Spherical linear interpolation on flattened tensors."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    orig_shape = a.shape
    va = a.reshape(-1)
    vb = b.reshape(-1)

    va = F.normalize(va, dim=0, eps=eps)
    vb = F.normalize(vb, dim=0, eps=eps)

    dot = torch.clamp(torch.dot(va, vb), -1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    if sin_omega.abs() < eps:
        out = (1.0 - lam) * va + lam * vb
    else:
        out = (
            torch.sin((1.0 - lam) * omega) / sin_omega
        ) * va + (torch.sin(lam * omega) / sin_omega) * vb

    return out.reshape(orig_shape)


def barycenter(tensors: Sequence[torch.Tensor], weights: Sequence[float] | None = None) -> torch.Tensor:
    """Latent barycenter; defaults to uniform weights."""
    items = _as_list(tensors)
    if weights is None:
        return uniform_soup(items)
    return weighted_soup(items, weights)


def merge_latents(
    tensors: Sequence[torch.Tensor],
    strategy: MergeStrategy,
    lam: float = 0.5,
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Dispatch merge operator over a list of latent tensors."""
    items = _as_list(tensors)

    if strategy == MergeStrategy.LERP:
        if len(items) != 2:
            raise ValueError("LERP requires exactly two tensors")
        return lerp(items[0], items[1], lam)

    if strategy == MergeStrategy.SLERP:
        if len(items) != 2:
            raise ValueError("SLERP requires exactly two tensors")
        return slerp(items[0], items[1], lam)

    if strategy == MergeStrategy.UNIFORM_SOUP:
        return uniform_soup(items)

    if strategy == MergeStrategy.WEIGHTED_SOUP:
        if weights is None:
            raise ValueError("weighted_soup requires weights")
        return weighted_soup(items, weights)

    if strategy == MergeStrategy.BARYCENTER:
        return barycenter(items, weights)

    raise ValueError(f"Unsupported strategy: {strategy}")


def merge_chunkwise(
    bundles_latents: Iterable[torch.Tensor],
    strategy: MergeStrategy,
    lam: float = 0.5,
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """
    Merge a sequence of per-chunk latent tensors.

    Each input tensor is expected to have shape (num_chunks, ...).
    Merging is applied independently for each chunk index.
    """
    latent_list = list(bundles_latents)
    if not latent_list:
        raise ValueError("No latent tensors provided")

    num_chunks = latent_list[0].shape[0]
    merged_chunks = []
    for chunk_idx in range(num_chunks):
        chunk_latents = [latents[chunk_idx] for latents in latent_list]
        merged_chunks.append(
            merge_latents(chunk_latents, strategy=strategy, lam=lam, weights=weights)
        )
    return torch.stack(merged_chunks, dim=0)
