from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

from src.merge.operations import lerp, merge_chunkwise
from src.merge.types import LatentBundle, MergeConfig, MergeStrategy
from .mapping import map_to_target_shape
from .ot_align import align_mean_std, gaussian_ot_align


@dataclass
class AlignConfig:
    """How to register source latents onto the target manifold."""

    enabled: bool = False
    method: str = "ot"  # "none" | "mean_std" | "ot"
    interpolate_after: bool = True
    eps: float = 1e-4


def _register_one(
    source: torch.Tensor,
    target: torch.Tensor,
    config: AlignConfig,
) -> torch.Tensor:
    mapped = map_to_target_shape(source, target)
    if not config.enabled or config.method == "none":
        return mapped
    if config.method == "mean_std":
        return align_mean_std(mapped, target, eps=config.eps)
    if config.method == "ot":
        return gaussian_ot_align(mapped, target, eps=config.eps)
    raise ValueError(f"Unknown align method: {config.method}")


def align_source_to_target(
    source: torch.Tensor,
    target: torch.Tensor,
    config: AlignConfig,
    *,
    lam: float = 0.1,
) -> torch.Tensor:
    """
    Map source → target shape, optionally OT-align, then interpolate.

    Paper interpolation after OT:
      Z_λ = (1 - λ) Z_tgt + λ T(Z_src)
    If interpolate_after is False, returns the aligned source only.
    """
    aligned = _register_one(source, target, config)
    if config.interpolate_after:
        return lerp(target, aligned, lam)
    return aligned


def merge_aligned_latents(
    bundles: Sequence[LatentBundle],
    merge_config: MergeConfig,
    align_config: AlignConfig,
    *,
    target_index: int = 0,
) -> torch.Tensor:
    """
    Register every source onto the target latent grid, then merge.

    Target is `bundles[target_index]` (default: first = decode architecture).
    """
    if len(bundles) < 2:
        raise ValueError("Need at least two bundles")
    if not 0 <= target_index < len(bundles):
        raise IndexError("target_index out of range")

    target = bundles[target_index].latent_tensor(deterministic=merge_config.deterministic)
    pair_merge = merge_config.strategy in {MergeStrategy.LERP, MergeStrategy.SLERP}

    if len(bundles) == 2 and pair_merge:
        source_index = 1 - target_index
        source = bundles[source_index].latent_tensor(deterministic=merge_config.deterministic)
        if align_config.enabled:
            cfg = AlignConfig(
                enabled=True,
                method=align_config.method,
                interpolate_after=True,
                eps=align_config.eps,
            )
            return align_source_to_target(source, target, cfg, lam=merge_config.lam)
        mapped = map_to_target_shape(source, target)
        return lerp(target, mapped, merge_config.lam)

    registered: List[torch.Tensor] = []
    for idx, bundle in enumerate(bundles):
        z = bundle.latent_tensor(deterministic=merge_config.deterministic)
        if idx == target_index:
            registered.append(target)
            continue
        registered.append(_register_one(z, target, align_config))

    return merge_chunkwise(
        registered,
        strategy=merge_config.strategy,
        lam=merge_config.lam,
        weights=merge_config.weights,
    )
