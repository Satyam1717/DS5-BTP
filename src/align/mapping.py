from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_ncd(z: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """
    Represent latents as (N, C) for resampling along the sample/chunk axis.
    Extra middle dims are flattened into C.
    """
    orig = z.shape
    if z.ndim == 1:
        return z.unsqueeze(0), orig
    return z.reshape(z.shape[0], -1), orig


def resample_along(z: torch.Tensor, target_n: int) -> torch.Tensor:
    """
    Linearly resample the chunk/sample axis to `target_n`.

    Used for proportional depth mapping when two models produce a different
    number of latent chunks.
    """
    if target_n < 1:
        raise ValueError("target_n must be >= 1")
    if z.shape[0] == target_n:
        return z

    flat, orig = _as_ncd(z)
    # (1, C, N) for interpolate along last dim
    seq = flat.T.unsqueeze(0)
    resampled = F.interpolate(seq, size=target_n, mode="linear", align_corners=True)
    out = resampled.squeeze(0).T
    if len(orig) > 2:
        feat = orig[1:]
        return out.reshape(target_n, *feat)
    if len(orig) == 1:
        return out.reshape(target_n)
    return out.reshape(target_n, orig[1])


def match_feature_dim(z: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pad or interpolate the last dimension to `target_dim`."""
    if z.shape[-1] == target_dim:
        return z
    leading = z.shape[:-1]
    seq = z.reshape(-1, 1, z.shape[-1])
    out = F.interpolate(seq, size=target_dim, mode="linear", align_corners=True)
    return out.reshape(*leading, target_dim)


def map_to_target_shape(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Map source latents onto the target tensor's (num_chunks, feature) grid.

    This is the paper's dimensionality-matching / proportional mapping step.
    """
    mapped = resample_along(source, target.shape[0])
    if mapped.shape[1:] != target.shape[1:]:
        mapped = match_feature_dim(mapped, target.shape[-1])
        if mapped.shape != target.shape:
            mapped = mapped.reshape(target.shape)
    return mapped
