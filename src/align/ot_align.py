from __future__ import annotations

from typing import Tuple

import torch


def _flatten_samples(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    """Treat dim 0 as samples, remaining dims as features."""
    if z.ndim == 1:
        z = z.unsqueeze(0)
    orig_shape = z.shape
    return z.reshape(z.shape[0], -1), orig_shape


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.T)


def _matrix_sqrt(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    """Symmetric PSD square root via eigen-decomposition."""
    matrix = _symmetrize(matrix)
    evals, evecs = torch.linalg.eigh(matrix)
    evals = torch.clamp(evals, min=eps)
    return evecs @ torch.diag(torch.sqrt(evals)) @ evecs.T


def _matrix_inv_sqrt(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    matrix = _symmetrize(matrix)
    evals, evecs = torch.linalg.eigh(matrix)
    evals = torch.clamp(evals, min=eps)
    return evecs @ torch.diag(1.0 / torch.sqrt(evals)) @ evecs.T


def empirical_gaussian(
    z: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and regularized covariance of samples (N, D)."""
    if z.shape[0] < 2:
        mean = z.mean(dim=0)
        dim = z.shape[1]
        cov = torch.eye(dim, dtype=z.dtype, device=z.device) * eps
        return mean, cov

    mean = z.mean(dim=0)
    centered = z - mean
    cov = (centered.T @ centered) / max(z.shape[0] - 1, 1)
    cov = _symmetrize(cov)
    dim = cov.shape[0]
    cov = cov + eps * torch.eye(dim, dtype=z.dtype, device=z.device)
    return mean, cov


def gaussian_ot_map(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Closed-form Gaussian OT (Monge) map parameters (mu_t, A).

    T(z) = mu_t + A (z - mu_s)
    A = Sigma_s^{-1/2} (Sigma_s^{1/2} Sigma_t Sigma_s^{1/2})^{1/2} Sigma_s^{-1/2}
    """
    src_flat, _ = _flatten_samples(source)
    tgt_flat, _ = _flatten_samples(target)
    if src_flat.shape[1] != tgt_flat.shape[1]:
        raise ValueError(
            f"OT requires matching feature dim, got {src_flat.shape[1]} vs {tgt_flat.shape[1]}"
        )

    mu_s, cov_s = empirical_gaussian(src_flat, eps=eps)
    mu_t, cov_t = empirical_gaussian(tgt_flat, eps=eps)

    cov_s_sqrt = _matrix_sqrt(cov_s, eps)
    cov_s_inv_sqrt = _matrix_inv_sqrt(cov_s, eps)
    inner = cov_s_sqrt @ cov_t @ cov_s_sqrt
    inner_sqrt = _matrix_sqrt(inner, eps)
    transform = cov_s_inv_sqrt @ inner_sqrt @ cov_s_inv_sqrt
    return mu_t, transform, mu_s


def apply_gaussian_ot(
    source: torch.Tensor,
    mu_t: torch.Tensor,
    transform: torch.Tensor,
    mu_s: torch.Tensor,
) -> torch.Tensor:
    src_flat, orig_shape = _flatten_samples(source)
    aligned = mu_t + (src_flat - mu_s) @ transform.T
    return aligned.reshape(orig_shape)


def gaussian_ot_align(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Push source latent distribution onto the target Gaussian."""
    mu_t, transform, mu_s = gaussian_ot_map(source, target, eps=eps)
    return apply_gaussian_ot(source, mu_t, transform, mu_s)


def align_mean_std(source: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cheap baseline: match per-feature mean and std (no full covariance)."""
    src_flat, orig_shape = _flatten_samples(source)
    tgt_flat, _ = _flatten_samples(target)
    src_mean = src_flat.mean(dim=0)
    tgt_mean = tgt_flat.mean(dim=0)
    src_std = src_flat.std(dim=0, unbiased=False).clamp_min(eps)
    tgt_std = tgt_flat.std(dim=0, unbiased=False).clamp_min(eps)
    aligned = (src_flat - src_mean) / src_std * tgt_std + tgt_mean
    return aligned.reshape(orig_shape)
