"""Tests for merge ops, OT alignment, and heterogeneous mapping (no VAE)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.align import AlignConfig, align_source_to_target, gaussian_ot_align, map_to_target_shape
from src.align.aligner import merge_aligned_latents
from src.merge.operations import lerp, merge_latents, uniform_soup
from src.merge.types import LatentBundle, MergeConfig, MergeStrategy
from src.merge.weight_baselines import task_vector_merge, weight_lerp, weight_uniform_soup


def test_uniform_soup() -> None:
    a = torch.ones(4, 8)
    b = torch.zeros(4, 8)
    c = torch.full((4, 8), 2.0)
    out = uniform_soup([a, b, c])
    assert torch.allclose(out, torch.full((4, 8), 1.0))


def test_lerp() -> None:
    a = torch.zeros(3)
    b = torch.ones(3)
    out = lerp(a, b, 0.25)
    assert torch.allclose(out, torch.full((3,), 0.25))


def test_merge_latents_dispatch() -> None:
    a = torch.randn(2, 4)
    b = torch.randn(2, 4)
    out = merge_latents([a, b], MergeStrategy.LERP, lam=0.5)
    assert out.shape == a.shape


def test_map_to_target_shape() -> None:
    source = torch.randn(12, 5)
    target = torch.randn(8, 5)
    mapped = map_to_target_shape(source, target)
    assert mapped.shape == target.shape


def test_gaussian_ot_mean_match() -> None:
    torch.manual_seed(0)
    target = torch.randn(64, 4)
    source = torch.randn(64, 4) * 2.5 + 3.0
    aligned = gaussian_ot_align(source, target)
    assert aligned.shape == source.shape
    assert torch.allclose(aligned.mean(0), target.mean(0), atol=1e-3)


def test_align_then_interpolate() -> None:
    torch.manual_seed(1)
    src = torch.randn(16, 4) + 2
    tgt = torch.randn(10, 4)
    cfg = AlignConfig(enabled=True, method="ot", interpolate_after=True)
    out = align_source_to_target(src, tgt, cfg, lam=0.1)
    assert out.shape == tgt.shape
    # λ=0.1 should stay close to target
    assert (out - tgt).abs().mean() < (src[:10] - tgt).abs().mean()


def test_merge_aligned_heterogeneous() -> None:
    src = LatentBundle(latents=torch.randn(12, 6), chunks=torch.randn(12, 6), source_id="src")
    tgt = LatentBundle(latents=torch.randn(8, 6), chunks=torch.randn(8, 6), source_id="tgt")
    merged = merge_aligned_latents(
        [tgt, src],
        MergeConfig(strategy=MergeStrategy.LERP, lam=0.1),
        AlignConfig(enabled=True, method="ot"),
        target_index=0,
    )
    assert merged.shape == tgt.latents.shape


def test_weight_baselines() -> None:
    a = {"w": torch.ones(3, 3), "bias": torch.zeros(3)}
    b = {"w": torch.zeros(3, 3), "bias": torch.ones(3)}
    soup = weight_uniform_soup([a, b])
    assert torch.allclose(soup["w"], torch.full((3, 3), 0.5))
    lerped = weight_lerp(a, b, 0.25)
    assert torch.allclose(lerped["w"], torch.full((3, 3), 0.75))
    arith = task_vector_merge(a, [b])
    assert arith["w"].shape == a["w"].shape


if __name__ == "__main__":
    test_uniform_soup()
    test_lerp()
    test_merge_latents_dispatch()
    test_map_to_target_shape()
    test_gaussian_ot_mean_match()
    test_align_then_interpolate()
    test_merge_aligned_heterogeneous()
    test_weight_baselines()
    print("All merge + alignment tests passed.")
