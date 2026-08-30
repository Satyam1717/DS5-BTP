from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from src.align.aligner import AlignConfig, merge_aligned_latents

from .operations import merge_chunkwise, merge_latents, uniform_soup
from .types import LatentBundle, MergeConfig, MergeStrategy
from .vae_bridge import VaeBridge


class LatentMerger:
    """
    High-level API for latent-space model merging.

    Typical flow:
      1. encode multiple checkpoints/adapters with the same VAE
      2. merge latents with soup / LERP / SLERP
      3. decode merged latents back to weights
    """

    def __init__(self, vae: VaeBridge) -> None:
        self.vae = vae

    def encode(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        source_id: str,
        scale: float = 1.0,
        deterministic: bool = True,
    ) -> LatentBundle:
        return self.vae.encode_state_dict(
            state_dict,
            source_id=source_id,
            scale=scale,
            deterministic=deterministic,
        )

    def merge_bundles(
        self,
        bundles: Sequence[LatentBundle],
        config: MergeConfig,
        align_config: AlignConfig | None = None,
    ) -> torch.Tensor:
        if len(bundles) < 2:
            raise ValueError("Need at least two LatentBundle objects to merge")

        align_config = align_config or AlignConfig(enabled=False)
        if align_config.enabled:
            return merge_aligned_latents(
                bundles,
                config,
                align_config,
                target_index=config.target_index,
            )

        for bundle in bundles:
            if bundle.num_chunks != bundles[0].num_chunks:
                raise ValueError(
                    "All bundles must have the same number of chunks "
                    "(enable AlignConfig for heterogeneous models)"
                )

        latent_tensors = [
            bundle.latent_tensor(deterministic=config.deterministic) for bundle in bundles
        ]

        if config.strategy in {MergeStrategy.LERP, MergeStrategy.SLERP}:
            if len(latent_tensors) != 2:
                raise ValueError(f"{config.strategy.value} requires exactly two bundles")
            return merge_latents(
                latent_tensors,
                strategy=config.strategy,
                lam=config.lam,
                weights=config.weights,
            )

        return merge_chunkwise(
            latent_tensors,
            strategy=config.strategy,
            lam=config.lam,
            weights=config.weights,
        )

    def self_merge(
        self,
        bundle: LatentBundle,
        *,
        n_samples: int = 4,
    ) -> torch.Tensor:
        """
        Sample multiple codes from one model's posterior and soup them.

        Falls back to repeating mu if logvar is unavailable.
        """
        if bundle.mu is None or bundle.logvar is None:
            return bundle.latent_tensor(deterministic=True)

        samples = []
        for _ in range(n_samples):
            std = torch.exp(0.5 * torch.clamp(bundle.logvar, min=-10, max=10))
            z = bundle.mu + torch.randn_like(std) * std
            samples.append(z)
        return uniform_soup(samples)

    def decode(
        self,
        merged_latents: torch.Tensor,
        reference_bundle: LatentBundle,
    ) -> Dict[str, torch.Tensor]:
        return self.vae.decode_bundle(merged_latents, reference_bundle)

    def merge_state_dicts(
        self,
        state_dicts: Sequence[Dict[str, torch.Tensor]],
        source_ids: Sequence[str],
        config: MergeConfig,
        *,
        scale: float = 1.0,
        align_config: AlignConfig | None = None,
    ) -> Dict[str, torch.Tensor]:
        bundles: List[LatentBundle] = []
        for state_dict, source_id in zip(state_dicts, source_ids):
            bundles.append(
                self.encode(
                    state_dict,
                    source_id=source_id,
                    scale=scale,
                    deterministic=config.deterministic,
                )
            )

        merged_latents = self.merge_bundles(bundles, config, align_config=align_config)
        target_index = config.target_index
        return self.decode(merged_latents, bundles[target_index])
