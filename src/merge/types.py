from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch


class MergeStrategy(str, Enum):
    """Supported latent-space merge operators."""

    LERP = "lerp"
    UNIFORM_SOUP = "uniform_soup"
    WEIGHTED_SOUP = "weighted_soup"
    SLERP = "slerp"
    BARYCENTER = "barycenter"


@dataclass
class LatentBundle:
    """
  Container for encoded weight chunks and bookkeeping needed to decode back.

  latents: tensor produced by the VAE encoder. Shape is typically
           (num_chunks, latent_dim) or (num_chunks, 1, latent_dim).
  chunks:  original weight chunks fed to the encoder, used for shape checks.
  mask:    optional padding mask aligned with chunks.
  mu/logvar: VAE moments when deterministic merging is preferred.
  metadata/total_numel: required to rebuild a HuggingFace state_dict.
  scale:   divisor applied during dataset loading in ls-merge training.
  """

    latents: torch.Tensor
    chunks: torch.Tensor
    source_id: str
    mask: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    total_numel: int = 0
    scale: float = 1.0
    original_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def latent_tensor(self, deterministic: bool = True) -> torch.Tensor:
        """Return latents used for merging."""
        if deterministic and self.mu is not None:
            return self.mu
        return self.latents

    @property
    def num_chunks(self) -> int:
        return int(self.latents.shape[0])


@dataclass
class MergeConfig:
    """Configuration for a latent merge run."""

    strategy: MergeStrategy = MergeStrategy.UNIFORM_SOUP
    lam: float = 0.5
    weights: Optional[List[float]] = None
    deterministic: bool = True
    device: str = "cpu"
    target_index: int = 0
