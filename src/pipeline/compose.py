from __future__ import annotations

from typing import Dict, Sequence

from src.align.aligner import AlignConfig, merge_aligned_latents
from src.merge.merger import LatentMerger
from src.merge.types import MergeConfig
from src.merge.weight_baselines import task_vector_merge, weight_lerp, weight_uniform_soup


class MergePipeline:
    """Encode → (optional align) → merge → decode."""

    def __init__(self, merger: LatentMerger) -> None:
        self.merger = merger

    def run(
        self,
        state_dicts: Sequence[Dict],
        source_ids: Sequence[str],
        merge_config: MergeConfig,
        align_config: AlignConfig | None = None,
        *,
        scale: float = 1.0,
        target_index: int = 0,
    ) -> Dict:
        align_config = align_config or AlignConfig(enabled=False)
        bundles = [
            self.merger.encode(
                sd,
                source_id=sid,
                scale=scale,
                deterministic=merge_config.deterministic,
            )
            for sd, sid in zip(state_dicts, source_ids)
        ]
        merged = merge_aligned_latents(
            bundles,
            merge_config,
            align_config,
            target_index=target_index,
        )
        return self.merger.decode(merged, bundles[target_index])

    def run_weight_space(
        self,
        state_dicts: Sequence[Dict],
        *,
        method: str = "uniform_soup",
        lam: float = 0.5,
        scaling: float = 1.0,
    ) -> Dict:
        if method == "uniform_soup":
            return weight_uniform_soup(state_dicts)
        if method == "lerp":
            if len(state_dicts) != 2:
                raise ValueError("weight lerp needs exactly two models")
            return weight_lerp(state_dicts[0], state_dicts[1], lam)
        if method == "task_arithmetic":
            return task_vector_merge(state_dicts[0], state_dicts[1:], scaling=scaling)
        raise ValueError(f"Unknown weight-space method: {method}")
