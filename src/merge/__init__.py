from .operations import barycenter, lerp, merge_latents, slerp, uniform_soup, weighted_soup
from .types import LatentBundle, MergeConfig, MergeStrategy
from .weight_baselines import task_vector_merge, weight_lerp, weight_uniform_soup

__all__ = [
    "LatentMerger",
    "LatentBundle",
    "MergeConfig",
    "MergeStrategy",
    "VaeBridge",
    "lerp",
    "slerp",
    "uniform_soup",
    "weighted_soup",
    "barycenter",
    "merge_latents",
    "weight_lerp",
    "weight_uniform_soup",
    "task_vector_merge",
]


def __getattr__(name: str):
    if name == "VaeBridge":
        from .vae_bridge import VaeBridge

        return VaeBridge
    if name == "LatentMerger":
        from .merger import LatentMerger

        return LatentMerger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
