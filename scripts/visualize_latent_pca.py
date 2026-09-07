#!/usr/bin/env python3
"""Visualize a trusted, locally saved LatentBundle or bundle-like ``.pt`` file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization import visualize_latent_pca


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA of ordered latent weight chunks")
    parser.add_argument("latent_file", type=Path, help="Trusted torch-saved LatentBundle or dict")
    parser.add_argument("--output-dir", type=Path, default=Path("results/latent_visualization"))
    args = parser.parse_args()

    artifact = torch.load(args.latent_file, map_location="cpu", weights_only=False)
    get = (lambda key, default=None: artifact.get(key, default)) if isinstance(artifact, dict) else (
        lambda key, default=None: getattr(artifact, key, default)
    )
    paths = visualize_latent_pca(
        get("mu") if get("mu") is not None else get("latents"),
        get("metadata", []),
        args.output_dir,
        source_id=get("source_id", args.latent_file.stem),
        total_numel=get("total_numel"),
        chunk_size=int(get("chunks").shape[1]) if get("chunks") is not None else None,
    )
    print("Saved " + ", ".join(str(path) for path in paths.values()))


if __name__ == "__main__":
    main()
