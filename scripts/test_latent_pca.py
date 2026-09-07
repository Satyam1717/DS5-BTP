"""Synthetic, offline test for the latent PCA visualization."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization import visualize_latent_pca


def test_synthetic_latent_pca() -> None:
    latents = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    metadata = [
        {"name": "blocks.0.attn.weight", "offset": 0, "total_numel": 5},
        {"name": "blocks.1.mlp.weight", "offset": 5, "total_numel": 9},
    ]
    with tempfile.TemporaryDirectory() as temporary_directory:
        paths = visualize_latent_pca(
            latents, metadata, temporary_directory, source_id="synthetic", total_numel=14, chunk_size=4
        )
        assert all(path.exists() for path in paths.values())
        with paths["csv"].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 4
        assert [int(row["chunk_index"]) for row in rows] == [0, 1, 2, 3]
        assert rows[1]["tensor_names"] == "blocks.0.attn.weight | blocks.1.mlp.weight"
        result_metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        assert result_metadata["latent_shape"] == [4, 2, 3]
        assert result_metadata["pca_feature_shape"] == [4, 6]
        assert result_metadata["chunk_records"][1]["tensor_count"] == 2


if __name__ == "__main__":
    test_synthetic_latent_pca()
    print("Synthetic latent PCA test passed.")
