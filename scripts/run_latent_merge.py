#!/usr/bin/env python3
"""
Merge two or more model checkpoints.

Examples:
  python scripts/run_latent_merge.py --dry-run
  python scripts/run_latent_merge.py --dry-run --align --strategy lerp --lam 0.1
  python scripts/run_latent_merge.py --space weight --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.align import AlignConfig, align_source_to_target
from src.merge import MergeConfig, MergeStrategy, merge_latents, weight_lerp, weight_uniform_soup
from src.merge.types import LatentBundle
from src.align.aligner import merge_aligned_latents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent-space / weight-space model merge")
    parser.add_argument("--config", default="configs/merge_default.yaml")
    parser.add_argument("--model-a", default=None, help="HF model id or local path")
    parser.add_argument("--model-b", default=None, help="HF model id or local path")
    parser.add_argument("--models", nargs="*", default=None, help="N-way merge")
    parser.add_argument("--strategy", default=None, choices=[s.value for s in MergeStrategy])
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--weights", nargs="*", type=float, default=None)
    parser.add_argument("--vae-config", default=None)
    parser.add_argument("--vae-checkpoint", default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--space", default=None, choices=["latent", "weight"])
    parser.add_argument("--align", action="store_true", help="Enable OT/mean-std alignment")
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument("--align-method", default=None, choices=["none", "mean_std", "ot"])
    parser.add_argument("--target-index", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_align_config(args: argparse.Namespace, cfg: dict) -> AlignConfig:
    block = cfg.get("align") or {}
    enabled = bool(block.get("enabled", False))
    if args.align:
        enabled = True
    if args.no_align:
        enabled = False
    return AlignConfig(
        enabled=enabled,
        method=args.align_method or block.get("method", "ot"),
        interpolate_after=True,
        eps=float(block.get("eps", 1e-4)),
    )


def load_state_dict(model_id: str) -> dict:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model
    return state_dict


def dry_run(strategy: MergeStrategy, lam: float, weights, align_config: AlignConfig) -> None:
    # Homogeneous merge
    a = torch.randn(8, 6)
    b = torch.randn(8, 6)
    merged = merge_latents([a, b], strategy=strategy, lam=lam, weights=weights)
    print(f"[dry-run] homogeneous {strategy.value} shape={tuple(merged.shape)}")

    # Heterogeneous: different chunk counts
    src = torch.randn(12, 6) * 3 + 2
    tgt = torch.randn(8, 6)
    aligned = align_source_to_target(src, tgt, align_config, lam=lam)
    print(
        f"[dry-run] align method={align_config.method} enabled={align_config.enabled} "
        f"src={tuple(src.shape)} -> aligned={tuple(aligned.shape)}"
    )

    dummy_src = LatentBundle(latents=src, chunks=src, source_id="src")
    dummy_tgt = LatentBundle(latents=tgt, chunks=tgt, source_id="tgt")
    cfg = MergeConfig(strategy=strategy, lam=lam, weights=weights)
    het = merge_aligned_latents(
        [dummy_tgt, dummy_src],
        cfg,
        align_config,
        target_index=0,
    )
    print(f"[dry-run] heterogeneous merge shape={tuple(het.shape)}")

    wa = {"w": torch.ones(4, 4), "skip": torch.zeros(2)}
    wb = {"w": torch.zeros(4, 4), "skip": torch.ones(2)}
    soup = weight_uniform_soup([wa, wb])
    lerp_w = weight_lerp(wa, wb, 0.25)
    print(f"[dry-run] weight soup mean={float(soup['w'].mean()):.3f} lerp={float(lerp_w['w'][0, 0]):.3f}")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(ROOT / args.config)

    strategy = MergeStrategy(args.strategy or cfg.get("strategy", "uniform_soup"))
    lam = args.lam if args.lam is not None else float(cfg.get("lam", 0.5))
    weights = args.weights if args.weights is not None else cfg.get("weights")
    scale = args.scale if args.scale is not None else float(cfg.get("scale", 1.0))
    device = args.device or cfg.get("device", "cpu")
    space = args.space or cfg.get("space", "latent")
    target_index = args.target_index if args.target_index is not None else int(cfg.get("target_index", 0))
    align_config = build_align_config(args, cfg)

    if args.dry_run:
        dry_run(strategy, lam, weights, align_config)
        return

    model_ids = args.models
    if model_ids is None:
        if not args.model_a or not args.model_b:
            raise SystemExit("Provide --model-a and --model-b, or --models, or use --dry-run")
        model_ids = [args.model_a, args.model_b]

    state_dicts = []
    source_ids = []
    for idx, model_id in enumerate(model_ids):
        print(f"Loading model {idx + 1}/{len(model_ids)}: {model_id}")
        state_dicts.append(load_state_dict(model_id))
        source_ids.append(f"model_{idx}")

    merge_config = MergeConfig(
        strategy=strategy,
        lam=lam,
        weights=weights,
        deterministic=True,
        device=device,
        target_index=target_index,
    )

    if space == "weight":
        from src.merge.weight_baselines import task_vector_merge, weight_lerp, weight_uniform_soup

        print(f"Merging in weight space: {strategy.value}")
        if strategy == MergeStrategy.LERP:
            merged_state_dict = weight_lerp(state_dicts[0], state_dicts[1], lam)
        elif strategy == MergeStrategy.UNIFORM_SOUP:
            merged_state_dict = weight_uniform_soup(state_dicts)
        else:
            merged_state_dict = task_vector_merge(state_dicts[0], state_dicts[1:])
    else:
        from src.merge import LatentMerger, VaeBridge
        from src.pipeline import MergePipeline

        vae_config = args.vae_config or cfg.get("vae_config")
        vae_checkpoint = args.vae_checkpoint or cfg.get("vae_checkpoint")
        if not vae_config:
            raise SystemExit("vae_config is required for latent merge")

        vae = VaeBridge(
            config_path=ROOT / vae_config,
            checkpoint_path=(ROOT / vae_checkpoint) if vae_checkpoint else None,
            device=device,
        )
        pipeline = MergePipeline(LatentMerger(vae))
        print(
            f"Merging in latent space strategy={strategy.value} "
            f"align={align_config.enabled}/{align_config.method}"
        )
        merged_state_dict = pipeline.run(
            state_dicts,
            source_ids,
            merge_config,
            align_config,
            scale=scale,
            target_index=target_index,
        )

    output_path = Path(args.output or cfg.get("output_dir", "data/checkpoints/merged"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix != ".pt":
        output_path = output_path / "merged_state_dict.pt"

    torch.save(merged_state_dict, output_path)
    print(f"Saved merged state_dict to {output_path}")


if __name__ == "__main__":
    main()
