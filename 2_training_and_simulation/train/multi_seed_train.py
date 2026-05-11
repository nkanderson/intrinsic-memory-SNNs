"""
Multi-seed retraining wrapper around train_fn.train().

Trains the same finalized config N times with different RNG seeds, captures
per-episode metrics for each seed, and writes an aggregate summary CSV with
mean ± std over the seeds.

Why: Deep RL on CartPole is high-variance across seeds. A single training run
is not statistically defensible (Henderson et al., 2018). 5 seeds is the
common minimum for mean ± std reporting.

Usage:
    python multi_seed_train.py --config configs/optimized-leaky-v2-top1.yaml
    python multi_seed_train.py --config configs/optimized-fractional.yaml \\
        --num-seeds 5 --base-seed 42

Outputs (relative to the training/ directory):
    metrics/multi-seed/<config_name>-seed<N>.csv     # per-episode rows per seed
    metrics/multi-seed/<config_name>-summary.csv     # one row per seed (final/best/convergence)
    models/<config_name>-seed<N>-best.pth            # if --save-best
    models/<config_name>-seed<N>-final.pth           # by default
"""

import argparse
import csv
import statistics
from pathlib import Path

import yaml

from train_fn import train


def get_device(hw_acceleration: bool) -> str:
    import torch
    if hw_acceleration and torch.cuda.is_available():
        return "cuda"
    if hw_acceleration and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def flatten_config(nested: dict) -> dict:
    """Flatten the nested {training: {...}, snn: {...}} YAML into the flat
    dict that train_fn.train() expects."""
    flat = {}
    for section in ("training", "snn"):
        if section in nested:
            flat.update(nested[section])
    return flat


def write_per_episode_csv(path: Path, result: dict) -> None:
    """Write one row per episode with the data needed for visualizations."""
    durations = result["episode_durations"]
    losses = result.get("episode_losses", [])
    n = len(durations)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "episode_steps",
            "running_avg_100",
            "avg_loss",
        ])
        for i in range(n):
            window = durations[max(0, i - 99): i + 1]
            running = sum(window) / len(window)
            loss = losses[i] if i < len(losses) else ""
            writer.writerow([i, durations[i], f"{running:.4f}", loss])


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "seed",
        "best_avg_reward",
        "final_avg_reward",
        "convergence_episode",
        "total_episodes",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Train the same config across multiple RNG seeds."
    )
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to YAML config (nested training/snn sections).",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=5,
        help="Number of seeds to run (default: 5; per Henderson et al., 2018).",
    )
    parser.add_argument(
        "--base-seed", type=int, default=42,
        help="First seed; subsequent seeds are base+1, base+2, ... (default: 42).",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=None,
        help="Override num_episodes from config (useful for smoke tests).",
    )
    parser.add_argument(
        "--save-best", action="store_true",
        help="Also save a -seedN-best.pth checkpoint per seed.",
    )
    parser.add_argument(
        "--no-save-final", action="store_true",
        help="Do NOT save -seedN-final.pth checkpoints.",
    )
    parser.add_argument(
        "--no-hw-acceleration", dest="hw_acceleration", action="store_false",
        help="Force CPU (disable CUDA/MPS).",
    )
    parser.set_defaults(hw_acceleration=True)
    parser.add_argument(
        "--output-dir", type=str, default="metrics/multi-seed",
        help="Directory for per-seed and summary CSVs.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config_name = config_path.stem
    with open(config_path) as f:
        nested = yaml.safe_load(f)
    flat = flatten_config(nested)
    if args.num_episodes is not None:
        flat["num_episodes"] = args.num_episodes

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.hw_acceleration)
    save_models = not args.no_save_final or args.save_best

    print(f"Multi-seed training for: {config_name}")
    print(f"  device       : {device}")
    print(f"  num_seeds    : {args.num_seeds}")
    print(f"  base_seed    : {args.base_seed}")
    print(f"  num_episodes : {flat['num_episodes']}")
    print(f"  output_dir   : {output_dir}")

    summary_rows = []
    for i in range(args.num_seeds):
        seed = args.base_seed + i
        prefix = f"{config_name}-seed{seed}"
        print(f"\n{'=' * 60}\nSeed {seed} ({i + 1}/{args.num_seeds})\n{'=' * 60}")

        result = train(
            config=flat,
            device=device,
            verbose=True,
            save_models=save_models,
            save_best_model=args.save_best,
            model_prefix=prefix,
            seed=seed,
        )

        per_seed_csv = output_dir / f"{prefix}.csv"
        write_per_episode_csv(per_seed_csv, result)
        print(f"  wrote per-episode CSV: {per_seed_csv}")

        summary_rows.append({
            "seed": seed,
            "best_avg_reward": result["best_avg_reward"],
            "final_avg_reward": result["final_avg_reward"],
            "convergence_episode": result.get("convergence_episode"),
            "total_episodes": result["total_episodes"],
        })

    summary_csv = output_dir / f"{config_name}-summary.csv"
    write_summary_csv(summary_csv, summary_rows)
    print(f"\nSummary CSV: {summary_csv}")

    def _stats(values):
        clean = [v for v in values if v is not None]
        if not clean:
            return None, None, 0
        if len(clean) == 1:
            return clean[0], 0.0, 1
        return statistics.mean(clean), statistics.stdev(clean), len(clean)

    final_mean, final_std, _ = _stats([r["final_avg_reward"] for r in summary_rows])
    best_mean, best_std, _ = _stats([r["best_avg_reward"] for r in summary_rows])
    conv_mean, conv_std, conv_n = _stats([r["convergence_episode"] for r in summary_rows])

    print(f"\nResults across {len(summary_rows)} seeds:")
    print(f"  final_avg_reward : {final_mean:.2f} ± {final_std:.2f}")
    print(f"  best_avg_reward  : {best_mean:.2f} ± {best_std:.2f}")
    if conv_n == 0:
        print(f"  convergence_ep   : never reached threshold in any seed")
    else:
        print(
            f"  convergence_ep   : {conv_mean:.1f} ± {conv_std:.1f}  "
            f"({conv_n}/{len(summary_rows)} seeds converged)"
        )


if __name__ == "__main__":
    main()
