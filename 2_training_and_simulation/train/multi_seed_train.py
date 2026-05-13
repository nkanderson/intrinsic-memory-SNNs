"""
Multi-seed retraining wrapper around train_fn.train().

Trains the same finalized config N times with different RNG seeds and writes
per-seed and aggregate CSVs.

Why
---
Deep RL training is high-variance across random seeds, and reporting from
a single run can be statistically misleading (Henderson et al., 2018, AAAI).
Retraining each Optuna-selected configuration with multiple seeds
characterizes that variance and lets us report how reliably the
configuration solves the task by the standard CartPole-v1 criterion
(mean reward >= 475 over the final 100 episodes).

Sample size
-----------
Henderson et al. (2018) explicitly decline to specify a recommended number
of seeds and instead argue that more is needed when variance is high.
Colas et al. (2018, arXiv:1806.08295) perform a statistical power analysis
and show that N <= 5 is under-powered for detecting small-to-medium effects
in deep RL — N ~ 20-25 is needed for 80% power on a representative
deep-RL effect size. We default to N = 10 here as a compute-feasible
improvement over the common N = 5 minimum; raise to 20+ for stronger
statistical claims when compute permits.

Reporting
---------
The primary outcome of this stage is the standard CartPole-v1 success
metric: mean reward over the final 100 episodes. We report it three ways
because each is informative at small N:

    1. K of N seeds where final_avg >= 475 — the success-count framing.
       Most direct CartPole-v1 outcome and informative even at small N.
    2. IQM with 95% percentile-bootstrap CI on final_avg across seeds —
       robust aggregate following Agarwal et al. (2021, NeurIPS).
    3. Mean ± std with 95% bootstrap CI — legacy comparability with older
       papers; bootstrap CI is reported alongside std because at small N
       the std is itself a noisy estimate (per Agarwal et al.).

The Optuna selection stage used a different metric (tail-IQM over the last
25% of training within a single run) to discourage selection of
configurations whose apparent success was concentrated in a single late
window. The two metrics serve different purposes and are intentionally
distinct: tail-IQM screens against unstable convergence during search,
while final_avg >= 475 is the established success criterion for individual
model reporting. The summary CSV records tail_iqm_avg_reward per seed as a
diagnostic so configurations whose final_avg clears 475 but whose tail-IQM
is low can still be flagged.

References
----------
- Henderson et al. (2018), AAAI: variance across seeds is large enough that
  single-run results are unreliable.
- Colas, Sigaud, Oudeyer (2018), arXiv:1806.08295: power analysis for
  choosing N; N <= 5 is under-powered for most deep-RL effect sizes.
- Agarwal et al. (2021), NeurIPS: IQM and stratified bootstrap CIs as
  robust aggregates that out-perform mean ± std at small N.

Usage
-----
    python multi_seed_train.py --config configs/optimized-leaky-v2-top1.yaml
    python multi_seed_train.py --config configs/optimized-fractional.yaml \\
        --num-seeds 10 --base-seed 42

Outputs (relative to the training/ directory)
---------------------------------------------
    metrics/multi-seed/<config_name>-seed<N>.csv     # per-episode rows per seed
    metrics/multi-seed/<config_name>-summary.csv     # one row per seed
    models/<config_name>-seed<N>-best.pth            # if --save-best
    models/<config_name>-seed<N>-final.pth           # by default
"""

import argparse
import csv
import random as _random
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


def _iqm(values):
    """Interquartile mean: sort, drop bottom 25% and top 25%, mean the middle 50%.

    For very small N the dropped count may collapse to 0 and IQM degenerates
    to the plain mean — explicitly preserved so callers don't crash.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    drop = n // 4
    middle = sorted_vals[drop : n - drop] or sorted_vals
    return sum(middle) / len(middle)


def bootstrap_ci(values, statistic=_iqm, n_resamples=10000, ci=0.95, rng_seed=0):
    """Percentile bootstrap CI for an arbitrary statistic over a 1-D sample.

    Returns (point_estimate, lo, hi). For N very small the CI is wide; this
    function does not pretend otherwise. Following Agarwal et al. (2021) we
    default the statistic to IQM, which is more robust than the mean for
    small samples in deep RL evaluation.
    """
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None, None
    point = statistic(clean)
    rng = _random.Random(rng_seed)
    n = len(clean)
    boots = []
    for _ in range(n_resamples):
        sample = [clean[rng.randrange(n)] for _ in range(n)]
        boots.append(statistic(sample))
    boots.sort()
    alpha = (1.0 - ci) / 2.0
    lo = boots[int(alpha * n_resamples)]
    hi = boots[int((1.0 - alpha) * n_resamples) - 1]
    return point, lo, hi


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "seed",
        "best_avg_reward",
        "final_avg_reward",
        "tail_iqm_avg_reward",
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
        "--num-seeds", type=int, default=10,
        help="Number of seeds to run (default: 10). Colas et al. (2018) show "
             "N <= 5 is under-powered for typical deep RL comparisons; N = 10 "
             "is a compute-feasible improvement, still below their 80%%-power "
             "recommendation for medium effects. Raise to 20+ for stronger "
             "statistical claims.",
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
        per_seed_csv = output_dir / f"{prefix}.csv"
        print(f"\n{'=' * 60}\nSeed {seed} ({i + 1}/{args.num_seeds})\n{'=' * 60}")

        # train_fn.train() streams per-episode rows to per_seed_csv in append
        # mode as each episode completes. This gives live progress visibility
        # (tail -f / wc -l on the CSV) and keeps data durable against kills.
        result = train(
            config=flat,
            device=device,
            verbose=True,
            save_models=save_models,
            save_best_model=args.save_best,
            model_prefix=prefix,
            seed=seed,
            metrics_csv_path=per_seed_csv,
        )

        print(f"  per-episode CSV streamed to: {per_seed_csv}")

        summary_rows.append({
            "seed": seed,
            "best_avg_reward": result["best_avg_reward"],
            "final_avg_reward": result["final_avg_reward"],
            "tail_iqm_avg_reward": result.get("tail_iqm_avg_reward"),
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
    iqm_mean, iqm_std, _ = _stats([r["tail_iqm_avg_reward"] for r in summary_rows])
    conv_mean, conv_std, conv_n = _stats([r["convergence_episode"] for r in summary_rows])

    # Primary CartPole-v1 success criterion: mean reward >= 475 over the
    # final 100 episodes. We report it as a count over seeds (K of N solved)
    # in addition to the mean ± std, since with small N a count is often
    # more informative than an aggregate.
    solved_count = sum(1 for r in summary_rows if r["final_avg_reward"] >= 475.0)

    print(f"\nResults across {len(summary_rows)} seeds:")
    print(
        f"  solved (final_avg >= 475) : {solved_count}/{len(summary_rows)} seeds"
    )
    print(f"  final_avg_reward  : {final_mean:.2f} ± {final_std:.2f}  (primary metric: CartPole-v1 standard)")
    print(f"  best_avg_reward   : {best_mean:.2f} ± {best_std:.2f}")
    if iqm_mean is not None:
        print(f"  tail_iqm_reward   : {iqm_mean:.2f} ± {iqm_std:.2f}  (diagnostic: matches Optuna selection metric)")
    if conv_n == 0:
        print(f"  convergence_ep    : never reached threshold in any seed")
    else:
        print(
            f"  convergence_ep    : {conv_mean:.1f} ± {conv_std:.1f}  "
            f"({conv_n}/{len(summary_rows)} seeds converged)"
        )

    # IQM + percentile bootstrap 95% CI on final_avg across seeds, following
    # Agarwal et al. (2021). Reported alongside mean ± std for transparency;
    # IQM and bootstrap CI are the more robust statistics at small N.
    finals = [r["final_avg_reward"] for r in summary_rows]
    iqm_point, iqm_lo, iqm_hi = bootstrap_ci(finals, statistic=_iqm)
    mean_point, mean_lo, mean_hi = bootstrap_ci(finals, statistic=lambda v: sum(v) / len(v))
    print("\nRobust reporting (final_avg_reward across seeds, 95% bootstrap CI):")
    if iqm_point is not None:
        print(f"  IQM   : {iqm_point:.2f}  [95% CI: {iqm_lo:.2f}, {iqm_hi:.2f}]")
        print(f"  Mean  : {mean_point:.2f}  [95% CI: {mean_lo:.2f}, {mean_hi:.2f}]")
    if len(finals) < 10:
        print(
            "  (note: with N < 10, bootstrap CIs and IQM are necessarily wide; "
            "per Colas et al. 2018 consider increasing --num-seeds.)"
        )


if __name__ == "__main__":
    main()
