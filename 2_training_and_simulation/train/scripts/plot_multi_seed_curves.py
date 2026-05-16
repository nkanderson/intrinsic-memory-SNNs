"""
Visualizations for multi-seed training runs.

Consumes per-seed CSVs written by multi_seed_train.py
(metrics/multi-seed/<config_name>-seed<N>.csv) plus the corresponding
summary CSV (<config_name>-summary.csv), and produces:

  1. learning_curve            Per-seed running_avg_100 lines + mean ± 1 std
                               band, with a dashed line at y=475 (the
                               CartPole-v1 "solved" threshold).
  2. cross_config              Same as (1) but overlaying the mean ± std
                               bands of multiple configs (use --config-name
                               more than once).
  3. convergence_strip         Strip / scatter plot of convergence_episode
                               per seed, one column per config.
  4. final_bars                Bar chart: mean final_avg_reward (and
                               best_avg_reward) across seeds, error bar = std.
  5. loss_curves               Mean ± std avg_loss per episode.
  6. performance_profile       Dolan-Moré curve: fraction of seeds scoring
                               at or above each threshold. Following
                               Agarwal et al. (2021), characterizes the full
                               distribution of run scores per config.

Usage:
    # Single config — produces plots 1, 4, 5 (cross-config + strip need 2+)
    python scripts/plot_multi_seed_curves.py \\
        --config-name optimized-leaky-v2-top1

    # Compare three candidates
    python scripts/plot_multi_seed_curves.py \\
        --config-name optimized-leaky-v2-top1 \\
        --config-name optimized-leaky-v2-top2 \\
        --config-name optimized-leaky-v2-top3

References:
    Henderson et al. (2018), "Deep Reinforcement Learning that Matters" —
    motivates per-seed bands and the mean ± std reporting.
    Agarwal et al. (2021), "Deep RL at the Edge of the Statistical
    Precipice" — recommends interquartile mean and bootstrap CIs (an
    extension; here we keep mean ± std for legibility).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reporting threshold for plots: the standard CartPole-v1 solved criterion
# (mean reward >= 475 over 100 consecutive episodes). This is the value
# the field compares against — not the Optuna selection metric. The Optuna
# stage used tail-IQM over the last 25% of training as the objective to
# discourage selecting configs whose apparent success was localized to a
# single late window. Multi-seed retraining and the plots produced here
# report against the standard 475 threshold instead.
SUCCESS_THRESHOLD = 475.0


def _config_dir(metrics_dir: Path, config_name: str) -> Path:
    """Resolve the per-config subdirectory under metrics_dir, falling back
    to the metrics_dir itself for backward compatibility with the old
    flat layout (where files lived directly in metrics/multi-seed/).
    """
    nested = metrics_dir / config_name
    if nested.is_dir() and any(nested.glob(f"{config_name}-seed*.csv")):
        return nested
    return metrics_dir


def load_per_seed_csvs(metrics_dir: Path, config_name: str):
    """Return (seeds, episodes, running_avg_100_matrix, avg_loss_matrix).

    The matrices are shape (num_seeds, num_episodes). Rows are padded with
    NaN if seeds have different episode counts (shouldn't happen in
    practice but keeps the plotting robust).

    Looks under metrics_dir/<config_name>/ first (the nested layout
    written by multi_seed_train.py); falls back to metrics_dir/ for older
    runs that used the flat layout.
    """
    config_dir = _config_dir(metrics_dir, config_name)
    pattern = f"{config_name}-seed*.csv"
    files = sorted(config_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No per-seed CSVs found at {config_dir / pattern}")

    seeds = []
    runs_running = []
    runs_loss = []
    for path in files:
        seed = int(path.stem.rsplit("-seed", 1)[1])
        seeds.append(seed)
        running, losses = [], []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                running.append(float(row["running_avg_100"]))
                loss = row.get("avg_loss", "")
                losses.append(float(loss) if loss else np.nan)
        runs_running.append(running)
        runs_loss.append(losses)

    max_len = max(len(r) for r in runs_running)
    running_mat = np.full((len(seeds), max_len), np.nan)
    loss_mat = np.full((len(seeds), max_len), np.nan)
    for i, (r, l) in enumerate(zip(runs_running, runs_loss)):
        running_mat[i, : len(r)] = r
        loss_mat[i, : len(l)] = l

    episodes = np.arange(max_len)
    return seeds, episodes, running_mat, loss_mat


def load_summary(metrics_dir: Path, config_name: str) -> list[dict]:
    config_dir = _config_dir(metrics_dir, config_name)
    path = config_dir / f"{config_name}-summary.csv"
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "best_avg_reward": float(row["best_avg_reward"]),
                    "final_avg_reward": float(row["final_avg_reward"]),
                    "convergence_episode": (
                        int(row["convergence_episode"])
                        if row["convergence_episode"] not in ("", "None")
                        else None
                    ),
                }
            )
    return rows


def plot_learning_curve(
    config_name: str,
    seeds,
    episodes,
    running_mat,
    output_dir: Path,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, seed in enumerate(seeds):
        ax.plot(
            episodes, running_mat[i], alpha=0.3, linewidth=0.8, label=f"seed {seed}"
        )

    mean = np.nanmean(running_mat, axis=0)
    std = np.nanstd(running_mat, axis=0)
    ax.plot(episodes, mean, color="black", linewidth=2.0, label="mean")
    ax.fill_between(
        episodes, mean - std, mean + std, color="black", alpha=0.15, label="± 1 std"
    )
    ax.axhline(
        SUCCESS_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label=f"solved ({SUCCESS_THRESHOLD:.0f})",
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Trailing 100-ep avg reward")
    ax.set_title(f"Learning curve — {config_name}  ({len(seeds)} seeds)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = output_dir / f"{config_name}-learning_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_cross_config(configs, episodes_by_cfg, mat_by_cfg, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg in configs:
        episodes = episodes_by_cfg[cfg]
        mat = mat_by_cfg[cfg]
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        (line,) = ax.plot(episodes, mean, linewidth=2.0, label=cfg)
        ax.fill_between(
            episodes, mean - std, mean + std, color=line.get_color(), alpha=0.15
        )
    ax.axhline(
        SUCCESS_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label=f"solved ({SUCCESS_THRESHOLD:.0f})",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Trailing 100-ep avg reward")
    ax.set_title("Cross-config comparison (mean ± std across seeds)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / "cross_config-learning_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_convergence_strip(configs, summary_by_cfg, output_dir: Path):
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(configs)), 5))
    rng = np.random.default_rng(0)
    for x, cfg in enumerate(configs):
        rows = summary_by_cfg[cfg]
        ys = [
            r["convergence_episode"]
            for r in rows
            if r["convergence_episode"] is not None
        ]
        # jitter for visibility
        xs = x + rng.uniform(-0.08, 0.08, size=len(ys))
        ax.scatter(xs, ys, alpha=0.8, s=50)
        n_conv = len(ys)
        n_total = len(rows)
        ax.text(
            x,
            ax.get_ylim()[0] if ys else 0,
            f"{n_conv}/{n_total}",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(
        "Convergence episode (first ep with trailing avg ≥ 475, never drops below)"
    )
    ax.set_title("Convergence-episode distribution by config")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = output_dir / "convergence_strip.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_final_bars(configs, summary_by_cfg, output_dir: Path):
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(configs)), 5))
    width = 0.35
    xs = np.arange(len(configs))

    final_means, final_stds, best_means, best_stds = [], [], [], []
    for cfg in configs:
        rows = summary_by_cfg[cfg]
        finals = np.array([r["final_avg_reward"] for r in rows])
        bests = np.array([r["best_avg_reward"] for r in rows])
        final_means.append(finals.mean())
        final_stds.append(finals.std(ddof=1) if len(finals) > 1 else 0.0)
        best_means.append(bests.mean())
        best_stds.append(bests.std(ddof=1) if len(bests) > 1 else 0.0)

    ax.bar(
        xs - width / 2,
        final_means,
        width,
        yerr=final_stds,
        capsize=4,
        label="final_avg",
    )
    ax.bar(
        xs + width / 2, best_means, width, yerr=best_stds, capsize=4, label="best_avg"
    )
    ax.axhline(SUCCESS_THRESHOLD, color="red", linestyle="--", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Reward (mean ± std across seeds)")
    ax.set_title("Final / best 100-ep avg reward by config")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = output_dir / "final_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_performance_profile(configs, summary_by_cfg, output_dir: Path):
    """Empirical performance profile (a Dolan-Moré curve) per configuration.

    x: score threshold τ ∈ [0, 500]
    y: fraction of seeds achieving final_avg_reward ≥ τ

    Per Agarwal et al. (2021), this characterizes the full distribution of
    run scores rather than collapsing it to a single aggregate; configurations
    whose curves stay flat longer (rightward shift) are uniformly better
    across the seed distribution. The vertical dashed line at τ = 475 marks
    the CartPole-v1 solved threshold; the y-value at that line is the
    'fraction of seeds that solved the task' for each configuration.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    thresholds = np.linspace(0, 500, 501)
    for cfg in configs:
        rows = summary_by_cfg[cfg]
        scores = np.array([r["final_avg_reward"] for r in rows])
        # fraction of seeds with score >= τ, for each τ
        fractions = [(scores >= t).mean() for t in thresholds]
        ax.plot(thresholds, fractions, linewidth=2.0, label=cfg)
    ax.axvline(
        SUCCESS_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label=f"solved ({SUCCESS_THRESHOLD:.0f})",
    )
    ax.set_xlabel("Score threshold τ (final 100-ep avg reward)")
    ax.set_ylabel("Fraction of seeds with score ≥ τ")
    ax.set_title("Performance profile across seeds")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / "performance_profile.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_loss_curves(config_name, episodes, loss_mat, output_dir: Path):
    if np.isnan(loss_mat).all():
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    mean = np.nanmean(loss_mat, axis=0)
    std = np.nanstd(loss_mat, axis=0)
    ax.plot(episodes, mean, color="C0", linewidth=2.0, label="mean")
    ax.fill_between(
        episodes, mean - std, mean + std, color="C0", alpha=0.2, label="± 1 std"
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg DQN loss per episode")
    ax.set_title(f"Loss curve — {config_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / f"{config_name}-loss_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Plot multi-seed training curves and summaries."
    )
    parser.add_argument(
        "--config-name",
        action="append",
        required=True,
        help="Config name (CSV stem). Repeatable for cross-config plots.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="metrics/multi-seed",
        help="Directory containing the per-seed and summary CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="images/multi-seed",
        help="Directory to write plots into.",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = args.config_name
    episodes_by_cfg = {}
    mat_by_cfg = {}
    summary_by_cfg = {}

    for cfg in configs:
        seeds, episodes, running_mat, loss_mat = load_per_seed_csvs(metrics_dir, cfg)
        episodes_by_cfg[cfg] = episodes
        mat_by_cfg[cfg] = running_mat
        summary_by_cfg[cfg] = load_summary(metrics_dir, cfg)

        out = plot_learning_curve(cfg, seeds, episodes, running_mat, output_dir)
        print(f"  wrote {out}")
        out = plot_loss_curves(cfg, episodes, loss_mat, output_dir)
        if out:
            print(f"  wrote {out}")

    if len(configs) >= 2:
        out = plot_cross_config(configs, episodes_by_cfg, mat_by_cfg, output_dir)
        print(f"  wrote {out}")
        out = plot_convergence_strip(configs, summary_by_cfg, output_dir)
        print(f"  wrote {out}")

    out = plot_final_bars(configs, summary_by_cfg, output_dir)
    print(f"  wrote {out}")

    # Performance profile (per Agarwal et al., 2021): shows the full
    # distribution of seed scores per config rather than collapsing to mean.
    out = plot_performance_profile(configs, summary_by_cfg, output_dir)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
