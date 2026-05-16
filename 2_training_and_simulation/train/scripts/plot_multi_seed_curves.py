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
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Path setup for shared modules ──────────────────────────────────────────
# Shared plot style module lives at <project_root>/common/scripts/plot_styles.py.
# multi_seed_train helpers live at <project_root>/2_training_and_simulation/train/.
ROOT_DIR = Path(__file__).resolve().parents[3]
TRAIN_DIR = Path(__file__).resolve().parents[1]
for _p in (ROOT_DIR, TRAIN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.scripts.plot_styles import (  # noqa: E402
    OKABE_ITO,
    COLOR_RAW,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    DEFAULT_FIGSIZE,
)
from multi_seed_train import _iqm, bootstrap_ci  # noqa: E402

# Reporting threshold for plots: the standard CartPole-v1 solved criterion
# (mean reward >= 475 over 100 consecutive episodes). This is the value
# the field compares against — not the Optuna selection metric. The Optuna
# stage used tail-IQM over the last 25% of training as the objective to
# discourage selecting configs whose apparent success was localized to a
# single late window. Multi-seed retraining and the plots produced here
# report against the standard 475 threshold instead.
SUCCESS_THRESHOLD = 475.0

# Semantic colors used throughout the plots, drawn from the Okabe-Ito
# colorblind-safe palette. Per-config series rotate through OKABE_ITO[0..7].
COLOR_THRESHOLD = OKABE_ITO[1]  # vermillion for the solved-threshold reference


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


def _per_episode_band(mat: np.ndarray, pct_lo: float = 25.0, pct_hi: float = 75.0):
    """Compute per-episode IQM (central) and IQR band edges across seeds.

    For each column (episode) of ``mat`` (shape ``[n_seeds, n_episodes]``):
      - IQM: mean of the middle 50% of seeds after sorting.
      - Band: 25th and 75th percentile of seed values at that episode.

    NaN entries are excluded per-column. Designed to be cheap (no bootstrap)
    so it scales to thousands of episodes; we use bootstrap CIs only for
    single-value statistics (e.g., the final-reward bars).
    """
    n_eps = mat.shape[1]
    iqm = np.full(n_eps, np.nan)
    lo = np.full(n_eps, np.nan)
    hi = np.full(n_eps, np.nan)
    for i in range(n_eps):
        col = mat[:, i]
        col = col[~np.isnan(col)]
        if col.size == 0:
            continue
        sorted_col = np.sort(col)
        drop = sorted_col.size // 4
        middle = sorted_col[drop : sorted_col.size - drop] if drop > 0 else sorted_col
        iqm[i] = middle.mean()
        lo[i] = np.percentile(col, pct_lo)
        hi[i] = np.percentile(col, pct_hi)
    return iqm, lo, hi


def _solved_count(summary_rows: list[dict]) -> tuple[int, int]:
    """Return (K, N): seeds reaching the success threshold, and total."""
    n = len(summary_rows)
    k = sum(1 for r in summary_rows if r["final_avg_reward"] >= SUCCESS_THRESHOLD)
    return k, n


def _savefig(fig, out_path: Path, fmt: str) -> Path:
    """Save a figure with format-appropriate options. Vector formats omit dpi."""
    if fmt == "png":
        fig.savefig(out_path, dpi=150)
    else:
        fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_learning_curve(
    config_name: str,
    seeds,
    episodes,
    running_mat,
    summary_rows: list[dict],
    output_dir: Path,
    fmt: str,
):
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, constrained_layout=True)

    color_central = OKABE_ITO[0]

    # Individual seed lines — single combined legend entry rather than one
    # per seed (avoids a 10+ item legend).
    for i, _seed in enumerate(seeds):
        label = f"individual seeds (N={len(seeds)})" if i == 0 else None
        ax.plot(
            episodes,
            running_mat[i],
            alpha=0.4,
            linewidth=0.8,
            color=COLOR_RAW,
            label=label,
        )

    # IQM + IQR band — matches the Agarwal-style reporting (robust to outliers).
    iqm, lo, hi = _per_episode_band(running_mat)
    ax.plot(episodes, iqm, color=color_central, linewidth=2.0, label="IQM")
    ax.fill_between(
        episodes, lo, hi, color=color_central, alpha=0.2, label="IQR (25–75th %ile)"
    )

    ax.axhline(
        SUCCESS_THRESHOLD,
        color=COLOR_THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label=f"solved ({SUCCESS_THRESHOLD:.0f})",
    )

    # K/N solved annotation, top-left so it doesn't fight the curve.
    k, n = _solved_count(summary_rows)
    ax.text(
        0.02,
        0.95,
        f"solved: {k}/{n} seeds",
        transform=ax.transAxes,
        fontsize=LEGEND_FONTSIZE,
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
        ),
    )

    ax.set_xlabel("Episode", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Trailing 100-ep avg reward", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3)

    return _savefig(fig, output_dir / f"{config_name}-learning_curve.{fmt}", fmt)


def plot_cross_config(
    configs,
    episodes_by_cfg,
    mat_by_cfg,
    summary_by_cfg,
    output_dir: Path,
    fmt: str,
):
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, constrained_layout=True)

    for idx, cfg in enumerate(configs):
        color = OKABE_ITO[idx % len(OKABE_ITO)]
        episodes = episodes_by_cfg[cfg]
        mat = mat_by_cfg[cfg]
        iqm, lo, hi = _per_episode_band(mat)
        k, n = _solved_count(summary_by_cfg[cfg])
        ax.plot(
            episodes,
            iqm,
            linewidth=2.0,
            color=color,
            label=f"{cfg}  [solved {k}/{n}]",
        )
        ax.fill_between(episodes, lo, hi, color=color, alpha=0.2)

    ax.axhline(
        SUCCESS_THRESHOLD,
        color=COLOR_THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label=f"solved ({SUCCESS_THRESHOLD:.0f})",
    )
    ax.set_xlabel("Episode", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Trailing 100-ep avg reward", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3)
    return _savefig(fig, output_dir / f"cross_config-learning_curves.{fmt}", fmt)


# Shared y-axis configuration for the stacked cross-config plot: every panel
# uses the same limits/ticks so that vertical comparison across configs is
# honest, even when one config never approaches 500. Matches the convention
# used by visualize_training_metrics.plot_comparison_stack_running.
STACK_Y_LIMITS = (-25, 525)
STACK_Y_TICKS = [100, 300, 500]


def _stack_panel_label(ax, label: str) -> None:
    """Place the config name in the upper-left corner of a stacked panel."""
    ax.text(
        0.012, 0.92, label,
        transform=ax.transAxes,
        fontsize=LEGEND_FONTSIZE, fontweight="bold",
        va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
    )


def plot_cross_config_stacked(
    configs,
    episodes_by_cfg,
    mat_by_cfg,
    summary_by_cfg,
    output_dir: Path,
    fmt: str,
):
    """Per-config stacked panels, one row per config.

    Each panel shows the same content as `plot_learning_curve` (per-seed
    faint lines, IQM line, IQR band, threshold line) on a shared y-axis so
    vertical comparison across configs is direct. Mirrors the
    `plot_comparison_stack_running` layout in visualize_training_metrics.py.
    """
    n = len(configs)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(DEFAULT_FIGSIZE[0], 1.9 * n + 1.0),
        sharex=True,
        constrained_layout=True,
    )
    if n == 1:
        axes = [axes]

    for idx, (cfg, ax) in enumerate(zip(configs, axes)):
        color = OKABE_ITO[idx % len(OKABE_ITO)]
        episodes = episodes_by_cfg[cfg]
        mat = mat_by_cfg[cfg]
        n_seeds = mat.shape[0]

        for i in range(n_seeds):
            ax.plot(
                episodes, mat[i],
                alpha=0.35, linewidth=0.7,
                color=COLOR_RAW, zorder=1,
            )

        iqm, lo, hi = _per_episode_band(mat)
        ax.fill_between(episodes, lo, hi, color=color, alpha=0.2, zorder=2)
        ax.plot(episodes, iqm, color=color, linewidth=1.8, zorder=3)

        ax.axhline(
            SUCCESS_THRESHOLD,
            color=COLOR_THRESHOLD,
            linestyle="--",
            linewidth=0.9,
            zorder=2,
        )

        k, n_total = _solved_count(summary_by_cfg[cfg])
        _stack_panel_label(ax, f"{cfg}  [solved {k}/{n_total}]")

        ax.set_ylim(*STACK_Y_LIMITS)
        ax.set_yticks(STACK_Y_TICKS)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Reward", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    axes[-1].set_xlabel("Episode", fontsize=AXIS_LABEL_FONTSIZE)

    # Shared legend in the bottom panel (every panel shares the same
    # encoding, so a single legend suffices). Use neutral gray for the
    # IQM/band entries since each panel uses its own config color.
    NEUTRAL = "#555555"
    legend_handles = [
        Line2D([0], [0], color=COLOR_RAW, linewidth=0.8, alpha=0.6,
               label="Individual seeds"),
        Line2D([0], [0], color=NEUTRAL, linewidth=1.8, label="IQM"),
        Patch(facecolor=NEUTRAL, alpha=0.2, label="IQR (25–75th %ile)"),
        Line2D([0], [0], color=COLOR_THRESHOLD, linestyle="--", linewidth=0.9,
               label=f"solved ({SUCCESS_THRESHOLD:.0f})"),
    ]
    axes[-1].legend(
        handles=legend_handles, loc="lower right",
        fontsize=LEGEND_FONTSIZE, framealpha=0.9,
    )

    return _savefig(
        fig, output_dir / f"cross_config-learning_curves_stacked.{fmt}", fmt
    )


def plot_convergence_strip(configs, summary_by_cfg, output_dir: Path, fmt: str):
    fig_w = max(DEFAULT_FIGSIZE[0], 1.5 * len(configs))
    fig, ax = plt.subplots(figsize=(fig_w, DEFAULT_FIGSIZE[1]), constrained_layout=True)
    rng = np.random.default_rng(0)
    for x, cfg in enumerate(configs):
        color = OKABE_ITO[x % len(OKABE_ITO)]
        rows = summary_by_cfg[cfg]
        ys = [
            r["convergence_episode"]
            for r in rows
            if r["convergence_episode"] is not None
        ]
        if ys:
            xs = x + rng.uniform(-0.08, 0.08, size=len(ys))
            ax.scatter(xs, ys, alpha=0.8, s=50, color=color)
            # median bar for the converged subset
            median_y = float(np.median(ys))
            ax.plot(
                [x - 0.18, x + 0.18],
                [median_y, median_y],
                color=color,
                linewidth=2.0,
            )
        n_conv = len(ys)
        n_total = len(rows)
        # K/N annotation just inside the bottom of the data area so it's
        # always visible regardless of the y-axis range (data coords for x,
        # axes-fraction for y).
        ax.text(
            x,
            0.06,
            f"{n_conv}/{n_total}",
            ha="center",
            va="bottom",
            fontsize=LEGEND_FONTSIZE,
            transform=ax.get_xaxis_transform(),
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="none",
                alpha=0.9,
            ),
        )
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=TICK_LABEL_FONTSIZE)
    ax.set_ylabel("Convergence episode", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3, axis="y")
    return _savefig(fig, output_dir / f"convergence_strip.{fmt}", fmt)


def plot_final_bars(configs, summary_by_cfg, output_dir: Path, fmt: str):
    """Per-config bar plot of final_avg / best_avg IQM with bootstrap CIs.

    Bar heights use the Interquartile Mean across seeds (matches the
    Agarwal-style reporting). Asymmetric error bars show the 95% bootstrap
    CI for the IQM, computed from the per-seed values.
    """
    fig_w = max(DEFAULT_FIGSIZE[0], 1.5 * len(configs))
    fig, ax = plt.subplots(figsize=(fig_w, DEFAULT_FIGSIZE[1]), constrained_layout=True)
    width = 0.35
    xs = np.arange(len(configs))

    final_iqms, final_err_lo, final_err_hi = [], [], []
    best_iqms, best_err_lo, best_err_hi = [], [], []
    ks: list[int] = []
    ns: list[int] = []
    for cfg in configs:
        rows = summary_by_cfg[cfg]
        finals = [r["final_avg_reward"] for r in rows]
        bests = [r["best_avg_reward"] for r in rows]

        f_iqm, f_lo, f_hi = bootstrap_ci(finals, statistic=_iqm)
        b_iqm, b_lo, b_hi = bootstrap_ci(bests, statistic=_iqm)
        final_iqms.append(f_iqm if f_iqm is not None else 0.0)
        final_err_lo.append((f_iqm - f_lo) if f_lo is not None else 0.0)
        final_err_hi.append((f_hi - f_iqm) if f_hi is not None else 0.0)
        best_iqms.append(b_iqm if b_iqm is not None else 0.0)
        best_err_lo.append((b_iqm - b_lo) if b_lo is not None else 0.0)
        best_err_hi.append((b_hi - b_iqm) if b_hi is not None else 0.0)

        k, n = _solved_count(rows)
        ks.append(k)
        ns.append(n)

    final_color = OKABE_ITO[0]
    best_color = OKABE_ITO[2]
    ax.bar(
        xs - width / 2,
        final_iqms,
        width,
        yerr=[final_err_lo, final_err_hi],
        capsize=4,
        color=final_color,
        label="final_avg (IQM)",
    )
    ax.bar(
        xs + width / 2,
        best_iqms,
        width,
        yerr=[best_err_lo, best_err_hi],
        capsize=4,
        color=best_color,
        label="best_avg (IQM)",
    )
    ax.axhline(
        SUCCESS_THRESHOLD,
        color=COLOR_THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label=f"solved ({SUCCESS_THRESHOLD:.0f})",
    )

    # K/N solved annotation above each config's bar pair.
    y_top = max(final_iqms + best_iqms + [SUCCESS_THRESHOLD]) * 1.06
    for x, k, n in zip(xs, ks, ns):
        ax.text(
            x,
            y_top,
            f"{k}/{n} solved",
            ha="center",
            va="bottom",
            fontsize=LEGEND_FONTSIZE,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=TICK_LABEL_FONTSIZE)
    ax.set_ylabel("Reward (IQM, 95% bootstrap CI)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylim(0, 520)
    ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3, axis="y")
    return _savefig(fig, output_dir / f"final_bars.{fmt}", fmt)


def plot_performance_profile(configs, summary_by_cfg, output_dir: Path, fmt: str):
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
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, constrained_layout=True)
    thresholds = np.linspace(0, 500, 501)
    for idx, cfg in enumerate(configs):
        color = OKABE_ITO[idx % len(OKABE_ITO)]
        rows = summary_by_cfg[cfg]
        scores = np.array([r["final_avg_reward"] for r in rows])
        fractions = [(scores >= t).mean() for t in thresholds]
        k, n = _solved_count(rows)
        ax.plot(
            thresholds,
            fractions,
            linewidth=2.0,
            color=color,
            label=f"{cfg}  [solved {k}/{n}]",
        )
    ax.axvline(
        SUCCESS_THRESHOLD,
        color=COLOR_THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label=f"solved ({SUCCESS_THRESHOLD:.0f})",
    )
    ax.set_xlabel(
        "Score threshold τ (final 100-ep avg reward)", fontsize=AXIS_LABEL_FONTSIZE
    )
    ax.set_ylabel("Fraction of seeds with score ≥ τ", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3)
    return _savefig(fig, output_dir / f"performance_profile.{fmt}", fmt)


def plot_loss_curves(config_name, episodes, loss_mat, output_dir: Path, fmt: str):
    if np.isnan(loss_mat).all():
        return None
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, constrained_layout=True)
    color = OKABE_ITO[0]
    iqm, lo, hi = _per_episode_band(loss_mat)
    ax.plot(episodes, iqm, color=color, linewidth=2.0, label="IQM")
    ax.fill_between(
        episodes, lo, hi, color=color, alpha=0.2, label="IQR (25–75th %ile)"
    )
    ax.set_xlabel("Episode", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Avg DQN loss per episode", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3)
    return _savefig(fig, output_dir / f"{config_name}-loss_curve.{fmt}", fmt)


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
    parser.add_argument(
        "--format",
        "-f",
        choices=["png", "svg", "pdf"],
        default="png",
        help="Output format. Use svg or pdf for paper-ready vector figures; "
             "png is the convenient development default.",
    )
    parser.add_argument(
        "--stacked",
        action="store_true",
        help="For the cross-config learning curve, produce a one-row-per-"
             "config stacked layout (shared x-axis, shared y-limits) instead "
             "of the overlaid plot. Mirrors the stacked variant in "
             "visualize_training_metrics.py.",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format

    configs = args.config_name
    episodes_by_cfg = {}
    mat_by_cfg = {}
    summary_by_cfg = {}

    for cfg in configs:
        seeds, episodes, running_mat, loss_mat = load_per_seed_csvs(metrics_dir, cfg)
        episodes_by_cfg[cfg] = episodes
        mat_by_cfg[cfg] = running_mat
        summary_by_cfg[cfg] = load_summary(metrics_dir, cfg)

        out = plot_learning_curve(
            cfg, seeds, episodes, running_mat, summary_by_cfg[cfg], output_dir, fmt
        )
        print(f"  wrote {out}")
        out = plot_loss_curves(cfg, episodes, loss_mat, output_dir, fmt)
        if out:
            print(f"  wrote {out}")

    if len(configs) >= 2:
        if args.stacked:
            out = plot_cross_config_stacked(
                configs, episodes_by_cfg, mat_by_cfg, summary_by_cfg, output_dir, fmt
            )
        else:
            out = plot_cross_config(
                configs, episodes_by_cfg, mat_by_cfg, summary_by_cfg, output_dir, fmt
            )
        print(f"  wrote {out}")
        out = plot_convergence_strip(configs, summary_by_cfg, output_dir, fmt)
        print(f"  wrote {out}")

    out = plot_final_bars(configs, summary_by_cfg, output_dir, fmt)
    print(f"  wrote {out}")

    # Performance profile (per Agarwal et al., 2021): shows the full
    # distribution of seed scores per config rather than collapsing to mean.
    out = plot_performance_profile(configs, summary_by_cfg, output_dir, fmt)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
