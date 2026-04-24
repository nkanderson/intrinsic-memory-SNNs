"""
Visualize training metrics from the SNN RL CSV logs written by train/main.py.

Given a single CSV or a directory of CSVs, produces:
  1. A comparison figure overlaying every model's running-average reward and
     generalization evaluations.
  2. One detail figure per CSV: a 2-panel view of (top) training reward with
     a rolling-std band and (bottom) generalization reward with per-seed
     min-max and IQR bands.

CSV columns consumed (all written by main.py):
    episode                          x-axis for every plot
    episode_reward                   raw per-episode reward; very noisy
    running_avg_100                  100-episode rolling mean of episode_reward
    generalization_avg               mean reward across 30 fixed eval seeds,
                                     recorded at periodic episodes only
    generalization_seeds             pipe-delimited list of the seeds used
    generalization_rewards           pipe-delimited per-seed rewards
    best_running_avg_100             running max of running_avg_100
    best_generalization_avg          running max of generalization_avg
    saved_best_running_model         1 when a new best running-avg
                                     checkpoint was written
    saved_best_generalization_model  1 when a new best-generalization
                                     checkpoint was written

Derivations (everything else is raw):
    gen_min / gen_max    min/max of the pipe-split generalization_rewards row
    gen_p25 / gen_p75    25th/75th percentile of the same row
    running_avg_band     running_avg_100 +/- rolling std over --roll-window

Usage:
    python scripts/visualize_training_metrics.py metrics/core-v2/ \\
        --output-dir images/core-v2/

    python scripts/visualize_training_metrics.py \\
        metrics/core-v2/leaky-64hl1-16hl2-training-metrics.csv \\
        --output-dir images/core-v2/ --format svg
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Okabe-Ito colorblind-safe palette. Deuteranopia-, protanopia-, and
# tritanopia-distinguishable. Used by both the comparison and detail plots.
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#F0E442",  # yellow
    "#000000",  # black
]

COLOR_TRAIN = OKABE_ITO[0]   # running average
COLOR_GEN = OKABE_ITO[2]     # generalization
COLOR_BEST = OKABE_ITO[1]    # best-so-far reference lines
COLOR_SAVE = OKABE_ITO[5]    # checkpoint save-event markers
COLOR_RAW = "#999999"        # faint raw episode-reward scatter

# Marker shapes: paired with Okabe-Ito colors so series are distinguishable
# without color.
COMPARISON_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]

DEFAULT_INPUT_RELATIVE = "metrics/core-v2"
DEFAULT_OUTPUT_RELATIVE = "images/core-v2"
DEFAULT_ROLL_WINDOW = 100


def parse_pipe_floats(cell: object) -> list[float] | None:
    """Parse a pipe-delimited cell like "500|498|..." into a list of floats."""
    if pd.isna(cell) or str(cell).strip() == "":
        return None
    try:
        return [float(v) for v in str(cell).split("|")]
    except ValueError:
        return None


def load(path: Path) -> pd.DataFrame:
    """Load a training-metrics CSV and derive per-row generalization stats."""
    df = pd.read_csv(path, header=0)
    df.columns = df.columns.str.strip()

    # generalization_rewards is a pipe-delimited "r1|r2|..." string per row
    # (one entry per eval seed, typically 30). Split it and take per-row
    # min / max / percentiles so we can draw uncertainty bands below the mean.
    if "generalization_rewards" in df.columns:
        reward_lists = df["generalization_rewards"].apply(parse_pipe_floats)
    else:
        reward_lists = pd.Series([None] * len(df), index=df.index)

    df["gen_min"] = reward_lists.apply(lambda r: float(np.min(r)) if r else np.nan)
    df["gen_max"] = reward_lists.apply(lambda r: float(np.max(r)) if r else np.nan)
    df["gen_p25"] = reward_lists.apply(
        lambda r: float(np.percentile(r, 25)) if r else np.nan
    )
    df["gen_p75"] = reward_lists.apply(
        lambda r: float(np.percentile(r, 75)) if r else np.nan
    )
    return df


def label_from_path(path: Path) -> str:
    """Derive a compact human-readable label from a metrics filename.

    Examples:
      leaky-64hl1-16hl2-training-metrics.csv            -> "leaky 64-16"
      fractional-32hl1-4hl2-16hist-training-metrics.csv -> "fractional 32-4 (hist=16)"
      bitshift-custom_slow_decay-32hl1-8hl2-8hist-...   -> "bitshift custom slow decay 32-8 (hist=8)"
    """
    stem = path.stem
    stem = re.sub(r"^dqn_", "", stem)
    stem = re.sub(r"-training-metrics$", "", stem)

    hl1 = re.search(r"(\d+)hl1", stem)
    hl2 = re.search(r"(\d+)hl2", stem)
    hist = re.search(r"(\d+)hist", stem)

    # Drop the architecture tokens from the prefix so only the neuron-type
    # descriptor (e.g. "bitshift-custom_slow_decay") remains.
    prefix = re.sub(r"-?\d+(hl1|hl2|hist)", "", stem).strip("-")
    neuron = prefix.replace("-", " ").replace("_", " ").strip()

    parts = [neuron] if neuron else []
    if hl1 and hl2:
        parts.append(f"{hl1.group(1)}-{hl2.group(1)}")
    if hist:
        parts.append(f"(hist={hist.group(1)})")

    return " ".join(parts) if parts else stem


def slug(label: str) -> str:
    """Filesystem-safe stem from a label."""
    return re.sub(r"[^a-zA-Z0-9]+", "-", label).strip("-").lower()


def save_fig(fig: plt.Figure, out_dir: Path, stem: str, fmt: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.{fmt}"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_comparison(frames: dict[str, pd.DataFrame], out_dir: Path, fmt: str) -> Path:
    """Overlay every model's running_avg_100 and generalization_avg."""
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

    model_handles: list[Line2D] = []

    for idx, (label, df) in enumerate(frames.items()):
        color = OKABE_ITO[idx % len(OKABE_ITO)]
        marker = COMPARISON_MARKERS[idx % len(COMPARISON_MARKERS)]

        # Running-average learning curve (solid line). Drop rows where the
        # column is NaN (main.py only starts writing running_avg_100 after
        # enough episodes have accumulated).
        run_mask = df["running_avg_100"].notna()
        if run_mask.any():
            ax.plot(
                df.loc[run_mask, "episode"],
                df.loc[run_mask, "running_avg_100"],
                color=color, linewidth=2.0, zorder=3,
            )

        # Generalization: in core-v2 runs, generalization_avg is populated
        # on ~every episode (not just periodic checkpoints), so rendering it
        # as hundreds of scatter points per model produces a solid "bar".
        # Instead, draw it as a thin dashed line in the same color and add a
        # sparse set of marker anchors so each model's shape is still
        # identifiable in the legend.
        gen_mask = df["generalization_avg"].notna()
        if gen_mask.any():
            gen_eps = df.loc[gen_mask, "episode"].to_numpy()
            gen_vals = df.loc[gen_mask, "generalization_avg"].to_numpy()
            ax.plot(
                gen_eps, gen_vals,
                color=color, linewidth=1.0, linestyle="--", alpha=0.75,
                zorder=2,
            )

            # Anchor markers at ~8 evenly-spaced positions along the eval
            # series so color+shape remain visible without clutter.
            n_anchors = min(8, len(gen_eps))
            if n_anchors >= 2:
                anchor_idx = np.linspace(0, len(gen_eps) - 1, n_anchors, dtype=int)
                ax.scatter(
                    gen_eps[anchor_idx], gen_vals[anchor_idx],
                    facecolors="none", edgecolors=color,
                    marker=marker, s=55, linewidths=1.6,
                    alpha=0.95, zorder=4,
                )

        # Single legend entry per model: colored line + hollow marker, so the
        # legend encodes both color and marker shape in one row.
        model_handles.append(
            Line2D(
                [0], [0],
                color=color, linewidth=2.0,
                marker=marker, markerfacecolor="none", markeredgecolor=color,
                markersize=8, markeredgewidth=1.5,
                label=label,
            )
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training comparison: running average (line) & generalization (markers)")
    ax.grid(True, alpha=0.3)

    if model_handles:
        ax.legend(
            handles=model_handles,
            loc="lower right",
            fontsize=9, framealpha=0.9,
            title="Model", title_fontsize=9,
        )

    return save_fig(fig, out_dir, "training_comparison", fmt)


def plot_detail(
    df: pd.DataFrame,
    label: str,
    out_dir: Path,
    fmt: str,
    *,
    roll_window: int,
    show_raw: bool,
) -> Path:
    """Two-panel detail figure for a single model."""
    eps = df["episode"]
    running_avg = df["running_avg_100"]

    # Band around the running average: ± rolling std over --roll-window
    # episodes of the raw episode_reward. Independent of main.py's own
    # running_avg_100 smoothing window, so --roll-window controls band
    # width without shifting the center line.
    roll_std = df["episode_reward"].rolling(roll_window, min_periods=5).std()
    band_hi = running_avg + roll_std
    band_lo = running_avg - roll_std

    # generalization_avg is NaN on non-eval episodes; mask down to eval rows.
    gen_mask = df["generalization_avg"].notna()
    gen_eps = eps[gen_mask]
    gen_avg = df.loc[gen_mask, "generalization_avg"]
    gen_min = df.loc[gen_mask, "gen_min"]
    gen_max = df.loc[gen_mask, "gen_max"]
    gen_p25 = df.loc[gen_mask, "gen_p25"]
    gen_p75 = df.loc[gen_mask, "gen_p75"]
    has_minmax = gen_min.notna().any()
    has_iqr = gen_p25.notna().any()

    saved_run_eps = eps[df.get("saved_best_running_model", 0) == 1]
    saved_gen_rows = df[df.get("saved_best_generalization_model", 0) == 1]

    fig, (ax_train, ax_gen) = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
        constrained_layout=True,
    )

    # ── Top panel: training reward ─────────────────────────────────────────
    if show_raw:
        ax_train.scatter(
            eps, df["episode_reward"],
            s=5, color=COLOR_RAW, alpha=0.18, linewidths=0,
            zorder=1, label="Episode reward (raw)",
        )
    ax_train.fill_between(
        eps, band_lo, band_hi,
        color=COLOR_TRAIN, alpha=0.2, zorder=2,
        label=f"Running avg ± rolling std ({roll_window} ep)",
    )
    ax_train.plot(
        eps, running_avg,
        color=COLOR_TRAIN, linewidth=2.0, zorder=3,
        label="Running avg (100 ep)",
    )
    if "best_running_avg_100" in df.columns:
        ax_train.plot(
            eps, df["best_running_avg_100"],
            color=COLOR_BEST, linewidth=1.2, linestyle="--", alpha=0.9,
            zorder=3, label="Best running avg to date",
        )
    # Short rug-style ticks at the top of the panel mark save events. Using
    # ymin just under 1.0 and ymax=1.0 in axes coords keeps them out of the
    # data region.
    for x in saved_run_eps:
        ax_train.axvline(
            x, color=COLOR_SAVE, linewidth=0.9, alpha=0.55,
            ymin=0.96, ymax=1.0, zorder=4,
        )
    if len(saved_run_eps):
        ax_train.plot(
            [], [],
            color=COLOR_SAVE, linewidth=1.5,
            label="Best running model saved",
        )

    ax_train.set_ylabel("Reward")
    ax_train.set_title(f"{label} — training reward")
    ax_train.grid(True, alpha=0.3)
    ax_train.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # ── Bottom panel: generalization ───────────────────────────────────────
    if has_minmax:
        ax_gen.fill_between(
            gen_eps, gen_min, gen_max,
            color=COLOR_GEN, alpha=0.15, zorder=1,
            label="Seed min–max",
        )
    if has_iqr:
        ax_gen.fill_between(
            gen_eps, gen_p25, gen_p75,
            color=COLOR_GEN, alpha=0.35, zorder=2,
            label="Seed IQR (25–75%)",
        )
    ax_gen.plot(
        gen_eps, gen_avg,
        color=COLOR_GEN, linewidth=2.0, marker="o", markersize=4,
        zorder=3, label="Generalization mean",
    )
    if "best_generalization_avg" in df.columns:
        ax_gen.plot(
            gen_eps, df.loc[gen_mask, "best_generalization_avg"],
            color=COLOR_BEST, linewidth=1.2, linestyle="--", alpha=0.9,
            zorder=3, label="Best generalization to date",
        )
    if not saved_gen_rows.empty and "best_generalization_avg" in saved_gen_rows.columns:
        ax_gen.scatter(
            saved_gen_rows["episode"],
            saved_gen_rows["best_generalization_avg"],
            color=COLOR_SAVE, s=90, marker="*", zorder=5,
            edgecolors="black", linewidths=0.5,
            label="Best generalization model saved",
        )

    ax_gen.set_xlabel("Episode")
    ax_gen.set_ylabel("Generalization reward")
    ax_gen.set_title("Generalization (across 30 fixed seeds)")
    ax_gen.grid(True, alpha=0.3)
    ax_gen.legend(loc="lower right", fontsize=9, framealpha=0.9)

    return save_fig(fig, out_dir, slug(label), fmt)


def discover_csvs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.glob("*.csv") if p.is_file())
    raise FileNotFoundError(f"Input path not found: {input_path}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    train_dir = script_dir.parent
    default_input = train_dir / DEFAULT_INPUT_RELATIVE
    default_output = train_dir / DEFAULT_OUTPUT_RELATIVE

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "input_path", nargs="?", default=str(default_input),
        help=f"CSV file or directory of CSVs (default: {DEFAULT_INPUT_RELATIVE})",
    )
    parser.add_argument(
        "--output-dir", default=str(default_output),
        help=f"Output directory for figures (default: {DEFAULT_OUTPUT_RELATIVE})",
    )
    parser.add_argument(
        "--format", choices=("png", "svg"), default="png",
        help="Output file format (default: png)",
    )
    parser.add_argument(
        "--roll-window", type=int, default=DEFAULT_ROLL_WINDOW,
        help=(
            f"Rolling-std window (episodes) for the band around the running "
            f"average in detail plots (default: {DEFAULT_ROLL_WINDOW})"
        ),
    )
    parser.add_argument(
        "--comparison-only", action="store_true",
        help="Only write the comparison figure, skip per-model detail figures.",
    )
    parser.add_argument(
        "--details-only", action="store_true",
        help="Only write per-model detail figures, skip the comparison figure.",
    )
    parser.add_argument(
        "--no-raw", action="store_true",
        help="Suppress the faint raw-episode-reward scatter in detail plots.",
    )
    args = parser.parse_args()

    if args.comparison_only and args.details_only:
        parser.error("--comparison-only and --details-only are mutually exclusive")

    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    try:
        csv_files = discover_csvs(input_path)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    if not csv_files:
        print(f"error: no CSV files found in {input_path}", file=sys.stderr)
        sys.exit(1)

    frames = {label_from_path(p): load(p) for p in csv_files}

    # Comparison figure only makes sense with >=2 models.
    want_comparison = not args.details_only and len(frames) >= 2
    want_details = not args.comparison_only

    if want_comparison:
        out = plot_comparison(frames, output_dir, args.format)
        print(f"Saved: {out}")

    if want_details:
        for label, df in frames.items():
            out = plot_detail(
                df, label, output_dir, args.format,
                roll_window=args.roll_window,
                show_raw=not args.no_raw,
            )
            print(f"Saved: {out}")


if __name__ == "__main__":
    main()
