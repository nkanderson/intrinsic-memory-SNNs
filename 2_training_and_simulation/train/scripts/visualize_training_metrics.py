"""
Visualize training metrics from the SNN RL CSV logs written by train/main.py.

Given a single CSV or a directory of CSVs, produces:
  1. A comparison output across models. Selected via --compare:
       running               overlaid running-average learning curves
                             with a peak marker per model (default).
       both                  running average plus generalization trace.
       efficiency            scatter of sample efficiency (first ep at
                             gen=500) vs. sustained performance (max
                             running avg).
       summary               markdown table to stdout + CSV on disk.
                             No figure.
       stack-running         small-multiples: one row per model showing
                             the training-reward rolling average panel.
       stack-generalization  small-multiples: one row per model showing
                             the generalization panel.
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
from matplotlib.patches import Patch

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


def plot_comparison_lines(
    frames: dict[str, pd.DataFrame],
    out_dir: Path,
    fmt: str,
    *,
    mode: str,
) -> Path:
    """Overlay every model's running_avg_100 and/or generalization_avg over episodes.

    ``mode`` is one of:
      - "running": running-average learning curves only (cleanest).
      - "both": running average solid + generalization dashed (busier, shows
        the train-vs-eval gap).
    """
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

    model_handles: list[Line2D] = []
    show_gen = mode == "both"

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
                solid_joinstyle="round", solid_capstyle="round",
            )
            # In "running" mode the line itself carries the trajectory but
            # nothing punctuates the peak. Drop the model's marker shape at
            # (episode, value) of max(running_avg_100), with the numeric
            # value next to it, so the eye can immediately read both *when*
            # and *how high* each model peaked. Hollow markers (only the
            # model's color as edge, no fill) match the legend style.
            # Skipped in "both" mode because the generalization anchor
            # markers already use that shape.
            if not show_gen:
                best_run_row = df.loc[run_mask, "running_avg_100"].idxmax()
                best_ep = df.loc[best_run_row, "episode"]
                best_val = float(df.loc[best_run_row, "running_avg_100"])
                ax.scatter(
                    [best_ep], [best_val],
                    marker=marker, s=110,
                    facecolors="none", edgecolors=color,
                    linewidths=1.8, zorder=5,
                )
                # Label sits above the marker with a semi-opaque white box
                # so it stays readable when it crosses other models' lines
                # in busy regions.
                ax.annotate(
                    f"{best_val:.0f}",
                    (best_ep, best_val),
                    xytext=(0, 9), textcoords="offset points",
                    fontsize=9, color="black",
                    va="bottom", ha="center", zorder=6,
                    bbox=dict(
                        facecolor="white", edgecolor="none",
                        alpha=0.7, pad=1.5,
                    ),
                )

        # In "both" mode, add a thin dashed generalization trace behind the
        # solid running-average line. In core-v2 runs generalization_avg is
        # populated on ~every episode, so rendering as scatter would blur
        # into a solid bar — a line with sparse anchor markers keeps each
        # model's shape readable.
        if show_gen:
            gen_mask = df["generalization_avg"].notna()
            if gen_mask.any():
                gen_eps = df.loc[gen_mask, "episode"].to_numpy()
                gen_vals = df.loc[gen_mask, "generalization_avg"].to_numpy()
                ax.plot(
                    gen_eps, gen_vals,
                    color=color, linewidth=1.0, linestyle="--", alpha=0.75,
                    zorder=2,
                    solid_joinstyle="round", solid_capstyle="round",
                )
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
    # No title: comparison figures are inserted with a figure caption.
    ax.grid(True, alpha=0.3)

    # Headroom for the peak-value labels (running mode): the labels sit
    # above their markers, and several models peak near the top of the
    # data range — without padding the text would clip past the axis.
    if mode == "running":
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.06 * (ymax - ymin))

    if model_handles:
        # Upper-left corner: models converge toward 500 at high episodes, so
        # the upper-left region (low rewards, early training) is the most
        # reliably empty area for legend placement.
        ax.legend(
            handles=model_handles,
            loc="upper left",
            fontsize=9, framealpha=0.9,
            title="Model", title_fontsize=9,
        )

    stem = {
        "running": "training_comparison_running",
        "both": "training_comparison",
    }[mode]
    return save_fig(fig, out_dir, stem, fmt)


CARTPOLE_MAX_REWARD = 500.0


def compute_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a one-row-per-model summary from the raw metrics frames.

    Columns are the handful of scalars that actually distinguish the models:
      - best_gen_avg         max(generalization_avg) seen during training
      - first_ep_at_gen_500  sample efficiency: episode at which
                             generalization_avg first reached the CartPole-v1
                             ceiling. NaN if the model never hit it.
      - best_running_avg     max(running_avg_100): best 100-ep smoothed
                             training reward — proxy for how well the policy
                             held up across training episodes, not just peak
                             evaluation runs.
      - final_running_avg    last non-NaN running_avg_100 — where the model
                             ended up at training end.
    """
    rows = []
    for label, df in frames.items():
        gen_mask = df["generalization_avg"].notna()
        run_mask = df["running_avg_100"].notna()

        best_gen = (
            float(df.loc[gen_mask, "generalization_avg"].max())
            if gen_mask.any() else float("nan")
        )
        ceiling = df[gen_mask & (df["generalization_avg"] >= CARTPOLE_MAX_REWARD)]
        first_500 = (
            int(ceiling["episode"].min()) if not ceiling.empty else None
        )
        best_run = (
            float(df.loc[run_mask, "running_avg_100"].max())
            if run_mask.any() else float("nan")
        )
        final_run = (
            float(df.loc[run_mask, "running_avg_100"].iloc[-1])
            if run_mask.any() else float("nan")
        )

        rows.append({
            "model": label,
            "best_gen_avg": best_gen,
            "first_ep_at_gen_500": first_500,
            "best_running_avg": best_run,
            "final_running_avg": final_run,
        })
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table.

    Hand-rolled to avoid a tabulate dependency. Numeric cells get one
    decimal; None/NaN renders as an em dash.
    """
    def fmt(v: object) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            if pd.isna(v):
                return "—"
            # Integer-valued floats render without trailing ".0".
            return f"{v:.1f}" if v != int(v) else f"{int(v)}"
        return str(v)

    headers = list(df.columns)
    rows = [[fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def pad_row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    lines = [
        pad_row(headers),
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend(pad_row(row) for row in rows)
    return "\n".join(lines)


def render_summary_table(
    frames: dict[str, pd.DataFrame],
    out_dir: Path,
) -> Path:
    """Print a markdown comparison table to stdout and write a CSV copy."""
    table = compute_summary(frames)
    print(_markdown_table(table))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "training_comparison_summary.csv"
    table.to_csv(csv_path, index=False)
    return csv_path


def plot_comparison_efficiency(
    frames: dict[str, pd.DataFrame],
    out_dir: Path,
    fmt: str,
) -> Path:
    """Sample-efficiency vs. sustained-performance scatter.

    One point per model at (first_ep_at_gen_500, max(running_avg_100)).
    Upper-left = best: hit perfect generalization early AND held a high
    running average. Lower-right = late convergence with weak sustained
    performance. Models that never reached gen=500 are omitted (with a
    warning to stderr) because there's no honest x-value for them.
    """
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    plotted_any = False
    skipped: list[str] = []

    for idx, (label, df) in enumerate(frames.items()):
        color = OKABE_ITO[idx % len(OKABE_ITO)]
        gen_mask = df["generalization_avg"].notna()
        run_mask = df["running_avg_100"].notna()

        ceiling = df[gen_mask & (df["generalization_avg"] >= CARTPOLE_MAX_REWARD)]
        if ceiling.empty or not run_mask.any():
            skipped.append(label)
            continue

        x = float(ceiling["episode"].min())
        y = float(df.loc[run_mask, "running_avg_100"].max())
        ax.scatter(
            [x], [y],
            color=color, s=180, marker="o", zorder=3,
            edgecolors="black", linewidths=0.8,
        )
        # Inline annotation labels each point directly, so no legend is
        # needed — and this avoids a legend box covering a data point.
        ax.annotate(
            label, (x, y),
            xytext=(9, 7), textcoords="offset points",
            fontsize=9, zorder=4,
        )
        plotted_any = True

    ax.set_xlabel("First episode reaching generalization avg = 500 (sample efficiency →)")
    ax.set_ylabel("Best running avg (sustained performance ↑)")
    # No title: comparison figures are inserted with a figure caption.
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Leave headroom on the right so long labels don't clip past the axis.
    if plotted_any:
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax + 0.08 * (xmax - xmin))

    if skipped:
        print(
            f"warning: skipped {len(skipped)} model(s) that never reached "
            f"generalization avg = {CARTPOLE_MAX_REWARD}: {', '.join(skipped)}",
            file=sys.stderr,
        )

    return save_fig(fig, out_dir, "training_comparison_efficiency", fmt)


# Shared y-axis configuration for the stacked comparison plots: every panel
# uses the same ticks and limits so that vertical comparisons across rows
# are honest, even when one model never gets close to 500.
STACK_Y_LIMITS = (-25, 525)
STACK_Y_TICKS = [100, 300, 500]


def _stack_panel_label(ax: plt.Axes, label: str) -> None:
    """Place a model name in the upper-left corner of a stacked panel."""
    ax.text(
        0.012, 0.92, label,
        transform=ax.transAxes,
        fontsize=9, fontweight="bold",
        va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
    )


def _apply_stack_y_axis(ax: plt.Axes) -> None:
    """Apply the shared y-axis limits and tick marks for stacked panels."""
    ax.set_ylim(*STACK_Y_LIMITS)
    ax.set_yticks(STACK_Y_TICKS)


def plot_comparison_stack_running(
    frames: dict[str, pd.DataFrame],
    out_dir: Path,
    fmt: str,
    *,
    roll_window: int,
    show_raw: bool,
) -> Path:
    """One row per model showing the same content as the detail top panel.

    Simplifications relative to the detail figure (per user request):
      - No save-event markers / rug ticks for the best-running model.
      - Same raw scatter / running-avg line / std band / best-to-date line.
    """
    n = len(frames)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(11, 1.7 * n + 1.0),
        sharex=True,
        constrained_layout=True,
    )
    # `subplots(1, 1)` returns a single Axes, not an array; normalize.
    if n == 1:
        axes = [axes]

    for idx, ((label, df), ax) in enumerate(zip(frames.items(), axes)):
        color = OKABE_ITO[idx % len(OKABE_ITO)]
        eps = df["episode"]
        running_avg = df["running_avg_100"]

        # Same band derivation as plot_detail: rolling std of raw
        # episode_reward over --roll-window episodes, centered on the
        # already-smoothed running_avg_100 line.
        roll_std = df["episode_reward"].rolling(roll_window, min_periods=5).std()
        band_hi = running_avg + roll_std
        band_lo = running_avg - roll_std

        if show_raw:
            ax.scatter(
                eps, df["episode_reward"],
                s=4, color=COLOR_RAW, alpha=0.18, linewidths=0, zorder=1,
            )
        ax.fill_between(eps, band_lo, band_hi, color=color, alpha=0.2, zorder=2)
        ax.plot(
            eps, running_avg,
            color=color, linewidth=1.8, zorder=3,
            solid_joinstyle="round", solid_capstyle="round",
        )
        if "best_running_avg_100" in df.columns:
            ax.plot(
                eps, df["best_running_avg_100"],
                color=COLOR_BEST, linewidth=1.0, linestyle="--",
                alpha=0.9, zorder=3,
            )

        _apply_stack_y_axis(ax)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Reward")
        _stack_panel_label(ax, label)

    axes[-1].set_xlabel("Episode")

    # Figure-wide legend goes in the bottom panel — every panel uses the
    # same encoding, so explaining it once is enough. Lower-right is the
    # empty corner once the leaky model plateaus near 500. Neutral gray
    # stands in for "the per-panel model color" since the legend is shared.
    NEUTRAL = "#555555"
    legend_handles = []
    if show_raw:
        legend_handles.append(Line2D(
            [0], [0], linestyle="none",
            marker="o", markersize=4, markerfacecolor=COLOR_RAW,
            markeredgecolor=COLOR_RAW, alpha=0.5,
            label="Episode reward (raw)",
        ))
    legend_handles.extend([
        Line2D([0], [0], color=NEUTRAL, linewidth=1.8,
               label="Running avg (100 ep)"),
        Patch(facecolor=NEUTRAL, alpha=0.2,
              label=f"Running avg ± rolling std ({roll_window} ep)"),
        Line2D([0], [0], color=COLOR_BEST, linewidth=1.0, linestyle="--",
               label="Best running avg to date"),
    ])
    axes[-1].legend(
        handles=legend_handles, loc="lower right",
        fontsize=8, framealpha=0.9,
    )

    return save_fig(fig, out_dir, "training_comparison_stack_running", fmt)


def plot_comparison_stack_generalization(
    frames: dict[str, pd.DataFrame],
    out_dir: Path,
    fmt: str,
) -> Path:
    """One row per model showing the same content as the detail bottom panel.

    Simplifications relative to the detail figure (per user request):
      - No seed min-max band, no IQR band.
      - Save-event markers only at episodes where the saved best_gen value
        was the CartPole ceiling (500) — the moments when the model first
        produced a "perfect" generalization checkpoint.
    """
    n = len(frames)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(11, 1.7 * n + 1.0),
        sharex=True,
        constrained_layout=True,
    )
    if n == 1:
        axes = [axes]

    for idx, ((label, df), ax) in enumerate(zip(frames.items(), axes)):
        color = OKABE_ITO[idx % len(OKABE_ITO)]
        gen_mask = df["generalization_avg"].notna()
        gen_eps = df.loc[gen_mask, "episode"]
        gen_avg = df.loc[gen_mask, "generalization_avg"]

        ax.plot(
            gen_eps, gen_avg,
            color=color, linewidth=1.6, zorder=3,
            solid_joinstyle="round", solid_capstyle="round",
        )
        if "best_generalization_avg" in df.columns:
            ax.plot(
                gen_eps, df.loc[gen_mask, "best_generalization_avg"],
                color=COLOR_BEST, linewidth=1.0, linestyle="--",
                alpha=0.9, zorder=3,
                solid_joinstyle="round", solid_capstyle="round",
            )

        # Save-event triangles only when the new best-gen value reached
        # the CartPole ceiling. saved_best_generalization_model fires every
        # time the running max increases, but we only mark "perfect" saves
        # to keep the stacked view uncluttered.
        saved = df[
            (df.get("saved_best_generalization_model", 0) == 1)
            & (df.get("best_generalization_avg", 0) >= CARTPOLE_MAX_REWARD)
        ]
        if not saved.empty:
            ax.scatter(
                saved["episode"], saved["best_generalization_avg"],
                color=COLOR_SAVE, s=55, marker="^", zorder=5,
                edgecolors="black", linewidths=0.5,
            )

        _apply_stack_y_axis(ax)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Reward")
        _stack_panel_label(ax, label)

    axes[-1].set_xlabel("Episode")

    # See plot_comparison_stack_running for why the legend lives in the
    # bottom panel only (all panels share the same encoding).
    NEUTRAL = "#555555"
    legend_handles = [
        Line2D([0], [0], color=NEUTRAL, linewidth=1.6,
               label="Generalization mean"),
        Line2D([0], [0], color=COLOR_BEST, linewidth=1.0, linestyle="--",
               label="Best generalization to date"),
        Line2D([0], [0], linestyle="none",
               marker="^", markersize=7,
               markerfacecolor=COLOR_SAVE, markeredgecolor="black",
               markeredgewidth=0.5,
               label="Best generalization model saved (= 500)"),
    ]
    axes[-1].legend(
        handles=legend_handles, loc="lower right",
        fontsize=8, framealpha=0.9,
    )

    return save_fig(fig, out_dir, "training_comparison_stack_generalization", fmt)


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
    # Disambiguates from the bottom panel: the top panel is *training*
    # episodes (the rolling-mean smoothing of episode_reward), not the
    # periodic generalization evaluations shown below.
    ax_train.set_title("Training reward (rolling average)")
    ax_train.grid(True, alpha=0.3)
    # Legends in upper-left of each panel: several models bottom out near
    # zero late in training, which would sit under a lower-right legend.
    ax_train.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # ── Bottom panel: generalization ───────────────────────────────────────
    # Bands are drawn at higher alpha (min-max 0.28, IQR 0.55) so spread is
    # visible even when seeds agree tightly; darker IQR overlays lighter
    # min-max for clear ordering.
    if has_minmax:
        ax_gen.fill_between(
            gen_eps, gen_min, gen_max,
            color=COLOR_GEN, alpha=0.28, zorder=1,
            label="Seed min–max",
        )
    if has_iqr:
        ax_gen.fill_between(
            gen_eps, gen_p25, gen_p75,
            color=COLOR_GEN, alpha=0.55, zorder=2,
            label="Seed IQR (25–75%)",
        )
    # Rounded line joins soften the right-angle corners that step-function
    # generalization data produces (consecutive evals often return the same
    # value, then jump sharply).
    ax_gen.plot(
        gen_eps, gen_avg,
        color=COLOR_GEN, linewidth=1.6, zorder=3,
        solid_joinstyle="round", solid_capstyle="round",
        label="Generalization mean",
    )
    if "best_generalization_avg" in df.columns:
        ax_gen.plot(
            gen_eps, df.loc[gen_mask, "best_generalization_avg"],
            color=COLOR_BEST, linewidth=1.2, linestyle="--", alpha=0.9,
            zorder=3,
            solid_joinstyle="round", solid_capstyle="round",
            label="Best generalization to date",
        )
    if not saved_gen_rows.empty and "best_generalization_avg" in saved_gen_rows.columns:
        ax_gen.scatter(
            saved_gen_rows["episode"],
            saved_gen_rows["best_generalization_avg"],
            color=COLOR_SAVE, s=70, marker="^", zorder=5,
            edgecolors="black", linewidths=0.5,
            label="Best generalization model saved",
        )

    ax_gen.set_xlabel("Episode")
    ax_gen.set_ylabel("Generalization reward")
    ax_gen.set_title("Generalization (across 30 fixed seeds)")
    ax_gen.grid(True, alpha=0.3)
    ax_gen.legend(loc="upper left", fontsize=9, framealpha=0.9)

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
        "--compare",
        choices=(
            "running", "both", "efficiency", "summary",
            "stack-running", "stack-generalization",
        ),
        default="running",
        help=(
            "What the comparison output is: 'running' (default) = overlaid "
            "running-average learning curves with a peak marker per model; "
            "'both' = running average plus generalization trace; "
            "'efficiency' = one-point-per-model scatter of sample efficiency "
            "(first ep at gen=500) vs sustained performance (max running "
            "avg); 'summary' = markdown table to stdout + CSV on disk, no "
            "figure; 'stack-running' = small-multiples view of every model's "
            "training-reward rolling average (one row each); "
            "'stack-generalization' = small-multiples view of every model's "
            "generalization (one row each)."
        ),
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
        if args.compare == "summary":
            out = render_summary_table(frames, output_dir)
        elif args.compare == "efficiency":
            out = plot_comparison_efficiency(frames, output_dir, args.format)
        elif args.compare == "stack-running":
            out = plot_comparison_stack_running(
                frames, output_dir, args.format,
                roll_window=args.roll_window,
                show_raw=not args.no_raw,
            )
        elif args.compare == "stack-generalization":
            out = plot_comparison_stack_generalization(
                frames, output_dir, args.format,
            )
        else:
            out = plot_comparison_lines(
                frames, output_dir, args.format, mode=args.compare,
            )
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
