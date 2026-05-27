"""Plot stage-3 PPA + inference-cycle metrics, with CSV side-tables.

Reads 3_benchmarking_on_FPGA/results/summary/ppa_cycles_combined.csv and
emits a mix of plots and CSV tables under
3_benchmarking_on_FPGA/results/summary/plots/ and .../tables/.

The aggregate CSV has one row per (config, profile). When more than one
profile is present for any config, bars are grouped by config with one
sub-bar per profile (profile distinguished by hatch pattern). When only
the default "baseline" profile is present, plots collapse to a single
bar per config.

Run from the repo root:

    python 3_benchmarking_on_FPGA/scripts/plot_ppa.py            # PNG (default)
    python 3_benchmarking_on_FPGA/scripts/plot_ppa.py --format svg
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Repo root is parents[2] of this file: 3_benchmarking_on_FPGA/scripts/plot_ppa.py
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.scripts.plot_styles import (  # noqa: E402
    OKABE_ITO,
    DEFAULT_FIGSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    get_latex_figsize,
)

# Stage colors for the top-level FSM stacked bar.
STAGE_COLORS = {
    "load_hl1": OKABE_ITO[4],  # sky blue
    "run_timesteps": OKABE_ITO[5],  # orange
    "finish_q": OKABE_ITO[6],  # yellow
}

# Per-timestep substate colors (6 distinct hues so the stack is readable).
TS_SUBSTATE_COLORS = {
    "hl1_step": OKABE_ITO[0],  # blue
    "fc2_start": OKABE_ITO[4],  # sky blue
    "fc2_wait": OKABE_ITO[1],  # vermillion
    "hl2_step": OKABE_ITO[2],  # bluish green
    "hl2_wait": OKABE_ITO[3],  # reddish purple
    "next": OKABE_ITO[5],  # orange
}

TS_SUBSTATE_LABELS = [
    ("hl1_step", "HL1_STEP"),
    ("fc2_start", "FC2_START"),
    ("fc2_wait", "FC2_WAIT"),
    ("hl2_step", "HL2_STEP"),
    ("hl2_wait", "HL2_WAIT"),
    ("next", "NEXT"),
]

# Hatch patterns cycled across profiles when N_profiles > 1.
# Baseline always renders solid (no hatch).
PROFILE_HATCHES = ["", "//", "xx", "..", "++", "\\\\"]


def _f(v) -> Optional[float]:
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _figsize(
    width_scale: float = 1.0, height_scale: Optional[float] = None
) -> tuple[float, float]:
    """Larger default than the previous LaTeX-column sizing.

    Uses DEFAULT_FIGSIZE (7.0 x 4.3) as the base and scales from there so
    these plots line up with the rest of the codebase's defaults.
    """
    w, h = DEFAULT_FIGSIZE
    if height_scale is None:
        return (w * width_scale, h * width_scale)
    return (w * width_scale, h * height_scale)


def _save(fig: plt.Figure, out_dir: Path, stem: str, fmt: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.{fmt}"
    fig.savefig(path, format=fmt, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def _save_table(
    rows: list[dict], out_dir: Path, stem: str, fields: Sequence[str]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"  {path}")


def _style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.xaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONTSIZE)


def _rows_with(rows: list[dict], key: str) -> list[dict]:
    return [r for r in rows if _f(r.get(key)) is not None]


def _group_by_config(
    rows: list[dict],
) -> tuple[list[str], list[str], list[list[Optional[dict]]]]:
    """Return (config_labels, profile_names, grid).

    config_labels: x-axis label per config (display_label of the first profile row found)
    profile_names: union of profiles, sorted with "baseline" first
    grid:          rows-of-cols where grid[i][j] is the row for (config_i, profile_j) or None
    """
    by_config: dict[str, dict[str, dict]] = {}
    order: list[str] = []
    for r in rows:
        cfg = r["config"]
        if cfg not in by_config:
            by_config[cfg] = {}
            order.append(cfg)
        by_config[cfg][r["profile"]] = r

    all_profiles = set()
    for cfg in order:
        all_profiles.update(by_config[cfg].keys())
    profile_names = sorted(all_profiles, key=lambda p: (p != "baseline", p))

    labels = [
        next(iter(by_config[cfg].values())).get("display_label") or cfg for cfg in order
    ]
    grid = [[by_config[cfg].get(p) for p in profile_names] for cfg in order]
    return labels, profile_names, grid


def _bar_positions(
    n_configs: int, n_profiles: int, group_width: float = 0.8
) -> tuple[np.ndarray, float]:
    """Compute centered x-offsets for grouped bars. Returns (offsets, bar_width).

    offsets[j] is the offset added to the per-config center for profile j.
    """
    if n_profiles == 1:
        return np.array([0.0]), group_width
    bar_width = group_width / n_profiles
    centers = (np.arange(n_profiles) - (n_profiles - 1) / 2.0) * bar_width
    return centers, bar_width


def _add_profile_legend(fig: plt.Figure, profile_names: list[str]) -> None:
    """Add a figure-level legend mapping hatch -> profile when N_profiles > 1.

    Placed below the axes so it doesn't collide with per-axes legends. No-op
    for the single-profile case (the visual collapses to the original layout).
    """
    if len(profile_names) <= 1:
        return
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="lightgray",
            edgecolor="black",
            hatch=PROFILE_HATCHES[i % len(PROFILE_HATCHES)],
        )
        for i in range(len(profile_names))
    ]
    # Reserve space at the bottom of the figure for the legend so it doesn't
    # collide with rotated xtick labels.
    fig.subplots_adjust(bottom=0.30)
    fig.legend(
        handles,
        profile_names,
        fontsize=LEGEND_FONTSIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(profile_names), 4),
        title="Profile",
        frameon=True,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_area(rows: list[dict], out_dir: Path, fmt: str) -> None:
    rows = _rows_with(rows, "slice_luts")
    if not rows:
        print("  area: no synth data; skipping")
        return
    labels, profiles, grid = _group_by_config(rows)
    n_cfg = len(labels)
    n_prof = len(profiles)
    offsets, bar_w = _bar_positions(n_cfg, n_prof)

    # Four resource groups laid out on x; within each group, n_cfg clusters
    # separated by n_prof profile bars. To keep the visual simple, we instead
    # cluster per config (each config is one cluster of 4 resource bars times
    # n_profiles), but this gets noisy. Use the original layout: x per config,
    # 4 resource bars per config, profiles via hatch overlay on each bar.
    x = np.arange(n_cfg)
    inner_offsets, inner_w = _bar_positions(n_cfg, 4, group_width=0.8)
    # The 4 resource bars per config sit at x + inner_offsets[k]; profiles
    # within each resource bar subdivide further.
    if n_prof > 1:
        sub_offsets, sub_w = _bar_positions(n_cfg, n_prof, group_width=inner_w * 0.9)
    else:
        sub_offsets, sub_w = np.array([0.0]), inner_w

    fig, ax = plt.subplots(figsize=_figsize(width_scale=1.2))

    # First pass: draw bars with the appropriate stack/hatch.
    for k, (resource_key, color, label) in enumerate(
        [
            ("slice_luts", OKABE_ITO[0], "LUTs"),
            ("slice_registers", OKABE_ITO[4], "FFs"),
            ("dsp", OKABE_ITO[5], "DSP"),
            ("bram_tiles", OKABE_ITO[2], "BRAM"),
        ]
    ):
        for j, prof in enumerate(profiles):
            hatch = PROFILE_HATCHES[j % len(PROFILE_HATCHES)]
            xs = x + inner_offsets[k] + sub_offsets[j]
            if resource_key == "slice_luts":
                # Stack LUT as Logic (solid) + LUT as Memory (hatched within
                # the same bar via a second draw above).
                logic = [
                    _f(grid[i][j]["lut_as_logic"]) if grid[i][j] else 0
                    for i in range(n_cfg)
                ]
                memory = [
                    _f(grid[i][j]["lut_as_memory"]) if grid[i][j] else 0
                    for i in range(n_cfg)
                ]
                logic = [v or 0 for v in logic]
                memory = [v or 0 for v in memory]
                # Bottom: logic, hatched by profile.
                ax.bar(
                    xs,
                    logic,
                    sub_w,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                    hatch=hatch,
                    label=label if j == 0 else None,
                )
                # Top: memory, drawn with a denser hatch so the split is visible
                # even within the same color.
                ax.bar(
                    xs,
                    memory,
                    sub_w,
                    bottom=logic,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.55,
                    hatch=(hatch + "**") if hatch else "**",
                    label="LUTs (Memory)" if (j == 0 and k == 0) else None,
                )
            else:
                vals = [
                    (_f(grid[i][j].get(resource_key)) or 0) if grid[i][j] else 0
                    for i in range(n_cfg)
                ]
                ax.bar(
                    xs,
                    vals,
                    sub_w,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                    hatch=hatch,
                    label=label if j == 0 else None,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Resource count")
    ax.set_yscale("log")
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
    _add_profile_legend(fig, profiles)
    _style_axes(ax)
    _save(fig, out_dir, "area", fmt)


def plot_power(rows: list[dict], out_dir: Path, fmt: str) -> None:
    rows = _rows_with(rows, "power_total_w")
    if not rows:
        print("  power: no power data; skipping")
        return
    labels, profiles, grid = _group_by_config(rows)
    n_cfg = len(labels)
    n_prof = len(profiles)
    x = np.arange(n_cfg)
    offsets, bar_w = _bar_positions(n_cfg, n_prof)

    fig, ax = plt.subplots(figsize=_figsize())
    for j, prof in enumerate(profiles):
        hatch = PROFILE_HATCHES[j % len(PROFILE_HATCHES)]
        xs = x + offsets[j]
        static = [
            (_f(grid[i][j]["power_static_w"]) or 0) if grid[i][j] else 0
            for i in range(n_cfg)
        ]
        dyn = [
            (_f(grid[i][j]["power_dynamic_w"]) or 0) if grid[i][j] else 0
            for i in range(n_cfg)
        ]
        total = [
            (_f(grid[i][j]["power_total_w"]) or 0) if grid[i][j] else 0
            for i in range(n_cfg)
        ]
        ax.bar(
            xs,
            static,
            bar_w,
            color=OKABE_ITO[7],
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
            label="Static" if j == 0 else None,
        )
        ax.bar(
            xs,
            dyn,
            bar_w,
            bottom=static,
            color=OKABE_ITO[5],
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
            label="Dynamic" if j == 0 else None,
        )
        for xi, t in zip(xs, total):
            if t > 0:
                ax.text(
                    xi,
                    t,
                    f"{t:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=TICK_LABEL_FONTSIZE,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("On-chip power (W)")
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
    _add_profile_legend(fig, profiles)
    _style_axes(ax)
    _save(fig, out_dir, "power_stacked", fmt)


def plot_cycles_per_stage(rows: list[dict], out_dir: Path, fmt: str) -> None:
    rows = _rows_with(rows, "total_cycles")
    if not rows:
        print("  cycles per stage: no cycle data; skipping")
        return
    labels, profiles, grid = _group_by_config(rows)
    n_cfg = len(labels)
    n_prof = len(profiles)
    x = np.arange(n_cfg)
    offsets, bar_w = _bar_positions(n_cfg, n_prof)

    fig, ax = plt.subplots(figsize=_figsize())
    for j in range(n_prof):
        hatch = PROFILE_HATCHES[j % len(PROFILE_HATCHES)]
        xs = x + offsets[j]
        load = [
            (_f(grid[i][j]["cycles_load_hl1"]) or 0) if grid[i][j] else 0
            for i in range(n_cfg)
        ]
        run = [
            (_f(grid[i][j]["cycles_run_timesteps"]) or 0) if grid[i][j] else 0
            for i in range(n_cfg)
        ]
        finish = [
            (_f(grid[i][j]["cycles_finish_q"]) or 0) if grid[i][j] else 0
            for i in range(n_cfg)
        ]
        ax.bar(
            xs,
            load,
            bar_w,
            color=STAGE_COLORS["load_hl1"],
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
            label="LOAD_HL1" if j == 0 else None,
        )
        ax.bar(
            xs,
            run,
            bar_w,
            bottom=load,
            color=STAGE_COLORS["run_timesteps"],
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
            label="RUN_TIMESTEPS" if j == 0 else None,
        )
        bot2 = [a + b for a, b in zip(load, run)]
        ax.bar(
            xs,
            finish,
            bar_w,
            bottom=bot2,
            color=STAGE_COLORS["finish_q"],
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
            label="FINISH_Q" if j == 0 else None,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Cycles per inference")
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left")
    _add_profile_legend(fig, profiles)
    _style_axes(ax)
    _save(fig, out_dir, "cycles_per_stage", fmt)


def plot_cycles_per_ts_substate(
    rows: list[dict],
    out_dir: Path,
    fmt: str,
    normalize: bool,
) -> None:
    """Stacked bar of cycles in each TS substate.

    normalize=False: total cycles across all timesteps (sum equals cycles_run_timesteps).
    normalize=True:  divided by num_timesteps so configs with different timestep counts compare directly.
    """
    rows = [
        r
        for r in rows
        if _f(r.get("cycles_ts_hl1_step")) is not None
        or _f(r.get("cycles_ts_next")) is not None
    ]
    if not rows:
        suffix = "per timestep" if normalize else "total"
        print(f"  cycles per ts-substate ({suffix}): no ts-state data; skipping")
        return
    labels, profiles, grid = _group_by_config(rows)
    n_cfg = len(labels)
    n_prof = len(profiles)
    x = np.arange(n_cfg)
    offsets, bar_w = _bar_positions(n_cfg, n_prof)

    fig, ax = plt.subplots(figsize=_figsize())
    for j in range(n_prof):
        hatch = PROFILE_HATCHES[j % len(PROFILE_HATCHES)]
        xs = x + offsets[j]
        bottom = np.zeros(n_cfg)
        for key, ts_label in TS_SUBSTATE_LABELS:
            vals = np.zeros(n_cfg)
            for i in range(n_cfg):
                if grid[i][j] is None:
                    continue
                raw = _f(grid[i][j].get(f"cycles_ts_{key}")) or 0
                if normalize:
                    nts = _f(grid[i][j].get("num_timesteps")) or 0
                    raw = raw / nts if nts > 0 else 0
                vals[i] = raw
            ax.bar(
                xs,
                vals,
                bar_w,
                bottom=bottom,
                color=TS_SUBSTATE_COLORS[key],
                edgecolor="black",
                linewidth=0.5,
                hatch=hatch,
                label=ts_label if j == 0 else None,
            )
            bottom = bottom + vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Cycles per timestep" if normalize else "Cycles in RUN_TIMESTEPS")
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left", ncol=2)
    _add_profile_legend(fig, profiles)
    _style_axes(ax)
    stem = (
        "cycles_per_ts_substate_per_timestep" if normalize else "cycles_per_ts_substate"
    )
    _save(fig, out_dir, stem, fmt)


def plot_figures_of_merit(rows: list[dict], out_dir: Path, fmt: str) -> None:
    rows = _rows_with(rows, "latency_us")
    if not rows:
        print("  figures of merit: no derived data; skipping")
        return
    labels, profiles, grid = _group_by_config(rows)
    n_cfg = len(labels)
    n_prof = len(profiles)
    x = np.arange(n_cfg)
    offsets, bar_w = _bar_positions(n_cfg, n_prof)

    fig, axes = plt.subplots(1, 3, figsize=_figsize(width_scale=1.5, height_scale=0.6))
    metrics = [
        ("Latency (us)", "latency_us"),
        ("Throughput (Hz)", "throughput_hz"),
        ("Energy (uJ/inf)", "energy_per_inference_uj"),
    ]
    for ax, (ylabel, key) in zip(axes, metrics):
        for j in range(n_prof):
            hatch = PROFILE_HATCHES[j % len(PROFILE_HATCHES)]
            xs = x + offsets[j]
            vals = [
                (_f(grid[i][j].get(key)) or 0) if grid[i][j] else 0
                for i in range(n_cfg)
            ]
            ax.bar(
                xs,
                vals,
                bar_w,
                color=OKABE_ITO[0],
                edgecolor="black",
                linewidth=0.5,
                hatch=hatch,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels, rotation=30, ha="right", fontsize=TICK_LABEL_FONTSIZE - 1
        )
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE - 1)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE - 1)
    _add_profile_legend(fig, profiles)
    fig.tight_layout()
    _save(fig, out_dir, "figures_of_merit", fmt)


# ---------------------------------------------------------------------------
# CSV tables
# ---------------------------------------------------------------------------


def write_timing_table(rows: list[dict], out_dir: Path) -> None:
    rows = [r for r in rows if _f(r.get("fmax_est_mhz")) is not None]
    if not rows:
        print("  performance_timing table: no timing data; skipping")
        return
    fields = [
        "config",
        "profile",
        "display_label",
        "fmax_est_mhz",
        "clock_period_ns",
        "wns_ns",
        "tns_ns",
        "whs_ns",
    ]
    _save_table(rows, out_dir, "performance_timing", fields)


def write_ppa_summary_table(rows: list[dict], out_dir: Path) -> None:
    rows = [r for r in rows if _f(r.get("slice_luts")) is not None]
    if not rows:
        print("  ppa_summary table: no synth data; skipping")
        return
    fields = [
        "config",
        "profile",
        "display_label",
        "slice_luts",
        "lut_as_logic",
        "lut_as_memory",
        "slice_registers",
        "dsp",
        "bram_tiles",
        "fmax_est_mhz",
        "power_total_w",
        "total_cycles",
        "latency_us",
        "throughput_hz",
        "energy_per_inference_uj",
    ]
    _save_table(rows, out_dir, "ppa_summary", fields)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT
        / "3_benchmarking_on_FPGA"
        / "results"
        / "summary"
        / "ppa_cycles_combined.csv",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=REPO_ROOT / "3_benchmarking_on_FPGA" / "results" / "summary" / "plots",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=REPO_ROOT / "3_benchmarking_on_FPGA" / "results" / "summary" / "tables",
    )
    parser.add_argument(
        "--format",
        choices=("png", "svg"),
        default="png",
        help="Plot output format (run twice with different values to get both)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input CSV not found: {args.input}", file=sys.stderr)
        print("Run aggregate_ppa.py first.", file=sys.stderr)
        return 2

    rows = load_rows(args.input)
    if not rows:
        print("No rows in input CSV", file=sys.stderr)
        return 2

    print(f"Plotting {len(rows)} (config x profile) rows from {args.input.name}")
    print(f"  plots -> {args.plot_dir} (.{args.format})")
    plot_area(rows, args.plot_dir, args.format)
    plot_power(rows, args.plot_dir, args.format)
    plot_cycles_per_stage(rows, args.plot_dir, args.format)
    plot_cycles_per_ts_substate(rows, args.plot_dir, args.format, normalize=False)
    plot_cycles_per_ts_substate(rows, args.plot_dir, args.format, normalize=True)
    plot_figures_of_merit(rows, args.plot_dir, args.format)

    print(f"  tables -> {args.table_dir}")
    write_timing_table(rows, args.table_dir)
    write_ppa_summary_table(rows, args.table_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
