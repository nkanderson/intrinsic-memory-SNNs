"""Plot stage-3 PPA + inference-cycle metrics, with CSV side-tables.

Reads 3_benchmarking_on_FPGA/results/summary/ppa_cycles_combined.csv and
emits a mix of plots and CSV tables under
3_benchmarking_on_FPGA/results/summary/plots/ and .../tables/.

The aggregate CSV has one row per (config, profile). When more than one
profile is present for any config, bars are grouped by config with one
sub-bar per profile (profile distinguished by hatch pattern). When only
the default "baseline" profile is present, plots collapse to a single
bar per config.

Profiles in DEFAULT_EXCLUDED_PROFILES are skipped unless re-included with
--include. Multiple --include flags are accepted:

    python 3_benchmarking_on_FPGA/scripts/plot_ppa.py --include buffer_bram

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

# Hatch patterns (retained for potential use; no longer applied by default).
PROFILE_HATCHES = ["", "//", "xx", "..", "++", "\\\\"]

# Human-readable labels for known profile names, used as per-bar tick labels.
PROFILE_DISPLAY_LABELS: dict[str, str] = {
    "baseline": "baseline",
    "fc1_8": "fc1 batch 8",
    "fc2_8": "fc2 batch 8",
    "q_1_fc2_8": "Q batch 1,\nfc2 batch 8",
    "q_batch_size_1": "Q batch 1",
    "reverse_case_neurons": "One-hot\nreverse case",
}

# Profiles excluded from plots by default. These were synthesis experiments
# where Vivado did not ultimately honor the directive (the buffer_bram /
# buffer_lutram memory-style hints), so they carry no useful comparison
# signal. Re-include any of them with --include <name>.
DEFAULT_EXCLUDED_PROFILES = ("buffer_bram", "buffer_lutram")

# Output resolution for raster plots. 150 DPI at 7" → ~1050 px wide, matching
# the range produced by the other project plotting scripts (150-180 DPI).
SAVE_DPI = 150

# Nexys A7-100T total available resources (for utilization % chart).
NEXYS_A7_100T_RESOURCES = {
    "slice_luts": 63_400,
    "slice_registers": 126_800,
    "dsp": 240,
    "bram_tiles": 135,
}


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
    """Return figure dimensions based on DEFAULT_FIGSIZE (7.0 × 4.3 inches).

    width_scale multiplies the width; height follows the same factor unless
    height_scale is given explicitly (then height = base_height × height_scale).
    """
    w, h = DEFAULT_FIGSIZE
    if height_scale is None:
        return (w * width_scale, h)
    return (w * width_scale, h * height_scale)


def _save(fig: plt.Figure, out_dir: Path, stem: str, fmt: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.{fmt}"
    # Pass dpi only for raster formats; SVG is resolution-independent.
    extra = {"dpi": SAVE_DPI} if fmt == "png" else {}
    fig.savefig(path, format=fmt, bbox_inches="tight", **extra)
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


def _profile_alpha(j: int) -> float:
    """Alpha opacity for profile bar index j. Baseline (j=0) is fully opaque."""
    return max(0.45, 1.0 - j * 0.18)


def _add_profile_legend(fig: plt.Figure, profile_names: list[str]) -> None:
    """Add a figure-level legend mapping alpha shade -> profile when N_profiles > 1.

    Placed below the axes so it doesn't collide with per-axes legends. No-op
    for the single-profile case (the visual collapses to the original layout).
    """
    if len(profile_names) <= 1:
        return
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor="dimgray",
            edgecolor="black",
            alpha=_profile_alpha(i),
        )
        for i in range(len(profile_names))
    ]
    fig.subplots_adjust(bottom=0.28)
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


def _set_two_level_xticks(
    ax: plt.Axes,
    x: np.ndarray,
    offsets: np.ndarray,
    profiles: list[str],
    config_labels: list[str],
    rotation: int = 38,
) -> None:
    """Angled per-bar profile tick labels with bold config group labels below.

    Used for multi-profile bar charts where the x-axis has a two-level
    hierarchy: individual profile bars (labeled by name) within config clusters
    (labeled by config display name).
    """
    n_cfg = len(x)
    n_prof = len(offsets)
    bar_positions = [float(x[i] + offsets[j]) for i in range(n_cfg) for j in range(n_prof)]
    bar_labels = [
        PROFILE_DISPLAY_LABELS.get(profiles[j], profiles[j])
        for _ in range(n_cfg)
        for j in range(n_prof)
    ]
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, rotation=rotation, ha="right",
                       fontsize=TICK_LABEL_FONTSIZE - 1)
    # Config group labels: centered under each cluster, below the profile labels.
    # get_xaxis_transform maps (data_x, axes_fraction_y); negative y is below the axes.
    for xi, cfg_label in zip(x, config_labels):
        ax.text(xi, -0.40, cfg_label, ha="center", va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=TICK_LABEL_FONTSIZE, fontweight="bold")


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

    any_bram = any(
        (_f(grid[i][j].get("bram_tiles")) or 0) > 0
        for i in range(n_cfg) for j in range(n_prof) if grid[i][j]
    )
    any_lut_memory = any(
        (_f(grid[i][j].get("lut_as_memory")) or 0) > 0
        for i in range(n_cfg) for j in range(n_prof) if grid[i][j]
    )

    # Build list of resources to show, suppressing any that are all-zero.
    resources_abs = [
        ("slice_luts", OKABE_ITO[0], "LUTs"),
        ("slice_registers", OKABE_ITO[4], "FFs"),
        ("dsp", OKABE_ITO[5], "DSP"),
    ]
    if any_bram:
        resources_abs.append(("bram_tiles", OKABE_ITO[2], "BRAM"))

    if n_prof == 1:
        # ── Single profile: absolute grouped bar chart (log scale). ──────────
        n_res = len(resources_abs)
        x = np.arange(n_cfg)
        inner_offsets, inner_w = _bar_positions(n_cfg, n_res, group_width=0.8)

        fig, ax = plt.subplots(figsize=_figsize(width_scale=1.2))
        for k, (res_key, color, label) in enumerate(resources_abs):
            xs = x + inner_offsets[k]
            if res_key == "slice_luts":
                logic = [(_f(grid[i][0]["lut_as_logic"]) or 0) if grid[i][0] else 0 for i in range(n_cfg)]
                memory = [(_f(grid[i][0]["lut_as_memory"]) or 0) if grid[i][0] else 0 for i in range(n_cfg)]
                ax.bar(xs, logic, inner_w, color=color, edgecolor="black", linewidth=0.5, label=label)
                if any_lut_memory:
                    ax.bar(xs, memory, inner_w, bottom=logic, color=color, edgecolor="black",
                           linewidth=0.5, alpha=0.55, hatch="**", label="LUTs (Memory)")
            else:
                vals = [(_f(grid[i][0].get(res_key)) or 0) if grid[i][0] else 0 for i in range(n_cfg)]
                ax.bar(xs, vals, inner_w, color=color, edgecolor="black", linewidth=0.5, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Resource count")
        ax.set_yscale("log")
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
        _style_axes(ax)
        _save(fig, out_dir, "area", fmt)

    else:
        # ── Multi-profile: % change from baseline, one panel per resource. ───
        baseline_idx = next((j for j, p in enumerate(profiles) if p == "baseline"), None)
        non_baseline = [(j, p) for j, p in enumerate(profiles) if p != "baseline"]
        n_nbp = len(non_baseline)
        if n_nbp == 0 or baseline_idx is None:
            print("  area: cannot render % change (need a 'baseline' profile + at least one other); skipping")
            return

        # Only include resources that have a nonzero baseline for at least one config
        # (avoids divide-by-zero and meaningless panels).
        resources_pct = [
            (label, res_key) for res_key, _, label in resources_abs
            if any((_f(grid[i][baseline_idx].get(res_key)) or 0) > 0 for i in range(n_cfg) if grid[i][baseline_idx])
        ]
        n_panels = len(resources_pct)
        if n_panels == 0:
            print("  area: no nonzero resources to compare; skipping")
            return

        fig, axes = plt.subplots(1, n_panels, figsize=_figsize(width_scale=0.55 + 0.45 * n_panels))
        if n_panels == 1:
            axes = [axes]

        x = np.arange(n_nbp)
        cfg_offsets, bar_w = _bar_positions(n_nbp, n_cfg)
        cfg_colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(n_cfg)]

        for panel_idx, (res_label, res_key) in enumerate(resources_pct):
            ax = axes[panel_idx]
            for i_cfg in range(n_cfg):
                base_row = grid[i_cfg][baseline_idx]
                base_val = (_f(base_row.get(res_key)) or 0) if base_row else None
                pcts = []
                for j, _ in non_baseline:
                    row = grid[i_cfg][j]
                    if row and base_val and base_val > 0:
                        pcts.append(((_f(row.get(res_key)) or 0) - base_val) / base_val * 100)
                    else:
                        pcts.append(float("nan"))
                xs = x + cfg_offsets[i_cfg]
                ax.bar(xs, pcts, bar_w, color=cfg_colors[i_cfg], edgecolor="black",
                       linewidth=0.5, label=labels[i_cfg] if panel_idx == 0 else None)

            ax.axhline(0, color="black", linewidth=0.8, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [PROFILE_DISPLAY_LABELS.get(p, p) for _, p in non_baseline],
                rotation=35, ha="right", fontsize=TICK_LABEL_FONTSIZE - 1,
            )
            ax.set_title(res_label, fontsize=AXIS_LABEL_FONTSIZE)
            if panel_idx == 0:
                ax.set_ylabel("% change vs baseline")
            _style_axes(ax)

        axes[0].legend(fontsize=LEGEND_FONTSIZE, loc="best")
        fig.tight_layout()
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
        xs = x + offsets[j]
        static = [(_f(grid[i][j]["power_static_w"]) or 0) if grid[i][j] else 0 for i in range(n_cfg)]
        dyn = [(_f(grid[i][j]["power_dynamic_w"]) or 0) if grid[i][j] else 0 for i in range(n_cfg)]
        total = [(_f(grid[i][j]["power_total_w"]) or 0) if grid[i][j] else 0 for i in range(n_cfg)]
        ax.bar(xs, static, bar_w, color=OKABE_ITO[7], edgecolor="black",
               linewidth=0.5, label="Static" if j == 0 else None)
        ax.bar(xs, dyn, bar_w, bottom=static, color=OKABE_ITO[5],
               edgecolor="black", linewidth=0.5, label="Dynamic" if j == 0 else None)
        if n_prof == 1:
            for xi, t in zip(xs, total):
                if t > 0:
                    ax.text(xi, t, f"{t:.3f}", ha="center", va="bottom",
                            fontsize=TICK_LABEL_FONTSIZE)

    ax.set_ylabel("On-chip power (W)")
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
    if n_prof == 1:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
    else:
        _set_two_level_xticks(ax, x, offsets, profiles, labels)
        fig.subplots_adjust(bottom=0.48)
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
        xs = x + offsets[j]
        load = [(_f(grid[i][j]["cycles_load_hl1"]) or 0) if grid[i][j] else 0 for i in range(n_cfg)]
        run = [(_f(grid[i][j]["cycles_run_timesteps"]) or 0) if grid[i][j] else 0 for i in range(n_cfg)]
        finish = [(_f(grid[i][j]["cycles_finish_q"]) or 0) if grid[i][j] else 0 for i in range(n_cfg)]
        ax.bar(xs, load, bar_w, color=STAGE_COLORS["load_hl1"], edgecolor="black",
               linewidth=0.5, label="LOAD_HL1" if j == 0 else None)
        ax.bar(xs, run, bar_w, bottom=load, color=STAGE_COLORS["run_timesteps"],
               edgecolor="black", linewidth=0.5, label="RUN_TIMESTEPS" if j == 0 else None)
        bot2 = [a + b for a, b in zip(load, run)]
        ax.bar(xs, finish, bar_w, bottom=bot2, color=STAGE_COLORS["finish_q"],
               edgecolor="black", linewidth=0.5, label="FINISH_Q" if j == 0 else None)

    ax.set_ylabel("Cycles per inference")
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left")
    if n_prof == 1:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
    else:
        _set_two_level_xticks(ax, x, offsets, profiles, labels)
        fig.subplots_adjust(bottom=0.48)
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
        alpha = _profile_alpha(j)
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
            ax.bar(xs, vals, bar_w, bottom=bottom, color=TS_SUBSTATE_COLORS[key],
                   edgecolor="black", linewidth=0.5, alpha=alpha,
                   label=ts_label if j == 0 else None)
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
            xs = x + offsets[j]
            vals = [(_f(grid[i][j].get(key)) or 0) if grid[i][j] else 0 for i in range(n_cfg)]
            ax.bar(xs, vals, bar_w, color=OKABE_ITO[0], edgecolor="black",
                   linewidth=0.5, alpha=_profile_alpha(j))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=TICK_LABEL_FONTSIZE - 1)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE - 1)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE - 1)
    fig.tight_layout()
    _add_profile_legend(fig, profiles)
    _save(fig, out_dir, "figures_of_merit", fmt)


def plot_utilization_pct(rows: list[dict], out_dir: Path, fmt: str) -> None:
    """Resource utilization as % of available on the Nexys A7-100T (baseline only).

    Always filters to the baseline profile so the chart is a clean reference
    for how much of the board each config occupies. Profile-to-profile
    differences are handled by the % change area chart.
    """
    baseline_rows = [r for r in rows if r.get("profile") == "baseline" and _f(r.get("slice_luts")) is not None]
    if not baseline_rows:
        print("  utilization %: no baseline synth data; skipping")
        return

    labels = [r["display_label"] for r in baseline_rows]
    n_cfg = len(labels)

    any_bram = any((_f(r.get("bram_tiles")) or 0) > 0 for r in baseline_rows)
    resources = [
        ("slice_luts", "LUTs", OKABE_ITO[0]),
        ("slice_registers", "FFs", OKABE_ITO[4]),
        ("dsp", "DSPs", OKABE_ITO[5]),
    ]
    if any_bram:
        resources.append(("bram_tiles", "BRAM", OKABE_ITO[2]))

    n_res = len(resources)
    x = np.arange(n_cfg)
    inner_offsets, inner_w = _bar_positions(n_cfg, n_res, group_width=0.8)

    fig, ax = plt.subplots(figsize=_figsize(width_scale=1.0))
    for k, (res_key, res_label, color) in enumerate(resources):
        total_avail = NEXYS_A7_100T_RESOURCES[res_key]
        vals = [(_f(r.get(res_key)) or 0) / total_avail * 100 for r in baseline_rows]
        xs = x + inner_offsets[k]
        ax.bar(xs, vals, inner_w, color=color, edgecolor="black", linewidth=0.5, label=res_label)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Utilization (% of available)")
    ax.set_ylim(0, 105)
    ax.axhline(100, color="black", linewidth=0.8, linestyle="--", alpha=0.4, label="100% limit")
    ax.legend(fontsize=LEGEND_FONTSIZE)
    _style_axes(ax)
    _save(fig, out_dir, "utilization_pct", fmt)


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
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="PROFILE",
        help="Re-include a profile that is excluded by default "
             f"(currently: {', '.join(DEFAULT_EXCLUDED_PROFILES)}). "
             "May be repeated.",
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

    # Apply profile exclusion filter. "baseline" is never excluded.
    excluded = set(DEFAULT_EXCLUDED_PROFILES) - set(args.include)
    if excluded:
        before = len(rows)
        rows = [r for r in rows if r.get("profile", "") not in excluded]
        dropped = before - len(rows)
        if dropped:
            print(f"  excluded profiles {sorted(excluded)}: dropped {dropped} rows "
                  f"(use --include <name> to re-add)")

    print(f"Plotting {len(rows)} (config x profile) rows from {args.input.name}")
    print(f"  plots -> {args.plot_dir} (.{args.format})")
    plot_area(rows, args.plot_dir, args.format)
    plot_utilization_pct(rows, args.plot_dir, args.format)
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
