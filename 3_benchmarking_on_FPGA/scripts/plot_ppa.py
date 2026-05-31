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
import matplotlib.patches as mpatches
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
    "load_hl1": OKABE_ITO[2],  # bluish green
    "run_timesteps": OKABE_ITO[5],  # orange
    "finish_q": OKABE_ITO[0],  # blue
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

# Human-readable labels for known profile names used in legends.
PROFILE_DISPLAY_LABELS: dict[str, str] = {
    "baseline": "baseline",
    "fc1_8": "fc1 batch 8",
    "fc2_8": "fc2 batch 8",
    "q_1_fc2_8": "Q batch 1, fc2 batch 8",
    "q_batch_size_1": "Q batch 1",
    "reverse_case_neurons": "One-hot reverse case",
}

# Per-profile colors used when color encodes the synthesis profile.
# Black for baseline (the reference); distinct Okabe-Ito hues for variants.
PROFILE_COLORS: dict[str, str] = {
    "baseline": OKABE_ITO[7],  # black
    "fc1_8": OKABE_ITO[0],  # blue
    "fc2_8": OKABE_ITO[2],  # bluish green
    "q_1_fc2_8": OKABE_ITO[3],  # reddish purple
    "q_batch_size_1": OKABE_ITO[4],  # sky blue
    "reverse_case_neurons": OKABE_ITO[1],  # vermillion
}

# Profiles excluded from plots by default. These were synthesis experiments
# where Vivado did not ultimately honor the directive (the buffer_bram /
# buffer_lutram memory-style hints), so they carry no useful comparison
# signal. Re-include any of them with --include <name>.
DEFAULT_EXCLUDED_PROFILES = ("buffer_bram", "buffer_lutram")

# Neuron type colors matching plot_optuna_results.py ("leaky" maps to "lif" here).
NEURON_TYPE_COLORS: dict[str, str] = {
    "lif": OKABE_ITO[0],  # blue
    "fractional": OKABE_ITO[2],  # bluish green
    "bitshift": OKABE_ITO[5],  # orange
}

STATIC_POWER_COLOR = "#aaaaaa"  # medium gray; keeps black edge visible in stacked bars

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


def _neuron_color(neuron_type: str) -> str:
    return NEURON_TYPE_COLORS.get(neuron_type, OKABE_ITO[7])


_CONFIG_NEURON_ORDER = {"lif": 0, "frac": 1, "bitshift": 2}


def _config_sort_order(config_name: str) -> int:
    for prefix, order in _CONFIG_NEURON_ORDER.items():
        if config_name.startswith(prefix):
            return order
    return 99


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

    order.sort(key=_config_sort_order)

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


def _profile_legend_handles(profiles: list[str]) -> list[mpatches.Patch]:
    """Legend patch handles mapping profile color → display label."""
    return [
        mpatches.Patch(
            facecolor=PROFILE_COLORS.get(p, OKABE_ITO[7]),
            label=PROFILE_DISPLAY_LABELS.get(p, p),
        )
        for p in profiles
    ]


def _set_numbered_xticks(
    ax: plt.Axes,
    x: np.ndarray,
    offsets: np.ndarray,
    profiles: list[str],
    config_labels: list[str],
    grid: Optional[list[list[Optional[dict]]]] = None,
) -> None:
    """Number labels (1, 2, …) at bar positions that have data; bold config labels below.

    If grid is provided, positions where grid[i][j] is None are silently skipped
    (no tick mark drawn for missing config/profile combinations).
    """
    n_cfg = len(x)
    n_prof = len(offsets)
    bar_positions = []
    bar_labels = []
    for i in range(n_cfg):
        for j in range(n_prof):
            if grid is None or grid[i][j] is not None:
                bar_positions.append(float(x[i] + offsets[j]))
                bar_labels.append(str(j + 1))
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, fontsize=TICK_LABEL_FONTSIZE - 1)
    for xi, cfg_label in zip(x, config_labels):
        ax.text(
            xi,
            -0.09,
            cfg_label,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=TICK_LABEL_FONTSIZE,
            fontweight="bold",
        )


def _profile_mapping_text(profiles: list[str]) -> str:
    """'1 = baseline    2 = fc1 batch 8    …' for the figure annotation."""
    return "    ".join(
        f"{i + 1} = {PROFILE_DISPLAY_LABELS.get(p, p)}" for i, p in enumerate(profiles)
    )


def _mapping_text_wrapped(profiles: list[str], per_line: int = 3) -> tuple[str, int]:
    """Wrapped profile mapping text. Returns (text, n_lines)."""
    parts = [
        f"{i + 1} = {PROFILE_DISPLAY_LABELS.get(p, p)}" for i, p in enumerate(profiles)
    ]
    lines = [
        "    ".join(parts[i : i + per_line]) for i in range(0, len(parts), per_line)
    ]
    return "\n".join(lines), len(lines)


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
        for i in range(n_cfg)
        for j in range(n_prof)
        if grid[i][j]
    )
    any_lut_memory = any(
        (_f(grid[i][j].get("lut_as_memory")) or 0) > 0
        for i in range(n_cfg)
        for j in range(n_prof)
        if grid[i][j]
    )

    # Build list of resources to show, suppressing any that are all-zero.
    resources_abs = [
        ("slice_luts", OKABE_ITO[0], "LUTs"),
        ("slice_registers", OKABE_ITO[4], "FFs"),
        ("dsp", OKABE_ITO[5], "DSP"),
    ]
    if any_bram:
        resources_abs.append(("bram_tiles", OKABE_ITO[2], "BRAM"))

    cfg_neuron_types = [
        next(
            (grid[i][j].get("neuron_type", "") for j in range(n_prof) if grid[i][j]), ""
        )
        for i in range(n_cfg)
    ]

    resources = [("slice_luts", "LUTs"), ("slice_registers", "FFs"), ("dsp", "DSPs")]
    if any_bram:
        resources.append(("bram_tiles", "BRAM"))
    n_res = len(resources)

    if n_prof == 1:
        # ── Single profile: x = resource type, bars = neuron-type-colored configs ──
        x = np.arange(n_res)
        cfg_offsets, bar_w = _bar_positions(n_res, n_cfg, group_width=0.8)

        fig, ax = plt.subplots(figsize=_figsize(width_scale=1.2))
        for i_cfg, (ntype, lbl) in enumerate(zip(cfg_neuron_types, labels)):
            color = _neuron_color(ntype)
            for k, (res_key, _) in enumerate(resources):
                xi = x[k] + cfg_offsets[i_cfg]
                row = grid[i_cfg][0]
                if res_key == "slice_luts":
                    logic = (_f(row["lut_as_logic"]) or 0) if row else 0
                    memory = (_f(row["lut_as_memory"]) or 0) if row else 0
                    ax.bar(
                        xi,
                        logic,
                        bar_w,
                        color=color,
                        edgecolor="black",
                        linewidth=0.5,
                        label=lbl if k == 0 else None,
                    )
                    if any_lut_memory and memory > 0:
                        ax.bar(
                            xi,
                            memory,
                            bar_w,
                            bottom=logic,
                            color=color,
                            edgecolor="black",
                            linewidth=0.5,
                            alpha=0.55,
                            hatch="//",
                        )
                else:
                    val = (_f(row.get(res_key)) or 0) if row else 0
                    ax.bar(
                        xi,
                        val,
                        bar_w,
                        color=color,
                        edgecolor="black",
                        linewidth=0.5,
                        label=lbl if k == 0 else None,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([res_label for _, res_label in resources])
        ax.set_ylabel("Resource count")
        ax.set_yscale("log")
        legend_handles = [
            mpatches.Patch(
                facecolor=_neuron_color(nt), edgecolor="black", linewidth=0.5, label=lbl
            )
            for nt, lbl in zip(cfg_neuron_types, labels)
        ]
        if any_lut_memory:
            legend_handles.append(
                mpatches.Patch(
                    facecolor="#888888",
                    alpha=0.55,
                    hatch="//",
                    edgecolor="black",
                    linewidth=0.5,
                    label="LUT (Memory, hatched)",
                )
            )
        ax.legend(handles=legend_handles, fontsize=LEGEND_FONTSIZE, loc="upper right")
        _style_axes(ax)
        _save(fig, out_dir, "area", fmt)

    else:
        # ── Multi-profile: % change from baseline, one panel per resource. ───
        baseline_idx = next(
            (j for j, p in enumerate(profiles) if p == "baseline"), None
        )
        non_baseline = [(j, p) for j, p in enumerate(profiles) if p != "baseline"]
        n_nbp = len(non_baseline)
        if n_nbp == 0 or baseline_idx is None:
            print(
                "  area: cannot render % change (need a 'baseline' profile + at least one other); skipping"
            )
            return

        resources_pct = [
            (res_label, res_key)
            for res_key, res_label in resources
            if any(
                (_f(grid[i][baseline_idx].get(res_key)) or 0) > 0
                for i in range(n_cfg)
                if grid[i][baseline_idx]
            )
        ]
        n_panels = len(resources_pct)
        if n_panels == 0:
            print("  area: no nonzero resources to compare; skipping")
            return

        fig, axes = plt.subplots(
            1, n_panels, figsize=_figsize(width_scale=0.55 + 0.45 * n_panels)
        )
        if n_panels == 1:
            axes = [axes]

        x = np.arange(n_nbp)
        cfg_offsets, bar_w = _bar_positions(n_nbp, n_cfg)
        # Non-baseline profiles numbered 1, 2, … (baseline is the reference, not shown).
        nb_nums = list(range(1, n_nbp + 1))

        # Scale fonts up to compensate for narrower panel width.
        _fs_label = AXIS_LABEL_FONTSIZE + 5
        _fs_tick = TICK_LABEL_FONTSIZE + 3
        _fs_legend = LEGEND_FONTSIZE + 3

        for panel_idx, (res_label, res_key) in enumerate(resources_pct):
            ax = axes[panel_idx]
            for i_cfg, ntype in enumerate(cfg_neuron_types):
                base_row = grid[i_cfg][baseline_idx]
                base_val = (_f(base_row.get(res_key)) or 0) if base_row else None
                pcts = []
                for j, _ in non_baseline:
                    row = grid[i_cfg][j]
                    if row and base_val and base_val > 0:
                        pcts.append(
                            ((_f(row.get(res_key)) or 0) - base_val) / base_val * 100
                        )
                    else:
                        pcts.append(float("nan"))
                xs = x + cfg_offsets[i_cfg]
                ax.bar(
                    xs,
                    pcts,
                    bar_w,
                    color=_neuron_color(ntype),
                    edgecolor="black",
                    linewidth=0.5,
                    label=labels[i_cfg] if panel_idx == 0 else None,
                )

            ax.axhline(0, color="black", linewidth=0.8, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([str(n) for n in nb_nums], fontsize=_fs_tick)
            ax.set_title(res_label, fontsize=_fs_label)
            if panel_idx == 0:
                ax.set_ylabel("% change vs baseline")
            _style_axes(ax)
            ax.tick_params(axis="both", labelsize=_fs_tick)
            ax.yaxis.label.set_size(_fs_label)

        axes[0].legend(fontsize=_fs_legend, loc="best")
        # Mapping: only non-baseline profiles numbered from 1; wrap at 3 per line.
        nb_parts = [
            f"{k + 1} = {PROFILE_DISPLAY_LABELS.get(p, p)}"
            for k, (_, p) in enumerate(non_baseline)
        ]
        mapping_lines = [
            "    ".join(nb_parts[i : i + 3]) for i in range(0, len(nb_parts), 3)
        ]
        n_map_lines = len(mapping_lines)
        bottom_frac = 0.12 + n_map_lines * 0.10
        fig.tight_layout()
        fig.subplots_adjust(bottom=min(bottom_frac, 0.32))
        fig.text(
            0.5,
            0.12,  # Adjust vertical position of profile list
            "\n".join(mapping_lines),
            ha="center",
            va="bottom",
            fontsize=AXIS_LABEL_FONTSIZE * 1.3,
        )
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

    fig, ax = plt.subplots(figsize=_figsize(height_scale=1.0 if n_prof == 1 else 1.2))
    for j, prof in enumerate(profiles):
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
            color=STATIC_POWER_COLOR,
            edgecolor="black",
            linewidth=0.5,
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
            label="Dynamic" if j == 0 else None,
        )
        if n_prof == 1:
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

    ax.set_ylabel("On-chip power (W)")
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left")
    if n_prof == 1:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
    else:
        _set_numbered_xticks(ax, x, offsets, profiles, labels, grid=grid)
        mapping_text, n_map_lines = _mapping_text_wrapped(profiles)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.12 + n_map_lines * 0.07)
        fig.text(
            0.5,
            0.1,
            mapping_text,
            ha="center",
            va="bottom",
            fontsize=LEGEND_FONTSIZE * 1.2,
        )
    _style_axes(ax)
    _save(fig, out_dir, "power_stacked", fmt)


def plot_cycles_stage_pct(rows: list[dict], out_dir: Path, fmt: str) -> None:
    """100% stacked bar: proportion of each top-level FSM stage, all profiles."""
    rows = _rows_with(rows, "total_cycles")
    if not rows:
        print("  cycles stage %: no cycle data; skipping")
        return
    labels, profiles, grid = _group_by_config(rows)
    n_cfg = len(labels)
    n_prof = len(profiles)
    x = np.arange(n_cfg)
    offsets, bar_w = _bar_positions(n_cfg, n_prof)

    fig, ax = plt.subplots(figsize=_figsize(height_scale=1.0 if n_prof == 1 else 1.2))
    for j in range(n_prof):
        xs = x + offsets[j]
        load_pct, run_pct, finish_pct = [], [], []
        for i in range(n_cfg):
            r = grid[i][j]
            total = (_f(r.get("total_cycles")) or 0) if r else 0
            if total > 0:
                load_pct.append((_f(r.get("cycles_load_hl1")) or 0) / total * 100)
                run_pct.append((_f(r.get("cycles_run_timesteps")) or 0) / total * 100)
                finish_pct.append((_f(r.get("cycles_finish_q")) or 0) / total * 100)
            else:
                load_pct.append(0)
                run_pct.append(0)
                finish_pct.append(0)
        ax.bar(
            xs,
            load_pct,
            bar_w,
            color=STAGE_COLORS["load_hl1"],
            label="LOAD_HL1" if j == 0 else None,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.bar(
            xs,
            run_pct,
            bar_w,
            bottom=load_pct,
            color=STAGE_COLORS["run_timesteps"],
            label="RUN_TIMESTEPS" if j == 0 else None,
            edgecolor="white",
            linewidth=0.3,
        )
        bot2 = [a + b for a, b in zip(load_pct, run_pct)]
        ax.bar(
            xs,
            finish_pct,
            bar_w,
            bottom=bot2,
            color=STAGE_COLORS["finish_q"],
            label="FINISH_Q" if j == 0 else None,
            edgecolor="white",
            linewidth=0.3,
        )

    ax.set_ylabel("Cycles (% of total per inference)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="center right")
    if n_prof == 1:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
    else:
        _set_numbered_xticks(ax, x, offsets, profiles, labels, grid=grid)
        mapping_text, n_map_lines = _mapping_text_wrapped(profiles)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.12 + n_map_lines * 0.07)
        fig.text(
            0.5,
            0.09,
            mapping_text,
            ha="center",
            va="bottom",
            fontsize=LEGEND_FONTSIZE * 1.2,
        )
    _style_axes(ax)
    _save(fig, out_dir, "cycles_stage_pct", fmt)


def _plot_cycles_bars(
    rows: list[dict],
    out_dir: Path,
    fmt: str,
    col: str,
    ylabel: str,
    stem: str,
) -> None:
    rows = _rows_with(rows, col)
    if not rows:
        print(f"  {stem}: no cycle data; skipping")
        return
    labels, profiles, grid = _group_by_config(rows)
    n_cfg = len(labels)
    n_prof = len(profiles)
    x = np.arange(n_cfg)
    offsets, bar_w = _bar_positions(n_cfg, n_prof)

    fig, ax = plt.subplots(figsize=_figsize())
    for j, prof in enumerate(profiles):
        color = PROFILE_COLORS.get(prof, OKABE_ITO[7])
        xs = x + offsets[j]
        vals = [
            (_f(grid[i][j].get(col)) or 0) if grid[i][j] else 0 for i in range(n_cfg)
        ]
        ax.bar(xs, vals, bar_w, color=color, edgecolor="white", linewidth=0.5)

    ax.set_ylabel(ylabel)
    if n_prof == 1:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.legend(
            handles=_profile_legend_handles(profiles),
            fontsize=LEGEND_FONTSIZE,
            loc="upper left",
        )
    else:
        _set_numbered_xticks(ax, x, offsets, profiles, labels, grid=grid)
        ax.legend(
            handles=_profile_legend_handles(profiles),
            fontsize=LEGEND_FONTSIZE,
            loc="upper left",
        )
        mapping_text, n_map_lines = _mapping_text_wrapped(profiles)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.12 + n_map_lines * 0.07)
        fig.text(
            0.5,
            0.08,
            mapping_text,
            ha="center",
            va="bottom",
            fontsize=LEGEND_FONTSIZE * 1.2,
        )
    _style_axes(ax)
    _save(fig, out_dir, stem, fmt)


def plot_cycles_total(rows: list[dict], out_dir: Path, fmt: str) -> None:
    _plot_cycles_bars(
        rows, out_dir, fmt, "total_cycles", "Cycles per inference", "cycles_total"
    )


def plot_cycles_run_timesteps(rows: list[dict], out_dir: Path, fmt: str) -> None:
    _plot_cycles_bars(
        rows,
        out_dir,
        fmt,
        "cycles_run_timesteps",
        "Cycles in RUN_TIMESTEPS",
        "cycles_run_timesteps",
    )


def plot_cycles_per_timestep(rows: list[dict], out_dir: Path, fmt: str) -> None:
    _plot_cycles_bars(
        rows,
        out_dir,
        fmt,
        "cycles_per_timestep",
        "Cycles per timestep",
        "cycles_per_timestep",
    )


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

    fig, axes = plt.subplots(1, 3, figsize=_figsize(width_scale=1.5, height_scale=0.7))
    metrics = [
        ("Latency (us)", "latency_us"),
        ("Throughput (Hz)", "throughput_hz"),
        ("Energy (uJ/inf)", "energy_per_inference_uj"),
    ]
    for ax, (ylabel, key) in zip(axes, metrics):
        for j, prof in enumerate(profiles):
            color = PROFILE_COLORS.get(prof, OKABE_ITO[7])
            xs = x + offsets[j]
            vals = [
                (_f(grid[i][j].get(key)) or 0) if grid[i][j] else 0
                for i in range(n_cfg)
            ]
            ax.bar(xs, vals, bar_w, color=color, edgecolor="white", linewidth=0.5)
        if n_prof == 1:
            ax.set_xticks(x)
            ax.set_xticklabels(
                labels, rotation=20, ha="right", fontsize=TICK_LABEL_FONTSIZE - 1
            )
        else:
            _set_numbered_xticks(ax, x, offsets, profiles, labels, grid=grid)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE - 1)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE - 1)
        _style_axes(ax)

    if n_prof == 1:
        fig.tight_layout()
    else:
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)
        fig.text(
            0.5,
            0.02,
            _profile_mapping_text(profiles),
            ha="center",
            va="bottom",
            fontsize=LEGEND_FONTSIZE,
        )
    _save(fig, out_dir, "figures_of_merit", fmt)


def plot_utilization_pct(rows: list[dict], out_dir: Path, fmt: str) -> None:
    """Resource utilization as % of available on the Nexys A7-100T (baseline only).

    Always filters to the baseline profile so the chart is a clean reference
    for how much of the board each config occupies. Profile-to-profile
    differences are handled by the % change area chart.
    """
    baseline_rows = [
        r
        for r in rows
        if r.get("profile") == "baseline" and _f(r.get("slice_luts")) is not None
    ]
    baseline_rows.sort(key=lambda r: _config_sort_order(r.get("config", "")))
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
        ax.bar(
            xs,
            vals,
            inner_w,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=res_label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Utilization (% of available)")
    ax.set_ylim(0, 105)
    ax.axhline(
        100, color="black", linewidth=0.8, linestyle="--", alpha=0.4, label="100% limit"
    )
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left")
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
            print(
                f"  excluded profiles {sorted(excluded)}: dropped {dropped} rows "
                f"(use --include <name> to re-add)"
            )

    print(f"Plotting {len(rows)} (config x profile) rows from {args.input.name}")
    print(f"  plots -> {args.plot_dir} (.{args.format})")
    plot_area(rows, args.plot_dir, args.format)
    plot_utilization_pct(rows, args.plot_dir, args.format)
    plot_power(rows, args.plot_dir, args.format)
    plot_cycles_stage_pct(rows, args.plot_dir, args.format)
    plot_cycles_total(rows, args.plot_dir, args.format)
    plot_cycles_run_timesteps(rows, args.plot_dir, args.format)
    plot_cycles_per_timestep(rows, args.plot_dir, args.format)
    plot_figures_of_merit(rows, args.plot_dir, args.format)

    print(f"  tables -> {args.table_dir}")
    write_timing_table(rows, args.table_dir)
    write_ppa_summary_table(rows, args.table_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
