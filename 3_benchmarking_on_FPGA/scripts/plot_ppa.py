"""Plot stage-3 PPA + inference-cycle metrics.

Reads 3_benchmarking_on_FPGA/results/summary/ppa_cycles_combined.csv and
emits five plots under 3_benchmarking_on_FPGA/results/summary/plots/.

Run from the repo root (the import of common.scripts.plot_styles expects this):

    python 3_benchmarking_on_FPGA/scripts/plot_ppa.py            # PNG (default)
    python 3_benchmarking_on_FPGA/scripts/plot_ppa.py --format svg
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Repo root is parents[2] of this file: 3_benchmarking_on_FPGA/scripts/plot_ppa.py
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.scripts.plot_styles import (  # noqa: E402
    OKABE_ITO,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    get_latex_figsize,
)


NEURON_COLOR = {
    "lif": OKABE_ITO[0],         # blue
    "fractional": OKABE_ITO[1],  # vermillion
    "bitshift": OKABE_ITO[2],    # bluish green
}

# Stage colors used in the per-stage stacked bar
STAGE_COLORS = {
    "load_hl1": OKABE_ITO[4],        # sky blue
    "run_timesteps": OKABE_ITO[5],   # orange
    "finish_q": OKABE_ITO[6],        # yellow
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


def _figsize() -> tuple[float, float]:
    fs = get_latex_figsize(width_scale=1.0)
    return (fs["width"], fs["height"])


def _save(fig: plt.Figure, out_dir: Path, stem: str, fmt: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.{fmt}"
    fig.savefig(path, format=fmt, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def _style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.xaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONTSIZE)


def _rows_with(rows: list[dict], key: str) -> list[dict]:
    return [r for r in rows if _f(r.get(key)) is not None]


def plot_area(rows: list[dict], out_dir: Path, fmt: str) -> None:
    rows = _rows_with(rows, "slice_luts")
    if not rows:
        print("  area: no synth data; skipping")
        return
    labels = [r["display_label"] for r in rows]
    luts = [_f(r["slice_luts"]) or 0 for r in rows]
    ffs = [_f(r["slice_registers"]) or 0 for r in rows]
    dsp = [_f(r["dsp"]) or 0 for r in rows]
    bram = [_f(r["bram_tiles"]) or 0 for r in rows]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=_figsize())
    ax.bar(x - 1.5 * width, luts, width, label="LUTs", color=OKABE_ITO[0])
    ax.bar(x - 0.5 * width, ffs, width, label="FFs", color=OKABE_ITO[4])
    ax.bar(x + 0.5 * width, dsp, width, label="DSP", color=OKABE_ITO[5])
    ax.bar(x + 1.5 * width, bram, width, label="BRAM", color=OKABE_ITO[2])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Resource count")
    ax.set_yscale("log")
    ax.legend(fontsize=LEGEND_FONTSIZE)
    _style_axes(ax)
    _save(fig, out_dir, "area", fmt)


def plot_performance(rows: list[dict], out_dir: Path, fmt: str) -> None:
    rows = _rows_with(rows, "fmax_est_mhz")
    if not rows:
        print("  performance: no fmax data; skipping")
        return
    labels = [r["display_label"] for r in rows]
    fmax = [_f(r["fmax_est_mhz"]) or 0 for r in rows]
    colors = [NEURON_COLOR.get(r.get("neuron_type", ""), OKABE_ITO[7]) for r in rows]

    fig, ax = plt.subplots(figsize=_figsize())
    ax.bar(np.arange(len(labels)), fmax, color=colors)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Estimated Fmax (MHz)")
    ax.axhline(100, color=OKABE_ITO[7], linestyle="--", linewidth=0.8, label="100 MHz target")
    ax.legend(fontsize=LEGEND_FONTSIZE)
    _style_axes(ax)
    _save(fig, out_dir, "performance_fmax", fmt)


def plot_power(rows: list[dict], out_dir: Path, fmt: str) -> None:
    rows = _rows_with(rows, "power_total_w")
    if not rows:
        print("  power: no power data; skipping")
        return
    labels = [r["display_label"] for r in rows]
    dyn = [_f(r["power_dynamic_w"]) or 0 for r in rows]
    static = [_f(r["power_static_w"]) or 0 for r in rows]
    total = [_f(r["power_total_w"]) or 0 for r in rows]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=_figsize())
    ax.bar(x, static, label="Static", color=OKABE_ITO[7])
    ax.bar(x, dyn, bottom=static, label="Dynamic", color=OKABE_ITO[5])
    for xi, t in zip(x, total):
        ax.text(xi, t, f"{t:.3f}", ha="center", va="bottom", fontsize=TICK_LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("On-chip power (W)")
    ax.legend(fontsize=LEGEND_FONTSIZE)
    _style_axes(ax)
    _save(fig, out_dir, "power_stacked", fmt)


def plot_cycles_per_stage(rows: list[dict], out_dir: Path, fmt: str) -> None:
    rows = _rows_with(rows, "total_cycles")
    if not rows:
        print("  cycles: no cycle data; skipping")
        return
    labels = [r["display_label"] for r in rows]
    load = [_f(r["cycles_load_hl1"]) or 0 for r in rows]
    run = [_f(r["cycles_run_timesteps"]) or 0 for r in rows]
    finish = [_f(r["cycles_finish_q"]) or 0 for r in rows]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=_figsize())
    ax.bar(x, load, label="LOAD_HL1", color=STAGE_COLORS["load_hl1"])
    ax.bar(x, run, bottom=load, label="RUN_TIMESTEPS", color=STAGE_COLORS["run_timesteps"])
    bottom2 = [a + b for a, b in zip(load, run)]
    ax.bar(x, finish, bottom=bottom2, label="FINISH_Q", color=STAGE_COLORS["finish_q"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Cycles per inference")
    ax.legend(fontsize=LEGEND_FONTSIZE)
    _style_axes(ax)
    _save(fig, out_dir, "cycles_per_stage", fmt)


def plot_figures_of_merit(rows: list[dict], out_dir: Path, fmt: str) -> None:
    rows = _rows_with(rows, "latency_us")
    if not rows:
        print("  figures of merit: no derived data; skipping")
        return
    labels = [r["display_label"] for r in rows]
    lat = [_f(r["latency_us"]) or 0 for r in rows]
    thr = [_f(r["throughput_hz"]) or 0 for r in rows]
    en = [_f(r["energy_per_inference_uj"]) or 0 for r in rows]
    colors = [NEURON_COLOR.get(r.get("neuron_type", ""), OKABE_ITO[7]) for r in rows]

    fs = get_latex_figsize(width_scale=1.0, height_scale=0.5)
    fig, axes = plt.subplots(1, 3, figsize=(fs["width"], fs["height"]))
    metrics = [
        ("Latency (us)", lat),
        ("Throughput (Hz)", thr),
        ("Energy (uJ/inf)", en),
    ]
    x = np.arange(len(labels))
    for ax, (ylabel, vals) in zip(axes, metrics):
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=TICK_LABEL_FONTSIZE - 2)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE - 1)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE - 1)
    fig.tight_layout()
    _save(fig, out_dir, "figures_of_merit", fmt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=REPO_ROOT / "3_benchmarking_on_FPGA" / "results" / "summary" / "ppa_cycles_combined.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "3_benchmarking_on_FPGA" / "results" / "summary" / "plots",
    )
    parser.add_argument("--format", choices=("png", "svg"), default="png",
                        help="Output format (run twice with different values to get both)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input CSV not found: {args.input}", file=sys.stderr)
        print("Run aggregate_ppa.py first.", file=sys.stderr)
        return 2

    rows = load_rows(args.input)
    if not rows:
        print("No rows in input CSV", file=sys.stderr)
        return 2

    print(f"Plotting {len(rows)} configs from {args.input.name} -> {args.out_dir} (.{args.format})")
    plot_area(rows, args.out_dir, args.format)
    plot_performance(rows, args.out_dir, args.format)
    plot_power(rows, args.out_dir, args.format)
    plot_cycles_per_stage(rows, args.out_dir, args.format)
    plot_figures_of_merit(rows, args.out_dir, args.format)
    return 0


if __name__ == "__main__":
    sys.exit(main())
