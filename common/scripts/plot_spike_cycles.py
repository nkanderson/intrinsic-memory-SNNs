import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path for the plot_styles import.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.scripts.plot_styles import (
    OKABE_ITO,
    COLOR_RAW,
    COLOR_MA,
    PHASE_COLORS,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    get_latex_figsize,
)


def load_spike_cycles(csv_path: Path):
    """Load spike-cycle data from CSV.

    Returns (cycles_arr, lengths_arr, phases_arr) where phases_arr is a string
    array of phase labels (empty string when the column is absent).
    """
    cycles = []
    lengths = []
    phases = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        has_phase = "phase" in (reader.fieldnames or [])
        for row in reader:
            cycle_num = row.get("cycle_num", "").strip()
            cycle_length = row.get("cycle_length", "").strip()
            if not cycle_num or not cycle_length:
                continue

            cycles.append(int(cycle_num))
            lengths.append(float(cycle_length))
            phases.append(row.get("phase", "constant").strip() if has_phase else "constant")

    if not cycles:
        raise ValueError(f"No usable rows found in {csv_path}")

    cycles_arr = np.array(cycles, dtype=float)
    lengths_arr = np.array(lengths, dtype=float)
    phases_arr = np.array(phases, dtype=object)
    return cycles_arr, lengths_arr, phases_arr


def moving_average(y, window):
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="valid")


def summarize_adaptation(lengths, phases, early_count=10, late_count=10):
    n = len(lengths)
    e = min(early_count, n)
    late_n = min(late_count, n)

    early_mean = float(np.mean(lengths[:e]))
    late_mean = float(np.mean(lengths[-late_n:]))
    gain = early_mean / late_mean if late_mean > 0 else float("nan")

    distinct_phases = list(dict.fromkeys(phases))
    multi_phase = len(distinct_phases) > 1

    phase_summaries = {}
    if multi_phase:
        for label in distinct_phases:
            mask = phases == label
            phase_lengths = lengths[mask]
            if len(phase_lengths) >= 4:
                ph_early = float(np.mean(phase_lengths[:min(early_count, len(phase_lengths))]))
                phase_summaries[label] = ph_early

    return early_mean, late_mean, gain, phase_summaries


def plot_spike_cycles(
    csv_file,
    output_file,
    title="",
    smooth_window=1,
    early_count=10,
    late_count=10,
    max_cycles=None,
    min_cycle_length=0,
    show_plot=False,
):
    csv_path = Path(csv_file)
    output_path = Path(output_file)

    cycles, cycle_lengths, phases = load_spike_cycles(csv_path)

    # Filter out very short cycles (e.g. the within-doublet ISI=2 cycles produced by
    # the fractional LIF). These compress the y-axis and obscure the adaptation in
    # the longer between-doublet ISIs that actually carry the trend.
    if min_cycle_length > 0:
        mask = cycle_lengths >= min_cycle_length
        dropped = len(cycles) - int(mask.sum())
        cycles = cycles[mask]
        cycle_lengths = cycle_lengths[mask]
        phases = phases[mask]
        if dropped > 0:
            print(f"Filtered out {dropped} cycles shorter than {min_cycle_length} steps")

    if max_cycles is not None:
        mask = cycles <= max_cycles
        cycles = cycles[mask]
        cycle_lengths = cycle_lengths[mask]
        phases = phases[mask]

    frequencies = 1.0 / cycle_lengths

    early_mean, late_mean, gain, phase_summaries = summarize_adaptation(
        cycle_lengths, phases, early_count=early_count, late_count=late_count,
    )

    print(f"Loaded cycles: {len(cycles)}")
    print(f"Early mean interval: {early_mean:.4f}")
    print(f"Late mean interval:  {late_mean:.4f}")
    print(f"Interval gain (early/late): {gain:.4f}")
    if phase_summaries:
        print(f"Per-phase early means: {phase_summaries}")

    distinct_phases = list(dict.fromkeys(phases))
    multi_phase = len(distinct_phases) > 1

    figsize = get_latex_figsize(width_scale=1.6, height_scale=0.85)
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(figsize["width"], figsize["height"]),
        sharex=True,
    )

    if multi_phase:
        for label in distinct_phases:
            mask = phases == label
            color = PHASE_COLORS.get(label, COLOR_RAW)
            ax1.scatter(cycles[mask], cycle_lengths[mask], color=color, s=8, alpha=0.7, label=label)
            ax1.plot(cycles[mask], cycle_lengths[mask], color=color, linewidth=0.8, alpha=0.5)
            ax2.scatter(cycles[mask], frequencies[mask], color=color, s=8, alpha=0.7, label=label)
            ax2.plot(cycles[mask], frequencies[mask], color=color, linewidth=0.8, alpha=0.5)

        # Draw vertical lines at phase transitions.
        for ax in (ax1, ax2):
            prev_label = phases[0]
            for i in range(1, len(phases)):
                if phases[i] != prev_label:
                    ax.axvline(cycles[i], color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
                prev_label = phases[i]

        ax1.legend(loc="best", fontsize=6)
    else:
        color = PHASE_COLORS.get(distinct_phases[0] if distinct_phases else "constant", OKABE_ITO[2])
        ax1.plot(cycles, cycle_lengths, color=color, linewidth=1.0, alpha=0.9)
        ax1.scatter(cycles, cycle_lengths, color=color, s=8, alpha=0.5)
        ax2.plot(cycles, frequencies, color=color, linewidth=1.0, alpha=0.9)
        ax2.scatter(cycles, frequencies, color=color, s=8, alpha=0.5)

    ax1.set_ylabel("Cycle Length")
    ax1.grid(True, linestyle=":", alpha=0.6)

    if smooth_window > 1 and len(cycles) >= smooth_window:
        smooth_lengths = moving_average(cycle_lengths, smooth_window)
        smooth_cycles = cycles[smooth_window - 1:]
        ax1.plot(
            smooth_cycles, smooth_lengths,
            color=COLOR_MA, linewidth=1.5, label=f"MA({smooth_window})",
        )
        ax1.legend(loc="best", fontsize=6)

    ax2.set_xlabel("Spike Cycle Index")
    ax2.set_ylabel("Spike Freq (1/steps)")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Title is opt-in: by default the figure is left untitled so the figure caption
    # can carry the summary statistics. Numeric summaries are still printed to stdout.
    if title:
        if phase_summaries:
            summary_parts = [f"{k} early={v:.1f}" for k, v in phase_summaries.items()]
            if "startup" in phase_summaries and "recovery" in phase_summaries:
                recovery_gain = phase_summaries["startup"] / phase_summaries["recovery"]
                summary_parts.append(f"recovery_gain={recovery_gain:.3f}")
            title_full = f"{title}\n" + ", ".join(summary_parts)
        else:
            title_full = f"{title}\nInterval gain (early/late): {gain:.3f}"
        fig.suptitle(title_full, fontsize=10)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = output_path.suffix.lstrip(".") or "svg"
    plt.savefig(output_path, format=output_format, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot spike-cycle adaptation from CSV exported by cocotb memory tests."
    )
    parser.add_argument("csv_file", help="Input CSV path (spike cycle export)")
    parser.add_argument(
        "--output",
        default="images/fractional_lif_memory_spike_cycles.svg",
        help="Output plot path",
    )
    parser.add_argument(
        "--title", default="",
        help="Figure suptitle. Defaults to empty (figure caption carries the summary).",
    )
    parser.add_argument("--smooth-window", type=int, default=1, help="Moving-average window (>=1)")
    parser.add_argument("--early-count", type=int, default=10)
    parser.add_argument("--late-count", type=int, default=10)
    parser.add_argument(
        "--max-cycles", type=int, default=None, metavar="N",
        help="Clip x-axis to the first N cycles (useful to focus on early adaptation)",
    )
    parser.add_argument(
        "--min-cycle-length", type=int, default=0, metavar="N",
        help="Drop cycles shorter than N steps. Useful to filter doublet within-pair "
             "ISIs (typically length 2) so the long-ISI adaptation trend is visible.",
    )
    parser.add_argument("--show", action="store_true", help="Show interactive window")
    args = parser.parse_args()

    plot_spike_cycles(
        csv_file=args.csv_file,
        output_file=args.output,
        title=args.title,
        smooth_window=max(1, args.smooth_window),
        early_count=max(1, args.early_count),
        late_count=max(1, args.late_count),
        max_cycles=args.max_cycles,
        min_cycle_length=max(0, args.min_cycle_length),
        show_plot=args.show,
    )


if __name__ == "__main__":
    main()
