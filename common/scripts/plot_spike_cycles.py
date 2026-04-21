import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_latex_figsize(width_scale=1.0, height_scale=None):
    doc_textwidth_mm = 117.0
    inches_per_mm = 1 / 25.4
    doc_textwidth_in = doc_textwidth_mm * inches_per_mm

    fig_width = doc_textwidth_in * width_scale

    if height_scale is None:
        golden_ratio = (np.sqrt(5) - 1.0) / 2.0
        fig_height = fig_width * golden_ratio
    else:
        fig_height = doc_textwidth_in * height_scale

    return {"width": fig_width, "height": fig_height}


def load_spike_cycles(csv_path: Path):
    cycles = []
    lengths = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cycle_num = row.get("cycle_num", "").strip()
            cycle_length = row.get("cycle_length", "").strip()
            if not cycle_num or not cycle_length:
                continue

            cycles.append(int(cycle_num))
            lengths.append(float(cycle_length))

    if not cycles:
        raise ValueError(f"No usable rows found in {csv_path}")

    cycles_arr = np.array(cycles, dtype=float)
    lengths_arr = np.array(lengths, dtype=float)
    return cycles_arr, lengths_arr


def moving_average(y, window):
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="valid")


def summarize_adaptation(lengths, early_count=10, late_count=10):
    n = len(lengths)
    e = min(early_count, n)
    late_n = min(late_count, n)

    early_mean = float(np.mean(lengths[:e]))
    late_mean = float(np.mean(lengths[-late_n:]))

    gain = early_mean / late_mean if late_mean > 0 else float("nan")
    return early_mean, late_mean, gain


def plot_spike_cycles(
    csv_file,
    output_file,
    title="Spike Interval Adaptation",
    smooth_window=1,
    early_count=10,
    late_count=10,
    show_plot=False,
):
    csv_path = Path(csv_file)
    output_path = Path(output_file)

    cycles, cycle_lengths = load_spike_cycles(csv_path)
    frequencies = 1.0 / cycle_lengths

    early_mean, late_mean, gain = summarize_adaptation(
        cycle_lengths,
        early_count=early_count,
        late_count=late_count,
    )

    print(f"Loaded cycles: {len(cycles)}")
    print(f"Early mean interval: {early_mean:.4f}")
    print(f"Late mean interval:  {late_mean:.4f}")
    print(f"Interval gain (early/late): {gain:.4f}")

    figsize = get_latex_figsize(width_scale=1.6, height_scale=0.85)
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(figsize["width"], figsize["height"]),
        sharex=True,
    )

    ax1.plot(cycles, cycle_lengths, color="dodgerblue", linewidth=1.0, alpha=0.9)
    ax1.scatter(cycles, cycle_lengths, color="dodgerblue", s=8, alpha=0.5)
    ax1.set_ylabel("Cycle Length")
    ax1.grid(True, linestyle=":", alpha=0.6)

    if smooth_window > 1 and len(cycles) >= smooth_window:
        smooth_lengths = moving_average(cycle_lengths, smooth_window)
        smooth_cycles = cycles[smooth_window - 1 :]
        ax1.plot(
            smooth_cycles,
            smooth_lengths,
            color="navy",
            linewidth=1.5,
            label=f"MA({smooth_window})",
        )
        ax1.legend(loc="best")

    ax2.plot(cycles, frequencies, color="seagreen", linewidth=1.0, alpha=0.9)
    ax2.scatter(cycles, frequencies, color="seagreen", s=8, alpha=0.5)
    ax2.set_xlabel("Spike Cycle Index")
    ax2.set_ylabel("Spike Freq (1/steps)")
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"{title}\nInterval gain (early/late): {gain:.3f}", fontsize=10)
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
        default="images/fractional_spike_cycles_adaptation.svg",
        help="Output plot path",
    )
    parser.add_argument(
        "--title",
        default="Spike Interval Adaptation",
        help="Plot title",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving-average window (>=1)",
    )
    parser.add_argument(
        "--early-count",
        type=int,
        default=10,
        help="Number of early cycles for summary",
    )
    parser.add_argument(
        "--late-count",
        type=int,
        default=10,
        help="Number of late cycles for summary",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window",
    )
    args = parser.parse_args()

    plot_spike_cycles(
        csv_file=args.csv_file,
        output_file=args.output,
        title=args.title,
        smooth_window=max(1, args.smooth_window),
        early_count=max(1, args.early_count),
        late_count=max(1, args.late_count),
        show_plot=args.show,
    )


if __name__ == "__main__":
    main()
