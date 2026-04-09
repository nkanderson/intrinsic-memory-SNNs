import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from vcdvcd import VCDVCD
import argparse


def get_latex_figsize(width_scale=1.0, height_scale=None):
    """
    Calculates the figure size in inches for a LaTeX document based on its
    standard text width.
    """
    doc_textwidth_mm = 117.0
    inches_per_mm = 1 / 25.4
    doc_textwidth_in = doc_textwidth_mm * inches_per_mm

    fig_width = doc_textwidth_in * width_scale

    if height_scale is None:
        # Use the golden ratio for a pleasing aspect ratio
        golden_ratio = (np.sqrt(5) - 1.0) / 2.0
        fig_height = fig_width * golden_ratio
    else:
        fig_height = doc_textwidth_in * height_scale

    return {"width": fig_width, "height": fig_height}


def plot_signal(
    vcd_file,
    signal_name="frac_order_lif.membrane_potential[7:0]",
    tmax=4_000_000,
    show_current=True,
):
    # Font size configuration
    axis_label_fontsize = 8
    tick_label_fontsize = 6

    # Parse VCD file (store_tvs=True keeps time-value changes)
    vcd = VCDVCD(vcd_file, store_tvs=True)

    # Match signal exactly
    if signal_name not in vcd.references_to_ids:
        print(f"Signal '{signal_name}' not found in {vcd_file}")
        print("Available signals:")
        for s in vcd.references_to_ids:
            print(" ", s)
        return

    sig_id = vcd.references_to_ids[signal_name]
    tv = vcd.data[sig_id].tv

    # Extract times/values and limit to tmax
    times = []
    values = []
    for t, v in tv:
        if t > tmax:
            break
        if v in ("x", "z"):
            continue
        try:
            val = int(v, 2)  # convert binary string (e.g., "10101010") to int
            values.append(val)
            times.append(t)
        except ValueError:
            continue

    if not times:
        print("No valid data points found.")
        return

    # Extend the last value to tmax if needed
    if times and times[-1] < tmax:
        times.append(tmax)
        values.append(values[-1])

    figsize = get_latex_figsize(
        width_scale=1.5, height_scale=0.75 if show_current else 0.5
    )

    if show_current:
        # Create subplot for membrane potential and current
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(figsize["width"], figsize["height"]), sharex=True
        )
    else:
        # fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize["width"], figsize["height"]))

    # Plot membrane potential
    # ax1.step(times, values, where="post", color="blue", linewidth=1.5, label="Membrane Potential")
    ax1.step(times, values, where="post", color="dodgerblue", label=None)
    ax1.set_ylabel("Membrane Potential", fontsize=axis_label_fontsize)
    ax1.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)

    # Custom formatter to show scientific notation without offset
    def scientific_formatter(x, pos):
        if x == 0:
            return "0"
        else:
            return f"{x:.1e}".replace("e+0", "e").replace("e+", "e")

    ax1.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
    # ax1.grid(True, alpha=0.3)
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)

    # Add x-axis label if not showing current (when showing current, it goes on ax2)
    if not show_current:
        ax1.set_xlabel("Time (ns)", fontsize=axis_label_fontsize)
        ax1.set_xlim(0, tmax)

    if show_current:
        # Plot current signal on bottom subplot
        current_signal = "frac_order_lif.current[7:0]"
        current_plotted = False
        if current_signal in vcd.references_to_ids:
            current_id = vcd.references_to_ids[current_signal]
            current_tv = vcd.data[current_id].tv
            # Extract current times and values
            current_times = []
            current_values = []
            for t, v in current_tv:
                if t > tmax:
                    break
                if v in ("x", "z"):
                    continue
                try:
                    val = int(v, 2)
                    current_values.append(val)
                    current_times.append(t)
                except ValueError:
                    continue
            if current_times:
                # Extend the last value to tmax if needed
                if current_times[-1] < tmax:
                    current_times.append(tmax)
                    current_values.append(current_values[-1])
                # ax2.step(current_times, current_values, where="post", color="red", linewidth=1.5, label="Input Current")
                ax2.step(
                    current_times,
                    current_values,
                    where="post",
                    color="darkorange",
                    # linewidth=1.5,
                )
                current_plotted = True
        if not current_plotted:
            # If no current signal found, create empty plot with message
            ax2.text(
                0.5,
                0.5,
                "Current signal not found in VCD file",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
            )
        ax2.set_ylabel("Input Current", fontsize=axis_label_fontsize)
        ax2.set_xlabel("Time (ns)", fontsize=axis_label_fontsize)
        ax2.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
        ax2.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        # ax2.grid(True, alpha=0.3)
        ax2.grid(True, which="both", linestyle=":", alpha=0.6)
        ax2.set_xlim(0, tmax)

    plt.tight_layout()
    plt.savefig(
        "images/hardware_simulation_hist256.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot membrane potential and optionally current from VCD file."
    )
    parser.add_argument("vcd_file", help="Path to VCD file")
    parser.add_argument(
        "signal_name",
        nargs="?",
        default="frac_order_lif.membrane_potential[7:0]",
        help="Signal name for membrane potential",
    )
    parser.add_argument(
        "--show-current", action="store_true", help="Show the current signal"
    )
    args = parser.parse_args()

    plot_signal(args.vcd_file, args.signal_name, show_current=args.show_current)
