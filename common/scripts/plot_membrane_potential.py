import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from vcdvcd import VCDVCD


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


def _to_int(value_bits: str, width: int, signed: bool):
    value = int(value_bits, 2)
    if signed and value >= (1 << (width - 1)):
        value -= 1 << width
    return value


def _extract_signal(vcd, signal_name: str, tmax: int, signed: bool, width: int):
    if signal_name not in vcd.references_to_ids:
        return None, None

    signal_id = vcd.references_to_ids[signal_name]
    tv = vcd.data[signal_id].tv

    times = []
    values = []
    for t, v in tv:
        if t > tmax:
            break
        if v in ("x", "z"):
            continue
        try:
            values.append(_to_int(v, width, signed))
            times.append(t)
        except ValueError:
            continue

    if not times:
        return [], []

    if times[-1] < tmax:
        times.append(tmax)
        values.append(values[-1])

    return times, values


def plot_signal(
    vcd_file,
    membrane_signal="fractional_lif.membrane_out[23:0]",
    current_signal="fractional_lif.current[15:0]",
    membrane_width=24,
    current_width=16,
    tmax=4_000_000,
    show_current=True,
    output_file="images/hardware_simulation_hist256.svg",
):
    axis_label_fontsize = 8
    tick_label_fontsize = 6

    vcd = VCDVCD(vcd_file, store_tvs=True)

    mem_times, mem_values = _extract_signal(
        vcd=vcd,
        signal_name=membrane_signal,
        tmax=tmax,
        signed=True,
        width=membrane_width,
    )

    if mem_times is None:
        print(f"Signal '{membrane_signal}' not found in {vcd_file}")
        print("Available signals:")
        for signal in vcd.references_to_ids:
            print(" ", signal)
        return

    if not mem_times:
        print("No valid membrane signal samples found.")
        return

    figsize = get_latex_figsize(
        width_scale=1.5, height_scale=0.75 if show_current else 0.5
    )

    if show_current:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(figsize["width"], figsize["height"]), sharex=True
        )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize["width"], figsize["height"]))

    ax1.step(mem_times, mem_values, where="post", color="dodgerblue", label=None)
    ax1.set_ylabel("Membrane Potential", fontsize=axis_label_fontsize)
    ax1.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)

    def scientific_formatter(x, _):
        if x == 0:
            return "0"
        return f"{x:.1e}".replace("e+0", "e").replace("e+", "e")

    ax1.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)

    if not show_current:
        ax1.set_xlabel("Time (ns)", fontsize=axis_label_fontsize)
        ax1.set_xlim(0, tmax)

    if show_current:
        cur_times, cur_values = _extract_signal(
            vcd=vcd,
            signal_name=current_signal,
            tmax=tmax,
            signed=True,
            width=current_width,
        )

        if cur_times is None or not cur_times:
            ax2.text(
                0.5,
                0.5,
                f"Current signal '{current_signal}' not found",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )
        else:
            ax2.step(cur_times, cur_values, where="post", color="darkorange")

        ax2.set_ylabel("Input Current", fontsize=axis_label_fontsize)
        ax2.set_xlabel("Time (ns)", fontsize=axis_label_fontsize)
        ax2.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
        ax2.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        ax2.grid(True, which="both", linestyle=":", alpha=0.6)
        ax2.set_xlim(0, tmax)

    plt.tight_layout()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format=output_path.suffix.lstrip(".") or "svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot membrane potential and current from VCD file for fractional_lif."
    )
    parser.add_argument("vcd_file", help="Path to VCD file")
    parser.add_argument(
        "--membrane-signal",
        default="fractional_lif.membrane_out[23:0]",
        help="Membrane potential signal name",
    )
    parser.add_argument(
        "--current-signal",
        default="fractional_lif.current[15:0]",
        help="Input current signal name",
    )
    parser.add_argument(
        "--tmax",
        type=int,
        default=4_000_000,
        help="Maximum simulation time to plot",
    )
    parser.add_argument(
        "--show-current",
        action="store_true",
        help="Show current subplot",
    )
    parser.add_argument(
        "--output",
        default="images/hardware_simulation_hist256.svg",
        help="Output image path",
    )
    args = parser.parse_args()

    plot_signal(
        vcd_file=args.vcd_file,
        membrane_signal=args.membrane_signal,
        current_signal=args.current_signal,
        tmax=args.tmax,
        show_current=args.show_current,
        output_file=args.output,
    )
