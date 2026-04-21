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


def _resolve_signal_name(vcd, requested: str, fallback_suffixes):
    if requested in vcd.references_to_ids:
        return requested

    requested_leaf = requested.split(".")[-1]
    if requested_leaf in vcd.references_to_ids:
        return requested_leaf

    for signal in vcd.references_to_ids:
        if signal.endswith(f".{requested_leaf}"):
            return signal

    for signal in vcd.references_to_ids:
        for suffix in fallback_suffixes:
            suffix_leaf = suffix.lstrip(".")
            if signal.endswith(suffix):
                return signal
            if signal == suffix_leaf:
                return signal
            if signal.endswith(f".{suffix_leaf}"):
                return signal
            if signal.endswith(suffix_leaf):
                return signal

    return None


def _extract_signal(vcd, signal_name: str, tmax: int, signed: bool, width: int):
    if signal_name is None or signal_name not in vcd.references_to_ids:
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


def _downsample(times, values, max_points: int):
    if not times or max_points <= 0 or len(times) <= max_points:
        return times, values

    idx = np.linspace(0, len(times) - 1, num=max_points, dtype=int)
    return [times[i] for i in idx], [values[i] for i in idx]


def _detect_spike_rising_edges(spike_times, spike_values):
    if not spike_times:
        return []

    edges = []
    prev = int(spike_values[0])
    for i in range(1, len(spike_values)):
        curr = int(spike_values[i])
        if prev == 0 and curr == 1:
            edges.append(spike_times[i])
        prev = curr

    return edges


def _compute_isi(spike_edge_times):
    if len(spike_edge_times) < 2:
        return [], []

    isi_times = spike_edge_times[1:]
    isi_values = [
        spike_edge_times[i] - spike_edge_times[i - 1]
        for i in range(1, len(spike_edge_times))
    ]
    return isi_times, isi_values


def _build_constant_segments(times, values, min_duration=0):
    if not times or len(times) < 2:
        return []

    segments = []
    for i in range(len(times) - 1):
        start = times[i]
        end = times[i + 1]
        if end <= start:
            continue

        duration = end - start
        if duration < min_duration:
            continue

        segments.append(
            {
                "start": start,
                "end": end,
                "duration": duration,
                "value": values[i],
            }
        )

    return segments


def _segment_spike_times(spike_edge_times, start, end):
    return [t for t in spike_edge_times if start <= t < end]


def _segment_gain(spike_times):
    if len(spike_times) < 6:
        return None

    intervals = np.diff(spike_times)
    if len(intervals) < 4:
        return None

    window = min(6, len(intervals) // 2)
    if window < 2:
        return None

    early = intervals[:window]
    late = intervals[-window:]
    late_mean = float(np.mean(late))
    if late_mean <= 0:
        return None

    return float(np.mean(early) / late_mean)


def _choose_best_current_segment(
    segments,
    spike_edge_times,
    target_current_raw=None,
):
    if not segments:
        return None

    nonzero_segments = [s for s in segments if s["value"] != 0]
    if not nonzero_segments:
        return None

    enriched = []
    for seg in nonzero_segments:
        seg_spikes = _segment_spike_times(spike_edge_times, seg["start"], seg["end"])
        gain = _segment_gain(seg_spikes)
        enriched.append(
            {
                **seg,
                "spike_count": len(seg_spikes),
                "gain": gain,
            }
        )

    if target_current_raw is not None:
        enriched.sort(
            key=lambda s: (
                abs(s["value"] - target_current_raw),
                -s["spike_count"],
                -s["duration"],
            )
        )
        return enriched[0]

    gain_candidates = [s for s in enriched if s["gain"] is not None]
    if gain_candidates:
        gain_candidates.sort(key=lambda s: (s["gain"], s["spike_count"], s["duration"]))
        return gain_candidates[-1]

    enriched.sort(key=lambda s: (s["spike_count"], s["duration"]))
    return enriched[-1]


def plot_signal(
    vcd_file,
    membrane_signal="fractional_lif.membrane_out[23:0]",
    current_signal="fractional_lif.current[15:0]",
    spike_signal="fractional_lif.spike_out",
    membrane_width=24,
    current_width=16,
    spike_width=1,
    tmax=4_000_000,
    show_current=True,
    show_spike=True,
    show_isi=True,
    max_points=200_000,
    x_min=None,
    x_max=None,
    auto_window_spikes=False,
    auto_window_best_current=False,
    auto_window_padding=50_000,
    min_current_segment_duration=0,
    current_frac_bits=13,
    target_current=None,
    output_file="images/hardware_simulation_hist256.svg",
    show_plot=False,
):
    axis_label_fontsize = 8
    tick_label_fontsize = 6

    vcd = VCDVCD(vcd_file, store_tvs=True)

    mem_name = _resolve_signal_name(
        vcd,
        membrane_signal,
        [
            ".membrane_out[23:0]",
            ".membrane_potential[23:0]",
            ".membrane_out",
            ".membrane_potential",
        ],
    )
    cur_name = _resolve_signal_name(
        vcd,
        current_signal,
        [".current[15:0]", ".current[7:0]", ".current"],
    )
    spk_name = _resolve_signal_name(
        vcd,
        spike_signal,
        [".spike_out", ".spike"],
    )

    print(f"Membrane signal: {mem_name}")
    print(f"Current signal:  {cur_name}")
    print(f"Spike signal:    {spk_name}")

    mem_times, mem_values = _extract_signal(
        vcd=vcd,
        signal_name=mem_name,
        tmax=tmax,
        signed=True,
        width=membrane_width,
    )

    if mem_times is None:
        print("Could not resolve membrane signal in VCD.")
        print("Available signals:")
        for signal in vcd.references_to_ids:
            print(" ", signal)
        return

    if not mem_times:
        print("No valid membrane signal samples found.")
        return

    cur_times, cur_values = _extract_signal(
        vcd=vcd,
        signal_name=cur_name,
        tmax=tmax,
        signed=True,
        width=current_width,
    )

    spike_edge_times = []
    if spk_name is not None:
        spk_times, spk_values = _extract_signal(
            vcd=vcd,
            signal_name=spk_name,
            tmax=tmax,
            signed=False,
            width=spike_width,
        )

        if spk_times is not None and spk_times:
            spike_edge_times = _detect_spike_rising_edges(spk_times, spk_values)
            print(f"Detected spike edges: {len(spike_edge_times)}")

    target_current_raw = None
    if target_current is not None:
        target_current_raw = int(round(target_current * (1 << current_frac_bits)))
        print(
            f"Requested target current: {target_current:.6f} -> raw {target_current_raw} "
            f"(Q*.{current_frac_bits})"
        )

    if auto_window_best_current and cur_times not in [(None, []), None] and cur_values not in [(None, []), None]:
        segments = _build_constant_segments(
            cur_times,
            cur_values,
            min_duration=min_current_segment_duration,
        )
        best_segment = _choose_best_current_segment(
            segments,
            spike_edge_times,
            target_current_raw=target_current_raw,
        )
        if best_segment is not None:
            seg_x_min = max(0, best_segment["start"] - auto_window_padding)
            seg_x_max = min(tmax, best_segment["end"] + auto_window_padding)
            if seg_x_max > seg_x_min:
                x_min = seg_x_min if x_min is None else x_min
                x_max = seg_x_max if x_max is None else x_max

            current_float = best_segment["value"] / float(1 << current_frac_bits)
            print(
                "Best current segment: "
                f"value_raw={best_segment['value']}, value_float={current_float:.6f}, "
                f"start={best_segment['start']}, end={best_segment['end']}, "
                f"spikes={best_segment['spike_count']}, gain={best_segment['gain']}"
            )

    mem_times, mem_values = _downsample(mem_times, mem_values, max_points)

    subplot_count = 1
    subplot_count += 1 if show_current else 0
    subplot_count += 1 if (show_spike and spk_name is not None) else 0
    subplot_count += 1 if (show_isi and spk_name is not None) else 0

    figsize = get_latex_figsize(width_scale=1.6, height_scale=0.35 * subplot_count)

    fig, axes = plt.subplots(
        subplot_count,
        1,
        figsize=(figsize["width"], figsize["height"]),
        sharex=True,
    )
    if subplot_count == 1:
        axes = [axes]

    def scientific_formatter(x, _):
        if x == 0:
            return "0"
        return f"{x:.1e}".replace("e+0", "e").replace("e+", "e")

    ax_index = 0

    ax_mem = axes[ax_index]
    ax_mem.step(mem_times, mem_values, where="post", color="dodgerblue")
    ax_mem.set_ylabel("Membrane", fontsize=axis_label_fontsize)
    ax_mem.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
    ax_mem.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
    ax_mem.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_index += 1

    if show_current:
        ax_cur = axes[ax_index]
        if cur_times is None or not cur_times:
            ax_cur.text(
                0.5,
                0.5,
                "Current signal unavailable",
                transform=ax_cur.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )
        else:
            cur_times, cur_values = _downsample(cur_times, cur_values, max_points)
            ax_cur.step(cur_times, cur_values, where="post", color="darkorange")

        ax_cur.set_ylabel("Current", fontsize=axis_label_fontsize)
        ax_cur.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
        ax_cur.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        ax_cur.grid(True, which="both", linestyle=":", alpha=0.6)
        ax_index += 1

    if show_spike and spk_name is not None:
        ax_spk = axes[ax_index]
        if spike_edge_times:
            y = np.ones(len(spike_edge_times))
            ax_spk.scatter(spike_edge_times, y, marker="|", s=80, color="crimson")
        else:
            ax_spk.text(
                0.5,
                0.5,
                "No spike rising edges detected",
                transform=ax_spk.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )

        ax_spk.set_ylabel("Spikes", fontsize=axis_label_fontsize)
        ax_spk.set_ylim(0.0, 1.5)
        ax_spk.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
        ax_spk.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        ax_spk.grid(True, which="both", linestyle=":", alpha=0.6)
        ax_index += 1

    if show_isi and spk_name is not None:
        ax_isi = axes[ax_index]
        isi_times, isi_values = _compute_isi(spike_edge_times)
        if isi_times:
            isi_times, isi_values = _downsample(isi_times, isi_values, max_points)
            ax_isi.plot(isi_times, isi_values, color="seagreen", linewidth=1.2)
        else:
            ax_isi.text(
                0.5,
                0.5,
                "Not enough spikes for ISI",
                transform=ax_isi.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )

        ax_isi.set_ylabel("ISI (ns)", fontsize=axis_label_fontsize)
        ax_isi.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
        ax_isi.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        ax_isi.grid(True, which="both", linestyle=":", alpha=0.6)

    if auto_window_spikes and len(spike_edge_times) >= 2 and not auto_window_best_current:
        computed_x_min = max(0, spike_edge_times[0] - auto_window_padding)
        computed_x_max = min(tmax, spike_edge_times[-1] + auto_window_padding)
        if computed_x_max > computed_x_min:
            x_min = computed_x_min if x_min is None else x_min
            x_max = computed_x_max if x_max is None else x_max
            print(f"Auto spike window: x_min={x_min}, x_max={x_max}")

    if x_min is None:
        x_min = 0
    if x_max is None:
        x_max = tmax

    axes[-1].set_xlabel("Time (ns)", fontsize=axis_label_fontsize)
    axes[-1].set_xlim(x_min, x_max)

    plt.tight_layout()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = output_path.suffix.lstrip(".") or "svg"
    plt.savefig(output_path, format=output_format, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot membrane/current/spike/ISI traces from a VCD file."
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
        help="Current signal name",
    )
    parser.add_argument(
        "--spike-signal",
        default="fractional_lif.spike_out",
        help="Spike signal name",
    )
    parser.add_argument(
        "--membrane-width",
        type=int,
        default=24,
        help="Membrane signal width",
    )
    parser.add_argument(
        "--current-width",
        type=int,
        default=16,
        help="Current signal width",
    )
    parser.add_argument(
        "--spike-width",
        type=int,
        default=1,
        help="Spike signal width",
    )
    parser.add_argument(
        "--tmax",
        type=int,
        default=4_000_000,
        help="Maximum simulation time to plot",
    )
    parser.add_argument(
        "--no-current",
        action="store_true",
        help="Disable current subplot",
    )
    parser.add_argument(
        "--no-spike",
        action="store_true",
        help="Disable spike subplot",
    )
    parser.add_argument(
        "--no-isi",
        action="store_true",
        help="Disable ISI subplot",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200_000,
        help="Maximum points per plotted trace",
    )
    parser.add_argument(
        "--x-min",
        type=int,
        default=None,
        help="Manual minimum x-axis time (ns)",
    )
    parser.add_argument(
        "--x-max",
        type=int,
        default=None,
        help="Manual maximum x-axis time (ns)",
    )
    parser.add_argument(
        "--auto-window-spikes",
        action="store_true",
        help="Auto-focus x-axis around first and last detected spike",
    )
    parser.add_argument(
        "--auto-window-best-current",
        action="store_true",
        help="Auto-focus x-axis to a single constant-current segment with strongest adaptation",
    )
    parser.add_argument(
        "--auto-window-padding",
        type=int,
        default=50_000,
        help="Padding (ns) added around auto spike window",
    )
    parser.add_argument(
        "--min-current-segment-duration",
        type=int,
        default=0,
        help="Minimum duration (ns) for constant-current segments considered in best-current mode",
    )
    parser.add_argument(
        "--current-frac-bits",
        type=int,
        default=13,
        help="Fractional bits used to convert raw current to float in logs",
    )
    parser.add_argument(
        "--target-current",
        type=float,
        default=None,
        help="Optional target current in float units to focus the nearest segment",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive window after saving figure",
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
        spike_signal=args.spike_signal,
        membrane_width=args.membrane_width,
        current_width=args.current_width,
        spike_width=args.spike_width,
        tmax=args.tmax,
        show_current=not args.no_current,
        show_spike=not args.no_spike,
        show_isi=not args.no_isi,
        max_points=args.max_points,
        x_min=args.x_min,
        x_max=args.x_max,
        auto_window_spikes=args.auto_window_spikes,
        auto_window_best_current=args.auto_window_best_current,
        auto_window_padding=args.auto_window_padding,
        min_current_segment_duration=args.min_current_segment_duration,
        current_frac_bits=args.current_frac_bits,
        target_current=args.target_current,
        output_file=args.output,
        show_plot=args.show,
    )
