"""Plot membrane potential, current, spike raster, and ISI traces from a VCD file.

Accepts VCD input only. To plot from a cocotb FST, first convert:
    fst2vcd results/sim_build/fractional_lif.fst -o fractional_lif.vcd
(fst2vcd is bundled with GTKWave.)

Typical usage (constant-current capture):
    python common/scripts/plot_membrane_potential.py fractional_lif.vcd \\
        --output common/images/fractional_lif_memory_constant.svg

Dropout/recovery (with phase shading from the spike-cycles CSV):
    python common/scripts/plot_membrane_potential.py fractional_lif.vcd \\
        --phase-csv common/sv/cocotb/results/fractional_lif_dropout_recovery_spike_cycles.csv \\
        --output common/images/fractional_lif_memory_dropout.svg
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from vcdvcd import VCDVCD


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
COLOR_RAW = "#999999"

COLOR_MEMBRANE = OKABE_ITO[0]   # blue
COLOR_CURRENT = OKABE_ITO[5]    # orange
COLOR_SPIKE = OKABE_ITO[1]      # vermillion
COLOR_ISI = OKABE_ITO[2]        # bluish green
COLOR_MA = OKABE_ITO[7]         # black

PHASE_COLORS = {
    "startup": COLOR_RAW,
    "dropout": OKABE_ITO[1],   # vermillion
    "recovery": OKABE_ITO[2],  # bluish green
}


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


_UNIT_TO_NS = {
    "fs": 1e-6, "ps": 1e-3, "ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9,
}


def _vcd_time_scale_to_ns(vcd) -> float:
    """Return the multiplier that converts raw vcdvcd time values to nanoseconds.

    Icarus's default `$timescale` is 1ps, so without scaling all VCD times are 1000×
    larger than the sidecar JSON's `get_sim_time(unit='ns')` values, and phase shading
    falls completely outside the data window.
    """
    ts = getattr(vcd, "timescale", None)
    if not ts:
        return 1.0
    if isinstance(ts, dict):
        magnitude = ts.get("magnitude", 1)
        unit = ts.get("unit", "ns")
    else:
        import re
        m = re.match(r"\s*(\d+)\s*([a-zA-Z]+)", str(ts))
        if not m:
            return 1.0
        magnitude = int(m.group(1))
        unit = m.group(2)
    return float(magnitude) * _UNIT_TO_NS.get(unit.lower(), 1.0)


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


def _extract_signal(vcd, signal_name: str, tmax_ns: float, signed: bool, width: int,
                    time_scale_ns: float = 1.0):
    """Extract a signal's (time, value) trace. tmax_ns and returned times are both in ns."""
    if signal_name is None or signal_name not in vcd.references_to_ids:
        return None, None

    signal_id = vcd.references_to_ids[signal_name]
    tv = vcd.data[signal_id].tv

    # vcd.tv stores times in the VCD's precision unit; the loop compares in that unit
    # to avoid per-sample multiplication, then scales kept timestamps to ns at the end.
    tmax_vcd = tmax_ns / time_scale_ns

    times = []
    values = []
    for t, v in tv:
        if t > tmax_vcd:
            break
        if v in ("x", "z"):
            continue
        try:
            values.append(_to_int(v, width, signed))
            times.append(t * time_scale_ns)
        except ValueError:
            continue

    if not times:
        return [], []

    if times[-1] < tmax_ns:
        times.append(tmax_ns)
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


def _phase_meta_sidecar_path(phase_csv_path: Path):
    """Return the conventional sidecar path for a spike-cycles CSV.

    e.g. foo_dropout_recovery_spike_cycles.csv  →  foo_dropout_recovery_phases.json
    """
    name = phase_csv_path.name.replace("_spike_cycles.csv", "_phases.json")
    return phase_csv_path.parent / name


def _load_phase_metadata(json_path: Path):
    """Load a phase-boundary sidecar written by the capture tests.

    Returns list of (t_start_ns, t_end_ns, label) tuples, or None if file is missing.
    These boundaries reflect when current actually changed, so phase shading lines up
    with the real dropout window rather than the gap between adjacent spikes.
    """
    if not json_path.exists():
        return None
    with json_path.open() as f:
        data = json.load(f)
    return [(p["t_start_ns"], p["t_end_ns"], p["label"]) for p in data.get("phases", [])]


def _compute_isi_excluding_cross_phase(spike_edge_times, phase_labels):
    """Compute ISI series, dropping intervals that span a phase boundary.

    The cross-dropout ISI (last startup spike → first recovery spike) is huge and
    compresses the y-axis so much that the actual ISI changes within each phase
    become invisible. Skipping it preserves the within-phase scale.
    """
    if len(spike_edge_times) < 2:
        return [], []
    isi_times = []
    isi_values = []
    n_paired = min(len(spike_edge_times), len(phase_labels))
    for i in range(1, len(spike_edge_times)):
        if i < n_paired and i - 1 < n_paired and phase_labels[i] != phase_labels[i - 1]:
            continue
        isi_times.append(spike_edge_times[i])
        isi_values.append(spike_edge_times[i] - spike_edge_times[i - 1])
    return isi_times, isi_values


def _load_phase_labels(phase_csv_path: Path):
    """Return an ordered list of phase labels from the CSV, one entry per spike cycle.

    Works with any CSV that has a 'phase' column, regardless of whether 'spike_time_ns'
    is present. Row K corresponds to the Kth spike detected in the VCD.
    """
    labels = []
    with phase_csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "phase" not in (reader.fieldnames or []):
            print("phase-csv: 'phase' column not found — skipping phase shading")
            return []
        for row in reader:
            labels.append(row.get("phase", "constant").strip())
    return labels


def _compute_phase_spans(spike_edge_times, phase_labels, vcd_end_ns: float):
    """Build (t_start_ns, t_end_ns, label) spans from VCD spike times + per-spike labels.

    spike_edge_times[i] is the actual nanosecond timestamp of the (i+1)th spike from
    the VCD. phase_labels[i] is the label for that spike's cycle from the CSV.
    Gaps between phases (e.g. a dropout period with no spikes) are filled automatically
    and labeled 'dropout'.
    """
    n = min(len(spike_edge_times), len(phase_labels))
    if n == 0:
        return []

    # Group consecutive spikes by label into coarse spans.
    # The first phase always starts at t=0 (simulation start).
    spans = []
    cur_label = phase_labels[0]
    cur_start = 0.0
    cur_end = spike_edge_times[0]

    for i in range(1, n):
        t = spike_edge_times[i]
        label = phase_labels[i]
        if label == cur_label:
            cur_end = t
        else:
            spans.append((cur_start, cur_end, cur_label))
            cur_start = t
            cur_end = t
            cur_label = label

    spans.append((cur_start, cur_end, cur_label))

    # Fill gaps between spans (e.g. the current-off dropout window with no spikes).
    full_spans = []
    prev_end = 0.0
    for start, end, label in spans:
        if start > prev_end + 1:
            full_spans.append((prev_end, start, "dropout"))
        full_spans.append((start, end, label))
        prev_end = end

    # Extend the final span to the end of the VCD.
    if full_spans:
        s, _, lbl = full_spans[-1]
        full_spans[-1] = (s, vcd_end_ns, lbl)

    return full_spans


def _plot_per_phase_panels(
    mem_times, mem_values,
    spike_edge_times, phase_labels, phase_spans,
    zoom_n, padding, max_points,
    output_file, show_plot,
):
    """1-row × N-phase figure: one membrane-trace panel per phase, each zoomed to first N spikes.

    For phases with no spikes (e.g. dropout) the full phase duration is shown.
    This layout is the clearest way to compare startup vs. recovery spiking density.
    """
    axis_label_fontsize = 8
    tick_label_fontsize = 6

    # Group VCD spike times by phase label (preserving order).
    n_paired = min(len(spike_edge_times), len(phase_labels))
    phase_spike_times: dict[str, list] = {}
    for i in range(n_paired):
        lbl = phase_labels[i]
        phase_spike_times.setdefault(lbl, []).append(spike_edge_times[i])

    n_panels = len(phase_spans)
    figsize = get_latex_figsize(width_scale=1.6, height_scale=0.55)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(figsize["width"], figsize["height"]),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    mem_arr = np.array(mem_times, dtype=float)
    val_arr = np.array(mem_values, dtype=float)

    for ax, (phase_start, phase_end, label) in zip(axes, phase_spans):
        spikes = phase_spike_times.get(label, [])
        color = PHASE_COLORS.get(label, COLOR_RAW)

        if spikes and zoom_n > 0:
            n_shown = min(zoom_n, len(spikes))
            xmin = max(phase_start, spikes[0] - padding)
            xmax = min(phase_end, spikes[n_shown - 1] + padding)
        else:
            xmin = phase_start
            xmax = phase_end

        # Slice membrane trace to this panel's window and downsample.
        mask = (mem_arr >= xmin) & (mem_arr <= xmax)
        t_w = mem_arr[mask].tolist()
        v_w = val_arr[mask].tolist()
        if t_w:
            t_w, v_w = _downsample(t_w, v_w, max_points // n_panels)
            ax.step(t_w, v_w, where="post", color=COLOR_MEMBRANE, linewidth=0.8)

        # Spike raster (using axes-fraction y so it stays at the top regardless of scale).
        spk_in_window = [t for t in spikes if xmin <= t <= xmax]
        if spk_in_window:
            ax.scatter(
                spk_in_window,
                np.ones(len(spk_in_window)) * 0.97,
                transform=ax.get_xaxis_transform(),
                marker="|", s=60, color=COLOR_SPIKE, linewidth=0.8,
            )

        ax.axvspan(xmin, xmax, alpha=0.12, color=color, linewidth=0)
        ax.set_xlim(xmin, xmax)
        ax.set_title(label, fontsize=axis_label_fontsize, color=color)
        ax.set_xlabel("Time (ns)", fontsize=axis_label_fontsize)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{x:.2e}".replace("e+0", "e").replace("e+", "e"))
        )
        ax.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
        ax.grid(True, which="both", linestyle=":", alpha=0.6)

    axes[0].set_ylabel("Membrane", fontsize=axis_label_fontsize)
    plt.tight_layout()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format=output_path.suffix.lstrip(".") or "svg", bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_signal(
    vcd_file,
    membrane_signal="fractional_lif.membrane_out[23:0]",
    current_signal="fractional_lif.current[15:0]",
    spike_signal="fractional_lif.spike_out",
    membrane_width=24,
    current_width=16,
    spike_width=1,
    tmax=None,  # None → use full VCD duration
    show_current=True,
    show_spike=True,
    show_isi=True,
    max_points=200_000,
    x_min=None,
    x_max=None,
    zoom_early_spikes=20,
    zoom_full=False,
    auto_window_padding=5_000,
    phase_csv=None,
    zoom_early_spikes_per_phase=0,
    phase_zoom_spikes=0,
    output_file="images/fractional_lif_memory_constant.svg",
    show_plot=False,
):
    if str(vcd_file).endswith(".fst"):
        print(
            "Error: input is an FST file. Convert it first with:\n"
            f"  fst2vcd {vcd_file} -o {Path(vcd_file).with_suffix('.vcd')}"
        )
        return

    axis_label_fontsize = 8
    tick_label_fontsize = 6

    vcd = VCDVCD(vcd_file, store_tvs=True)

    # Convert VCD's native time unit (often picoseconds for Icarus default 1ps timescale)
    # to nanoseconds so phase metadata from the sidecar JSON (which uses get_sim_time(unit="ns"))
    # lines up with VCD-derived timestamps.
    time_scale_ns = _vcd_time_scale_to_ns(vcd)
    print(f"VCD timescale: {getattr(vcd, 'timescale', '?')}, multiplier→ns = {time_scale_ns}")

    if tmax is None:
        tmax = float(vcd.endtime) * time_scale_ns
        print(f"tmax auto-detected from VCD: {tmax} ns")

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
        tmax_ns=tmax,
        signed=True,
        width=membrane_width,
        time_scale_ns=time_scale_ns,
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
        tmax_ns=tmax,
        signed=True,
        width=current_width,
        time_scale_ns=time_scale_ns,
    )

    # Load phase labels before spike detection so we know whether to force zoom_full.
    phase_labels = []
    if phase_csv is not None:
        phase_labels = _load_phase_labels(Path(phase_csv))
        # Multi-phase CSV (e.g. dropout/recovery): show the full simulation by default
        # so all three regions are visible. The user can override with --zoom-early-spikes.
        distinct = set(phase_labels)
        if len(distinct) > 1 and not zoom_full:
            zoom_full = True
            print(f"Multi-phase CSV detected ({distinct}); switching to full-span zoom.")

    spike_edge_times = []
    if spk_name is not None:
        spk_times, spk_values = _extract_signal(
            vcd=vcd,
            signal_name=spk_name,
            tmax_ns=tmax,
            signed=False,
            width=spike_width,
            time_scale_ns=time_scale_ns,
        )

        if spk_times is not None and spk_times:
            spike_edge_times = _detect_spike_rising_edges(spk_times, spk_values)
            print(f"Detected spike edges: {len(spike_edge_times)}")

    # Phase spans: prefer the sidecar JSON (written by capture tests) since it has the
    # actual current-on/current-off boundaries. Fall back to inference from VCD spike
    # times + CSV labels for older runs that predate the sidecar.
    phase_spans = []
    if phase_csv is not None:
        sidecar_path = _phase_meta_sidecar_path(Path(phase_csv))
        sidecar = _load_phase_metadata(sidecar_path)
        if sidecar:
            phase_spans = sidecar
            print(f"Phase spans (from sidecar {sidecar_path.name}): "
                  f"{[(round(s), round(e), l) for s, e, l in phase_spans]}")

    if not phase_spans and phase_labels and spike_edge_times:
        phase_spans = _compute_phase_spans(spike_edge_times, phase_labels, tmax)
        print(f"Phase spans (inferred from spikes): "
              f"{[(round(s), round(e), l) for s, e, l in phase_spans]}")

    # 3-panel per-phase view: one membrane panel per phase, each zoomed to first N spikes.
    if zoom_early_spikes_per_phase > 0 and phase_spans:
        _plot_per_phase_panels(
            mem_times=mem_times,
            mem_values=mem_values,
            spike_edge_times=spike_edge_times,
            phase_labels=phase_labels,
            phase_spans=phase_spans,
            zoom_n=zoom_early_spikes_per_phase,
            padding=auto_window_padding,
            max_points=max_points,
            output_file=output_file,
            show_plot=show_plot,
        )
        return

    # Determine x window.
    if x_min is None and x_max is None:
        if zoom_full and len(spike_edge_times) >= 2:
            x_min = max(0, spike_edge_times[0] - auto_window_padding)
            x_max = min(tmax, spike_edge_times[-1] + auto_window_padding)
            print(f"Full spike window: x_min={x_min}, x_max={x_max}")
        elif zoom_early_spikes > 0 and len(spike_edge_times) >= 1:
            n = min(zoom_early_spikes, len(spike_edge_times))
            x_min = max(0, spike_edge_times[0] - auto_window_padding)
            x_max = min(tmax, spike_edge_times[n - 1] + auto_window_padding)
            print(f"Early-spike window (N={zoom_early_spikes}): x_min={x_min}, x_max={x_max}")

    # --phase-zoom-spikes N: trim x_max to the Nth spike of the last active (non-dropout) phase.
    # Useful for cutting the long stable tail of recovery while keeping the full x_min context.
    if phase_zoom_spikes > 0 and phase_labels and spike_edge_times:
        n_paired = min(len(spike_edge_times), len(phase_labels))
        last_active_spikes = [
            spike_edge_times[i] for i in range(n_paired)
            if phase_labels[i] != "dropout"
        ]
        # Spikes of the last active phase only.
        last_phase_label = next(
            (lbl for _, _, lbl in reversed(phase_spans) if lbl != "dropout"), None
        )
        if last_phase_label:
            last_phase_spikes = [
                spike_edge_times[i] for i in range(n_paired)
                if phase_labels[i] == last_phase_label
            ]
            n_trim = min(phase_zoom_spikes, len(last_phase_spikes))
            if n_trim > 0:
                x_max = min(tmax, last_phase_spikes[n_trim - 1] + auto_window_padding)
                print(f"Phase-zoom trim: x_max capped at spike {n_trim} of {last_phase_label} ({x_max} ns)")

    if x_min is None:
        x_min = 0
    if x_max is None:
        x_max = tmax

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

    def _draw_phase_spans(ax):
        for t_start, t_end, label in phase_spans:
            color = PHASE_COLORS.get(label, COLOR_RAW)
            ax.axvspan(t_start, t_end, alpha=0.12, color=color, linewidth=0)

    ax_index = 0

    ax_mem = axes[ax_index]
    ax_mem.step(mem_times, mem_values, where="post", color=COLOR_MEMBRANE)
    _draw_phase_spans(ax_mem)
    if phase_spans:
        # Annotate phase labels at the top of the membrane subplot.
        for t_start, t_end, label in phase_spans:
            mid = (t_start + t_end) / 2
            ax_mem.text(
                mid, 0.97, label,
                transform=ax_mem.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=tick_label_fontsize,
                color=PHASE_COLORS.get(label, COLOR_RAW),
            )
    ax_mem.set_ylabel("Membrane", fontsize=axis_label_fontsize)
    ax_mem.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
    ax_mem.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
    ax_mem.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_index += 1

    if show_current:
        ax_cur = axes[ax_index]
        if cur_times is None or not cur_times:
            ax_cur.text(
                0.5, 0.5, "Current signal unavailable",
                transform=ax_cur.transAxes,
                ha="center", va="center", fontsize=10, color="gray",
            )
        else:
            cur_times, cur_values = _downsample(cur_times, cur_values, max_points)
            ax_cur.step(cur_times, cur_values, where="post", color=COLOR_CURRENT)
        _draw_phase_spans(ax_cur)
        ax_cur.set_ylabel("Current", fontsize=axis_label_fontsize)
        ax_cur.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
        ax_cur.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        ax_cur.grid(True, which="both", linestyle=":", alpha=0.6)
        ax_index += 1

    if show_spike and spk_name is not None:
        ax_spk = axes[ax_index]
        if spike_edge_times:
            y = np.ones(len(spike_edge_times))
            ax_spk.scatter(spike_edge_times, y, marker="|", s=80, color=COLOR_SPIKE)
        else:
            ax_spk.text(
                0.5, 0.5, "No spike rising edges detected",
                transform=ax_spk.transAxes,
                ha="center", va="center", fontsize=10, color="gray",
            )
        _draw_phase_spans(ax_spk)
        ax_spk.set_ylabel("Spikes", fontsize=axis_label_fontsize)
        ax_spk.set_ylim(0.0, 1.5)
        ax_spk.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
        ax_spk.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        ax_spk.grid(True, which="both", linestyle=":", alpha=0.6)
        ax_index += 1

    if show_isi and spk_name is not None:
        ax_isi = axes[ax_index]
        # Skip cross-phase ISIs (e.g. the huge gap across the dropout) when phase
        # data is available — those single intervals dominate the y-axis otherwise.
        if phase_labels:
            isi_times, isi_values = _compute_isi_excluding_cross_phase(
                spike_edge_times, phase_labels
            )
        else:
            isi_times, isi_values = _compute_isi(spike_edge_times)
        if isi_times:
            isi_times, isi_values = _downsample(isi_times, isi_values, max_points)
            ax_isi.plot(isi_times, isi_values, color=COLOR_ISI, linewidth=1.2)
            ax_isi.scatter(isi_times, isi_values, color=COLOR_ISI, s=4, alpha=0.7)
        else:
            ax_isi.text(
                0.5, 0.5, "Not enough spikes for ISI",
                transform=ax_isi.transAxes,
                ha="center", va="center", fontsize=10, color="gray",
            )
        _draw_phase_spans(ax_isi)
        ax_isi.set_ylabel("ISI (ns)", fontsize=axis_label_fontsize)
        ax_isi.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
        ax_isi.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        ax_isi.grid(True, which="both", linestyle=":", alpha=0.6)

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
        description=(
            "Plot membrane/current/spike/ISI traces from a VCD file. "
            "Accepts VCD only — convert FST first with: "
            "fst2vcd <file>.fst -o <file>.vcd"
        )
    )
    parser.add_argument("vcd_file", help="Path to VCD file (.fst input prints a conversion hint)")
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
    parser.add_argument("--membrane-width", type=int, default=24)
    parser.add_argument("--current-width", type=int, default=16)
    parser.add_argument("--spike-width", type=int, default=1)
    parser.add_argument(
        "--tmax", type=int, default=None,
        help="Maximum simulation time to plot (ns). Defaults to the full VCD duration.",
    )
    parser.add_argument("--no-current", action="store_true", help="Disable current subplot")
    parser.add_argument("--no-spike", action="store_true", help="Disable spike subplot")
    parser.add_argument("--no-isi", action="store_true", help="Disable ISI subplot")
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--x-min", type=int, default=None, help="Manual x-axis minimum (ns)")
    parser.add_argument("--x-max", type=int, default=None, help="Manual x-axis maximum (ns)")
    parser.add_argument(
        "--zoom-early-spikes", type=int, default=20, metavar="N",
        help="Window x-axis from the first to the Nth detected spike (default: 20). "
             "Set to 0 to disable.",
    )
    parser.add_argument(
        "--zoom-full", action="store_true",
        help="Window x-axis from first to last spike (overrides --zoom-early-spikes)",
    )
    parser.add_argument(
        "--auto-window-padding", type=int, default=5_000,
        help="Padding (ns) added around the auto-computed spike window (default: 5000)",
    )
    parser.add_argument(
        "--phase-csv", default=None, metavar="PATH",
        help="Path to spike-cycles CSV with a 'phase' column; adds shaded phase regions",
    )
    parser.add_argument(
        "--phase-zoom-spikes", type=int, default=0, metavar="N",
        help="Trim x_max to the Nth spike of the last active phase (e.g. recovery). "
             "Leaves x_min at the default full-span start. Use with --phase-csv.",
    )
    parser.add_argument(
        "--zoom-early-spikes-per-phase", type=int, default=0, metavar="N",
        help="3-panel layout: one membrane panel per phase, each showing the first N spikes. "
             "Requires --phase-csv. Overrides --zoom-early-spikes / --zoom-full.",
    )
    parser.add_argument("--show", action="store_true", help="Display interactive window after saving")
    parser.add_argument(
        "--output",
        default="images/fractional_lif_memory_constant.svg",
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
        zoom_early_spikes=args.zoom_early_spikes,
        zoom_full=args.zoom_full,
        auto_window_padding=args.auto_window_padding,
        phase_csv=args.phase_csv,
        zoom_early_spikes_per_phase=args.zoom_early_spikes_per_phase,
        phase_zoom_spikes=args.phase_zoom_spikes,
        output_file=args.output,
        show_plot=args.show,
    )
