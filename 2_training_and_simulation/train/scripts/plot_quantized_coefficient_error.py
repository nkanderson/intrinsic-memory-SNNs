"""
Visualize how fixed-point precision controls GL coefficient approximation error.

Compares the analytical Grunwald-Letnikov coefficient magnitudes |g_k| against
two unsigned fixed-point quantizations:

  - UQ0.16  (16 fractional bits, resolution 2^-16)
  - UQ0.8   ( 8 fractional bits, resolution 2^-8)

The story this figure tells: because |g_k| decays with history step k, the
coarser UQ0.8 grid runs out of resolution early -- its relative error departs
from zero much sooner than UQ0.16, grows as the coefficients shrink toward the
quantization step, and finally collapses to -100% once |g_k| rounds to zero
(underflow). UQ0.16 stays effectively exact over the same window.

Two stacked panels share the x-axis (history step k, starting at k=1; g_0=1 is
not operationally stored):

  Top    -- coefficient magnitude vs k (log scale), with each scheme's
            quantization resolution drawn as a reference line. Visually, the
            UQ0.8 series flattens into a staircase and then drops out (NaN)
            when it underflows.
  Bottom -- signed relative error (%) vs k for both schemes, with the first
            nonzero-error step annotated for each, plus a -100% underflow line.

Run:
    python plot_quantized_coefficient_error.py \
        --max-history 16 \
        --output ../images/coefficients/quantized-coeff-error-16-vs-8.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on sys.path when running from train/scripts/.
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.scripts.utils import compute_gl_coefficients  # type: ignore  # noqa: E402
from common.scripts.plot_styles import (  # type: ignore  # noqa: E402
    OKABE_ITO,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    DEFAULT_FIGSIZE,
)

# Per-scheme styling. Index into OKABE_ITO so the two precisions stay visually
# distinct in color *and* marker shape (important for grayscale printing).
COLOR_ANALYTICAL = OKABE_ITO[7]  # black
COLOR_HIGH = OKABE_ITO[0]  # blue   -> UQ0.16
COLOR_LOW = OKABE_ITO[1]  # vermillion -> UQ0.8


def quantize_magnitude(value: float, frac_bits: int) -> float:
    """
    Quantize a magnitude to unsigned fixed-point with 0 integer bits.

    UQ0.{frac_bits}: total width == frac_bits, so the largest representable
    value is (2^frac_bits - 1) / 2^frac_bits < 1 (fine here since |g_k| <= 0.5
    for k >= 1). Values below half the LSB round to exactly zero (underflow).

    Returns the reconstructed float (0.0 on underflow).
    """
    scale = 1 << frac_bits
    max_val = (1 << frac_bits) - 1  # 0 integer bits -> total bits == frac_bits
    q = int(round(abs(value) * scale))
    q = max(0, min(q, max_val))
    return q / scale


def build_error_series(
    alpha: float,
    history_length: int,
    frac_bits_high: int,
    frac_bits_low: int,
) -> Dict[str, np.ndarray]:
    """
    Compute analytical |g_k| and both quantizations for k = 1..history_length.

    g_0 = 1 is skipped (not operationally stored). Relative error is
    (approx - analytical) / analytical * 100, signed.
    """
    coeffs = compute_gl_coefficients(alpha, history_length + 1)
    coeffs_np = coeffs.numpy() if hasattr(coeffs, "numpy") else np.array(coeffs)

    k = np.arange(1, history_length + 1)
    analytical = np.abs(coeffs_np[1 : history_length + 1].astype(float))

    high = np.array([quantize_magnitude(v, frac_bits_high) for v in analytical])
    low = np.array([quantize_magnitude(v, frac_bits_low) for v in analytical])

    err_high = (high - analytical) / analytical * 100.0
    err_low = (low - analytical) / analytical * 100.0

    return {
        "k": k,
        "analytical": analytical,
        "high": high,
        "low": low,
        "err_high": err_high,
        "err_low": err_low,
    }


def _first_nonzero_step(
    k: np.ndarray, err: np.ndarray, tol: float = 1e-9
) -> int | None:
    """Return the first k at which |relative error| exceeds tol, else None."""
    nz = np.flatnonzero(np.abs(err) > tol)
    return int(k[nz[0]]) if nz.size else None


def plot_quantized_coefficient_error(
    alpha: float = 0.5,
    history_length: int = 32,
    frac_bits_high: int = 16,
    frac_bits_low: int = 8,
    output_path: Path | None = None,
    use_svg: bool = False,
):
    s = build_error_series(alpha, history_length, frac_bits_high, frac_bits_low)
    k = s["k"]
    marker_stride = max(1, history_length // 24)

    lsb_high = 1.0 / (1 << frac_bits_high)  # quantization resolution (LSB)
    lsb_low = 1.0 / (1 << frac_bits_low)

    label_high = f"UQ0.{frac_bits_high}"
    label_low = f"UQ0.{frac_bits_low}"

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(DEFAULT_FIGSIZE[0], DEFAULT_FIGSIZE[1] * 1.5),
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.08},
    )

    # ---- Top panel: coefficient magnitude (log y) ----------------------
    ax_top.plot(
        k,
        s["analytical"],
        color=COLOR_ANALYTICAL,
        linewidth=1.6,
        label=rf"analytical ($\alpha={alpha:g}$)",
        zorder=3,
    )
    # NaN-mask underflowed (zero) samples so the log axis simply drops them.
    high_y = np.where(s["high"] == 0.0, np.nan, s["high"])
    low_y = np.where(s["low"] == 0.0, np.nan, s["low"])
    ax_top.plot(
        k,
        high_y,
        color=COLOR_HIGH,
        linestyle="--",
        linewidth=1.3,
        marker="s",
        markersize=4,
        markevery=marker_stride,
        label=f"{label_high} quantized",
        zorder=4,
    )
    ax_top.plot(
        k,
        low_y,
        color=COLOR_LOW,
        linestyle="none",
        marker="o",
        markersize=4.5,
        drawstyle="steps-mid",
        label=f"{label_low} quantized",
        zorder=5,
    )
    # Connect the UQ0.8 markers with a staircase to make the quantization grid
    # explicit (each plateau is one representable level).
    ax_top.step(
        k, low_y, where="mid", color=COLOR_LOW, linewidth=1.0, alpha=0.6, zorder=2
    )

    # Quantization resolution reference lines (the LSB of each scheme).
    ax_top.axhline(
        lsb_low,
        color=COLOR_LOW,
        linestyle=":",
        linewidth=1.1,
        alpha=0.8,
        label=f"{label_low} resolution ($2^{{-{frac_bits_low}}}$)",
    )
    ax_top.axhline(
        lsb_high,
        color=COLOR_HIGH,
        linestyle=":",
        linewidth=1.1,
        alpha=0.8,
        label=f"{label_high} resolution ($2^{{-{frac_bits_high}}}$)",
    )

    ax_top.set_yscale("log")
    ax_top.set_ylabel("Coefficient magnitude", fontsize=AXIS_LABEL_FONTSIZE)
    ax_top.grid(True, which="both", alpha=0.22, linestyle=":")
    ax_top.set_axisbelow(True)
    ax_top.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    # Keep the UQ0.16 resolution line visible but don't let it dominate the axis.
    ax_top.set_ylim(bottom=lsb_high * 0.5, top=float(s["analytical"][0]) * 2.0)
    ax_top.legend(loc="lower left", fontsize=LEGEND_FONTSIZE, framealpha=0.9, ncol=2)

    # ---- Bottom panel: signed relative error (%) -----------------------
    ax_bot.axhline(0.0, color="#888888", linewidth=0.9, zorder=1)
    ax_bot.plot(
        k,
        s["err_high"],
        color=COLOR_HIGH,
        linestyle="--",
        linewidth=1.3,
        marker="s",
        markersize=4,
        markevery=marker_stride,
        label=f"{label_high} rel. error",
        zorder=4,
    )
    ax_bot.plot(
        k,
        s["err_low"],
        color=COLOR_LOW,
        linewidth=1.4,
        marker="o",
        markersize=4.5,
        markevery=marker_stride,
        label=f"{label_low} rel. error",
        zorder=5,
    )

    first_high = _first_nonzero_step(k, s["err_high"])
    first_low = _first_nonzero_step(k, s["err_low"])

    # Does UQ0.8 underflow (round to 0 -> -100%) anywhere in this window?
    underflow_idx = np.flatnonzero(s["low"] == 0.0)
    underflows = underflow_idx.size > 0

    # Y-limits adapt to the window: reserve room for the -100% floor only when
    # underflow actually happens, otherwise hug the data so a short window does
    # not show a misleading expanse of empty space down to -100%.
    y_top = float(np.max(s["err_low"]))
    y_hi = max(20.0, y_top * 1.18)
    y_lo = -112.0 if underflows else min(-5.0, float(np.min(s["err_low"])) * 1.18)
    ax_bot.set_ylim(y_lo, y_hi)
    ax_bot.set_xlim(0.5, history_length + 0.5)

    # Annotate where each scheme first departs from exact (0%) error. Heights
    # are taken as fractions of the (window-dependent) axis top so the labels
    # stay well placed whether the window is short or long.
    for kk, color, name, text_x, text_y, err in (
        (
            first_low,
            COLOR_LOW,
            label_low,
            (first_low or 0) - 1.0,
            y_hi * 0.46,
            s["err_low"],
        ),
        (
            first_high,
            COLOR_HIGH,
            label_high,
            (first_high or 0) - 0.8,
            y_hi * 0.46,
            s["err_high"],
        ),
    ):
        if kk is None:
            continue
        ax_bot.axvline(
            kk, color=color, linestyle="-", linewidth=0.8, alpha=0.35, zorder=1
        )
        # Point the arrow at this scheme's own marker (its error value at k=kk),
        # not at y=0 -- otherwise it appears to indicate the flat UQ0.16 line.
        ax_bot.annotate(
            f"{name} first error\nat k={kk}",
            xy=(kk, float(err[kk - 1])),
            xytext=(text_x, text_y),
            fontsize=LEGEND_FONTSIZE,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8, alpha=0.7),
            ha="left",
            va="center",
        )

    # -100% floor + label, only when UQ0.8 actually underflows in this window.
    # The label ends just left of the first underflow so it clears the line.
    if underflows:
        ax_bot.axhline(
            -100.0, color=COLOR_LOW, linestyle=":", linewidth=1.0, alpha=0.7, zorder=1
        )
        underflow_start = int(k[underflow_idx[0]])
        ax_bot.text(
            underflow_start - 0.4,
            -100.0,
            "underflow (rounds to 0)",
            fontsize=LEGEND_FONTSIZE,
            color=COLOR_LOW,
            ha="right",
            va="bottom",
        )

    ax_bot.set_xlabel("History step (k)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_bot.set_ylabel("Relative error (%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_bot.grid(True, alpha=0.22, linestyle=":")
    ax_bot.set_axisbelow(True)
    ax_bot.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax_bot.legend(loc="upper left", fontsize=LEGEND_FONTSIZE, framealpha=0.9)

    # No tight_layout(): the annotations/reference lines are outside the data
    # area and trip its compatibility check. hspace is set in gridspec_kw and
    # savefig(bbox_inches="tight") trims the final margins.

    if output_path is not None:
        final = output_path.with_suffix(".svg") if use_svg else output_path
        final.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(final, dpi=150, bbox_inches="tight")
        print(f"Wrote quantized coefficient error plot to {final}")

    plt.close(fig)
    return s


def print_table(s: Dict[str, np.ndarray], frac_bits_high: int, frac_bits_low: int):
    head_high = f"UQ0.{frac_bits_high}"
    head_low = f"UQ0.{frac_bits_low}"
    print(
        f"{'k':>3} {'analytical':>12} {head_high:>12} "
        f"{'err%':>9} {head_low:>12} {'err%':>9}"
    )
    print("-" * 64)
    for i in range(len(s["k"])):
        print(
            f"{int(s['k'][i]):>3} {s['analytical'][i]:>12.8f} "
            f"{s['high'][i]:>12.8f} {s['err_high'][i]:>9.3f} "
            f"{s['low'][i]:>12.8f} {s['err_low'][i]:>9.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot GL coefficient approximation error for two fixed-point "
            "precisions (default UQ0.16 vs UQ0.8)."
        )
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--max-history",
        "--history-length",
        dest="max_history",
        type=int,
        default=16,
        help=(
            "Number of history steps (k) to plot, i.e. the x-axis extent. "
            "Shorter windows end before UQ0.8 underflows. (alias: --history-length)"
        ),
    )
    parser.add_argument("--frac-bits-high", type=int, default=16)
    parser.add_argument("--frac-bits-low", type=int, default=8)
    parser.add_argument(
        "--output",
        type=str,
        default="../images/coefficients/quantized-coeff-error-16-vs-8.png",
        help="Output image path (relative to this script's directory by default).",
    )
    parser.add_argument(
        "--svg", action="store_true", help="Save as SVG instead of PNG."
    )
    parser.add_argument(
        "--no-table", action="store_true", help="Suppress the printed numeric table."
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (Path(__file__).resolve().parent / output_path).resolve()

    s = plot_quantized_coefficient_error(
        alpha=args.alpha,
        history_length=args.max_history,
        frac_bits_high=args.frac_bits_high,
        frac_bits_low=args.frac_bits_low,
        output_path=output_path,
        use_svg=args.svg,
    )

    if not args.no_table:
        print()
        print_table(s, args.frac_bits_high, args.frac_bits_low)


if __name__ == "__main__":
    main()
