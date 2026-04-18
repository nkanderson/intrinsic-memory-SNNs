"""
Analyze GL coefficients: produce table data and compare to bit-shift sequences.

Produces two datasets:
- Quantized comparison: analytical |g_k|, quantized fixed-point integer, hex,
  quantized float, and percent difference (default alpha=0.5, H=16).
- Bit-shift comparison: analytical |g_k| vs several bit-shift-derived sequences
  (simple, slow_decay, custom, custom_slow_decay) with percent differences.

Outputs CSV files when requested and prints a compact table to stdout.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on sys.path when running this script from train/scripts/.
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.scripts.utils import compute_gl_coefficients  # type: ignore

try:
    import history_coefficients as hc  # type: ignore
except ModuleNotFoundError:
    from scripts import history_coefficients as hc  # type: ignore


def quantize_magnitude(
    value: float, bits: int = 16, frac_bits: int = 15
) -> Tuple[int, float]:
    """
    Quantize an absolute coefficient magnitude to unsigned fixed-point.

    Returns (quantized_int, quantized_float).
    """
    scale = 1 << frac_bits
    max_val = (1 << bits) - 1

    q = int(round(abs(value) * scale))
    q = max(0, min(q, max_val))
    return q, q / scale


def hex_width_for_bits(bits: int) -> int:
    if bits <= 8:
        return 2
    if bits <= 16:
        return 4
    return 8


def build_quantized_table(
    alpha: float = 0.5,
    history_length: int = 16,
    coeff_bits: int = 16,
    coeff_frac_bits: int = 15,
    precision: int = 8,
) -> List[Dict[str, str]]:
    """
    Build table rows containing index, analytical value, quantized int, hex,
    quantized float, and percent difference.
    """
    coeffs = compute_gl_coefficients(alpha, history_length)
    coeffs_np = coeffs.numpy() if hasattr(coeffs, "numpy") else np.array(coeffs)

    rows: List[Dict[str, str]] = []
    hw = hex_width_for_bits(coeff_bits)

    for k in range(history_length):
        analytical = float(coeffs_np[k])
        mag = abs(analytical)
        q_int, q_float = quantize_magnitude(
            mag, bits=coeff_bits, frac_bits=coeff_frac_bits
        )
        q_hex = f"{q_int:0{hw}X}"

        pct_diff = ((q_float - mag) / mag * 100) if mag != 0 else 0.0

        rows.append(
            {
                "index": str(k),
                "analytical": f"{mag:.{precision}f}",
                "quantized_int": str(q_int),
                "quantized_hex": q_hex,
                "quantized_float": f"{q_float:.{precision}f}",
                "pct_diff": f"{pct_diff:.6f}",
            }
        )

    return rows


def build_bitshift_comparison(
    alpha: float = 0.5, history_length: int = 16, precision: int = 8
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Compare analytical |g_k| to several bit-shift sequences. Returns rows and
    the ordered list of sequence names used.
    """
    coeffs = compute_gl_coefficients(alpha, history_length)
    coeffs_np = coeffs.numpy() if hasattr(coeffs, "numpy") else np.array(coeffs)

    sequences = {
        "simple": hc.simple_sequence(history_length),
        "slow_decay": hc.slow_decay_sequence(history_length),
        "custom": hc.custom_sequence(history_length, decay_rate=3),
        # Future work could include investigating alternate rates here
        # "custom": hc.custom_sequence(history_length, decay_rate=4),
        "custom_slow": hc.custom_slow_decay_sequence(history_length),
    }

    seq_names = list(sequences.keys())
    rows: List[Dict[str, str]] = []

    for k in range(history_length):
        analytical = float(abs(coeffs_np[k]))
        row: Dict[str, str] = {
            "index": str(k),
            "analytical": f"{analytical:.{precision}f}",
        }

        for name, seq in sequences.items():
            approx = float(seq[k])
            # Use common convention:
            # difference = approximation - analytical
            # relative error % = difference / analytical * 100
            diff = approx - analytical
            pct = (diff / analytical * 100) if analytical != 0 else 0.0
            row[f"{name}"] = f"{approx:.{precision}f}"
            row[f"{name}_diff"] = f"{diff:.{precision}f}"
            row[f"{name}_pct"] = f"{pct:.6f}"

        rows.append(row)

    return rows, seq_names


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def resolve_plot_path(path: Path, use_svg: bool) -> Path:
    """
    Resolve output plot path for selected image format.

    If use_svg is True, force `.svg` extension.
    """
    if use_svg:
        return path.with_suffix(".svg") if path.suffix else Path(f"{path}.svg")
    return path


def build_relative_error_series(
    alpha: float = 0.5,
    history_length: int = 16,
    coeff_bits: int = 16,
    coeff_frac_bits: int = 15,
) -> Dict[str, np.ndarray]:
    """
    Build aligned relative-error series (%):
    (approximation - analytical) / analytical * 100
    for quantized and 4 bit-shift variants.
    """
    coeffs = compute_gl_coefficients(alpha, history_length)
    coeffs_np = coeffs.numpy() if hasattr(coeffs, "numpy") else np.array(coeffs)

    analytical = np.abs(coeffs_np.astype(float))
    eps = np.finfo(float).eps
    denom = np.where(analytical == 0.0, eps, analytical)

    quantized_vals = []
    for value in analytical:
        _, q_float = quantize_magnitude(
            float(value), bits=coeff_bits, frac_bits=coeff_frac_bits
        )
        quantized_vals.append(q_float)
    quantized = np.array(quantized_vals, dtype=float)

    simple = np.array(hc.simple_sequence(history_length), dtype=float)
    slow_decay = np.array(hc.slow_decay_sequence(history_length), dtype=float)
    custom = np.array(hc.custom_sequence(history_length, decay_rate=3), dtype=float)
    custom_slow = np.array(hc.custom_slow_decay_sequence(history_length), dtype=float)

    quantized_pct = (quantized - analytical) / denom * 100.0
    simple_pct = (simple - analytical) / denom * 100.0
    slow_decay_pct = (slow_decay - analytical) / denom * 100.0
    custom_pct = (custom - analytical) / denom * 100.0
    custom_slow_pct = (custom_slow - analytical) / denom * 100.0

    return {
        "history": np.arange(history_length),
        "quantized_pct": quantized_pct,
        "simple_pct": simple_pct,
        "slow_decay_pct": slow_decay_pct,
        "custom_pct": custom_pct,
        "custom_slow_pct": custom_slow_pct,
    }


def plot_relative_error_comparison(
    alpha: float = 0.5,
    history_length: int = 16,
    coeff_bits: int = 16,
    coeff_frac_bits: int = 15,
    output_path: Path | None = None,
    use_svg: bool = False,
):
    """
    Plot relative error (%) for quantized and bit-shift variant coefficients.

    X-axis: history step index
    Y-axis: relative error (%)
    """
    series = build_relative_error_series(
        alpha=alpha,
        history_length=history_length,
        coeff_bits=coeff_bits,
        coeff_frac_bits=coeff_frac_bits,
    )

    x = series["history"]
    marker_stride = max(1, history_length // 12)

    plt.figure(figsize=(12, 7))

    plt.plot(
        x,
        series["quantized_pct"],
        label="quantized (QU1.15)",
        linewidth=1.8,
        marker="s",
        markersize=4,
        markevery=marker_stride,
        alpha=0.9,
    )
    plt.plot(
        x,
        series["simple_pct"],
        label="bitshift: simple",
        linewidth=1.6,
        marker="o",
        markersize=3.8,
        markevery=marker_stride,
        alpha=0.9,
    )
    plt.plot(
        x,
        series["slow_decay_pct"],
        label="bitshift: slow_decay",
        linewidth=1.6,
        marker="^",
        markersize=4,
        markevery=marker_stride,
        alpha=0.9,
    )
    plt.plot(
        x,
        series["custom_pct"],
        label="bitshift: custom",
        linewidth=1.6,
        marker="D",
        markersize=3.6,
        markevery=marker_stride,
        alpha=0.9,
    )
    plt.plot(
        x,
        series["custom_slow_pct"],
        label="bitshift: custom_slow",
        linewidth=1.6,
        marker="v",
        markersize=4,
        markevery=marker_stride,
        alpha=0.9,
    )

    # plt.title(
    #     f"Coefficient Relative Error Comparison (alpha={alpha}, H={history_length})"
    # )
    plt.xlabel("History Step (k)")
    plt.ylabel("Relative Error (%)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()

    if output_path is not None:
        final_path = resolve_plot_path(output_path, use_svg)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if use_svg:
            plt.savefig(final_path, format="svg")
        else:
            plt.savefig(final_path, dpi=150)
        print(f"Wrote relative error plot to {final_path}")

    plt.close()


def _method_series_from_relative_error(
    series: Dict[str, np.ndarray],
) -> Tuple[List[str], np.ndarray]:
    method_names = ["quantized", "simple", "slow_decay", "custom", "custom_slow"]
    matrix = np.vstack(
        [
            series["quantized_pct"],
            series["simple_pct"],
            series["slow_decay_pct"],
            series["custom_pct"],
            series["custom_slow_pct"],
        ]
    )
    return method_names, matrix


def plot_relative_error_abslog(
    alpha: float = 0.5,
    history_length: int = 16,
    coeff_bits: int = 16,
    coeff_frac_bits: int = 15,
    output_path: Path | None = None,
    use_svg: bool = False,
):
    """
    Plot absolute relative error (%) using a log-like scale that includes zero.

    Notes:
    - Exact zeros are shown at y=0.
    - Uses symlog scale so near-zero is linear and larger values are logarithmic.
    """
    series = build_relative_error_series(
        alpha=alpha,
        history_length=history_length,
        coeff_bits=coeff_bits,
        coeff_frac_bits=coeff_frac_bits,
    )
    method_names, matrix = _method_series_from_relative_error(series)

    x = series["history"]
    marker_stride = max(1, history_length // 12)
    abs_matrix = np.abs(matrix)

    markers = {
        "quantized": "s",
        "simple": "o",
        "slow_decay": "^",
        "custom": "D",
        "custom_slow": "v",
    }

    plt.figure(figsize=(12, 7))
    for idx, name in enumerate(method_names):
        plt.plot(
            x,
            abs_matrix[idx],
            label=name,
            linewidth=1.8 if name == "quantized" else 1.6,
            marker=markers[name],
            markersize=4,
            markevery=marker_stride,
            alpha=0.9,
        )

    # symlog = symmetric log scale:
    # - linear region around 0, so exact zeros are displayed as y=0
    # - logarithmic spacing outside that region for larger magnitudes
    #
    # Tuning notes for future adjustments:
    # - linthresh: width of the linear near-zero region (smaller -> more log-like,
    #   larger -> more linear around zero)
    # - linscale: visual height allocated to the linear region
    # - base: logarithm base for the outer log region (10 is standard)
    plt.yscale("symlog", linthresh=1.0, linscale=1.0, base=10)
    y_max = float(np.max(abs_matrix))
    plt.ylim(0.0, max(1.0, y_max * 1.1))

    plt.xlabel("History Step (k)")
    # NOTE: This is symlog scale, see comments above. The plot visually shows the
    # absolute relative error, but the y-axis is not a pure percentage scale due
    # to the symlog transformation.
    plt.ylabel("Absolute Relative Error (%)")
    plt.grid(True, alpha=0.25, which="both")
    plt.legend(loc="best")
    plt.tight_layout()

    if output_path is not None:
        final_path = resolve_plot_path(output_path, use_svg)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if use_svg:
            plt.savefig(final_path, format="svg")
        else:
            plt.savefig(final_path, dpi=150)
        print(f"Wrote absolute relative error log plot to {final_path}")

    plt.close()


def plot_mean_absolute_error_bar(
    alpha: float = 0.5,
    history_length: int = 16,
    coeff_bits: int = 16,
    coeff_frac_bits: int = 15,
    output_path: Path | None = None,
    use_svg: bool = False,
):
    """
    Plot mean absolute relative error (%) as a bar chart by method.
    """
    series = build_relative_error_series(
        alpha=alpha,
        history_length=history_length,
        coeff_bits=coeff_bits,
        coeff_frac_bits=coeff_frac_bits,
    )
    method_names, matrix = _method_series_from_relative_error(series)
    mae = np.mean(np.abs(matrix), axis=1)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(method_names, mae, width=0.65)
    plt.ylabel("Mean Absolute Relative Error (%)")
    plt.xlabel("Method")
    plt.grid(True, axis="y", alpha=0.25)

    for bar, value in zip(bars, mae):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if output_path is not None:
        final_path = resolve_plot_path(output_path, use_svg)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if use_svg:
            plt.savefig(final_path, format="svg")
        else:
            plt.savefig(final_path, dpi=150)
        print(f"Wrote mean absolute error bar chart to {final_path}")

    plt.close()


def print_compact_table(rows: List[Dict[str, str]], max_rows: int = 32):
    # Print header from keys of first row
    if not rows:
        print("(no rows)")
        return
    keys = list(rows[0].keys())
    # Column widths
    widths = {k: max(len(k), max((len(r.get(k, "")) for r in rows))) for k in keys}
    header = " ".join(f"{k:<{widths[k]}}" for k in keys)
    print(header)
    print("-" * len(header))
    for r in rows[:max_rows]:
        line = " ".join(f"{r.get(k, ''):<{widths[k]}}" for k in keys)
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GL coefficients and bit-shift approximations"
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--history-length", type=int, default=16)
    parser.add_argument("--coeff-bits", type=int, default=16)
    parser.add_argument("--coeff-frac-bits", type=int, default=15)
    parser.add_argument("--precision", type=int, default=8)
    parser.add_argument(
        "--out-quant-csv",
        type=str,
        default=None,
        help="CSV path for quantized comparison",
    )
    parser.add_argument(
        "--out-bitshift-csv",
        type=str,
        default=None,
        help="CSV path for bitshift comparison",
    )
    parser.add_argument(
        "--out-relerr-plot",
        type=str,
        default=None,
        help="Output image path for relative-error plot (quantized + bitshift variants)",
    )
    parser.add_argument(
        "--out-relerr-abslog",
        type=str,
        default=None,
        help="Output image path for absolute relative-error log-scale line plot",
    )
    parser.add_argument(
        "--out-relerr-mae-bar",
        type=str,
        default=None,
        help="Output image path for mean absolute relative-error bar chart",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Save plot outputs as SVG instead of PNG",
    )

    args = parser.parse_args()

    q_rows = build_quantized_table(
        alpha=args.alpha,
        history_length=args.history_length,
        coeff_bits=args.coeff_bits,
        coeff_frac_bits=args.coeff_frac_bits,
        precision=args.precision,
    )

    print("Quantized comparison (analytical |g_k| vs quantized magnitude):")
    print_compact_table(q_rows)

    if args.out_quant_csv:
        fieldnames = [
            "index",
            "analytical",
            "quantized_int",
            "quantized_hex",
            "quantized_float",
            "pct_diff",
        ]
        write_csv(Path(args.out_quant_csv), q_rows, fieldnames)
        print(f"Wrote quantized CSV to {args.out_quant_csv}")

    bs_rows, seq_names = build_bitshift_comparison(
        alpha=args.alpha, history_length=args.history_length, precision=args.precision
    )
    print("\nBit-shift comparisons:")
    print_compact_table(bs_rows)

    if args.out_bitshift_csv:
        # Build fieldnames: index, analytical, for each seq: seq, seq_diff, seq_pct
        fieldnames = ["index", "analytical"]
        for name in seq_names:
            fieldnames += [name, f"{name}_diff", f"{name}_pct"]

        write_csv(Path(args.out_bitshift_csv), bs_rows, fieldnames)
        print(f"Wrote bit-shift CSV to {args.out_bitshift_csv}")

    if args.out_relerr_plot:
        plot_relative_error_comparison(
            alpha=args.alpha,
            history_length=args.history_length,
            coeff_bits=args.coeff_bits,
            coeff_frac_bits=args.coeff_frac_bits,
            output_path=Path(args.out_relerr_plot),
            use_svg=args.svg,
        )

    if args.out_relerr_abslog:
        plot_relative_error_abslog(
            alpha=args.alpha,
            history_length=args.history_length,
            coeff_bits=args.coeff_bits,
            coeff_frac_bits=args.coeff_frac_bits,
            output_path=Path(args.out_relerr_abslog),
            use_svg=args.svg,
        )

    if args.out_relerr_mae_bar:
        plot_mean_absolute_error_bar(
            alpha=args.alpha,
            history_length=args.history_length,
            coeff_bits=args.coeff_bits,
            coeff_frac_bits=args.coeff_frac_bits,
            output_path=Path(args.out_relerr_mae_bar),
            use_svg=args.svg,
        )


if __name__ == "__main__":
    main()
