"""
Calculate maximum history sizes for fractional-order LIF neuron GL weights.

This script computes Grünwald-Letnikov (GL) fractional derivative weights
for various alpha values and determines the maximum practical history size
before weights become too small to represent in the given fixed-point format.

For practical-sized analyses (default max_k ranges), GL coefficients are
generated via the shared recurrence implementation in `common.scripts.utils`.
For extreme-k searches in `max_history_json`, scipy's binomial form is retained.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from scipy.special import binom

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.scripts.utils import compute_gl_coefficients


def fixed_point_threshold(frac_bits: int) -> float:
    """
    Calculate the smallest representable positive value in fixed-point format.

    Args:
            frac_bits: Number of fractional bits

    Returns:
            Smallest positive value (1 LSB)
    """
    return 1.0 / (2**frac_bits)


def generate_alpha_values(alpha_bits: int) -> List[float]:
    """
    Generate all possible alpha values for given bit width.

    Args:
            alpha_bits: Bit width for alpha representation

    Returns:
            List of all possible alpha values
    """
    max_val = 2**alpha_bits - 1
    return [i / max_val for i in range(1, max_val)]  # Exclude 0.0 and 1.0


def _weights_shared(alpha: float, max_k: int, unsigned_magnitude: bool) -> List[float]:
    """Compute weights up to max_k using shared GL recurrence implementation."""
    coeffs = compute_gl_coefficients(alpha, max_k + 1)
    weights = coeffs.tolist()
    if unsigned_magnitude:
        for k in range(1, len(weights)):
            weights[k] = abs(weights[k])
    return weights


def calculate_max_history(
    alpha: float,
    threshold: float,
    max_k: int = 100,
    unsigned_magnitude: bool = False,
    include_weights: bool = False,
) -> Tuple[int, List[float] | None]:
    """
    Calculate maximum history size and optionally weight values for given alpha.

    Weights are calculated as: weight[k] = (-1)^k * C(α,k)
    This gives the weights used in the Grünwald-Letnikov fractional derivative.

    Args:
            alpha: Fractional order parameter
            threshold: Minimum weight magnitude threshold
            max_k: Maximum k value to test
            unsigned_magnitude: If True, return unsigned magnitudes for k≥1 (saves sign bit)
            include_weights: If True, calculate and return weights; else return None

    Returns:
            Tuple of (max_history, weight_list or None)
    """
    if max_k < 1:
        return 0, [1.0] if include_weights else None

    # Use shared recurrence for normal analysis range.
    # Keep scipy/binom fallback for very large max_k to avoid heavy full-vector generation.
    use_shared = max_k <= 200_000

    if use_shared:
        weights_all = _weights_shared(alpha, max_k, unsigned_magnitude)
        max_history = 0
        for k in range(1, max_k + 1):
            if abs(weights_all[k]) >= threshold:
                max_history = k

        if include_weights:
            return max_history, weights_all[: max_history + 1]
        return max_history, None

    # Fallback: binary search for very large max_k.
    left = 1
    right = max_k
    max_history = 0

    while left <= right:
        mid = (left + right) // 2
        binom_coeff = binom(alpha, mid)
        weight = ((-1) ** mid) * binom_coeff
        if unsigned_magnitude and mid >= 1:
            weight = abs(weight)
        if abs(weight) >= threshold:
            max_history = mid
            left = mid + 1
        else:
            right = mid - 1

    if include_weights:
        weights = []
        for k in range(max_history + 1):
            binom_coeff = binom(alpha, k)
            weight = ((-1) ** k) * binom_coeff
            if unsigned_magnitude and k >= 1:
                weight = abs(weight)
            weights.append(weight)
    else:
        weights = None

    return max_history, weights


def max_history_json(bit_width: int, data_dir: str = "data"):
    """
    Write exact maximum history values for given bit width to a JSON file.

    Args:
            bit_width: Number of fractional bits (unsigned, 0 integer bits)
            data_dir: Directory to save the JSON file (default: "data")
    """
    os.makedirs(data_dir, exist_ok=True)

    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold = fixed_point_threshold(bit_width)

    if 32 < bit_width <= 64:
        high = 4_700_000_000_000_000
    else:
        high = 41_750_000
    low = 1

    results = []

    for alpha_val in alpha_values:
        left = low
        right = high
        max_history = 0

        while left <= right:
            mid = (left + right) // 2
            coeff = abs(binom(alpha_val, mid))
            if coeff >= threshold:
                max_history = mid
                left = mid + 1
            else:
                right = mid - 1

        results.append({"alpha": alpha_val, "max_history": max_history})

    filename = os.path.join(data_dir, f"max_history_{bit_width}bit.json")
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exact max history values written to {filename}")


def format_coefficient(coeff: float, precision: int = 6) -> str:
    """Format coefficient for display with appropriate precision."""
    if abs(coeff) < 1e-10:
        return "0.000000"
    return f"{coeff:+.{precision}f}"


def print_analysis_table(
    results: Dict[float, Tuple[int, List[float]]],
    threshold: float,
    format_name: str,
    max_display_k: int = 25,
):
    """Print formatted analysis table to console."""
    print(f"\n{'='*80}")
    print(f"Binomial Coefficient Analysis - {format_name} Format")
    print(f"Threshold: {threshold:.8f} (1 LSB)")
    print(f"{'='*80}")

    header = (
        ["α"]
        + [f"k={k}" for k in range(1, min(max_display_k + 1, 21))]
        + ["Max History"]
    )
    print(f"{header[0]:>6}", end="")
    for h in header[1:-1]:
        print(f"{h:>10}", end="")
    print(f"{header[-1]:>12}")
    print("-" * 80)

    for alpha in sorted(results.keys()):
        max_history, coefficients = results[alpha]
        print(f"{alpha:>6.2f}", end="")

        for k in range(1, min(max_display_k + 1, 21)):
            if k < len(coefficients):
                coeff_str = format_coefficient(coefficients[k], 4)
                if abs(coefficients[k]) < threshold:
                    coeff_str = f"({coeff_str[1:]})"
                print(f"{coeff_str:>10}", end="")
            else:
                print(f"{'---':>10}", end="")

        print(f"{'**' + str(max_history) + '**':>12}")


def save_csv(
    results: Dict[float, Tuple[int, List[float]]], filename: str, max_k: int = 50
):
    """Save results to CSV file."""
    with open(filename, "w", newline="") as csvfile:
        actual_max_k = min(
            max_k, max(len(coeffs) - 1 for _, coeffs in results.values())
        )

        writer = csv.writer(csvfile)
        header = (
            ["alpha"] + [f"k_{k}" for k in range(actual_max_k + 1)] + ["max_history"]
        )
        writer.writerow(header)

        for alpha in sorted(results.keys()):
            max_history, coefficients = results[alpha]
            row = [alpha]

            for k in range(actual_max_k + 1):
                if k < len(coefficients):
                    row.append(coefficients[k])
                else:
                    row.append(0.0)

            row.append(max_history)
            writer.writerow(row)

    print(f"\nResults saved to: {filename}")


def save_format_comparison_csv(filename: str):
    """Generate CSV comparing different fixed-point formats."""
    formats = [
        ("0.8", False),
        ("u0.8", True),
        ("0.12", False),
        ("u0.12", True),
        ("0.16", False),
        ("u0.16", True),
    ]

    test_alphas = [0.1, 0.9]
    results = []

    for format_str, unsigned_magnitude in formats:
        if format_str.startswith("u"):
            int_bits, frac_bits = map(int, format_str[1:].split("."))
        else:
            int_bits, frac_bits = map(int, format_str.split("."))

        total_bits = int_bits + frac_bits

        if unsigned_magnitude:
            threshold = fixed_point_threshold(total_bits)
            format_desc = f"k=0: 1.{total_bits-1}, k≥1: 0.{total_bits}"
        else:
            if int_bits == 0:
                actual_frac_bits = total_bits - 1
                threshold = fixed_point_threshold(actual_frac_bits)
                format_desc = f"S0.{actual_frac_bits}"
            else:
                threshold = fixed_point_threshold(frac_bits)
                format_desc = f"S{int_bits}.{frac_bits}"

        max_histories = {}
        for alpha in test_alphas:
            max_history, _ = calculate_max_history(
                alpha, threshold, max_k=500, unsigned_magnitude=unsigned_magnitude
            )
            max_histories[alpha] = max_history

        result = {
            "format": format_str,
            "format_desc": format_desc,
            "total_bits": total_bits,
            "threshold": threshold,
        }

        for alpha in test_alphas:
            result[f"max_history_alpha_{alpha}"] = max_histories[alpha]

        results.append(result)

    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["format", "format_desc", "total_bits", "threshold"] + [
            f"max_history_alpha_{alpha}" for alpha in test_alphas
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result)

    print(f"\nFormat comparison saved to: {filename}")

    print(f"\n{'='*90}")
    print("Fixed-Point Format Comparison")
    print(f"{'='*90}")
    print(
        f"{'Format':<8} {'Description':<25} {'Bits':<5} {'Threshold':<12} {'α=0.1':<6} {'α=0.9':<6}"
    )
    print("-" * 90)

    for result in results:
        print(
            f"{result['format']:<8} {result['format_desc']:<25} {result['total_bits']:<5} "
            f"{result['threshold']:<12.8f} {result['max_history_alpha_0.1']:<6} {result['max_history_alpha_0.9']:<6}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Calculate maximum history for fractional-order LIF coefficients"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="1.15",
        help='Fixed-point format as "integer.fractional" (default: 1.15)',
    )
    parser.add_argument(
        "--alpha-list",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9],
        help="List of alpha values to analyze",
    )
    parser.add_argument(
        "--alpha-bits",
        type=int,
        help="Generate all possible alpha values for given bit width",
    )
    parser.add_argument("--csv", type=str, help="Save results to CSV file")
    parser.add_argument(
        "--max-k", type=int, default=50, help="Maximum k value to test (default: 50)"
    )
    parser.add_argument(
        "--display-k",
        type=int,
        default=20,
        help="Maximum k value to display in table (default: 20)",
    )
    parser.add_argument(
        "--unsigned-magnitude",
        action="store_true",
        help="Store k≥1 weights as unsigned magnitudes (saves sign bit)",
    )
    parser.add_argument(
        "--format-comparison",
        type=str,
        help="Generate CSV comparing different fixed-point formats",
    )

    args = parser.parse_args()

    if args.format_comparison:
        save_format_comparison_csv(args.format_comparison)
        return

    try:
        int_bits, frac_bits = map(int, args.format.split("."))
        total_bits = int_bits + frac_bits
    except ValueError:
        print(f"Error: Invalid format '{args.format}'. Use format like '1.15'")
        sys.exit(1)

    if args.unsigned_magnitude:
        k0_frac_bits = total_bits - 1
        k1_frac_bits = total_bits

        threshold = fixed_point_threshold(k1_frac_bits)
        format_name = f"k=0: 1.{k0_frac_bits}, k≥1: 0.{k1_frac_bits}"

        print("Using unsigned magnitude mode:")
        print(
            f"  k=0: 1.{k0_frac_bits} signed format (threshold: {fixed_point_threshold(k0_frac_bits):.8f})"
        )
        print(
            f"  k≥1: 0.{k1_frac_bits} unsigned magnitude format (threshold: {threshold:.8f})"
        )
    else:
        sign_bits = 1 if int_bits == 0 else 0
        actual_frac_bits = total_bits - sign_bits - int_bits
        threshold = fixed_point_threshold(actual_frac_bits)
        format_name = f"{int_bits + sign_bits}.{actual_frac_bits}"

        print(f"Using signed format: {format_name}")
        print(f"Threshold: {threshold:.8f}")

    if args.alpha_bits:
        alpha_values = generate_alpha_values(args.alpha_bits)
        print(
            f"Generated {len(alpha_values)} alpha values from {args.alpha_bits}-bit representation"
        )
    else:
        alpha_values = args.alpha_list

    print(f"Analyzing {len(alpha_values)} alpha values...")
    print(f"Fixed-point format: {format_name} ({total_bits} bits total)")

    results = {}
    for alpha in alpha_values:
        max_history, weights = calculate_max_history(
            alpha, threshold, args.max_k, args.unsigned_magnitude, True
        )
        results[alpha] = (max_history, weights)

    print_analysis_table(results, threshold, format_name, args.display_k)

    if args.csv:
        save_csv(results, args.csv, args.max_k)

    if args.unsigned_magnitude:
        print("\nUnsigned Magnitude Analysis:")
        print(f"Precision gain for k≥1: {k1_frac_bits - k0_frac_bits} bit(s)")
        print(
            f"Threshold improvement: {fixed_point_threshold(k0_frac_bits)/threshold:.1f}x better for k≥1"
        )

        print(f"k=0 LSB: {fixed_point_threshold(k0_frac_bits):.8f}")
        print(f"k≥1 LSB: {threshold:.8f}")

        for hist_size in [4, 8, 16]:
            if hist_size <= 50:
                precision_gain_coeffs = hist_size
                print(
                    f"History size {hist_size}: {precision_gain_coeffs} coefficients gain {k1_frac_bits - k0_frac_bits} bit(s) precision each"
                )


if __name__ == "__main__":
    main()
