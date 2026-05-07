"""Visualize GL coefficient magnitude and alpha-history relationships.

Extracted from 1_design_space/v1-092025/scripts/plot.py (archive).
Updated to use the shared common/scripts/plot_styles module.

Provides:
  - plot_coefficient_magnitude     log-log coefficient magnitude vs. k
  - plot_normalized_coefficient_magnitude  normalized version of the above
  - plot_alpha_sweep               history length vs. alpha for UQ formats
  - Markdown-table and convenience save wrappers

Typical usage:
    python 1_design_space/scripts/visualize_alpha_history.py \\
        --plot coefficient-magnitude --output images/coefficient_magnitude.svg

    python 1_design_space/scripts/visualize_alpha_history.py \\
        --plot alpha-sweep --output images/alpha_sweep.svg
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

# FIXME: Is binom just for integers?? Is this plotting incorrect?
from scipy.special import binom

# Resolve project root so that common.scripts.plot_styles is importable.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.scripts.plot_styles import (  # noqa: E402
    OKABE_ITO,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    DEFAULT_FIGSIZE,
)

# 1_design_space/scripts/ already contains max_history.py.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from max_history import (  # noqa: E402
    fixed_point_threshold,
    calculate_max_history,
    generate_alpha_values,
)

# ---------------------------------------------------------------------------
# Color / marker sequences for multi-series plots (draw from Okabe-Ito)
# ---------------------------------------------------------------------------

_LINE_STYLES = [
    {"color": OKABE_ITO[0], "marker": "o", "linestyle": "-"},        # blue
    {"color": OKABE_ITO[5], "marker": "s", "linestyle": "--"},       # orange
    {"color": OKABE_ITO[2], "marker": "^", "linestyle": "-."},       # bluish green
    {"color": OKABE_ITO[1], "marker": "D", "linestyle": ":"},        # vermillion
    {"color": OKABE_ITO[3], "marker": "v", "linestyle": (0, (1, 1, 2, 1, 3, 1))},  # purple
]


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


# TODO: Consider consolidating generation of alpha sweep and max history data
def generate_bar_chart_comparison_data(max_k: int = 500) -> Dict:
    """Generate format comparison data for bar chart visualization.

    Tests specific alpha values (0.1, 0.5, 0.9) across different formats.

    Returns:
        Dictionary with format comparison data
    """
    formats = [
        ("0.8", False),   # 8-bit signed
        ("u0.8", True),   # 8-bit unsigned magnitude
        ("0.12", False),  # 12-bit signed
        ("u0.12", True),  # 12-bit unsigned magnitude
        ("0.16", False),  # 16-bit signed
        ("u0.16", True),  # 16-bit unsigned magnitude
    ]

    test_alphas = [0.1, 0.5, 0.9]
    results = {}

    for format_str, unsigned_magnitude in formats:
        if format_str.startswith("u"):
            int_bits, frac_bits = map(int, format_str[1:].split("."))
        else:
            int_bits, frac_bits = map(int, format_str.split("."))

        total_bits = int_bits + frac_bits

        if unsigned_magnitude:
            threshold = fixed_point_threshold(total_bits)
        else:
            if int_bits == 0:
                actual_frac_bits = total_bits - 1
                threshold = fixed_point_threshold(actual_frac_bits)
            else:
                threshold = fixed_point_threshold(frac_bits)

        format_results = {}
        for alpha in test_alphas:
            max_history, _ = calculate_max_history(
                alpha, threshold, max_k=max_k, unsigned_magnitude=unsigned_magnitude
            )
            format_results[f"alpha_{alpha}"] = max_history

        results[format_str] = format_results

    return results


def generate_alpha_sweep_data(alpha_bits: int = 4, max_k: int = 3000) -> Dict:
    """Generate data for alpha sweep across all possible values for given bit width.

    Uses unsigned magnitude formats only.

    Args:
        alpha_bits: Number of bits to represent alpha (default: 4)
        max_k: Maximum k value to test (default: 3000)

    Returns:
        Dictionary with alpha sweep data
    """
    alpha_values = generate_alpha_values(alpha_bits)

    formats = [
        ("UQ0.8", 8),
        ("UQ0.12", 12),
        ("UQ0.16", 16),
        ("UQ0.32", 32),
        ("UQ0.64", 64),
    ]

    results = {}
    for format_name, total_bits in formats:
        threshold = fixed_point_threshold(total_bits)
        format_results = {
            "alpha_values": alpha_values,
            "max_histories": [],
            "threshold": threshold,
        }
        for alpha in alpha_values:
            max_history, _ = calculate_max_history(
                alpha, threshold, max_k=max_k, unsigned_magnitude=True
            )
            format_results["max_histories"].append(max_history)

        results[format_name] = format_results

    return results


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_coefficient_magnitude(save_path: str = None, max_k: int = 20):
    """Plot coefficient magnitude vs. history step k for several alpha values.

    Uses a log-log scale so the power-law decay is visible and comparable
    across orders. Series start at k=1 (g_0 = 1 is skipped).

    Args:
        save_path: Output path. If None, displays interactively.
        max_k: Maximum k value to plot (default: 20).
    """
    alpha_values = [0.1, 0.5, 0.9]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    k_values = list(range(1, max_k + 1))

    for i, alpha in enumerate(alpha_values):
        coefficient_magnitudes = []
        for k in k_values:
            binom_coeff = binom(alpha, k)
            sign_factor = -1.0 if (k % 2 == 1) else 1.0
            final_weight = sign_factor * binom_coeff
            coefficient_magnitudes.append(abs(final_weight))

        style = _LINE_STYLES[i]
        ax.plot(
            k_values,
            coefficient_magnitudes,
            label=f"α = {alpha}",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
            alpha=0.8,
        )

    ax.set_xlabel("History Step (k)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Coefficient Magnitude", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1, max_k)
    ax.set_xticks(range(1, max_k + 1, max(1, max_k // 10)))

    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)

    ax.legend(loc="upper right", fontsize=LEGEND_FONTSIZE, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        if save_path.lower().endswith(".svg"):
            plt.savefig(save_path, format="svg", bbox_inches="tight")
        elif save_path.lower().endswith(".pdf"):
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
        else:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Coefficient magnitude plot saved to: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_normalized_coefficient_magnitude(save_path: str = None, max_k: int = 20):
    """Plot normalized coefficient magnitude vs. k.

    Each sequence is normalized by its k=1 value so all sequences start at 1.0,
    enabling direct comparison of decay rates across alpha values.

    Args:
        save_path: Output path. If None, displays interactively.
        max_k: Maximum k value to plot (default: 20).
    """
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    k_values = list(range(1, max_k + 1))

    for i, alpha in enumerate(alpha_values):
        coefficient_magnitudes = []
        for k in k_values:
            binom_coeff = binom(alpha, k)
            sign_factor = -1.0 if (k % 2 == 1) else 1.0
            final_weight = sign_factor * binom_coeff
            coefficient_magnitudes.append(abs(final_weight))

        k1_magnitude = coefficient_magnitudes[0]
        normalized_magnitudes = [mag / k1_magnitude for mag in coefficient_magnitudes]

        style = _LINE_STYLES[i % len(_LINE_STYLES)]
        ax.plot(
            k_values,
            normalized_magnitudes,
            label=f"α = {alpha}",
            color=style["color"],
            linestyle=style["linestyle"],
        )

    ax.set_xlabel("History Step (k)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Normalized Coefficient Magnitude", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

    ax.set_yscale("log")
    ax.set_xlim(1, max_k)
    ax.set_xticks(range(1, max_k + 1, max(1, max_k // 10)))

    ax.grid(True, alpha=0.6, linestyle=":")
    ax.set_axisbelow(True)

    ax.legend(loc="upper right", fontsize=LEGEND_FONTSIZE, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        if save_path.lower().endswith(".svg"):
            plt.savefig(save_path, format="svg", bbox_inches="tight")
        elif save_path.lower().endswith(".pdf"):
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
        else:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Normalized coefficient magnitude plot saved to: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_alpha_sweep(save_path: str = None, alpha_bits: int = 4, max_k: int = 3000):
    """Line plot of maximum history length vs. alpha for several UQ formats.

    Args:
        save_path: Output path. If None, displays interactively.
        alpha_bits: Number of bits used to represent alpha (default: 4).
        max_k: Upper bound on history length tested (default: 3000).
    """
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    format_styles = {
        "UQ0.8": {
            "color": OKABE_ITO[0],
            "linestyle": "-",
        },
        "UQ0.16": {
            "color": OKABE_ITO[2],
            "linestyle": "-.",
        },
        "UQ0.32": {
            "color": OKABE_ITO[5],
            "linestyle": "--",
        },
    }

    # Compute max history for each format / alpha combination.
    results: Dict = {}
    for format_str in format_styles:
        frac_bits = int(format_str.split(".")[1])
        threshold = fixed_point_threshold(frac_bits)
        format_results: Dict = {}
        for alpha in alpha_values:
            max_history, _ = calculate_max_history(
                alpha, threshold, max_k=max_k, unsigned_magnitude=True
            )
            format_results[f"{alpha}"] = max_history
        results[format_str] = format_results

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    for format_name in format_styles:
        data = results[format_name]
        x = [float(k) for k in data]
        display_y = [h if h < max_k else max_k for h in data.values()]
        style = format_styles[format_name]
        ax.plot(
            x,
            display_y,
            label=format_name,
            color=style["color"],
            linestyle=style["linestyle"],
        )

    ax.set_xlabel("Fractional Order (α)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Maximum History Size", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

    ax.set_xlim(0, 1)
    ax.set_yscale("log")
    ax.set_ylim(1, max_k * 2)

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    ax.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)

    note_text = f"Values at {max_k} indicate max history ≥ {max_k} (test limit)"  # noqa: F841

    if save_path:
        if save_path.lower().endswith(".svg"):
            plt.savefig(save_path, format="svg", bbox_inches="tight")
        elif save_path.lower().endswith(".pdf"):
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
        else:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Alpha sweep plot saved to: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown table helpers
# ---------------------------------------------------------------------------


def generate_alpha_sweep_markdown_table(alpha_bits: int = 4, max_k: int = 3000) -> str:
    """Generate a markdown table showing history length for all alpha values.

    Args:
        alpha_bits: Bits used to represent alpha (default: 4).
        max_k: Maximum k value to test.

    Returns:
        Formatted markdown table string.
    """
    sweep_data = generate_alpha_sweep_data(alpha_bits, max_k)
    alpha_values = sweep_data["UQ0.8"]["alpha_values"]

    lines = ["| α Value | UQ0.8 | UQ0.12 | UQ0.16 |", "|---------|-------|--------|--------|"]
    for i, alpha in enumerate(alpha_values):
        alpha_str = f"{alpha:.3f}"
        uq8 = sweep_data["UQ0.8"]["max_histories"][i]
        uq12 = sweep_data["UQ0.12"]["max_histories"][i]
        uq16 = sweep_data["UQ0.16"]["max_histories"][i]
        lines.append(f"| {alpha_str} | {uq8} | {uq12} | {uq16} |")

    return "\n".join(lines)


def save_alpha_sweep_markdown_table(
    output_path: str = "alpha_sweep_table.md",
    alpha_bits: int = 4,
    max_k: int = 3000,
):
    """Generate and save a markdown table of alpha sweep results.

    Args:
        output_path: Path to save the markdown file.
        alpha_bits: Bits used to represent alpha.
        max_k: Maximum k value to test.
    """
    table_content = generate_alpha_sweep_markdown_table(alpha_bits, max_k)
    sweep_data = generate_alpha_sweep_data(alpha_bits, max_k)

    full_content = f"""# Alpha Sweep Analysis - Maximum History Length

Analysis of maximum history length for fractional-order LIF neurons using different fixed-point formats.
All formats use unsigned magnitude representation for k≥1 coefficients.

## Format Details
- **UQ0.8**: k=0 uses UQ1.7, k≥1 uses UQ0.8 (8-bit total)
- **UQ0.12**: k=0 uses UQ1.11, k≥1 uses UQ0.12 (12-bit total)
- **UQ0.16**: k=0 uses UQ1.15, k≥1 uses UQ0.16 (16-bit total)

## Results Table

{table_content}

## Notes
- Alpha values from {alpha_bits}-bit representation: {len(sweep_data['UQ0.8']['alpha_values'])} total values
- Values marked with '+' indicate maximum history ≥ {max_k} (test limit)
- Higher precision formats (more bits) support longer history lengths
- Lower alpha values generally support longer history lengths
"""

    with open(output_path, "w") as f:
        f.write(full_content)

    print(f"Alpha sweep markdown table saved to: {output_path}")
    return table_content


# ---------------------------------------------------------------------------
# Convenience save wrappers
# ---------------------------------------------------------------------------


def save_coefficient_magnitude_plot(
    output_path: str = "coefficient_magnitude.svg", max_k: int = 20
):
    """Save a coefficient magnitude plot."""
    plot_coefficient_magnitude(save_path=output_path, max_k=max_k)


def save_normalized_coefficient_magnitude_plot(
    output_path: str = "normalized_coefficient_magnitude.svg", max_k: int = 20
):
    """Save a normalized coefficient magnitude plot."""
    plot_normalized_coefficient_magnitude(save_path=output_path, max_k=max_k)


def save_alpha_sweep_plot(
    output_path: str = "alpha_sweep.svg", alpha_bits: int = 4, max_k: int = 3000
):
    """Save an alpha sweep plot."""
    plot_alpha_sweep(save_path=output_path, alpha_bits=alpha_bits, max_k=max_k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PLOT_CHOICES = ["coefficient-magnitude", "normalized-coefficient-magnitude", "alpha-sweep"]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate alpha-history visualization plots. "
            "Extracted from 1_design_space/v1-092025/scripts/plot.py."
        )
    )
    parser.add_argument(
        "--plot",
        choices=PLOT_CHOICES,
        default="coefficient-magnitude",
        help="Which plot to generate (default: coefficient-magnitude)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path. If not given, displays the plot interactively.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=20,
        help="Maximum k / history value to plot (default: 20).",
    )
    parser.add_argument(
        "--alpha-bits",
        type=int,
        default=4,
        help="Bits used to represent alpha for the alpha-sweep plot (default: 4).",
    )

    args = parser.parse_args()

    if args.plot == "coefficient-magnitude":
        plot_coefficient_magnitude(save_path=args.output, max_k=args.max_k)
    elif args.plot == "normalized-coefficient-magnitude":
        plot_normalized_coefficient_magnitude(save_path=args.output, max_k=args.max_k)
    elif args.plot == "alpha-sweep":
        plot_alpha_sweep(
            save_path=args.output,
            alpha_bits=args.alpha_bits,
            max_k=args.max_k,
        )


if __name__ == "__main__":
    main()
