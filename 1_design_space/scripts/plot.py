import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import os
import sys

# FIXME: Is binom just for integers?? Is this plotting incorrect?
from scipy.special import binom
from plot_membrane_potential import get_latex_figsize

# Add the scripts directory to the path so we can import max_history
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from max_history import (
    fixed_point_threshold,
    calculate_max_history,
    generate_alpha_values,
)


# TODO: Consider consolidating generation of alpha sweep and max history data
def generate_bar_chart_comparison_data(max_k: int = 500) -> Dict:
    """
    Generate format comparison data for bar chart visualization.
    Tests specific alpha values (0.1, 0.5, 0.9) across different formats.

    Returns:
        Dictionary with format comparison data
    """

    # Define formats to compare
    formats = [
        ("0.8", False),  # 8-bit signed
        ("u0.8", True),  # 8-bit unsigned magnitude
        ("0.12", False),  # 12-bit signed
        ("u0.12", True),  # 12-bit unsigned magnitude
        ("0.16", False),  # 16-bit signed
        ("u0.16", True),  # 16-bit unsigned magnitude
    ]

    # Test alpha values (including 0.5)
    test_alphas = [0.1, 0.5, 0.9]

    results = {}

    for format_str, unsigned_magnitude in formats:
        # Parse format
        if format_str.startswith("u"):
            int_bits, frac_bits = map(int, format_str[1:].split("."))
        else:
            int_bits, frac_bits = map(int, format_str.split("."))

        total_bits = int_bits + frac_bits

        # Calculate threshold based on format
        if unsigned_magnitude:
            # k≥1 coefficients use unsigned format (all bits fractional)
            threshold = fixed_point_threshold(total_bits)
        else:
            # Signed format: need sign bit
            if int_bits == 0:
                actual_frac_bits = total_bits - 1  # Reserve 1 bit for sign
                threshold = fixed_point_threshold(actual_frac_bits)
            else:
                threshold = fixed_point_threshold(frac_bits)

        # Calculate max history for each alpha
        format_results = {}
        for alpha in test_alphas:
            max_history, _ = calculate_max_history(
                alpha, threshold, max_k=max_k, unsigned_magnitude=unsigned_magnitude
            )
            format_results[f"alpha_{alpha}"] = max_history

        results[format_str] = format_results

    return results


def generate_alpha_sweep_data(alpha_bits: int = 4, max_k: int = 3000) -> Dict:
    """
    Generate data for alpha sweep across all possible values for given bit width.
    Uses unsigned magnitude formats only.

    Args:
        alpha_bits: Number of bits to represent alpha (default: 4)
        max_k: Maximum k value to test (default: 300)

    Returns:
        Dictionary with alpha sweep data
    """
    # Generate all possible alpha values for the given bit width
    alpha_values = generate_alpha_values(alpha_bits)

    # Define unsigned magnitude formats to compare
    formats = [
        ("UQ0.8", 8),  # 8-bit unsigned
        ("UQ0.12", 12),  # 12-bit unsigned
        ("UQ0.16", 16),  # 16-bit unsigned
        ("UQ0.32", 32),  # 32-bit unsigned
        ("UQ0.64", 64),  # 64-bit unsigned
    ]

    results = {}

    for format_name, total_bits in formats:
        # For unsigned magnitude: all bits are fractional for k≥1
        threshold = fixed_point_threshold(total_bits)

        format_results = {
            "alpha_values": alpha_values,
            "max_histories": [],
            "threshold": threshold,
        }

        # Calculate max history for each alpha
        for alpha in alpha_values:
            max_history, _ = calculate_max_history(
                alpha, threshold, max_k=max_k, unsigned_magnitude=True
            )
            format_results["max_histories"].append(max_history)

        results[format_name] = format_results

    return results


def plot_alpha_sweep(save_path: str = None, alpha_bits: int = 4, max_k: int = 3000):
    """
    Create a line plot showing history length for all possible alpha values
    using different unsigned magnitude formats.

    Args:
        save_path: Path to save the plot (if None, displays plot)
        alpha_bits: Number of bits to represent alpha (default: 4)
        max_k: Maximum k value to test
    """
    # Alpha values to compare
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    figsize = get_latex_figsize()

    # Font size configuration
    axis_label_fontsize = 8
    tick_label_fontsize = 6

    # Create the plot
    fig, ax = plt.subplots(figsize=(figsize["width"], figsize["height"]))
    # Create the plot
    # fig, ax = plt.subplots(figsize=(12, 8))

    # Define line styles and markers for each format
    format_styles = {
        "UQ0.8": {
            "color": "dodgerblue",
            "marker": "o",
            "linestyle": "-",
            "markersize": 4,
        },
        # "UQ0.12": {
        #     "color": "#ff7f0e",
        #     "marker": "s",
        #     "linestyle": "--",
        #     "markersize": 4,
        # },
        "UQ0.16": {
            "color": "#2ca02c",
            "marker": "^",
            "linestyle": "-.",
            "markersize": 4,
        },
        # The max history length for UQ0.32 is around 41_750_000 for alpha=0.15
        "UQ0.32": {
            "color": "darkorange",
            "marker": "s",
            "linestyle": "--",
            "markersize": 4,
        },
        # NOTE: The max history length for UQ0.64 can be extremely large,
        # around 4.7 quadrillion (4.7x10^15) for alpha=0.15 and around 3.98 billion (3.98x10^9)
        # for alpha=0.9.
        # "UQ0.64": {
        #     "color": "#9467bd",
        #     "marker": "D",
        #     "linestyle": ":",
        #     "markersize": 4,
        # },
    }

    # Use generate_alpha_sweep_data when plotting alpha values with a specific
    # bit width
    # Generate alpha sweep data with determined max_k
    # sweep_data = generate_alpha_sweep_data(alpha_bits, max_k)
    results = {}
    for format_str in format_styles.keys():
        # Parse format
        frac_bits = int(format_str.split(".")[1])

        threshold = fixed_point_threshold(frac_bits)

        # Calculate max history for each alpha
        format_results = {}
        for alpha in alpha_values:
            max_history, _ = calculate_max_history(
                alpha, threshold, max_k=max_k, unsigned_magnitude=True
            )
            format_results[f"{alpha}"] = max_history

        results[format_str] = format_results

    # Plot lines for each format
    # for format_name in ["UQ0.8", "UQ0.12", "UQ0.16", "UQ0.32", "UQ0.64"]:
    for format_name in ["UQ0.8", "UQ0.16", "UQ0.32"]:
        data = results[format_name]
        alpha_values = [float(key) for key in data.keys()]
        max_histories = list(data.values())

        # Create display values capped at max_k for better visualization
        display_histories = [h if h < max_k else max_k for h in max_histories]

        style = format_styles[format_name]
        ax.plot(
            alpha_values,
            display_histories,
            label=format_name,
            color=style["color"],
            # marker=style["marker"],
            linestyle=style["linestyle"],
            # markersize=style["markersize"],
            # markevery=1,  # Show markers on all points
            # linewidth=1,
        )

    # Customize the plot
    ax.set_xlabel("Fractional Order (α)", fontsize=axis_label_fontsize)
    ax.set_ylabel("Maximum History Size", fontsize=axis_label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
    # ax.set_title(
    #     f"History Length vs Alpha Value ({alpha_bits}-bit Alpha Representation)\n"
    #     f"Unsigned Magnitude Fixed-Point Formats",
    #     fontsize=14,
    #     fontweight="bold",
    #     pad=20,
    # )

    # Set axis limits and ticks
    ax.set_xlim(0, 1)
    ax.set_yscale("log")  # Use logarithmic scale for y-axis
    ax.set_ylim(1, max_k * 2)  # Start from 1 and go above max_k for log scale

    # Add grid for better readability
    # ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(loc="upper right", fontsize=tick_label_fontsize)

    # Add format descriptions as text box (moved to bottom left to avoid covering plot)
    # description_text = (
    #     f"Format Details (all unsigned magnitude):\n"
    #     f"• UQ0.8: k=0 uses UQ1.7, k≥1 uses UQ0.8\n"
    #     f"• UQ0.12: k=0 uses UQ1.11, k≥1 uses UQ0.12\n"
    #     f"• UQ0.16: k=0 uses UQ1.15, k≥1 uses UQ0.16\n"
    #     f"• Alpha values: {len(results['UQ0.8']['alpha_values'])} points from {alpha_bits}-bit representation"
    # )

    # Leaving this out for now - it may be more appropriate in the figure caption
    # ax.text(0.02, 0.15, description_text, transform=ax.transAxes, fontsize=9,
    #        verticalalignment='top',
    #        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Add note about the max_k limit
    # Exclude this while we're actually showing the max values
    note_text = f"Values at {max_k} indicate max history ≥ {max_k} (test limit)"
    # ax.text(0.98, 0.02, note_text, transform=ax.transAxes, fontsize=8,
    #        horizontalalignment='right', verticalalignment='bottom',
    #        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Adjust layout
    # plt.tight_layout()

    # Save or show the plot
    if save_path:
        # Determine format based on file extension
        if save_path.lower().endswith(".svg"):
            plt.savefig(save_path, format="svg", bbox_inches="tight")
        elif save_path.lower().endswith(".pdf"):
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
        else:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Alpha sweep plot saved to: {save_path}")
    else:
        plt.show()


def plot_format_comparison(save_path: str = None, max_k: int = 500):
    """
    Create a visualization comparing maximum history sizes across different
    fixed-point formats and alpha values.

    Args:
        save_path: Path to save the plot (if None, displays plot)
        max_k: Maximum k value to test (should match max_history.py setting)
    """

    # Generate format comparison data
    formats_data = generate_bar_chart_comparison_data(max_k)

    # Alpha values to compare
    alpha_values = [0.1, 0.5, 0.9]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set up grouped bar chart
    formats = list(formats_data.keys())
    n_formats = len(formats)
    n_alphas = len(alpha_values)

    # Bar width and positions
    bar_width = 0.25
    positions = np.arange(n_formats)

    # Colors for different alpha values
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

    # Create bars for each alpha value
    for i, alpha in enumerate(alpha_values):
        values = []
        for format_name in formats:
            max_hist = formats_data[format_name][f"alpha_{alpha}"]
            values.append(max_hist)

        bars = ax.bar(
            positions + i * bar_width,
            values,
            bar_width,
            label=f"α = {alpha}",
            color=colors[i],
            alpha=0.8,
        )

        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            label_text = f"{val}" if val < max_k else f"{val}+"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                label_text,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Customize the plot
    ax.set_xlabel("Fixed-Point Format", fontsize=12, fontweight="bold")
    ax.set_ylabel("Maximum History Size", fontsize=12, fontweight="bold")
    ax.set_title(
        "Fixed-Point Format Comparison:\nMaximum History Size for Fractional-Order LIF",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Set x-axis labels
    format_labels = []
    for fmt in formats:
        if fmt.startswith("u"):
            format_labels.append(f"{fmt}\n(unsigned mag.)")
        else:
            format_labels.append(f"{fmt}\n(signed)")

    ax.set_xticks(positions + bar_width)
    ax.set_xticklabels(format_labels)

    # Add legend
    ax.legend(title="Alpha Value", loc="upper right", fontsize=10)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    # Add format descriptions as text box
    description_text = (
        "Format Descriptions:\n"
        "• 0.8 signed: S0.7 (1 sign + 7 fractional bits)\n"
        "• u0.8 unsigned mag.: k=0: 1.7, k≥1: 0.8\n"
        "• 0.12 signed: S0.11 (1 sign + 11 fractional bits)\n"
        "• u0.12 unsigned mag.: k=0: 1.11, k≥1: 0.12\n"
        "• 0.16 signed: S0.15 (1 sign + 15 fractional bits)\n"
        "• u0.16 unsigned mag.: k=0: 1.15, k≥1: 0.16"
    )

    ax.text(
        0.02,
        0.98,
        description_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    # Add note about the "+" symbol
    note_text = f"Values marked with '+' indicate max history ≥ {max_k} (test limit)"
    ax.text(
        0.98,
        0.02,
        note_text,
        transform=ax.transAxes,
        fontsize=8,
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    # Adjust layout
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def save_format_comparison_plot(
    output_path: str = "format_comparison.png", max_k: int = 500
):
    """
    Convenience function to generate and save the format comparison plot.

    Args:
        output_path: Path where to save the plot
        max_k: Maximum k value to test
    """
    plot_format_comparison(save_path=output_path, max_k=max_k)


def save_alpha_sweep_plot(
    output_path: str = "alpha_sweep.svg", alpha_bits: int = 4, max_k: int = 3000
):
    """
    Convenience function to generate and save the alpha sweep plot.

    Args:
        output_path: Path where to save the plot (default: SVG format)
        alpha_bits: Number of bits to represent alpha
        max_k: Maximum k value to test
    """
    plot_alpha_sweep(save_path=output_path, alpha_bits=alpha_bits, max_k=max_k)


def generate_alpha_sweep_markdown_table(alpha_bits: int = 4, max_k: int = 3000) -> str:
    """
    Generate a markdown table showing history length for all alpha values across different formats.

    Args:
        alpha_bits: Number of bits to represent alpha (default: 4)
        max_k: Maximum k value to test (default: 1000)

    Returns:
        Formatted markdown table as string
    """
    # Generate alpha sweep data
    sweep_data = generate_alpha_sweep_data(alpha_bits, max_k)

    # Get alpha values (they're the same for all formats)
    alpha_values = sweep_data["UQ0.8"]["alpha_values"]

    # Start building the markdown table
    lines = []

    # Table header
    lines.append("| α Value | UQ0.8 | UQ0.12 | UQ0.16 |")
    lines.append("|---------|-------|--------|--------|")

    # Table rows
    for i, alpha in enumerate(alpha_values):
        alpha_str = f"{alpha:.3f}"
        uq8_hist = sweep_data["UQ0.8"]["max_histories"][i]
        uq12_hist = sweep_data["UQ0.12"]["max_histories"][i]
        uq16_hist = sweep_data["UQ0.16"]["max_histories"][i]

        # Format history values (show exact values, no "+" limit)
        uq8_str = f"{uq8_hist}"
        uq12_str = f"{uq12_hist}"
        uq16_str = f"{uq16_hist}"

        lines.append(f"| {alpha_str} | {uq8_str} | {uq12_str} | {uq16_str} |")

    return "\n".join(lines)


def save_alpha_sweep_markdown_table(
    output_path: str = "alpha_sweep_table.md", alpha_bits: int = 4, max_k: int = 3000
):
    """
    Generate and save a markdown table showing alpha sweep results.

    Args:
        output_path: Path to save the markdown file
        alpha_bits: Number of bits to represent alpha
        max_k: Maximum k value to test
    """
    table_content = generate_alpha_sweep_markdown_table(alpha_bits, max_k)

    # Get sweep data for metadata
    sweep_data = generate_alpha_sweep_data(alpha_bits, max_k)

    # Add header and description
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


def plot_coefficient_magnitude(save_path: str = None, max_k: int = 20):
    """
    Create a plot showing coefficient magnitude vs history step k for different alpha values.

    Args:
        save_path: Path to save the plot (if None, displays plot)
        max_k: Maximum k value to plot (default: 20)
    """
    # Alpha values to compare
    # alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    alpha_values = [0.1, 0.5, 0.9]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors and markers for different alpha values (for black/white compatibility)
    line_styles = [
        # {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        {"color": "dodgerblue", "marker": "o", "linestyle": "-"},
        # {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
        {"color": "darkorange", "marker": "s", "linestyle": "--"},
        {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
        # {"color": "#228B22", "marker": "^", "linestyle": "-."},
        {"color": "#d62728", "marker": "D", "linestyle": ":"},
        # {"color": "#DC143C", "marker": "D", "linestyle": ":"},
        # {"color": "#9467bd", "marker": "v", "linestyle": "-"},
        {"color": "#9370DB", "marker": "v", "linestyle": "-"},
        # {"color": "#DAA520", "marker": "v", "linestyle": "-"},
        # {"color": "#3CB371", "marker": "v", "linestyle": "-"},
        # {"color": "#483D8B", "marker": "v", "linestyle": "-"},
    ]

    # Generate k values (starting from 1)
    k_values = list(range(1, max_k + 1))

    # Plot coefficient magnitudes for each alpha
    for i, alpha in enumerate(alpha_values):
        coefficient_magnitudes = []

        for k in k_values:
            # Calculate binomial coefficient
            # Compute the generalized binomial coefficient (alpha choose k)
            binom_coeff = binom(alpha, k)

            # Apply (-1)^k factor to get final weight: (-1)^k * (α choose k)
            sign_factor = -1.0 if (k % 2 == 1) else 1.0
            final_weight = sign_factor * binom_coeff

            # Store magnitude (all final weights are negative for 0 < α < 1)
            weight_magnitude = abs(final_weight)
            coefficient_magnitudes.append(weight_magnitude)

        # Plot the line
        style = line_styles[i]
        ax.plot(
            k_values,
            coefficient_magnitudes,
            label=f"α = {alpha}",
            color=style["color"],
            # marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=4,
            linewidth=2,
            alpha=0.8,
        )

    # Customize the plot
    ax.set_xlabel("History Step (k)", fontsize=12)
    ax.set_ylabel("Coefficient Magnitude", fontsize=12)

    # Set logarithmic scale for y-axis to better show the decay
    ax.set_yscale("log")
    # Set logarithmic scale for x-axis to better space out the k values
    ax.set_xscale("log")

    # Set x-axis to start from 1 and show only integer ticks
    ax.set_xlim(1, max_k)
    ax.set_xticks(range(1, max_k + 1, max(1, max_k // 10)))  # Show ~10 ticks max

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(
        # title="Alpha Value",
        loc="upper right",
        fontsize=12,
        title_fontsize=12,
        framealpha=0.9,
    )

    # Adjust layout
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        # Determine format based on file extension
        if save_path.lower().endswith(".svg"):
            plt.savefig(save_path, format="svg", bbox_inches="tight")
        elif save_path.lower().endswith(".pdf"):
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
        else:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Coefficient magnitude plot saved to: {save_path}")
    else:
        plt.show()


def plot_normalized_coefficient_magnitude(save_path: str = None, max_k: int = 20):
    """
    Create a plot showing normalized coefficient magnitude vs history step k for different alpha values.
    Each sequence is normalized by its k=1 value, so all sequences start at 1.0.

    Args:
        save_path: Path to save the plot (if None, displays plot)
        max_k: Maximum k value to plot (default: 20)
    """
    # Alpha values to compare
    # alpha_values = [0.1, 0.5, 0.9]
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    figsize = get_latex_figsize()

    # Font size configuration
    axis_label_fontsize = 8
    tick_label_fontsize = 6

    # Create the plot
    # fig, ax = plt.subplots(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(figsize["width"], figsize["height"]))

    # Define colors and markers for different alpha values (for black/white compatibility)
    line_styles = [
        # {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        {"color": "dodgerblue", "marker": "o", "linestyle": "-"},
        # {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
        {"color": "darkorange", "marker": "s", "linestyle": "--"},
        {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
        # {"color": "#228B22", "marker": "^", "linestyle": "-."},
        {"color": "#d62728", "marker": "D", "linestyle": ":"},
        # {"color": "#DC143C", "marker": "D", "linestyle": ":"},
        # {"color": "#9467bd", "marker": "v", "linestyle": "-"},
        {"color": "#9370DB", "marker": "v", "linestyle": (0, (1, 1, 2, 1, 3, 1))},
        # {"color": "#DAA520", "marker": "v", "linestyle": "-"},
        # {"color": "#3CB371", "marker": "v", "linestyle": "-"},
        # {"color": "#483D8B", "marker": "v", "linestyle": "-"},
    ]

    # Generate k values (starting from 1)
    k_values = list(range(1, max_k + 1))

    # Plot normalized coefficient magnitudes for each alpha
    for i, alpha in enumerate(alpha_values):
        coefficient_magnitudes = []

        # First, calculate all magnitudes
        for k in k_values:
            # Calculate binomial coefficient
            # Compute the generalized binomial coefficient (alpha choose k)
            binom_coeff = binom(alpha, k)

            # Apply (-1)^k factor to get final weight: (-1)^k * (α choose k)
            sign_factor = -1.0 if (k % 2 == 1) else 1.0
            final_weight = sign_factor * binom_coeff

            # Store magnitude (all final weights are negative for 0 < α < 1)
            weight_magnitude = abs(final_weight)
            coefficient_magnitudes.append(weight_magnitude)

        # Normalize by k=1 value (first element)
        k1_magnitude = coefficient_magnitudes[0]  # This is |α choose 1| = |α|
        normalized_magnitudes = [mag / k1_magnitude for mag in coefficient_magnitudes]

        # Plot the line
        style = line_styles[i]
        ax.plot(
            k_values,
            normalized_magnitudes,
            label=f"α = {alpha}",
            color=style["color"],
            # marker=style["marker"],
            linestyle=style["linestyle"],
            # markersize=4,
            # linewidth=2,
            # alpha=0.8,
        )

    # Customize the plot
    ax.set_xlabel("History Step (k)", fontsize=axis_label_fontsize)
    ax.set_ylabel("Normalized Coefficient Magnitude", fontsize=axis_label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)

    # Set logarithmic scale for y-axis to better show the decay
    ax.set_yscale("log")

    # Set x-axis to start from 1 and show only integer ticks
    ax.set_xlim(1, max_k)
    ax.set_xticks(range(1, max_k + 1, max(1, max_k // 10)))  # Show ~10 ticks max

    # Add grid
    # ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.grid(True, alpha=0.6, linestyle=":")
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(
        # title="Alpha Value",
        loc="upper right",
        fontsize=tick_label_fontsize,
        # title_fontsize=12,
        framealpha=0.9,
    )

    # Add note about normalization
    note_text = "Each sequence normalized by its k=1 value"
    # ax.text(
    #     0.02,
    #     0.02,
    #     note_text,
    #     transform=ax.transAxes,
    #     fontsize=10,
    #     bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    # )

    # Adjust layout
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        # Determine format based on file extension
        if save_path.lower().endswith(".svg"):
            plt.savefig(save_path, format="svg", bbox_inches="tight")
        elif save_path.lower().endswith(".pdf"):
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
        else:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Normalized coefficient magnitude plot saved to: {save_path}")
    else:
        plt.show()


def save_coefficient_magnitude_plot(
    output_path: str = "coefficient_magnitude.svg", max_k: int = 20
):
    """
    Convenience function to generate and save the coefficient magnitude plot.

    Args:
        output_path: Path where to save the plot (default: SVG format)
        max_k: Maximum k value to plot
    """
    plot_coefficient_magnitude(save_path=output_path, max_k=max_k)


def save_normalized_coefficient_magnitude_plot(
    output_path: str = "normalized_coefficient_magnitude.svg", max_k: int = 20
):
    """
    Convenience function to generate and save the normalized coefficient magnitude plot.

    Args:
        output_path: Path where to save the plot (default: SVG format)
        max_k: Maximum k value to plot
    """
    plot_normalized_coefficient_magnitude(save_path=output_path, max_k=max_k)


if __name__ == "__main__":
    # Example usage - generate both plots
    # plot_format_comparison(save_path="images/format_comparison.png")
    plot_alpha_sweep(
        save_path="images/alpha_sweep_32bit_max.svg",
        alpha_bits=4,
        max_k=41_750_000,
    )
    # save_coefficient_magnitude_plot(
    #     output_path="images/coefficient_magnitude_log_1000.svg", max_k=1000
    # )
    # save_normalized_coefficient_magnitude_plot(
    #     output_path="images/normalized_coefficient_magnitude_250.svg", max_k=250
    # )
