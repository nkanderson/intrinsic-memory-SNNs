#!/usr/bin/env python3
"""
Resource Usage Comparison Tool for Fractional-Order LIF Neuron

This script provides comprehensive visualization tools to compare resource usage
between the standard Lapicque LIF neuron and the fractional-order LIF neuron
implementations after FPGA synthesis.

Author: Generated for Teuscher Lab
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, List
import os
import re

# Global path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# Default statistics file paths
LAPICQUE_STATS_PATH = os.path.join(DATA_DIR, "lapicque_stats.txt")
FRAC_LIF_STATS_PATH = os.path.join(DATA_DIR, "frac_order_lif_stats.txt")


def parse_stats_file(file_path: str) -> Dict:
    """
    Parse synthesis statistics from Yosys output file.

    Args:
        file_path: Path to the statistics file

    Returns:
        Dictionary containing parsed statistics for both logic-level and cell-level
    """
    try:
        with open(file_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Warning: Stats file not found: {file_path}")
        return {}

    stats = {"logic_level": {}, "cell_level": {}, "module_name": ""}

    # Split into logic-level and cell-level sections
    sections = content.split("===============================================")

    if len(sections) >= 2:
        logic_section = sections[0]

        # Parse logic-level statistics
        stats["logic_level"] = _parse_stats_section(logic_section)

        # Parse cell-level statistics - look for the post-synthesis section
        # The post-synthesis section is usually after the delimiter, so check all sections
        for i, section in enumerate(sections):
            if "Post-synthesis" in section:
                # Get the full section including the next part which contains the actual stats
                if i + 1 < len(sections):
                    full_section = (
                        section
                        + "==============================================="
                        + sections[i + 1]
                    )
                    stats["cell_level"] = _parse_stats_section(full_section)
                else:
                    stats["cell_level"] = _parse_stats_section(section)
                break

    return stats


def _parse_stats_section(section: str) -> Dict:
    """Helper function to parse a statistics section."""
    stats = {}

    # Extract module name
    module_match = re.search(r"=== (\w+) ===", section)
    if module_match:
        stats["module_name"] = module_match.group(1)

    # Parse basic statistics
    patterns = {
        "wires": r"Number of wires:\s+(\d+)",
        "wire_bits": r"Number of wire bits:\s+(\d+)",
        "ports": r"Number of ports:\s+(\d+)",
        "port_bits": r"Number of port bits:\s+(\d+)",
        "memories": r"Number of memories:\s+(\d+)",
        "memory_bits": r"Number of memory bits:\s+(\d+)",
        "processes": r"Number of processes:\s+(\d+)",
        "cells": r"Number of cells:\s+(\d+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, section)
        if match:
            stats[key] = int(match.group(1))

    # Parse cell breakdown - fixed regex to handle the actual format
    cell_breakdown = {}
    lines = section.split("\n")

    for line in lines:
        # Original line with leading spaces and cell type followed by count
        # Match "     $add                            1" format
        if line.strip():
            match = re.search(r"^\s+(\$?\w+)\s+(\d+)\s*$", line)
            if match:
                cell_type = match.group(1)
                count = int(match.group(2))
                cell_breakdown[cell_type] = count

    stats["cell_breakdown"] = cell_breakdown

    return stats


def plot_resource_comparison(
    lapicque_stats_path: str = None,
    frac_lif_stats_path: str = None,
    save_path: str = None,
):
    """
    Create comprehensive resource usage comparison plots.

    Args:
        lapicque_stats_path: Path to lapicque statistics file
        frac_lif_stats_path: Path to frac_order_lif statistics file
        save_path: Optional path to save the plot
    """
    # Default paths if not provided
    if lapicque_stats_path is None:
        lapicque_stats_path = LAPICQUE_STATS_PATH
    if frac_lif_stats_path is None:
        frac_lif_stats_path = FRAC_LIF_STATS_PATH

    # Parse statistics files
    lapicque_stats = parse_stats_file(lapicque_stats_path)
    frac_lif_stats = parse_stats_file(frac_lif_stats_path)

    if not lapicque_stats or not frac_lif_stats:
        print("Error: Could not parse statistics files")
        return

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Resource Usage Comparison: Lapicque vs Fractional-Order LIF",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Basic Resource Comparison (Logic Level)
    _plot_basic_resources(
        ax1,
        lapicque_stats["logic_level"],
        frac_lif_stats["logic_level"],
        "Logic-Level Resources",
    )

    # Plot 2: Cell Breakdown (Logic Level)
    _plot_cell_breakdown(
        ax2,
        lapicque_stats["logic_level"],
        frac_lif_stats["logic_level"],
        "Logic-Level Cell Types",
    )

    # Plot 3: Basic Resource Comparison (Cell Level)
    _plot_basic_resources(
        ax3,
        lapicque_stats["cell_level"],
        frac_lif_stats["cell_level"],
        "FPGA Cell Resources",
    )

    # Plot 4: FPGA Cell Breakdown
    _plot_fpga_cells(ax4, lapicque_stats["cell_level"], frac_lif_stats["cell_level"])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Resource comparison plot saved to: {save_path}")
    else:
        plt.show()


def _plot_basic_resources(ax, lapicque_stats, frac_lif_stats, title):
    """Plot basic resource comparison (wires, bits, ports, etc.)"""
    resources = [
        "wires",
        "wire_bits",
        "ports",
        "port_bits",
        "memories",
        "memory_bits",
        "cells",
    ]
    lapicque_values = [lapicque_stats.get(res, 0) for res in resources]
    frac_lif_values = [frac_lif_stats.get(res, 0) for res in resources]

    x = np.arange(len(resources))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        lapicque_values,
        width,
        label="Lapicque",
        color="skyblue",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        frac_lif_values,
        width,
        label="Fractional-Order LIF",
        color="lightcoral",
        alpha=0.8,
    )

    ax.set_xlabel("Resource Type")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [r.replace("_", "\n") for r in resources], rotation=45, ha="right"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    _add_bar_labels(ax, bars1)
    _add_bar_labels(ax, bars2)


def _plot_cell_breakdown(ax, lapicque_stats, frac_lif_stats, title):
    """Plot cell type breakdown comparison"""
    lap_cells = lapicque_stats.get("cell_breakdown", {})
    frac_cells = frac_lif_stats.get("cell_breakdown", {})

    # Get all unique cell types
    all_cells = set(lap_cells.keys()) | set(frac_cells.keys())
    all_cells = sorted(all_cells)

    lapicque_values = [lap_cells.get(cell, 0) for cell in all_cells]
    frac_lif_values = [frac_cells.get(cell, 0) for cell in all_cells]

    x = np.arange(len(all_cells))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        lapicque_values,
        width,
        label="Lapicque",
        color="skyblue",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        frac_lif_values,
        width,
        label="Fractional-Order LIF",
        color="lightcoral",
        alpha=0.8,
    )

    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(all_cells, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    _add_bar_labels(ax, bars1)
    _add_bar_labels(ax, bars2)


def _plot_fpga_cells(ax, lapicque_stats, frac_lif_stats):
    """Plot FPGA-specific cell comparison"""
    lap_cells = lapicque_stats.get("cell_breakdown", {})
    frac_cells = frac_lif_stats.get("cell_breakdown", {})

    # Focus on FPGA cells (SB_ prefix for iCE40)
    fpga_cells = ["SB_LUT4", "SB_CARRY", "SB_DFFESR", "SB_DFFER", "SB_MAC16"]

    lapicque_values = [lap_cells.get(cell, 0) for cell in fpga_cells]
    frac_lif_values = [frac_cells.get(cell, 0) for cell in fpga_cells]

    x = np.arange(len(fpga_cells))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        lapicque_values,
        width,
        label="Lapicque",
        color="skyblue",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        frac_lif_values,
        width,
        label="Fractional-Order LIF",
        color="lightcoral",
        alpha=0.8,
    )

    ax.set_xlabel("FPGA Cell Type")
    ax.set_ylabel("Count")
    ax.set_title("FPGA Resource Usage")
    ax.set_xticks(x)
    ax.set_xticklabels(fpga_cells, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels
    _add_bar_labels(ax, bars1)
    _add_bar_labels(ax, bars2)


def _add_bar_labels(ax, bars):
    """Add value labels on top of bars"""
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def plot_resource_efficiency_summary(
    lapicque_stats_path: str = None,
    frac_lif_stats_path: str = None,
    save_path: str = None,
):
    """
    Create a summary plot showing resource efficiency metrics.
    """
    # Default paths if not provided
    if lapicque_stats_path is None:
        lapicque_stats_path = LAPICQUE_STATS_PATH
    if frac_lif_stats_path is None:
        frac_lif_stats_path = FRAC_LIF_STATS_PATH

    # Parse statistics files
    lapicque_stats = parse_stats_file(lapicque_stats_path)
    frac_lif_stats = parse_stats_file(frac_lif_stats_path)

    if not lapicque_stats or not frac_lif_stats:
        print("Error: Could not parse statistics files")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Resource Efficiency Analysis", fontsize=16, fontweight="bold")

    # Calculate key metrics
    metrics = _calculate_efficiency_metrics(lapicque_stats, frac_lif_stats)

    # Plot 1: Resource multipliers
    _plot_resource_multipliers(ax1, metrics)

    # Plot 2: Summary table
    _plot_summary_table(ax2, lapicque_stats, frac_lif_stats, metrics)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Resource efficiency plot saved to: {save_path}")
    else:
        plt.show()


def _calculate_efficiency_metrics(lapicque_stats, frac_lif_stats):
    """Calculate efficiency metrics comparing the two implementations"""
    metrics = {}

    # Cell-level comparison (most important for FPGA)
    lap_cell = lapicque_stats.get("cell_level", {})
    frac_cell = frac_lif_stats.get("cell_level", {})

    key_resources = ["cells", "wire_bits"]

    for resource in key_resources:
        lap_val = lap_cell.get(resource, 1)
        frac_val = frac_cell.get(resource, 1)
        metrics[f"{resource}_multiplier"] = (
            frac_val / lap_val if lap_val > 0 else float("inf")
        )

    # FPGA cell multipliers
    lap_cells = lap_cell.get("cell_breakdown", {})
    frac_cells = frac_cell.get("cell_breakdown", {})

    fpga_cells = ["SB_LUT4", "SB_CARRY", "SB_DFFESR", "SB_DFFER"]
    for cell in fpga_cells:
        lap_val = lap_cells.get(cell, 0)
        frac_val = frac_cells.get(cell, 0)
        if lap_val > 0:
            metrics[f"{cell}_multiplier"] = frac_val / lap_val
        else:
            metrics[f"{cell}_multiplier"] = float("inf") if frac_val > 0 else 1.0

    return metrics


def _plot_resource_multipliers(ax, metrics):
    """Plot resource multiplier comparison"""
    # Filter out infinite values and focus on key metrics
    key_metrics = {
        "Total Cells": metrics.get("cells_multiplier", 1),
        "Wire Bits": metrics.get("wire_bits_multiplier", 1),
        "LUT4s": metrics.get("SB_LUT4_multiplier", 1),
        "Carry Chains": metrics.get("SB_CARRY_multiplier", 1),
        "Flip-Flops": max(
            metrics.get("SB_DFFESR_multiplier", 1),
            metrics.get("SB_DFFER_multiplier", 1),
        ),
    }

    # Filter out infinite values
    filtered_metrics = {k: v for k, v in key_metrics.items() if v != float("inf")}

    names = list(filtered_metrics.keys())
    values = list(filtered_metrics.values())

    colors = [
        "red" if v > 10 else "orange" if v > 5 else "yellow" if v > 2 else "lightgreen"
        for v in values
    ]

    bars = ax.bar(names, values, color=colors, alpha=0.7)
    ax.set_ylabel("Resource Multiplier (Fractional/Lapicque)")
    ax.set_title("Resource Usage Multipliers")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1f}x",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add horizontal reference lines
    ax.axhline(y=1, color="green", linestyle="--", alpha=0.7, label="Same as Lapicque")
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.7, label="10x More Resources")
    ax.legend()


def _plot_summary_table(ax, lapicque_stats, frac_lif_stats, metrics):
    """Plot summary comparison table"""
    ax.axis("off")

    # Extract key data for table
    lap_cell = lapicque_stats.get("cell_level", {})
    frac_cell = frac_lif_stats.get("cell_level", {})

    # Prepare table data
    table_data = [
        ["Resource", "Lapicque", "Fractional-Order", "Multiplier"],
        [
            "Total Cells",
            f"{lap_cell.get('cells', 0)}",
            f"{frac_cell.get('cells', 0)}",
            f"{metrics.get('cells_multiplier', 0):.1f}x",
        ],
        [
            "Wire Bits",
            f"{lap_cell.get('wire_bits', 0)}",
            f"{frac_cell.get('wire_bits', 0)}",
            f"{metrics.get('wire_bits_multiplier', 0):.1f}x",
        ],
        [
            "LUT4s",
            f"{lap_cell.get('cell_breakdown', {}).get('SB_LUT4', 0)}",
            f"{frac_cell.get('cell_breakdown', {}).get('SB_LUT4', 0)}",
            f"{metrics.get('SB_LUT4_multiplier', 0):.1f}x",
        ],
        [
            "Carry Chains",
            f"{lap_cell.get('cell_breakdown', {}).get('SB_CARRY', 0)}",
            f"{frac_cell.get('cell_breakdown', {}).get('SB_CARRY', 0)}",
            f"{metrics.get('SB_CARRY_multiplier', 0):.1f}x",
        ],
        [
            "MAC16",
            f"{lap_cell.get('cell_breakdown', {}).get('SB_MAC16', 0)}",
            f"{frac_cell.get('cell_breakdown', {}).get('SB_MAC16', 0)}",
            "N/A",
        ],
    ]

    # Create table
    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.2, 0.3, 0.25],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code the multiplier column
    for i in range(1, len(table_data)):
        if i < len(table_data) - 1:  # Skip MAC16 row
            mult_str = table_data[i][3]
            if "x" in mult_str:
                mult_val = float(mult_str.replace("x", ""))
                if mult_val > 10:
                    color = "lightcoral"
                elif mult_val > 5:
                    color = "orange"
                elif mult_val > 2:
                    color = "yellow"
                else:
                    color = "lightgreen"
                table[(i, 3)].set_facecolor(color)

    ax.set_title("Resource Usage Summary", fontsize=12, fontweight="bold")


def generate_resource_report(
    lapicque_stats_path: str = None,
    frac_lif_stats_path: str = None,
    output_dir: str = None,
):
    """
    Generate comprehensive resource comparison report with multiple visualizations.

    Args:
        lapicque_stats_path: Path to lapicque statistics file
        frac_lif_stats_path: Path to frac_order_lif statistics file
        output_dir: Directory to save output images
    """
    # Use global paths if not provided
    if lapicque_stats_path is None:
        lapicque_stats_path = LAPICQUE_STATS_PATH
    if frac_lif_stats_path is None:
        frac_lif_stats_path = FRAC_LIF_STATS_PATH
    if output_dir is None:
        output_dir = IMAGES_DIR

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("Generating resource usage comparison report...")

    # Generate main comparison plot
    print("1. Creating detailed resource comparison...")
    plot_resource_comparison(
        lapicque_stats_path=lapicque_stats_path,
        frac_lif_stats_path=frac_lif_stats_path,
        save_path=os.path.join(output_dir, "resource_comparison.png"),
    )

    # Generate efficiency summary
    print("2. Creating efficiency summary...")
    plot_resource_efficiency_summary(
        lapicque_stats_path=lapicque_stats_path,
        frac_lif_stats_path=frac_lif_stats_path,
        save_path=os.path.join(output_dir, "resource_efficiency.png"),
    )

    print(f"\nResource comparison report generated in: {output_dir}")
    print("Generated files:")
    print("  - resource_comparison.png: Detailed 4-panel comparison")
    print("  - resource_efficiency.png: Summary metrics and table")


if __name__ == "__main__":
    # Test parsing functionality with clean output
    print("=== PARSING TEST ===")

    print("\nLapicque Stats:")
    lapicque_stats = parse_stats_file(LAPICQUE_STATS_PATH)
    print(
        f"  Logic cells: {len(lapicque_stats.get('logic_level', {}).get('cell_breakdown', {}))}"
    )
    print(
        f"  Logic breakdown: {list(lapicque_stats.get('logic_level', {}).get('cell_breakdown', {}).keys())}"
    )
    print(
        f"  FPGA cells: {len(lapicque_stats.get('cell_level', {}).get('cell_breakdown', {}))}"
    )
    print(
        f"  FPGA breakdown: {list(lapicque_stats.get('cell_level', {}).get('cell_breakdown', {}).keys())}"
    )

    print("\nFractional Order Stats:")
    frac_stats = parse_stats_file(FRAC_LIF_STATS_PATH)
    print(
        f"  Logic cells: {len(frac_stats.get('logic_level', {}).get('cell_breakdown', {}))}"
    )
    print(
        f"  Logic breakdown: {list(frac_stats.get('logic_level', {}).get('cell_breakdown', {}).keys())}"
    )
    print(
        f"  FPGA cells: {len(frac_stats.get('cell_level', {}).get('cell_breakdown', {}))}"
    )
    print(
        f"  FPGA breakdown: {list(frac_stats.get('cell_level', {}).get('cell_breakdown', {}).keys())}"
    )

    # Generate the complete resource comparison report
    print("\n=== GENERATING PLOTS ===")
    generate_resource_report()
