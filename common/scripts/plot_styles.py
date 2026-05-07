"""Shared plotting style constants for all visualization scripts.

A single source of truth for:
  - Okabe-Ito colorblind-safe palette
  - Semantic color aliases (VCD signal plots, training metrics)
  - Font size constants
  - LaTeX-compatible figure size helper
  - Marker shapes for multi-series comparison plots

Import example:
    from common.scripts.plot_styles import (
        OKABE_ITO,
        COLOR_MEMBRANE,
        AXIS_LABEL_FONTSIZE,
        TICK_LABEL_FONTSIZE,
        get_latex_figsize,
    )
"""

import numpy as np

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

# Okabe-Ito colorblind-safe palette. Distinguishable under deuteranopia,
# protanopia, and tritanopia. Indexed 0-7.
OKABE_ITO = [
    "#0072B2",  # 0 — blue
    "#D55E00",  # 1 — vermillion
    "#009E73",  # 2 — bluish green
    "#CC79A7",  # 3 — reddish purple
    "#56B4E9",  # 4 — sky blue
    "#E69F00",  # 5 — orange
    "#F0E442",  # 6 — yellow
    "#000000",  # 7 — black
]

# Neutral gray for raw/background data points
COLOR_RAW = "#999999"

# ---------------------------------------------------------------------------
# Semantic aliases — VCD signal plots
# (used by plot_membrane_potential.py and plot_spike_cycles.py)
# ---------------------------------------------------------------------------

COLOR_MEMBRANE = OKABE_ITO[0]   # blue
COLOR_CURRENT = OKABE_ITO[5]    # orange
COLOR_SPIKE = OKABE_ITO[1]      # vermillion
COLOR_ISI = OKABE_ITO[2]        # bluish green
COLOR_MA = OKABE_ITO[7]         # black (moving average)

PHASE_COLORS = {
    "startup": COLOR_RAW,
    "constant": OKABE_ITO[2],   # bluish green
    "dropout": OKABE_ITO[1],    # vermillion
    "recovery": OKABE_ITO[2],   # bluish green
}

# ---------------------------------------------------------------------------
# Semantic aliases — training metrics plots
# (used by visualize_training_metrics.py)
# ---------------------------------------------------------------------------

COLOR_TRAIN = OKABE_ITO[0]   # running average line
COLOR_GEN = OKABE_ITO[2]     # generalization line
COLOR_BEST = OKABE_ITO[1]    # best-so-far reference lines
COLOR_SAVE = OKABE_ITO[5]    # checkpoint save-event markers

# ---------------------------------------------------------------------------
# Font sizes
# Sized for ~7" wide figures that scale down cleanly in LaTeX.
# ---------------------------------------------------------------------------

AXIS_LABEL_FONTSIZE: int = 12
TICK_LABEL_FONTSIZE: int = 10
LEGEND_FONTSIZE: int = 8

# ---------------------------------------------------------------------------
# Marker shapes — paired with Okabe-Ito colors so series are distinguishable
# in both color and shape (important for black-and-white printing).
# ---------------------------------------------------------------------------

COMPARISON_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]

# ---------------------------------------------------------------------------
# Figure sizing
# ---------------------------------------------------------------------------

# Default figure size: 7" wide, golden-ratio height (~4.3").
# Large enough for detail; scales down cleanly to LaTeX column width.
DEFAULT_FIGSIZE: tuple[float, float] = (7.0, 4.3)


def get_latex_figsize(width_scale: float = 1.0, height_scale: float | None = None) -> dict:
    """Return figure dimensions (inches) sized for a LaTeX document.

    Assumes a text column width of 117 mm (standard single-column IEEE /
    conference layout). Adjust ``width_scale`` for multi-column figures.

    Args:
        width_scale: Fraction of the text width (1.0 = full column).
        height_scale: If given, height = doc_textwidth_in * height_scale.
                      If None, height is set by the golden ratio.

    Returns:
        ``{"width": float, "height": float}`` in inches.
    """
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
