"""
Reference implementation for generalized binomial coefficients.
Uses scipy.special for mathematical accuracy.
"""

from scipy.special import binom
from typing import List


def generalized_binomial_coeff(alpha: float, k: int) -> float:
    """
    Calculate generalized binomial coefficient (alpha choose k).

    For fractional alpha, uses the generalized formula:
    (alpha choose k) = alpha * (alpha-1) * ... * (alpha-k+1) / k!

    Args:
        alpha: Fractional order parameter (0.0 to 1.0 typically)
        k: Integer index (k >= 0)

    Returns:
        Binomial coefficient value
    """
    return binom(alpha, k)


def fixed_point_to_float(fixed_val: int, frac_bits: int = 8) -> float:
    """Convert fixed-point integer to floating point."""
    return float(fixed_val) / (2 ** frac_bits)


def float_to_fixed_point(float_val: float, frac_bits: int = 8, width: int = 8) -> int:
    """Convert floating point to unsigned fixed-point integer."""
    scaled = float_val * (2 ** frac_bits)
    max_val = 2 ** width - 1
    min_val = 0

    # Clamp to representable range
    clamped = max(min_val, min(max_val, scaled))
    return int(clamped)


def alpha_4bit_to_float(alpha_4bit: int) -> float:
    """Convert 4-bit alpha parameter to float (1-14 maps to 1/15 - 14/15)."""
    if alpha_4bit < 1 or alpha_4bit > 14:
        raise ValueError(f"Alpha 4-bit value must be 1-14, got {alpha_4bit}")
    return float(alpha_4bit) / 15.0


def decode_lut_coefficient(raw_coeff: int, k: int) -> float:
    """
    Decode coefficient from LUT based on k value and storage format.

    Args:
        raw_coeff: Raw unsigned coefficient value from LUT
        k: Index value (determines format)

    Returns:
        Actual coefficient value as float (unsigned magnitude for k≥1)
    """
    if k == 0:
        # k=0: Stored directly as UQ1.7 format (1 integer bit + 7 fractional bits)
        # 128 = 1.0, so coefficient is raw_coeff / 128.0
        return fixed_point_to_float(raw_coeff, frac_bits=7)  # UQ1.7 format
    else:
        # k≥1: Stored as unsigned magnitude in UQ0.8 format
        # Return unsigned magnitude (parent applies signs as needed)
        return fixed_point_to_float(raw_coeff, frac_bits=8)  # UQ0.8 format


def calculate_reference_coeffs(alpha: float, max_k: int) -> List[float]:
    """Calculate reference coefficients for k=0 to max_k."""
    return [generalized_binomial_coeff(alpha, k) for k in range(max_k + 1)]


def calculate_tolerance(coeff_width: int, k: int = 0) -> float:
    """
    Calculate reasonable tolerance for fixed-point comparison.

    Args:
        coeff_width: Bit width of coefficient storage (8-bit)
        k: Index value (determines format: k=0 uses UQ1.7, k≥1 uses UQ0.8)

    Returns:
        Tolerance value for comparison
    """
    if k == 0:
        # k=0: UQ1.7 format (7 fractional bits)
        frac_bits = 7
    else:
        # k≥1: UQ0.8 format (8 fractional bits)
        frac_bits = 8

    lsb = 1.0 / (2 ** frac_bits)  # Least significant bit value
    quantization_error = 0.5 * lsb  # ±0.5 LSB quantization
    calculation_error = 1e-6  # Small calculation precision buffer

    return quantization_error + calculation_error


# Test data sets for 4-bit alpha values (1-14 mapping to 1/15 - 14/15)
COMMON_ALPHA_VALUES = [
    (1/15, 1),    # α ≈ 0.067
    (2/15, 2),    # α ≈ 0.133
    (3/15, 3),    # α = 0.2
    (4/15, 4),    # α ≈ 0.267
    (8/15, 8),    # α ≈ 0.533
    (12/15, 12),  # α = 0.8
    (14/15, 14),  # α ≈ 0.933
]

# Expected coefficient patterns for validation
EXPECTED_PATTERNS = {
    0.5: {
        # For alpha=0.5, coefficients should decrease in magnitude
        # and alternate signs when used with (-1)^k
        "decreasing": True,
        "k1_negative": True,  # When multiplied by (-1)^1
    }
}
