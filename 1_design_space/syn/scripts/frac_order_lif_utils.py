"""
Utility functions for fractional-order LIF neuron parameter calculation.
Generates fixed-point constants and coefficient memory files for synthesis.
"""

from typing import List


def calculate_frac_lif_parameters(
    alpha: float,
    tau_scaled: float,
    h_scaled: float,
    tau_width: int = 16,
    norm_factor_width: int = 8,
):
    """
    Calculate fractional-order LIF parameters from physical values.

    Args:
        alpha (float): Fractional order parameter (0 < alpha < 1)
        tau_scaled (float): Time constant in ms (scaled units)
        h_scaled (float): Time step in ms (scaled units)
        tau_width (int): Bit width for tau fixed-point representation (default: 16)
        norm_factor_width (int): Bit width for norm_factor fixed-point representation (default: 8)

    Returns:
        dict: Dictionary containing:
            - tau_over_h_alpha: tau / h^alpha ratio
            - norm_factor: Normalization factor 1/(1 + tau/h^alpha)
            - tau_over_h_alpha_fixed: Fixed-point value (Q(tau_width/2).(tau_width/2) format)
            - norm_factor_fixed: Fixed-point value (Q0.norm_factor_width format)
    """
    # Calculate fractional powers
    h_alpha = h_scaled**alpha
    tau_over_h_alpha = tau_scaled / h_alpha

    # Calculate normalization factor
    norm_factor = 1.0 / (1.0 + tau_over_h_alpha)

    # Convert to fixed-point representations
    # TAU_OVER_H_ALPHA in Q(tau_width/2).(tau_width/2) format
    tau_fractional_bits = tau_width // 2
    tau_scale_factor = 2**tau_fractional_bits
    tau_over_h_alpha_scaled = tau_over_h_alpha * tau_scale_factor
    tau_over_h_alpha_fixed = int(tau_over_h_alpha_scaled)
    tau_max_value = (2**tau_width) - 1
    if tau_over_h_alpha_fixed > tau_max_value:
        tau_over_h_alpha_fixed = tau_max_value

    # NORM_FACTOR in Q0.norm_factor_width format
    norm_factor_scale_factor = 2**norm_factor_width
    norm_factor_scaled = norm_factor * norm_factor_scale_factor
    norm_factor_fixed = int(norm_factor_scaled)
    norm_factor_max_value = (2**norm_factor_width) - 1
    if norm_factor_fixed > norm_factor_max_value:
        norm_factor_fixed = norm_factor_max_value

    return {
        "tau_over_h_alpha": tau_over_h_alpha,
        "norm_factor": norm_factor,
        "tau_over_h_alpha_fixed": tau_over_h_alpha_fixed,
        "norm_factor_fixed": norm_factor_fixed,
    }


def calculate_binomial_coefficient(alpha: float, k: int) -> float:
    """
    Calculate generalized binomial coefficient (alpha choose k).

    Args:
        alpha (float): Fractional order parameter
        k (int): Index value

    Returns:
        float: Binomial coefficient value
    """
    # TODO: We may wish to use scipy.special.binom instead. For now, we'll keep
    # this more literal translation of the original SV implementation.
    if k == 0:
        return 1.0

    result = 1.0
    # Calculate: alpha * (alpha-1) * ... * (alpha-k+1) / k!
    # Interleave multiplication and division to prevent overflow
    for i in range(k):
        result *= (alpha - i) / (i + 1)

    return result


def binomal_coefficients_scaled(
    alpha: float, history_size: int, coeff_width: int
) -> List[int]:
    """
    Calculate all binomial coefficients for k=1 to history_size.

    Args:
        alpha (float): Fractional order parameter
        history_size (int): Number of history terms to generate
        coeff_width (int): Bit width for coefficients (typically 8)

    Returns:
        List[float]: List of binomial coefficients
    """
    coefficients = []
    max_value = (2**coeff_width) - 1

    # Generate coefficients for k=1 to history_size
    for k in range(1, history_size + 1):
        # Calculate binomial coefficient
        binom_coeff = calculate_binomial_coefficient(alpha, k)

        # Apply (-1)^k factor: (-1)^k * (alpha choose k)
        sign_factor = -1.0 if (k % 2 == 1) else 1.0
        final_weight = sign_factor * binom_coeff

        # For 0 < alpha < 1, all final weights are negative, so store magnitude
        weight_mag = abs(final_weight)

        # Convert to fixed-point (UQ0.coeff_width format)
        scaled = weight_mag * (2.0**coeff_width)

        # Clamp to representable range
        if scaled > max_value:
            scaled = max_value
        elif scaled < 0:
            scaled = 0

        coeff_fixed = int(scaled)
        coefficients.append(coeff_fixed)

    return coefficients


def generate_coefficient_mem_file(
    alpha: float,
    coeff_width: int,
    history_size: int,
    output_file: str = "src/coefficients.mem",
):
    """
    Generate a .mem file containing fractional-order LIF coefficients.

    Args:
        alpha (float): Fractional order parameter (0 < alpha < 1)
        coeff_width (int): Bit width for coefficients (typically 8)
        history_size (int): Number of history terms to generate
        output_file (str): Output filename for .mem file

    Returns:
        list: List of coefficient values (for verification)
    """
    coefficients = binomal_coefficients_scaled(alpha, history_size, coeff_width)

    # Write .mem file
    with open(output_file, "w") as f:
        f.write("// Fractional-order LIF coefficients\n")
        f.write(f"// Alpha = {alpha:.6f}\n")
        f.write(f"// Coefficient width = {coeff_width} bits\n")
        f.write(f"// History size = {history_size}\n")
        f.write(f"// Format: UQ0.{coeff_width}\n")
        f.write("//\n")
        f.write("// Coefficients represent magnitudes of (-1)^k * (alpha choose k)\n")
        f.write("// All weights are negative for 0 < alpha < 1\n")

        for i, coeff in enumerate(coefficients):
            f.write(f"{coeff:0{(coeff_width + 3) // 4}X}\n")  # Hex format, padded

    print(f"Generated coefficient file: {output_file}")
    print(f"Coefficients (decimal): {coefficients}")
    print(f"Alpha = {alpha:.6f}, History size = {history_size}")

    return coefficients


def print_synthesis_constants(
    alpha: float,
    tau_scaled: float,
    h_scaled: float,
    coeff_width,
    history_size,
):
    """
    Print SystemVerilog constants for synthesis (for reference/debugging).

    Args:
        alpha (float): Fractional order parameter
        tau_scaled (float): Time constant in ms
        h_scaled (float): Time step in ms
        coeff_width (int): Coefficient bit width
        history_size (int): History buffer size
    """
    # Calculate parameters
    params = calculate_frac_lif_parameters(alpha, tau_scaled, h_scaled)
    coefficients = binomal_coefficients_scaled(alpha, history_size, coeff_width)

    print(f"\n// SystemVerilog constants for alpha = {alpha:.6f}")
    print(
        f"localparam [15:0] TAU_OVER_H_ALPHA = 16'd{params['tau_over_h_alpha_fixed']};"
    )
    print(f"localparam [7:0] NORM_FACTOR = 8'd{params['norm_factor_fixed']};")
    print("\n// Coefficient constants:")
    for i, coeff in enumerate(coefficients):
        print(f"localparam [7:0] COEFF_{i+1} = 8'd{coeff};")


if __name__ == "__main__":
    # Example usage
    alpha = 5.0 / 15.0  # Default alpha
    tau_scaled = 25.0  # in ms
    h_scaled = 1.0  # 1 ms
    # NOTE: r_scaled is not directly used, but this value represents a reasonable
    # default given the units used. This is the same set of values used by default
    # for the lapicque neuron module.
    r_scaled = 9.0  # M-ohms
    coeff_width = 16
    history_size = 256

    print("Fractional-order LIF Parameter Calculator")
    print("=========================================")

    # Calculate and display parameters
    params = calculate_frac_lif_parameters(alpha, tau_scaled, h_scaled)
    print("\nPhysical parameters:")
    print(f"  Alpha: {alpha:.6f}")
    print(f"  Tau (scaled): {tau_scaled} ms")
    print(f"  H (scaled): {h_scaled} ms")
    print(f"  R (scaled): {r_scaled} M-ohms")

    print("\nCalculated values:")
    print(f"  tau / h^alpha: {params['tau_over_h_alpha']:.6f}")
    print(f"  Normalization factor: {params['norm_factor']:.6f}")

    print("\nFixed-point representations:")
    print(f"  TAU_OVER_H_ALPHA (Q8.8): {params['tau_over_h_alpha_fixed']}")
    print(f"  NORM_FACTOR (Q0.8): {params['norm_factor_fixed']}")

    # Generate coefficient file
    print("\nGenerating coefficient file...")
    coefficients = generate_coefficient_mem_file(
        alpha, coeff_width, history_size, "src/coefficients.mem"
    )

    # Print synthesis constants
    print_synthesis_constants(alpha, tau_scaled, h_scaled, coeff_width, history_size)
