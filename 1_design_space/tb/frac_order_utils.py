"""
Utility functions for fractional-order LIF neuron testing.
Contains standalone functions that mirror the SystemVerilog implementation
for testing and verification purposes.
"""

from utils import to_fixed_point, from_fixed_point


def calculate_updated_potential(
    current_input,
    fractional_sum,
    r_scaled,
    tau_over_h_alpha,
    norm_factor
):
    """
    Calculate updated membrane potential using fractional-order LIF model.

    This function mirrors the combinational logic for these calculations in frac_order_lif.sv.

    Args:
        current_input (int): Input current (8-bit, represents 10pA per unit)
        fractional_sum (int): Sum of fractional derivative terms (32-bit, Q24.8 format)
        r_scaled (int): Resistance scaling factor (8-bit, default 100)
        tau_over_h_alpha (int): Fractional parameter (16-bit, Q8.8 format)
        norm_factor (int): Normalization factor (8-bit, Q0.8 format)
        
    Returns:
        int: Updated membrane potential (16-bit, Q16.0 format, will be truncated to 8-bit)
        
    Implementation Details:
        - current_term = R_SCALED * current (16-bit result)
        - history_term = TAU_OVER_H_ALPHA * fractional_sum (Q16.16 format)
        - unnormalized_potential = (current_term << 16) + history_term (Q16.16 format)
        - updated_potential = (unnormalized_potential * NORM_FACTOR) >> 24 (Q16.0 format)
    """
    
    # Current term: R * I[n]
    current_term = r_scaled * current_input
    
    # History term: (tau/h^alpha) * fractional_sum
    # TAU_OVER_H_ALPHA (Q8.8) * fractional_sum (Q24.8) = Q32.16
    temp_history_mult = tau_over_h_alpha * fractional_sum
    
    # Check for saturation in history term (48-bit to 32-bit)
    if temp_history_mult > 0xFFFFFFFF:
        history_term = 0xFFFFFFFF  # Saturate to max Q16.16
    else:
        history_term = temp_history_mult & 0xFFFFFFFF  # Extract lower 32 bits
    
    # Combined: R*I[n] + (tau/h^alpha) * fractional_sum
    # We add because the fractional sum represents negative contributions
    # Check for overflow when combining current_term and history_term
    if current_term == 0xFFFF and history_term > 0x0000FFFF:
        # Would cause overflow, saturate the addition
        unnormalized_potential = 0xFFFFFFFF
    else:
        # Shift current_term to match Q16.16 format for addition with history_term
        unnormalized_potential = (current_term << 16) + history_term
    
    # Normalize: multiply by 1/(1+(tau/h^alpha))
    # NORM_FACTOR is in Q0.8 format, 32×8 = 40-bit result
    # temp_result is Q16.24
    temp_result = unnormalized_potential * norm_factor
    
    # Take only the upper bits for Q16.0 format (effectively right shift by 24)
    updated_potential = (temp_result >> 24) & 0xFFFF
    
    return updated_potential


def calculate_fractional_sum(coefficients, history_values):
    """
    Calculate the fractional derivative sum from coefficients and history values.

    This mirrors the parallel multiplication and summation logic from frac_order_lif.sv.

    Args:
        coefficients (list): List of binomial coefficient magnitudes (8-bit each, UQ0.8 format)
        history_values (list): List of history membrane potential values (8-bit each)
        
    Returns:
        int: Fractional sum (32-bit, Q24.8 format)
        
    Implementation Details:
        - Each product: coefficients[k] * history_values[k] = Q8.8 format (16-bit)
        - Sum all products with zero-extension: Q24.8 format (32-bit)
    """
    fractional_sum = 0
    
    for coeff, hist_val in zip(coefficients, history_values):
        # Calculate product: coefficient (UQ0.8) * history_value (Q8.0) = Q8.8
        product = coeff * hist_val  # 16-bit result
        
        # Add to sum with zero-extension (maintains Q8.8 -> Q24.8 format)
        fractional_sum += product
    
    return fractional_sum & 0xFFFFFFFF  # Ensure 32-bit result


def verify_fixed_point_calculation():
    """
    Verify the fixed-point calculations with known test values.
    This can be used as a unit test to ensure the calculations are correct.
    """
    # Test parameters (using default values from frac_order_lif.sv)
    r_scaled = 100
    tau_over_h_alpha = 512  # 2.0 * 256 = 512 (new default)
    norm_factor = 85  # Approximately 1/(1+2.0) * 256 ≈ 85
    
    # Test case 1: Zero input, zero history
    current_input = 0
    fractional_sum = 0
    result = calculate_updated_potential(
        current_input, fractional_sum, r_scaled, tau_over_h_alpha, norm_factor
    )
    assert result == 0, f"Expected 0 for zero input, got {result}"
    
    # Test case 2: Small input, zero history
    current_input = 10
    fractional_sum = 0
    result = calculate_updated_potential(
        current_input, fractional_sum, r_scaled, tau_over_h_alpha, norm_factor
    )
    # Expected: (100 * 10) << 16 = 0x03E80000 (65536000), then * 105 >> 24 ≈ 410
    expected = ((r_scaled * current_input) << 16) * norm_factor >> 24
    assert result == expected, f"Expected {expected} for small input, got {result}"
    
    # Test case 3: Fractional sum calculation
    coefficients = [128, 64, 32]  # Example UQ0.8 coefficients (0.5, 0.25, 0.125)
    history_values = [10, 20, 30]  # Example history values
    frac_sum = calculate_fractional_sum(coefficients, history_values)
    # Expected: 128*10 + 64*20 + 32*30 = 1280 + 1280 + 960 = 3520
    expected_sum = 128*10 + 64*20 + 32*30
    assert frac_sum == expected_sum, f"Expected {expected_sum} for fractional sum, got {frac_sum}"
    
    print("All fixed-point calculations verified successfully!")


if __name__ == "__main__":
    # Run verification when script is executed directly
    verify_fixed_point_calculation()
