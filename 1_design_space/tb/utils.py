"""
Utility functions for testing neuron modules.
Contains helper functions for fixed-point arithmetic, signal verification, and test utilities.
"""

import pytest


def to_fixed_point(value, integer_bits=8, fractional_bits=8, signed=False):
    """
    Convert a floating-point value to fixed-point representation.

    Args:
        value (float): The floating-point value to convert
        integer_bits (int): Number of integer bits (default: 8)
        fractional_bits (int): Number of fractional bits (default: 8)
        signed (bool): Whether the fixed-point format is signed (default: False)

    Returns:
        int: Fixed-point representation as an integer

    Raises:
        ValueError: If the value is out of range for the specified format

    Examples:
        >>> to_fixed_point(1.5, 8, 8, False)  # UQ8.8 format
        384  # 1.5 * 256 = 384

        >>> to_fixed_point(-0.5, 8, 8, True)  # SQ8.8 format
        65408  # Two's complement representation
    """
    total_bits = integer_bits + fractional_bits
    scale_factor = 2 ** fractional_bits

    # Get the representable range
    min_value, max_value, _ = verify_fixed_point_range(integer_bits, fractional_bits, signed)

    # Check if value is within representable range
    if value > max_value or value < min_value:
        raise ValueError(f"Value {value} is out of range [{min_value}, {max_value}] "
                        f"for {'S' if signed else 'U'}Q{integer_bits}.{fractional_bits} format")

    # Scale the value
    scaled_value = value * scale_factor

    # Round to nearest integer
    rounded_value = round(scaled_value)

    # Handle two's complement for negative values
    if signed and rounded_value < 0:
        rounded_value = (2 ** total_bits) + rounded_value

    return int(rounded_value)


def from_fixed_point(fixed_value, integer_bits=8, fractional_bits=8, signed=False):
    """
    Convert a fixed-point integer back to floating-point representation.

    Args:
        fixed_value (int): The fixed-point value as an integer
        integer_bits (int): Number of integer bits (default: 8)
        fractional_bits (int): Number of fractional bits (default: 8)
        signed (bool): Whether the fixed-point format is signed (default: False)

    Returns:
        float: Floating-point representation

    Examples:
        >>> from_fixed_point(384, 8, 8, False)  # UQ8.8 format
        1.5

        >>> from_fixed_point(65408, 8, 8, True)  # SQ8.8 format
        -0.5
    """
    total_bits = integer_bits + fractional_bits
    scale_factor = 2 ** fractional_bits

    # Handle two's complement for signed values
    if signed:
        # Check if the sign bit is set
        if fixed_value >= (2 ** (total_bits - 1)):
            # Convert from two's complement
            fixed_value = fixed_value - (2 ** total_bits)

    # Convert back to floating point
    return fixed_value / scale_factor


def verify_fixed_point_range(integer_bits=8, fractional_bits=8, signed=False):
    """
    Return the valid range for a fixed-point format.

    Args:
        integer_bits (int): Number of integer bits (default: 8)
        fractional_bits (int): Number of fractional bits (default: 8)
        signed (bool): Whether the fixed-point format is signed (default: False)

    Returns:
        tuple: (min_value, max_value, resolution)
    """
    total_bits = integer_bits + fractional_bits
    scale_factor = 2 ** fractional_bits
    resolution = 1.0 / scale_factor

    if signed:
        max_value = (2 ** (total_bits - 1) - 1) / scale_factor
        min_value = -(2 ** (total_bits - 1)) / scale_factor
    else:
        max_value = (2 ** total_bits - 1) / scale_factor
        min_value = 0.0

    return min_value, max_value, resolution


# Unit tests for the fixed-point conversion functions
class TestFixedPointConversion:
    """Unit tests for fixed-point conversion functions."""

    def test_unsigned_basic_conversion(self):
        """Test basic unsigned fixed-point conversions."""
        # UQ8.8 format tests
        assert to_fixed_point(1.0, 8, 8, False) == 256
        assert to_fixed_point(0.5, 8, 8, False) == 128
        assert to_fixed_point(1.5, 8, 8, False) == 384
        assert to_fixed_point(0.0, 8, 8, False) == 0

        # Test maximum value for UQ8.8
        max_val = 255.99609375  # (2^16-1)/256
        assert to_fixed_point(max_val, 8, 8, False) == 65535

    def test_signed_basic_conversion(self):
        """Test basic signed fixed-point conversions."""
        # SQ8.8 format tests
        assert to_fixed_point(1.0, 8, 8, True) == 256
        assert to_fixed_point(-1.0, 8, 8, True) == 65280  # Two's complement
        assert to_fixed_point(0.5, 8, 8, True) == 128
        assert to_fixed_point(-0.5, 8, 8, True) == 65408  # Two's complement
        assert to_fixed_point(0.0, 8, 8, True) == 0

    def test_reverse_conversion(self):
        """Test that conversion is reversible."""
        test_values = [0.0, 0.5, 1.0, 1.5, 2.5, 10.25]

        for value in test_values:
            # Unsigned conversion
            fixed = to_fixed_point(value, 8, 8, False)
            recovered = from_fixed_point(fixed, 8, 8, False)
            assert abs(recovered - value) < 1e-6, f"Failed for unsigned {value}"

            # Signed positive conversion
            fixed = to_fixed_point(value, 8, 8, True)
            recovered = from_fixed_point(fixed, 8, 8, True)
            assert abs(recovered - value) < 1e-6, f"Failed for signed positive {value}"

        # Test negative values (signed only)
        negative_values = [-0.5, -1.0, -1.5, -10.25]
        for value in negative_values:
            fixed = to_fixed_point(value, 8, 8, True)
            recovered = from_fixed_point(fixed, 8, 8, True)
            assert abs(recovered - value) < 1e-6, f"Failed for signed negative {value}"

    def test_different_formats(self):
        """Test different fixed-point formats."""
        # UQ4.4 format
        assert to_fixed_point(1.5, 4, 4, False) == 24  # 1.5 * 16 = 24

        # UQ0.8 format (pure fractional)
        assert to_fixed_point(0.5, 0, 8, False) == 128  # 0.5 * 256 = 128

        # SQ15.8 format
        large_val = 100.5
        fixed = to_fixed_point(large_val, 15, 8, True)
        recovered = from_fixed_point(fixed, 15, 8, True)
        assert abs(recovered - large_val) < 1e-6

    def test_range_checking(self):
        """Test that out-of-range values raise ValueError."""
        # Test unsigned overflow
        with pytest.raises(ValueError):
            to_fixed_point(256.0, 8, 8, False)  # Max for UQ8.8 is ~255.996

        # Test signed overflow
        with pytest.raises(ValueError):
            to_fixed_point(128.0, 8, 8, True)  # Max for SQ8.8 is ~127.996

        # Test signed underflow
        with pytest.raises(ValueError):
            to_fixed_point(-129.0, 8, 8, True)  # Min for SQ8.8 is -128.0

        # Test unsigned negative
        with pytest.raises(ValueError):
            to_fixed_point(-1.0, 8, 8, False)  # Unsigned can't be negative

    def test_rounding(self):
        """Test proper rounding behavior."""
        # Test values that require rounding
        # For Q8.8, resolution is 1/256 ≈ 0.00390625

        # Should round to nearest
        assert to_fixed_point(1.001, 8, 8, False) == 256  # Rounds down to 1.0
        assert to_fixed_point(1.003, 8, 8, False) == 257  # Rounds up to ~1.00390625

    def test_range_verification(self):
        """Test the range verification helper function."""
        # UQ8.8 format
        min_val, max_val, res = verify_fixed_point_range(8, 8, False)
        assert min_val == 0.0
        assert abs(max_val - 255.99609375) < 1e-6
        assert abs(res - (1.0/256)) < 1e-10

        # SQ8.8 format
        min_val, max_val, res = verify_fixed_point_range(8, 8, True)
        assert min_val == -128.0
        assert abs(max_val - 127.99609375) < 1e-6
        assert abs(res - (1.0/256)) < 1e-10


if __name__ == "__main__":
    # Run the tests if this file is executed directly
    pytest.main([__file__, "-v"])
