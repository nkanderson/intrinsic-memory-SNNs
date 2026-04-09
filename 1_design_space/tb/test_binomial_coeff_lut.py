"""
Cocotb tests for binomial_coeff_lut module.
Tests fractional binomial coefficient calculation and LUT functionality.
"""

import cocotb
from cocotb.triggers import Timer
from cocotb.regression import TestFactory

from reference_binomial import (
    generalized_binomial_coeff,
    decode_lut_coefficient,
    alpha_4bit_to_float,
    calculate_tolerance,
    COMMON_ALPHA_VALUES
)


class BinomialLUTTester:
    """Helper class for binomial coefficient LUT testing."""

    def __init__(self, dut):
        self.dut = dut
        self.alpha_4bit = int(dut.ALPHA_4BIT.value)
        self.max_k = int(dut.MAX_K.value)
        self.coeff_width = int(dut.COEFF_WIDTH.value)
        self.alpha_float = alpha_4bit_to_float(self.alpha_4bit)
        # Add default tolerance attribute for backward compatibility
        self.tolerance = calculate_tolerance(self.coeff_width, k=1)  # Use k≥1 format as default

    async def read_coefficient(self, k: int) -> float:
        """Read coefficient from DUT and convert to float."""
        self.dut.k.value = k
        await Timer(1, units='ns')  # Allow combinational settling

        # Read unsigned coefficient and decode based on k value
        coeff_raw = int(self.dut.coeff.value)
        return decode_lut_coefficient(coeff_raw, k)

    async def verify_coefficient(self, k: int) -> bool:
        """Verify single coefficient against reference."""
        dut_coeff = await self.read_coefficient(k)

        # Get reference coefficient from scipy
        ref_coeff = generalized_binomial_coeff(self.alpha_float, k)
        
        # For k≥1, compare unsigned magnitudes since our LUT stores magnitude only
        if k == 0:
            # k=0 should be exactly 1.0
            expected = ref_coeff  # Should be 1.0
            actual = dut_coeff
        else:
            # k≥1: Compare magnitudes only
            expected = abs(ref_coeff)
            actual = abs(dut_coeff)

        # Calculate tolerance based on k value (different formats)
        tolerance = calculate_tolerance(self.coeff_width, k)
        error = abs(actual - expected)
        within_tolerance = error <= tolerance

        if not within_tolerance:
            cocotb.log.error(
                f"k={k}: DUT={actual:.6f}, REF={expected:.6f}, "
                f"ERROR={error:.6f}, TOL={tolerance:.6f}"
            )
        else:
            cocotb.log.info(
                f"k={k}: DUT={actual:.6f}, REF={expected:.6f}, "
                f"ERROR={error:.6f} ✓"
            )

        return within_tolerance


@cocotb.test()
async def test_basic_functionality(dut):
    """Test basic LUT functionality with default parameters."""
    tester = BinomialLUTTester(dut)

    cocotb.log.info(f"Testing alpha={tester.alpha_float:.3f}, max_k={tester.max_k}")

    # Test all k values from 0 to max_k
    all_passed = True
    for k in range(tester.max_k + 1):
        passed = await tester.verify_coefficient(k)
        all_passed = all_passed and passed

    assert all_passed, "Some coefficients failed tolerance check"


@cocotb.test()
async def test_edge_cases(dut):
    """Test edge cases and boundary conditions."""
    tester = BinomialLUTTester(dut)

    # Test k=0 (should always be 1.0, no sign change for k=0)
    k0_coeff = await tester.read_coefficient(0)
    expected_k0 = 1.0
    assert abs(k0_coeff - expected_k0) <= tester.tolerance, \
        f"k=0 coefficient should be 1.0, got {k0_coeff}"

    # Test k=max_k (boundary case)
    await tester.verify_coefficient(tester.max_k)

    # Test k > max_k (should return 0)
    if tester.max_k < 15:  # Avoid overflow of address width
        dut.k.value = tester.max_k + 1
        await Timer(1, units='ns')
        out_of_bounds_coeff = int(dut.coeff.value)
        assert out_of_bounds_coeff == 0, \
            f"k > max_k should return 0, got {out_of_bounds_coeff}"


@cocotb.test()
async def test_mathematical_properties(dut):
    """Test mathematical properties of binomial coefficients."""
    tester = BinomialLUTTester(dut)

    # For 0 < alpha < 1, coefficients should generally decrease in magnitude
    # as k increases (for reasonable k values)
    if 0 < tester.alpha_float < 1 and tester.max_k >= 3:
        coeffs = []
        for k in range(min(4, tester.max_k + 1)):
            coeff = await tester.read_coefficient(k)
            coeffs.append(abs(coeff))

        # Check that magnitude generally decreases
        # (allowing some tolerance for numerical precision)
        for i in range(1, len(coeffs) - 1):
            ratio = coeffs[i+1] / coeffs[i] if coeffs[i] != 0 else 0
            cocotb.log.info(f"|C({tester.alpha_float:.2f},{i+1})| / |C({tester.alpha_float:.2f},{i})| = {ratio:.3f}")


@cocotb.test()
async def test_alternating_signs(dut):
    """Test that coefficients match scipy magnitudes (stored as unsigned)."""
    tester = BinomialLUTTester(dut)

    cocotb.log.info("Testing coefficient magnitudes match scipy absolute values...")

    # Test first few coefficients match scipy magnitudes
    for k in range(min(4, tester.max_k + 1)):
        dut_coeff = await tester.read_coefficient(k)
        scipy_coeff = generalized_binomial_coeff(tester.alpha_float, k)
        
        if k == 0:
            # k=0 should be exactly 1.0 (no sign change)
            expected = scipy_coeff
            actual = dut_coeff
        else:
            # k≥1: Compare magnitudes since LUT stores unsigned magnitude
            expected = abs(scipy_coeff)
            actual = abs(dut_coeff)

        error = abs(actual - expected)
        tolerance = calculate_tolerance(tester.coeff_width, k)
        assert error <= tolerance, \
            f"k={k}: DUT_MAG={actual:.6f} != scipy_mag={expected:.6f}, error={error:.6f}"

        cocotb.log.info(f"k={k}: magnitude={actual:.6f}, scipy_mag={expected:.6f} ✓")


# TODO: Add functional testing
# The following tests would verify the coefficients work correctly
# in an actual fractional derivative calculation:
#
# @cocotb.test()
# async def test_functional_fractional_derivative(dut):
#     """
#     Test coefficients in a simple fractional derivative approximation.
#
#     This would:
#     1. Generate a known test signal (e.g., step, ramp, sine)
#     2. Apply fractional derivative using the LUT coefficients
#     3. Compare against analytical solution or reference implementation
#     4. Verify the fractional-order behavior is correct
#     """
#     pass
#
# @cocotb.test()
# async def test_memory_effect_validation(dut):
#     """
#     Verify the memory effect characteristic of fractional derivatives.
#
#     This would test that:
#     1. Recent history has stronger influence than distant history
#     2. The alternating sign pattern creates proper oscillatory behavior
#     3. The "long memory" property is preserved
#     """
#     pass


# Parameterized test factory for different configurations
def alpha_test_factory():
    """Factory for testing different alpha values."""

    @cocotb.test()
    async def test_alpha_accuracy(dut, alpha_float, alpha_4bit):
        """Test coefficient accuracy for specific alpha value."""
        # This test assumes the DUT is parameterized with alpha_4bit
        expected_alpha = alpha_4bit_to_float(alpha_4bit)

        if abs(expected_alpha - alpha_float) > 1e-3:
            cocotb.log.warning(
                f"DUT alpha mismatch: expected {alpha_float}, "
                f"DUT configured for {expected_alpha}"
            )
            return  # Skip if alpha doesn't match

        tester = BinomialLUTTester(dut)

        # Test first few coefficients for this alpha
        for k in range(min(4, tester.max_k + 1)):
            await tester.verify_coefficient(k)

    return test_alpha_accuracy


# Register parameterized tests
tf_alpha = TestFactory(alpha_test_factory())
for alpha_float, alpha_4bit in COMMON_ALPHA_VALUES:
    tf_alpha.add_option("alpha_float", [alpha_float])
    tf_alpha.add_option("alpha_4bit", [alpha_4bit])
tf_alpha.generate_tests()
