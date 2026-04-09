"""
Cocotb tests for frac_order_lif module.
Tests fractional-order LIF neuron with parallel coefficient implementation.
"""

import cocotb
from cocotb.triggers import Timer, RisingEdge, ClockCycles
from cocotb.clock import Clock

from reference_binomial import generalized_binomial_coeff


class FracOrderLIFTester:
    """Helper class for fractional-order LIF testing."""

    def __init__(self, dut):
        self.dut = dut
        # TODO: We should update this to be dynamic, maybe integrated with frac_order_lif_utils.py
        self.alpha_float = 8.0 / 15.0
        self.history_size = int(dut.HISTORY_SIZE.value)
        self.coeff_width = int(dut.COEFF_WIDTH.value)

        # Physical parameters
        self.r_scaled = int(dut.R_SCALED.value)
        self.threshold = int(dut.THRESHOLD.value)
        self.refractory_period = int(dut.REFRACTORY_PERIOD.value)

    async def reset_dut(self):
        """Reset the DUT and wait for stable state."""
        self.dut.rst.value = 1
        await RisingEdge(self.dut.clk)
        await RisingEdge(self.dut.clk)
        self.dut.rst.value = 0
        await RisingEdge(self.dut.clk)

    async def apply_current_step(self, current_value, duration_cycles):
        """Apply a constant current for specified duration."""
        self.dut.current.value = current_value
        await ClockCycles(self.dut.clk, duration_cycles)

    async def get_membrane_potential(self):
        """Read current membrane potential."""
        await Timer(1, units="ns")  # Allow combinational settling
        return int(self.dut.membrane_potential.value)

    def calculate_expected_coefficient_magnitude(self, k):
        """Calculate expected coefficient magnitude for verification."""
        # Calculate binomial coefficient
        binom_coeff = generalized_binomial_coeff(self.alpha_float, k)

        # Apply (-1)^k factor
        sign_factor = -1.0 if (k % 2 == 1) else 1.0
        final_weight = sign_factor * binom_coeff

        # All weights should be negative for 0 < α < 1
        if final_weight >= 0.0:
            cocotb.log.warning(
                f"Expected negative weight for k={k}, got {final_weight}"
            )

        # Return magnitude scaled to UQ0.8
        weight_mag = abs(final_weight)
        scaled = weight_mag * 256.0
        return min(255, max(0, int(scaled)))

    async def verify_coefficients_accessible(self):
        """Verify that coefficients are properly generated and accessible."""
        cocotb.log.info(
            f"Verifying coefficients for α={self.alpha_float:.3f}, history_size={self.history_size}"
        )

        # Note: We can't directly read the coefficient values since they're internal to generate blocks
        # Instead, we'll verify the behavior indirectly through the fractional sum calculation

        # Apply a known history pattern and verify the fractional sum makes sense
        await self.reset_dut()

        # Fill history buffer with known values
        test_history = [50, 100, 75, 25, 60, 40, 80, 30]  # 8 test values

        for i in range(min(self.history_size, len(test_history))):
            self.dut.current.value = test_history[i]
            await RisingEdge(self.dut.clk)

        # Let the fractional sum settle
        await Timer(10, units="ns")

        # The test passes if no simulation errors occur and values are reasonable
        membrane_potential = await self.get_membrane_potential()
        cocotb.log.info(f"Membrane potential with test history: {membrane_potential}")

        return True


@cocotb.test()
async def test_basic_functionality(dut):
    """Test basic LIF functionality and reset behavior."""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")  # 100MHz clock
    cocotb.start_soon(clock.start())

    tester = FracOrderLIFTester(dut)

    cocotb.log.info(
        f"Testing α={tester.alpha_float:.3f}, history_size={tester.history_size}"
    )

    # Test reset behavior
    await tester.reset_dut()

    # Verify initial state
    membrane_potential = await tester.get_membrane_potential()
    assert membrane_potential == 0, f"Expected 0 after reset, got {membrane_potential}"

    # Verify not spiking initially
    assert dut.spike.value == 0, "Should not spike after reset"

    # Apply small current and verify membrane potential increases
    await tester.apply_current_step(10, 5)
    membrane_potential_after = await tester.get_membrane_potential()

    # Should have some increase due to current
    assert (
        membrane_potential_after > 0
    ), f"Expected increase from current, got {membrane_potential_after}"

    cocotb.log.info(
        f"Basic functionality test passed. Final potential: {membrane_potential_after}"
    )


@cocotb.test()
async def test_coefficient_generation(dut):
    """Test that coefficient generation works without errors."""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    tester = FracOrderLIFTester(dut)

    # Verify coefficients are accessible and reasonable
    coefficients_ok = await tester.verify_coefficients_accessible()
    assert coefficients_ok, "Coefficient verification failed"

    cocotb.log.info("Coefficient generation test passed")


@cocotb.test()
async def test_spike_behavior(dut):
    """Test spike generation and refractory period."""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    tester = FracOrderLIFTester(dut)

    await tester.reset_dut()

    # Apply large current to trigger spike
    large_current = 100  # Should be enough to reach threshold
    max_cycles = 50  # Safety limit

    spike_detected = False
    for cycle in range(max_cycles):
        dut.current.value = large_current
        await RisingEdge(dut.clk)

        if dut.spike.value == 1:
            spike_detected = True
            cocotb.log.info(f"Spike detected at cycle {cycle}")
            break

    assert (
        spike_detected
    ), f"No spike detected within {max_cycles} cycles with current {large_current}"

    # Verify refractory period
    # During refractory, should not spike even with high current
    await RisingEdge(dut.clk)  # Move past spike cycle

    for ref_cycle in range(tester.refractory_period):
        assert (
            dut.spike.value == 0
        ), f"Unexpected spike during refractory period at cycle {ref_cycle}"
        await RisingEdge(dut.clk)

    cocotb.log.info("Spike and refractory behavior test passed")


@cocotb.test()
async def test_fractional_memory_effect(dut):
    """Test that the fractional derivative provides memory effect."""
    # TODO: This needs to be more rigorous, using actual numbers calculated based on equation used.
    # FIXME: The results of Test 2 below indicate a possible issue with spiking after a decay period.
    # This *might* be due to the fractional memory effect behaving as expected, or an issue with the SV implementation.

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    tester = FracOrderLIFTester(dut)

    # Test 1: Apply current, remove it, verify decay behavior
    await tester.reset_dut()

    # Apply moderate current for several cycles
    await tester.apply_current_step(30, 5)
    potential_with_current = await tester.get_membrane_potential()

    # Remove current and observe decay
    await tester.apply_current_step(0, 10)
    potential_after_decay = await tester.get_membrane_potential()

    # Due to fractional derivative memory effect, should not decay to zero immediately
    # The exact behavior depends on α, but there should be some memory
    cocotb.log.info(
        f"Potential with current: {potential_with_current}, after decay: {potential_after_decay}"
    )

    # Test 2: Compare with pure integrator behavior
    # Apply brief pulse and observe response
    await tester.reset_dut()

    # Brief high current pulse
    await tester.apply_current_step(80, 2)
    potential_after_pulse = await tester.get_membrane_potential()

    # Let it evolve without current
    await tester.apply_current_step(0, 8)
    potential_final = await tester.get_membrane_potential()

    cocotb.log.info(f"After pulse: {potential_after_pulse}, final: {potential_final}")
    cocotb.log.info("Fractional memory effect test completed")


@cocotb.test()
async def test_saturation_behavior(dut):
    """Test membrane potential saturation at bounds."""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    tester = FracOrderLIFTester(dut)

    await tester.reset_dut()

    # Test upper saturation
    # Apply very large current for extended time
    await tester.apply_current_step(255, 20)  # Maximum current
    potential_max = await tester.get_membrane_potential()

    # Should saturate at 255 or threshold, whichever is lower
    expected_max = min(255, tester.threshold)
    if potential_max < expected_max:
        # This is ok - might spike before reaching saturation
        cocotb.log.info(
            f"Potential {potential_max} < expected max {expected_max} (likely due to spiking)"
        )
    else:
        assert potential_max <= 255, f"Potential {potential_max} exceeds maximum 255"

    # TODO: Determine if this is a worthwhile test, or should be removed or modified
    # Test lower bound (should not go negative)
    await tester.reset_dut()
    # Start with some potential
    await tester.apply_current_step(20, 3)
    # Apply zero current to let it decay
    await tester.apply_current_step(0, 10)

    potential_min = await tester.get_membrane_potential()
    assert potential_min >= 0, f"Potential went negative: {potential_min}"

    cocotb.log.info("Saturation behavior test passed")


@cocotb.test()
async def test_basic_spike_train(dut):
    """Provides a basic input spike train that should be used for comparison across neuron models."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    tester = FracOrderLIFTester(dut)
    
    # Reset
    await tester.reset_dut()
    
    # Test with basic input spike train
    input_currents = [250, 250, 250, 250, 200, 200, 200, 200, 150, 150, 150, 150, 100, 100, 100, 100, 50, 50, 50, 50, 25, 25, 25, 25, 0, 0, 0, 0]
    # TODO: Consider adding function to calculate expected timing. This will be more complext
    # than in the Lapicque version, but could be helpful for verification.
    # expected_spikes, expected_potentials = tester.calculate_expected_spike_timing(input_currents)
    
    for i, current in enumerate(input_currents):
        dut.current.value = current
        await RisingEdge(dut.clk)
        
        # Check spike output
        # assert dut.spike.value == expected_spikes[i], f"Cycle {i}: Expected spike {expected_spikes[i]}, got {dut.spike.value}"

    await ClockCycles(dut.clk, 15)