import cocotb
from cocotb.triggers import Timer, RisingEdge, ClockCycles
from cocotb.clock import Clock
from utils import to_fixed_point, from_fixed_point


class LapicqueTester:
    """Helper class for Lapicque LIF testing."""

    def __init__(self, dut):
        self.dut = dut

        # Extract Lapicque parameters
        self.decay = int(dut.DECAY.value)
        self.k = int(dut.K.value)
        self.threshold = int(dut.THRESHOLD.value)
        self.refractory_period = int(dut.REFRACTORY_PERIOD.value)

    def calculate_new_potential(self, current_potential, input_current):
        """
        Calculate new membrane potential using Lapicque model.
        V[n] = decay * V[n-1] + K * I[n]

        Args:
            current_potential (int): Current membrane potential (8-bit)
            input_current (int): Input current (8-bit)

        Returns:
            int: New membrane potential (8-bit, clamped to 0-255)
        """
        # Both DECAY and K are in Q0.8 format, so both need >> 8 bit shifts
        # updated_potential = (({8'b0, K} * current) >> 8) + ((membrane_potential * DECAY) >> 8)

        # Calculate K * I[n] term
        current_term = (self.k * input_current) >> 8

        # Calculate decay * V[n-1] term
        decay_term = (current_potential * self.decay) >> 8

        # Combine terms
        updated_potential = current_term + decay_term

        # Clamp to 8-bit range [0, 255]
        return min(255, max(0, updated_potential))

    def calculate_expected_spike_timing(self, input_currents):
        """
        Calculate expected spike timing and membrane potentials for given input currents.

        Args:
            input_currents (list): List of input current values for each clock cycle

        Returns:
            tuple: (spike_outputs, membrane_potentials)
                - spike_outputs: List of 0s and 1s indicating spikes
                - membrane_potentials: List of membrane potential values
        """
        spike_outputs = []
        membrane_potentials = []

        # Initialize state
        membrane_potential = 0
        refractory_counter = 0

        for current in input_currents:
            # Check if in refractory period
            in_refractory = refractory_counter > 0

            # Check for spike (before potential update)
            spike = (membrane_potential >= self.threshold) and not in_refractory

            if in_refractory:
                # Decrement refractory counter, don't update potential
                refractory_counter -= 1
            elif spike:
                # Spike occurred: reset potential and enter refractory
                membrane_potential = membrane_potential - self.threshold
                refractory_counter = self.refractory_period
            else:
                # Normal update: calculate new potential
                membrane_potential = self.calculate_new_potential(
                    membrane_potential, current
                )

            # Record outputs
            spike_outputs.append(1 if spike else 0)
            membrane_potentials.append(membrane_potential)

        return spike_outputs, membrane_potentials


# Basic test functions
@cocotb.test()
async def test_no_input(dut):
    """Test that no input produces no spikes."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    tester = LapicqueTester(dut)

    # Reset
    dut.rst.value = 1
    await ClockCycles(dut.clk, 1)
    dut.rst.value = 0

    # Test with zero input
    input_currents = [0] * 20
    expected_spikes, expected_potentials = tester.calculate_expected_spike_timing(
        input_currents
    )

    for i, current in enumerate(input_currents):
        dut.current.value = current
        await RisingEdge(dut.clk)

        # Check spike output
        assert (
            dut.spike.value == expected_spikes[i]
        ), f"Cycle {i}: Expected spike {expected_spikes[i]}, got {dut.spike.value}"

        # Note: We can't directly check membrane_potential as it's internal


@cocotb.test()
async def test_constant_input(dut):
    """Test response to constant input current."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    tester = LapicqueTester(dut)

    # Reset
    dut.rst.value = 1
    await ClockCycles(dut.clk, 1)
    dut.rst.value = 0

    # Test with constant moderate input
    input_currents = [50] * 30  # Should eventually cause spike
    expected_spikes, expected_potentials = tester.calculate_expected_spike_timing(
        input_currents
    )

    for i, current in enumerate(input_currents):
        dut.current.value = current
        await RisingEdge(dut.clk)

        # Check spike output
        assert (
            dut.spike.value == expected_spikes[i]
        ), f"Cycle {i}: Expected spike {expected_spikes[i]}, got {dut.spike.value}"


@cocotb.test()
async def test_high_input_immediate_spike(dut):
    """Test that high input causes immediate spike."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    tester = LapicqueTester(dut)

    # Reset
    dut.rst.value = 1
    await ClockCycles(dut.clk, 1)
    dut.rst.value = 0

    # Test with very high input that should cause immediate spike
    input_currents = [255, 0, 0, 0, 0, 0, 0, 255]  # High pulse, then gap, then another
    expected_spikes, expected_potentials = tester.calculate_expected_spike_timing(
        input_currents
    )

    for i, current in enumerate(input_currents):
        dut.current.value = current
        await RisingEdge(dut.clk)

        # Check spike output
        assert (
            dut.spike.value == expected_spikes[i]
        ), f"Cycle {i}: Expected spike {expected_spikes[i]}, got {dut.spike.value}"


@cocotb.test()
async def test_refractory_period(dut):
    """Test that refractory period prevents spikes."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    tester = LapicqueTester(dut)

    # Reset
    dut.rst.value = 1
    await ClockCycles(dut.clk, 1)
    dut.rst.value = 0

    # Test with repeated high inputs to verify refractory behavior
    input_currents = [255] * 10  # Continuous high input
    expected_spikes, expected_potentials = tester.calculate_expected_spike_timing(
        input_currents
    )

    spike_count = 0
    for i, current in enumerate(input_currents):
        dut.current.value = current
        await RisingEdge(dut.clk)

        if dut.spike.value:
            spike_count += 1

        # Check spike output
        assert (
            dut.spike.value == expected_spikes[i]
        ), f"Cycle {i}: Expected spike {expected_spikes[i]}, got {dut.spike.value}"

    # Should not spike on every cycle due to refractory period
    assert spike_count < len(
        input_currents
    ), "Expected fewer spikes due to refractory period"


@cocotb.test()
async def test_basic_spike_train(dut):
    """Provides a basic input spike train that should be used for comparison across neuron models."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    tester = LapicqueTester(dut)

    # Reset
    dut.rst.value = 1
    await ClockCycles(dut.clk, 1)
    dut.rst.value = 0

    # Test with basic input spike train
    input_currents = [
        250,
        250,
        250,
        250,
        200,
        200,
        200,
        200,
        150,
        150,
        150,
        150,
        100,
        100,
        100,
        100,
        50,
        50,
        50,
        50,
        25,
        25,
        25,
        25,
        0,
        0,
        0,
        0,
    ]
    expected_spikes, expected_potentials = tester.calculate_expected_spike_timing(
        input_currents
    )

    for i, current in enumerate(input_currents):
        dut.current.value = current
        await RisingEdge(dut.clk)

        # Check spike output
        assert (
            dut.spike.value == expected_spikes[i]
        ), f"Cycle {i}: Expected spike {expected_spikes[i]}, got {dut.spike.value}"

    await ClockCycles(dut.clk, 15)
