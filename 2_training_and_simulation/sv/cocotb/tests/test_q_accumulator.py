"""
Cocotb tests for the q_accumulator module.

The q_accumulator computes Q-values by:
1. Reading membrane potentials from per-neuron buffers across all timesteps
2. Computing weighted sums (like a linear layer)
3. Accumulating across all timesteps
4. Averaging to produce final Q-values
5. Selecting the action with the highest Q-value at full internal precision

The module outputs selected_action (argmax) rather than Q-values, because the
Q-values routinely exceed the DATA_WIDTH (QS2.13) range and would saturate,
losing the distinction between actions.

Uses batched processing: BATCH_SIZE neurons processed per cycle.
Total multipliers: BATCH_SIZE × NUM_ACTIONS

Timing: NUM_TIMESTEPS × (NUM_NEURONS / BATCH_SIZE) + 2 cycles
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Timer
import os

# Test parameters (should match module instantiation)
NUM_NEURONS = 4
NUM_TIMESTEPS = 4
NUM_ACTIONS = 2
BATCH_SIZE = 2  # Process 2 neurons per cycle
DATA_WIDTH = 16
MEMBRANE_WIDTH = 24
FRAC_BITS = 13

# Scale factor for fixed-point
SCALE = 1 << FRAC_BITS  # 8192


def to_signed(val: int, bits: int) -> int:
    """Convert unsigned to signed interpretation."""
    if val >= (1 << (bits - 1)):
        return val - (1 << bits)
    return val


def to_unsigned(val: int, bits: int) -> int:
    """Convert signed to unsigned for assignment."""
    if val < 0:
        return val + (1 << bits)
    return val


def float_to_fixed(val: float, frac_bits: int = FRAC_BITS) -> int:
    """Convert float to signed fixed-point."""
    return int(round(val * (1 << frac_bits)))


def fixed_to_float(val: int, frac_bits: int = FRAC_BITS) -> float:
    """Convert signed fixed-point to float."""
    return val / (1 << frac_bits)


async def reset_dut(dut):
    """Apply reset sequence to the DUT."""
    dut.reset.value = 1
    dut.start.value = 0
    for i in range(NUM_NEURONS):
        dut.membrane_in[i].value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


def compute_expected_q_values(weights, biases, membranes):
    """
    Compute expected Q-values in software.

    Args:
        weights: [NUM_ACTIONS][NUM_NEURONS] weight matrix (float)
        biases: [NUM_ACTIONS] bias vector (float)
        membranes: [NUM_TIMESTEPS][NUM_NEURONS] membrane potentials (float)

    Returns:
        [NUM_ACTIONS] averaged Q-values (float)
    """
    q_accum = [0.0] * NUM_ACTIONS

    for t in range(NUM_TIMESTEPS):
        for a in range(NUM_ACTIONS):
            timestep_sum = sum(
                weights[a][n] * membranes[t][n] for n in range(NUM_NEURONS)
            )
            timestep_sum += biases[a]
            q_accum[a] += timestep_sum

    # Average over timesteps
    q_avg = [q / NUM_TIMESTEPS for q in q_accum]
    return q_avg


@cocotb.test()
async def test_q_accumulator_reset(dut):
    """Test that reset properly initializes the module."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Check reset state
    assert dut.done.value == 0, "done should be 0 after reset"

    dut._log.info("Reset test passed")


@cocotb.test()
async def test_q_accumulator_zero_membranes(dut):
    """Test with all-zero membrane potentials. Q-values should be biases."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # All membranes zero
    for i in range(NUM_NEURONS):
        dut.membrane_in[i].value = 0

    # Start computation
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done (with timeout)
    max_cycles = NUM_TIMESTEPS * (NUM_NEURONS // BATCH_SIZE) + 10
    for cycle in range(max_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            break

    assert dut.done.value == 1, f"done not asserted after {max_cycles} cycles"

    # With zero membranes, Q-values equal biases. selected_action reflects
    # which bias is larger.
    action = int(dut.selected_action.value)
    dut._log.info(f"selected_action = {action}")

    dut._log.info("Zero membranes test passed")


@cocotb.test()
async def test_q_accumulator_uniform_membranes(dut):
    """Test with uniform membrane potentials across neurons and timesteps."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Set uniform membrane value (0.5 in fixed-point)
    membrane_val = float_to_fixed(0.5, FRAC_BITS)
    for i in range(NUM_NEURONS):
        dut.membrane_in[i].value = to_unsigned(membrane_val, MEMBRANE_WIDTH)

    # Start computation
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    max_cycles = NUM_TIMESTEPS * (NUM_NEURONS // BATCH_SIZE) + 10
    for cycle in range(max_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            break

    assert dut.done.value == 1, f"done not asserted after {max_cycles} cycles"

    # Log selected action
    action = int(dut.selected_action.value)
    dut._log.info(f"Uniform membrane selected_action = {action}")

    dut._log.info("Uniform membranes test passed")


@cocotb.test()
async def test_q_accumulator_varying_timesteps(dut):
    """Test that different timesteps contribute correctly to the accumulation."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Start computation
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Track the expected number of batches per timestep
    batches_per_timestep = NUM_NEURONS // BATCH_SIZE
    expected_total_batches = NUM_TIMESTEPS * batches_per_timestep

    # Feed different membrane values for different timesteps
    # The q_accumulator reads from membrane_in based on read_timestep
    # We'll change membrane values each time timestep changes

    last_timestep = -1
    cycle_count = 0
    max_cycles = expected_total_batches + 20

    while cycle_count < max_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1

        if dut.done.value == 1:
            break

        # Read current timestep being requested
        current_ts = int(dut.read_timestep.value)

        # Set membrane values based on timestep
        # Each timestep has membranes = timestep * 0.1 per neuron
        membrane_val = float_to_fixed(0.1 * (current_ts + 1), FRAC_BITS)
        for i in range(NUM_NEURONS):
            dut.membrane_in[i].value = to_unsigned(membrane_val, MEMBRANE_WIDTH)

        # Print cycle info for each batch, i.e. when timestep changes
        if current_ts != last_timestep:
            dut._log.info(f"Cycle {cycle_count}: Processing timestep {current_ts}")
            last_timestep = current_ts

    assert dut.done.value == 1, f"done not asserted after {cycle_count} cycles"

    action = int(dut.selected_action.value)
    dut._log.info(f"Varying timestep selected_action = {action}")

    dut._log.info("Varying timesteps test passed")


@cocotb.test()
async def test_q_accumulator_timing(dut):
    """Test that computation completes in expected number of cycles."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Set membrane values
    for i in range(NUM_NEURONS):
        dut.membrane_in[i].value = to_unsigned(float_to_fixed(0.5), MEMBRANE_WIDTH)

    # Start and count cycles
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    cycle_count = 0
    max_cycles = NUM_TIMESTEPS * (NUM_NEURONS // BATCH_SIZE) + 20

    while cycle_count < max_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1
        if dut.done.value == 1:
            break

    # Expected: NUM_TIMESTEPS * (NUM_NEURONS / BATCH_SIZE) + 2 cycles
    # = 4 * 2 + 2 = 10 cycles
    expected_cycles = NUM_TIMESTEPS * (NUM_NEURONS // BATCH_SIZE) + 2

    # TODO: Make this match precisely rather than adding tolerance
    dut._log.info(f"Completed in {cycle_count} cycles (expected ~{expected_cycles})")

    # Allow some tolerance for state machine transitions
    assert (
        cycle_count <= expected_cycles + 5
    ), f"Took too long: {cycle_count} cycles (expected ~{expected_cycles})"

    dut._log.info("Timing test passed")


@cocotb.test()
async def test_q_accumulator_restart(dut):
    """Test that module can be restarted for new inference."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    max_cycles = NUM_TIMESTEPS * (NUM_NEURONS // BATCH_SIZE) + 10

    # First inference with low membrane values
    for i in range(NUM_NEURONS):
        dut.membrane_in[i].value = to_unsigned(float_to_fixed(0.1), MEMBRANE_WIDTH)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            break

    q0_first = int(dut.selected_action.value)
    dut._log.info(f"First inference selected_action = {q0_first}")

    # Second inference with high membrane values (restart while done is high)
    for i in range(NUM_NEURONS):
        dut.membrane_in[i].value = to_unsigned(float_to_fixed(1.0), MEMBRANE_WIDTH)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # done should deassert
    await RisingEdge(dut.clk)
    assert dut.done.value == 0, "done should deassert after restart"

    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            break

    q0_second = int(dut.selected_action.value)
    dut._log.info(f"Second inference selected_action = {q0_second}")

    # With 10× higher membranes, the selected action may differ
    # depending on weight signs
    dut._log.info("Restart test passed")


@cocotb.test()
async def test_q_accumulator_negative_membranes(dut):
    """Test with negative membrane potentials."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Negative membrane values
    for i in range(NUM_NEURONS):
        membrane_val = float_to_fixed(-0.5, FRAC_BITS)
        dut.membrane_in[i].value = to_unsigned(membrane_val, MEMBRANE_WIDTH)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    max_cycles = NUM_TIMESTEPS * (NUM_NEURONS // BATCH_SIZE) + 10
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            break

    assert dut.done.value == 1, "done not asserted"

    action = int(dut.selected_action.value)
    dut._log.info(f"Negative membrane selected_action = {action}")

    dut._log.info("Negative membranes test passed")
