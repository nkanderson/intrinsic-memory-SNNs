"""
Cocotb tests for the single-step LIF (Leaky Integrate-and-Fire) neuron module
based on snnTorch's Leaky neuron class.
Tests membrane dynamics, spike generation, and reset behavior.

Tests the single-step LIF that processes one timestep per 'enable' signal,
allowing external control of timestep timing by the neural_network module.

For tests of the legacy version with internal timestep loop, see test_lif_timestep.py.

Fixed-point format: QS2.13 (16-bit signed, 2 integer bits, 13 fractional bits)
- Scale factor: 2^13 = 8192
- Threshold 1.0 = 8192
- Beta 0.9 in Q1.7 = 115
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

# Fixed-point format constants
TOTAL_BITS = 16  # Total bits in QS2.13 format
FRAC_BITS = 13  # Fractional bits

# Derived constants
SCALE_FACTOR = 2**FRAC_BITS  # 8192 for QS2.13
THRESHOLD = 2**FRAC_BITS  # 1.0 in fixed-point = 8192
MAX_SIGNED = 2 ** (TOTAL_BITS - 1) - 1  # 32767
MIN_SIGNED = -(2 ** (TOTAL_BITS - 1))  # -32768
UNSIGNED_RANGE = 2**TOTAL_BITS  # 65536

BETA = 0.9
NUM_TIMESTEPS = 30  # Default number of timesteps for tests


def float_to_fixed(value: float) -> int:
    """Convert float to fixed-point (TOTAL_BITS-bit signed)."""
    scaled = int(round(value * SCALE_FACTOR))
    # Clamp to signed range
    if scaled > MAX_SIGNED:
        scaled = MAX_SIGNED
    elif scaled < MIN_SIGNED:
        scaled = MIN_SIGNED
    # Convert to unsigned for assignment (two's complement)
    if scaled < 0:
        scaled = scaled + UNSIGNED_RANGE
    return scaled


def membrane_to_float(value: int) -> float:
    """Convert 24-bit membrane potential to float."""
    if value >= (1 << 23):
        value = value - (1 << 24)
    return value / SCALE_FACTOR


async def reset_dut(dut):
    """Apply reset sequence to the DUT and wait for it to stabilize."""
    dut.reset.value = 1
    dut.clear.value = 0
    dut.enable.value = 0
    dut.current.value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


async def run_timesteps(dut, current_value: int, num_timesteps: int = NUM_TIMESTEPS):
    """Run the LIF for multiple timesteps with the same current.

    Returns list of (spike, membrane) tuples for each timestep.
    """
    results = []
    dut.current.value = current_value

    for _ in range(num_timesteps):
        dut.enable.value = 1
        await RisingEdge(dut.clk)
        dut.enable.value = 0
        await RisingEdge(dut.clk)  # Wait for registered outputs

        spike = int(dut.spike_out.value)
        membrane = int(dut.membrane_out.value)
        results.append((spike, membrane))

    return results


@cocotb.test()
async def test_lif_reset(dut):
    """Test that reset properly initializes the LIF neuron."""

    # Start clock (10ns period = 100MHz)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Apply reset
    dut.reset.value = 1
    dut.enable.value = 0
    dut.current.value = 0

    await ClockCycles(dut.clk, 5)

    # Check reset state
    assert dut.spike_out.value == 0, "spike_out should be 0 after reset"
    assert dut.membrane_out.value == 0, "membrane_out should be 0 after reset"

    dut._log.info("Reset test passed")


@cocotb.test()
async def test_lif_clear(dut):
    """Test that clear properly resets membrane without full reset."""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Build up some membrane potential
    results = await run_timesteps(dut, float_to_fixed(0.5), 5)
    _, membrane_before = results[-1]
    assert membrane_before != 0, "Should have non-zero membrane before clear"

    # Apply clear
    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0
    await RisingEdge(dut.clk)

    # Check membrane is cleared
    assert dut.membrane_out.value == 0, "membrane_out should be 0 after clear"
    assert dut.spike_out.value == 0, "spike_out should be 0 after clear"

    dut._log.info("Clear test passed")


@cocotb.test()
async def test_lif_no_spike_below_threshold(dut):
    """Test that neuron doesn't spike when input is below threshold."""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Apply small input (0.1) - membrane should accumulate but not spike
    # With beta=0.9, membrane = 0.1 + 0.9*0.1 + 0.9^2*0.1 + ... < 1.0
    small_input = float_to_fixed(0.1)

    results = await run_timesteps(dut, small_input, NUM_TIMESTEPS)

    # Check all timesteps - should never spike
    for i, (spike, membrane) in enumerate(results):
        dut._log.info(
            f"Timestep {i}: spike={spike}, membrane={membrane_to_float(membrane):.4f}"
        )
        assert spike == 0, f"Should not spike on timestep {i} with small input"

    dut._log.info("No spike below threshold test passed")


@cocotb.test()
async def test_lif_spike_above_threshold(dut):
    """Test that neuron spikes when membrane exceeds threshold."""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Apply large input (1.5) - should spike on first timestep
    large_input = float_to_fixed(1.5)

    results = await run_timesteps(dut, large_input, 1)

    spike, membrane = results[0]
    dut._log.info(
        f"Timestep 0: spike={spike}, membrane={membrane_to_float(membrane):.4f}"
    )
    assert spike == 1, "Should spike on first timestep when input exceeds threshold"

    dut._log.info("Spike above threshold test passed")


@cocotb.test()
async def test_lif_membrane_accumulation(dut):
    """Test that membrane potential accumulates over time and eventually spikes."""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Apply moderate input (0.3) repeatedly
    # With beta=0.9, membrane accumulates: 0.3, 0.3+0.27, 0.3+0.27+0.243, ...
    # Converges to 0.3/(1-0.9) = 3.0, so should spike
    moderate_input = float_to_fixed(0.3)

    results = await run_timesteps(dut, moderate_input, NUM_TIMESTEPS)

    spike_detected = False
    for i, (spike, membrane) in enumerate(results):
        if spike == 1:
            dut._log.info(f"Spike detected on timestep {i}")
            spike_detected = True
            break

    assert spike_detected, "Should have spiked with accumulated membrane potential"
    dut._log.info("Membrane accumulation test passed")


@cocotb.test()
async def test_lif_consecutive_spiking(dut):
    """Test that neuron can spike on consecutive timesteps with high sustained input."""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Apply input that will cause spike (1.2)
    input_val = float_to_fixed(1.2)

    results = await run_timesteps(dut, input_val, 3)

    # Timestep 0: membrane = 1.2, should spike
    spike1, mem1 = results[0]
    dut._log.info(
        f"Timestep 0 spike: {spike1}, membrane: {membrane_to_float(mem1):.4f}"
    )
    assert spike1 == 1, "Should spike on first timestep with input 1.2"

    # Timestep 1: membrane = beta * (1.2 - 1.0) + 1.2 = 0.9*0.2 + 1.2 = 1.38
    # Should spike again
    spike2, mem2 = results[1]
    dut._log.info(
        f"Timestep 1 spike: {spike2}, membrane: {membrane_to_float(mem2):.4f}"
    )

    # With continuous 1.2 input, neuron should keep spiking
    assert spike2 == 1, "Should spike again with sustained high input"

    dut._log.info("Consecutive spiking test passed")


@cocotb.test()
async def test_lif_reset_by_subtraction(dut):
    """Test that reset-by-subtraction prevents immediate re-spike.

    With input=0.55 and beta=0.9:
    - Timestep 0: membrane = 0.55 (no spike)
    - Timestep 1: membrane = 0.9*0.55 + 0.55 = 1.045 (spike)
    - Timestep 2: membrane = 0.9*(1.045 - 1.0) + 0.55 = 0.59 (no spike due to reset)
    This demonstrates that reset-by-subtraction reduces membrane by threshold.
    """

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Apply input that will spike after accumulation but not immediately re-spike
    input_val = float_to_fixed(0.55)

    results = await run_timesteps(dut, input_val, 3)

    # Timestep 0: membrane = 0.55, no spike
    spike1, mem1 = results[0]
    dut._log.info(
        f"Timestep 0 spike: {spike1}, membrane: {membrane_to_float(mem1):.6f}"
    )
    assert spike1 == 0, "Should not spike on timestep 0 (membrane = 0.55)"

    # Timestep 1: membrane = 0.9*0.55 + 0.55 = 1.045, spike
    spike2, mem2 = results[1]
    dut._log.info(
        f"Timestep 1 spike: {spike2}, membrane: {membrane_to_float(mem2):.6f}"
    )
    assert spike2 == 1, "Should spike on timestep 1 (membrane = 1.045)"

    # Timestep 2: membrane = 0.9*(1.045 - 1.0) + 0.55 = 0.59, no spike
    # This is the key assertion - reset-by-subtraction prevents immediate re-spike
    spike3, mem3 = results[2]
    dut._log.info(
        f"Timestep 2 spike: {spike3}, membrane: {membrane_to_float(mem3):.6f}"
    )
    assert spike3 == 0, (
        f"Should NOT spike on timestep 2 due to reset-by-subtraction "
        f"(membrane={membrane_to_float(mem3):.6f})"
    )

    dut._log.info("Reset by subtraction test passed")


@cocotb.test()
async def test_lif_multiple_inferences(dut):
    """Test that module can be cleared and run multiple times."""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # First inference with moderate input
    results1 = await run_timesteps(dut, float_to_fixed(0.3), NUM_TIMESTEPS)
    spikes1 = sum(s for s, _ in results1)
    dut._log.info(f"First inference: {spikes1} spikes")

    # Clear for new inference
    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0
    await RisingEdge(dut.clk)

    # Second inference with large input
    results2 = await run_timesteps(dut, float_to_fixed(1.5), 1)
    spike, _ = results2[0]
    assert spike == 1, "Should spike with large input on second inference"

    dut._log.info("Multiple inferences test passed")


@cocotb.test()
async def test_lif_negative_input(dut):
    """Test neuron behavior with negative input current."""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Apply negative input (-0.5)
    negative_input = float_to_fixed(-0.5)

    results = await run_timesteps(dut, negative_input, NUM_TIMESTEPS)

    # Should never spike with negative input
    for i, (spike, _) in enumerate(results):
        assert spike == 0, f"Should not spike with negative input at timestep {i}"

    dut._log.info("Negative input test passed")


@cocotb.test()
async def test_lif_beta_decay(dut):
    """Test that membrane decays with zero input."""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Apply zero input - membrane should stay at 0
    results = await run_timesteps(dut, 0, NUM_TIMESTEPS)

    # Verify no spike occurs throughout
    for i, (spike, membrane) in enumerate(results):
        assert spike == 0, f"Should not spike with zero input at timestep {i}"
        assert membrane == 0, f"Membrane should be 0 with zero input at timestep {i}"

    dut._log.info("Beta decay test passed")


@cocotb.test()
async def test_lif_enable_hold(dut):
    """Test that state holds when enable is not asserted."""

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Run one timestep with significant input
    dut.current.value = float_to_fixed(0.5)
    dut.enable.value = 1
    await RisingEdge(dut.clk)
    dut.enable.value = 0
    await RisingEdge(dut.clk)

    membrane_after_enable = int(dut.membrane_out.value)
    dut._log.info(
        f"Membrane after enable: {membrane_to_float(membrane_after_enable):.4f}"
    )

    # Wait several cycles without enable
    for _ in range(10):
        await RisingEdge(dut.clk)

    # Membrane should be unchanged
    membrane_after_wait = int(dut.membrane_out.value)
    assert membrane_after_wait == membrane_after_enable, (
        f"Membrane should hold when enable=0: was {membrane_to_float(membrane_after_enable):.4f}, "
        f"now {membrane_to_float(membrane_after_wait):.4f}"
    )

    dut._log.info("Enable hold test passed")


@cocotb.test()
async def test_lif_varying_current(dut):
    """Test LIF with varying current each timestep (like HL2 neurons receive)."""

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Simulate varying currents (like fc2 would produce)
    currents = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1]

    results = []
    for i, current in enumerate(currents):
        dut.current.value = float_to_fixed(current)
        dut.enable.value = 1
        await RisingEdge(dut.clk)
        dut.enable.value = 0
        await RisingEdge(dut.clk)

        spike = int(dut.spike_out.value)
        membrane = int(dut.membrane_out.value)
        results.append((spike, membrane))
        dut._log.info(
            f"Timestep {i}: current={current:.2f}, spike={spike}, "
            f"membrane={membrane_to_float(membrane):.4f}"
        )

    # Verify that spikes occurred (exact timing depends on accumulation)
    total_spikes = sum(s for s, _ in results)
    dut._log.info(f"Total spikes: {total_spikes}")
    assert total_spikes > 0, "Should have spiked at least once with increasing current"

    dut._log.info("Varying current test passed")
