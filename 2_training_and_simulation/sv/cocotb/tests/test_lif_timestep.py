"""
Cocotb tests for the LIF (Leaky Integrate-and-Fire) neuron module
based on snnTorch's Leaky neuron class.
Tests membrane dynamics, spike generation, and reset behavior.

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
FRAC_BITS = 13   # Fractional bits

# Derived constants
SCALE_FACTOR = 2 ** FRAC_BITS           # 8192 for QS2.13
THRESHOLD = 2 ** FRAC_BITS              # 1.0 in fixed-point = 8192
MAX_SIGNED = 2 ** (TOTAL_BITS - 1) - 1  # 32767
MIN_SIGNED = -(2 ** (TOTAL_BITS - 1))   # -32768
UNSIGNED_RANGE = 2 ** TOTAL_BITS        # 65536

BETA = 0.9


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


async def reset_dut(dut):
    """Apply reset sequence to the DUT and wait for it to stabilize."""
    dut.reset.value = 1
    dut.start.value = 0
    dut.current.value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


# def fixed_to_float(value: int) -> float:
#     """Convert fixed-point to float."""
#     # Handle as signed
#     if value >= 2 ** (TOTAL_BITS - 1):
#         value = value - UNSIGNED_RANGE
#     return value / SCALE_FACTOR


@cocotb.test()
async def test_lif_reset(dut):
    """Test that reset properly initializes the LIF neuron."""

    # Start clock (10ns period = 100MHz)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Apply reset
    dut.reset.value = 1
    dut.start.value = 0
    dut.current.value = 0

    await ClockCycles(dut.clk, 5)

    # Check reset state
    assert dut.spike_out.value == 0, "spike_out should be 0 after reset"
    assert dut.done.value == 0, "done should be 0 after reset"

    dut._log.info("Reset test passed")


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
    dut.current.value = small_input
    dut.start.value = 1

    await RisingEdge(dut.clk)
    dut.start.value = 0  # Deassert after one cycle

    # Check all 30 timesteps - should never spike
    for i in range(30):
        await RisingEdge(dut.clk)
        spike = int(dut.spike_out.value)
        timestep = int(dut.timestep.value)
        dut._log.info(f"Timestep {timestep}: spike={spike}")
        assert spike == 0, f"Should not spike on timestep {timestep} with small input"

    # Wait one more cycle for done signal to be asserted
    await RisingEdge(dut.clk)
    assert dut.done.value == 1, "Done should be asserted after 30 timesteps"

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
    dut.current.value = large_input
    dut.start.value = 1

    await RisingEdge(dut.clk)
    dut.start.value = 0

    # First timestep should produce a spike
    await RisingEdge(dut.clk)
    spike = int(dut.spike_out.value)
    timestep = int(dut.timestep.value)
    dut._log.info(f"Timestep {timestep}: spike={spike}")
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
    dut.current.value = moderate_input
    dut.start.value = 1

    await RisingEdge(dut.clk)
    dut.start.value = 0

    spike_detected = False
    for i in range(30):
        await RisingEdge(dut.clk)
        spike = int(dut.spike_out.value)
        timestep = int(dut.timestep.value)
        if spike == 1:
            dut._log.info(f"Spike detected on timestep {timestep}")
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
    dut.current.value = input_val
    dut.start.value = 1

    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Timestep 0: membrane = 1.2, should spike
    await RisingEdge(dut.clk)
    first_spike = int(dut.spike_out.value)
    first_timestep = int(dut.timestep.value)
    dut._log.info(f"Timestep {first_timestep} spike: {first_spike}")
    assert first_spike == 1, "Should spike on first timestep with input 1.2"

    # Timestep 1: membrane = beta * (1.2 - 1.0) + 1.2 = 0.9*0.2 + 1.2 = 1.38
    # Should spike again
    await RisingEdge(dut.clk)
    second_spike = int(dut.spike_out.value)
    second_timestep = int(dut.timestep.value)
    dut._log.info(f"Timestep {second_timestep} spike: {second_spike}")

    # With continuous 1.2 input, neuron should keep spiking
    assert second_spike == 1, "Should spike again with sustained high input"

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
    dut.current.value = input_val
    dut.start.value = 1

    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Timestep 0: membrane = 0.55, no spike
    await RisingEdge(dut.clk)
    spike1 = int(dut.spike_out.value)
    timestep1 = int(dut.timestep.value)
    mem1 = int(dut.membrane_out.value)
    mem1_float = mem1 / 8192.0 if mem1 < (1 << 23) else (mem1 - (1 << 24)) / 8192.0
    dut._log.info(f"Timestep {timestep1} spike: {spike1}, membrane: {mem1_float:.6f}")
    assert spike1 == 0, "Should not spike on timestep 0 (membrane = 0.55)"

    # Timestep 1: membrane = 0.9*0.55 + 0.55 = 1.045, spike
    await RisingEdge(dut.clk)
    spike2 = int(dut.spike_out.value)
    timestep2 = int(dut.timestep.value)
    mem2 = int(dut.membrane_out.value)
    mem2_float = mem2 / 8192.0 if mem2 < (1 << 23) else (mem2 - (1 << 24)) / 8192.0
    dut._log.info(f"Timestep {timestep2} spike: {spike2}, membrane: {mem2_float:.6f}")
    assert spike2 == 1, "Should spike on timestep 1 (membrane = 1.045)"

    # Timestep 2: membrane = 0.9*(1.045 - 1.0) + 0.55 = 0.59, no spike
    # This is the key assertion - reset-by-subtraction prevents immediate re-spike
    await RisingEdge(dut.clk)
    spike3 = int(dut.spike_out.value)
    timestep3 = int(dut.timestep.value)
    mem3 = int(dut.membrane_out.value)
    mem3_float = mem3 / 8192.0 if mem3 < (1 << 23) else (mem3 - (1 << 24)) / 8192.0
    dut._log.info(f"Timestep {timestep3} spike: {spike3}, membrane: {mem3_float:.6f}")
    assert spike3 == 0, f"Should NOT spike on timestep 2 due to reset-by-subtraction (membrane={mem3_float:.6f})"

    dut._log.info("Reset by subtraction test passed")


@cocotb.test()
async def test_lif_multiple_starts(dut):
    """Test that module can be started multiple times."""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # First run with moderate input
    dut.current.value = float_to_fixed(0.3)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for completion
    while int(dut.done.value) == 0:
        await RisingEdge(dut.clk)

    dut._log.info("First run completed")

    # Wait a few cycles
    await ClockCycles(dut.clk, 5)

    # Second run with different input
    dut.current.value = float_to_fixed(1.5)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Should spike on first timestep
    await RisingEdge(dut.clk)
    spike = int(dut.spike_out.value)
    assert spike == 1, "Should spike with large input on second run"

    dut._log.info("Multiple starts test passed")


@cocotb.test()
async def test_lif_negative_input(dut):
    """Test neuron behavior with negative input current."""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Apply negative input (-0.5)
    negative_input = float_to_fixed(-0.5)
    dut.current.value = negative_input
    dut.start.value = 1

    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Should never spike with negative input
    for i in range(30):
        await RisingEdge(dut.clk)
        spike = int(dut.spike_out.value)
        timestep = int(dut.timestep.value)
        assert spike == 0, f"Should not spike with negative input at timestep {timestep}"

    dut._log.info("Negative input test passed")


@cocotb.test()
async def test_lif_beta_decay(dut):
    """Test that membrane decays with zero input."""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Apply zero input - membrane should decay toward 0 with beta=0.9
    # Starting from 0, it should stay at 0
    dut.current.value = 0
    dut.start.value = 1

    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Verify no spike occurs throughout
    for i in range(30):
        await RisingEdge(dut.clk)
        spike = int(dut.spike_out.value)
        membrane = int(dut.membrane_out.value)
        timestep = int(dut.timestep.value)

        # With zero input and starting from zero, membrane should stay at 0
        assert spike == 0, f"Should not spike with zero input at timestep {timestep}"
        assert membrane == 0, f"Membrane should be 0 with zero input at timestep {timestep}"

    dut._log.info("Beta decay test passed")
