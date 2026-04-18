"""
Cocotb tests for the linear_layer module.
Tests fully connected layer computation with fixed-point arithmetic.

Fixed-point format: QS2.13 (16-bit signed, 2 integer bits, 13 fractional bits)
- Scale factor: 2^13 = 8192
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

# Fixed-point format constants
TOTAL_BITS = 16  # Total bits in QS2.13 format
FRAC_BITS = 13  # Fractional bits

# Derived constants
SCALE_FACTOR = 2**FRAC_BITS  # 8192 for QS2.13
MAX_SIGNED = 2 ** (TOTAL_BITS - 1) - 1  # 32767
MIN_SIGNED = -(2 ** (TOTAL_BITS - 1))  # -32768
UNSIGNED_RANGE = 2**TOTAL_BITS  # 65536

# Test configuration (must match Verilog parameters)
NUM_INPUTS = 4
NUM_OUTPUTS = 4


def float_to_fixed(value: float) -> int:
    """Convert float to fixed-point (TOTAL_BITS-bit signed, unsigned representation)."""
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


def fixed_to_float(value: int) -> float:
    """Convert fixed-point (unsigned representation) to float."""
    # Handle as signed
    if value >= 2 ** (TOTAL_BITS - 1):
        value = value - UNSIGNED_RANGE
    return value / SCALE_FACTOR


async def reset_dut(dut):
    """Apply reset sequence to the DUT and wait for it to stabilize."""
    dut.reset.value = 1
    dut.start.value = 0
    for i in range(NUM_INPUTS):
        dut.inputs[i].value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_linear_layer_reset(dut):
    """Test that reset properly initializes the linear layer."""

    # Start clock (10ns period = 100MHz)
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Apply reset
    dut.reset.value = 1
    dut.start.value = 0
    for i in range(NUM_INPUTS):
        dut.inputs[i].value = 0

    await ClockCycles(dut.clk, 5)

    # Check reset state
    assert dut.output_valid.value == 0, "output_valid should be 0 after reset"
    assert dut.done.value == 0, "done should be 0 after reset"

    dut.reset.value = 0
    await RisingEdge(dut.clk)

    dut._log.info("Reset test passed")


@cocotb.test()
async def test_linear_layer_identity_weights(dut):
    """Test with identity-like weights (diagonal 1.0, rest 0.0).

    With identity weights and zero bias:
    output[i] = input[i] (for i < NUM_INPUTS, assuming NUM_INPUTS == NUM_OUTPUTS)
    """

    # Start clock
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Set inputs: [0.5, -0.25, 1.0, -1.5]
    test_inputs = [0.5, -0.25, 1.0, -1.5]
    for i in range(NUM_INPUTS):
        dut.inputs[i].value = float_to_fixed(test_inputs[i])

    # Start computation
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait 2 cycles for registered outputs:
    # Cycle 1: inputs latched, state->COMPUTING, neuron_idx=0
    # Cycle 2: output_valid=1, output_current=result[0], output_idx=0
    await ClockCycles(dut.clk, 2)

    # Collect outputs - now outputs should be valid
    outputs = []
    for i in range(NUM_OUTPUTS):
        assert dut.output_valid.value == 1, f"output_valid should be 1 for output {i}"
        assert (
            int(dut.output_idx.value) == i
        ), f"output_idx should be {i}, got {int(dut.output_idx.value)}"

        output_val = int(dut.output_current.value)
        output_float = fixed_to_float(output_val)
        outputs.append(output_float)
        dut._log.info(f"Output {i}: {output_float:.6f} (expected {test_inputs[i]:.6f})")

        if i == NUM_OUTPUTS - 1:
            assert dut.done.value == 1, "done should be 1 on last output"
        else:
            await RisingEdge(dut.clk)  # Advance to next output

    # Verify outputs match inputs (identity weights)
    for i in range(NUM_OUTPUTS):
        expected = test_inputs[i] if i < len(test_inputs) else 0.0
        assert (
            abs(outputs[i] - expected) < 0.01
        ), f"Output {i}: expected {expected:.6f}, got {outputs[i]:.6f}"

    dut._log.info("Identity weights test passed")


@cocotb.test()
async def test_linear_layer_multiple_runs(dut):
    """Test that module can be started multiple times correctly."""

    # Start clock
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # First run
    test_inputs_1 = [0.5, 0.5, 0.5, 0.5]
    for i in range(NUM_INPUTS):
        dut.inputs[i].value = float_to_fixed(test_inputs_1[i])

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for completion: 1 cycle latency + NUM_OUTPUTS cycles for registered outputs
    await ClockCycles(dut.clk, 1 + NUM_OUTPUTS)

    # done should be asserted on the last output cycle
    assert dut.done.value == 1, "done should be 1 after first run"
    dut._log.info("First run completed")

    # Wait one cycle, then start second run
    await RisingEdge(dut.clk)

    # Second run with different inputs
    test_inputs_2 = [1.0, -1.0, 0.5, -0.5]
    for i in range(NUM_INPUTS):
        dut.inputs[i].value = float_to_fixed(test_inputs_2[i])

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for completion: 1 cycle latency + NUM_OUTPUTS cycles for registered outputs
    await ClockCycles(dut.clk, 1 + NUM_OUTPUTS)

    assert dut.done.value == 1, "done should be 1 after second run"
    dut._log.info("Multiple runs test passed")


@cocotb.test()
async def test_linear_layer_timing(dut):
    """Test that timing matches specification: NUM_OUTPUTS valid outputs."""

    # Start clock
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Set arbitrary inputs
    for i in range(NUM_INPUTS):
        dut.inputs[i].value = float_to_fixed(0.1 * (i + 1))

    # Start computation
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Count valid outputs until done
    valid_count = 0
    cycle_count = 0
    while dut.done.value != 1:
        await RisingEdge(dut.clk)
        cycle_count += 1
        if dut.output_valid.value == 1:
            valid_count += 1
            dut._log.info(
                f"Cycle {cycle_count}: output_idx={int(dut.output_idx.value)}, valid={valid_count}"
            )
        if cycle_count > NUM_OUTPUTS + 5:
            assert False, f"Timeout: done not asserted after {cycle_count} cycles"

    # Should have exactly NUM_OUTPUTS valid outputs
    assert (
        valid_count == NUM_OUTPUTS
    ), f"Expected {NUM_OUTPUTS} valid outputs, got {valid_count}"

    # done should be asserted with the last valid output
    assert (
        dut.output_valid.value == 1
    ), "output_valid should still be 1 when done asserts"

    dut._log.info(
        f"Timing test passed: {valid_count} valid outputs in {cycle_count} cycles"
    )
