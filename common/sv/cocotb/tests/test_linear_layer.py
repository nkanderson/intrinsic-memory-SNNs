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

    Handshake-based: collect each output by waiting for `output_valid`,
    so the test is robust against changes in the layer's per-neuron
    cycle count (e.g. serial-MAC vs single-cycle-reduction).
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

    # Collect outputs as they stream out, sampling on each rising edge
    # and capturing whenever output_valid pulses high.
    outputs_by_idx: dict[int, float] = {}
    last_idx = NUM_OUTPUTS - 1
    timeout = 1 + NUM_OUTPUTS * (NUM_INPUTS + 1) + 10  # generous handshake budget

    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if int(dut.output_valid.value) == 1:
            idx = int(dut.output_idx.value)
            val = fixed_to_float(int(dut.output_current.value))
            outputs_by_idx[idx] = val
            dut._log.info(
                f"output_valid: idx={idx} value={val:.6f} (expected {test_inputs[idx]:.6f})"
            )
            if idx == last_idx:
                assert dut.done.value == 1, "done should be 1 on last output"
                break
    else:
        assert False, f"timed out before seeing output for idx={last_idx}"

    # Verify every output index was emitted exactly once and matches the
    # corresponding input (identity weights).
    assert sorted(outputs_by_idx.keys()) == list(range(NUM_OUTPUTS)), (
        f"missing/extra outputs: got {sorted(outputs_by_idx.keys())}"
    )
    for i in range(NUM_OUTPUTS):
        expected = test_inputs[i] if i < len(test_inputs) else 0.0
        actual = outputs_by_idx[i]
        assert abs(actual - expected) < 0.01, (
            f"Output {i}: expected {expected:.6f}, got {actual:.6f}"
        )

    dut._log.info("Identity weights test passed")


async def wait_for_done(dut, timeout_cycles: int):
    """Tick the clock until `done` is observed, asserting on timeout."""
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            return
    assert False, f"done not asserted after {timeout_cycles} cycles"


@cocotb.test()
async def test_linear_layer_multiple_runs(dut):
    """Test that module can be started multiple times correctly."""

    # Start clock
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    timeout = 1 + NUM_OUTPUTS * (NUM_INPUTS + 1) + 10

    # First run
    test_inputs_1 = [0.5, 0.5, 0.5, 0.5]
    for i in range(NUM_INPUTS):
        dut.inputs[i].value = float_to_fixed(test_inputs_1[i])

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    await wait_for_done(dut, timeout)
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

    await wait_for_done(dut, timeout)
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

    # Count valid outputs until done. Per the layer's contract, done and the
    # final output_valid pulse coincide on the same cycle, and total latency
    # is bounded by 1 + NUM_OUTPUTS * (NUM_INPUTS + 1).
    valid_count = 0
    cycle_count = 0
    timeout = 1 + NUM_OUTPUTS * (NUM_INPUTS + 1) + 10
    while dut.done.value != 1:
        await RisingEdge(dut.clk)
        cycle_count += 1
        if dut.output_valid.value == 1:
            valid_count += 1
            dut._log.info(
                f"Cycle {cycle_count}: output_idx={int(dut.output_idx.value)}, valid={valid_count}"
            )
        if cycle_count > timeout:
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
