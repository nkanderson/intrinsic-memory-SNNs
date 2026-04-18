"""
Cocotb tests for the neural_network top-level module.

Tests signal timing and dataflow through the complete SNN inference pipeline.
Does NOT test model accuracy - uses simple test weights to verify connectivity.

Test configuration uses reduced parameters for faster simulation:
- NUM_INPUTS = 4
- HL1_SIZE = 8 (reduced from 64)
- HL2_SIZE = 4 (reduced from 16)
- NUM_ACTIONS = 2
- NUM_TIMESTEPS = 4 (reduced from 30)
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

# Fixed-point format constants
TOTAL_BITS = 16
FRAC_BITS = 13
SCALE_FACTOR = 2**FRAC_BITS  # 8192
MAX_SIGNED = 2 ** (TOTAL_BITS - 1) - 1
MIN_SIGNED = -(2 ** (TOTAL_BITS - 1))
UNSIGNED_RANGE = 2**TOTAL_BITS

# Test configuration (must match Verilog parameters)
NUM_INPUTS = 4
HL1_SIZE = 8
HL2_SIZE = 4
NUM_ACTIONS = 2
NUM_TIMESTEPS = 4


def float_to_fixed(value: float) -> int:
    """Convert float to fixed-point (unsigned representation for cocotb)."""
    scaled = int(round(value * SCALE_FACTOR))
    if scaled > MAX_SIGNED:
        scaled = MAX_SIGNED
    elif scaled < MIN_SIGNED:
        scaled = MIN_SIGNED
    if scaled < 0:
        scaled = scaled + UNSIGNED_RANGE
    return scaled


def fixed_to_float(value: int) -> float:
    """Convert fixed-point (unsigned representation) to float."""
    if value >= 2 ** (TOTAL_BITS - 1):
        value = value - UNSIGNED_RANGE
    return value / SCALE_FACTOR


async def reset_dut(dut):
    """Apply reset sequence to the DUT."""
    dut.reset.value = 1
    dut.start.value = 0
    for i in range(NUM_INPUTS):
        dut.observations[i].value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


async def wait_for_done(dut, timeout_cycles=5000):
    """Wait for done signal with timeout."""
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            return True
    return False


@cocotb.test()
async def test_neural_network_reset(dut):
    """Test that reset properly initializes the neural network."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Apply reset
    dut.reset.value = 1
    dut.start.value = 0
    for i in range(NUM_INPUTS):
        dut.observations[i].value = 0

    await ClockCycles(dut.clk, 5)

    # Check reset state
    assert dut.done.value == 0, "done should be 0 after reset"

    dut._log.info("Reset test passed")


@cocotb.test()
async def test_neural_network_start_signal(dut):
    """Test that start signal initiates inference."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Provide observations
    test_obs = [0.1, -0.2, 0.3, -0.1]
    for i in range(NUM_INPUTS):
        dut.observations[i].value = float_to_fixed(test_obs[i])

    # Assert start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # done should still be 0 (inference in progress)
    await RisingEdge(dut.clk)
    assert dut.done.value == 0, "done should be 0 during inference"

    dut._log.info("Start signal test passed")


@cocotb.test()
async def test_neural_network_completes(dut):
    """Test that inference completes and done is asserted."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Provide observations
    test_obs = [0.5, -0.3, 0.2, 0.1]
    for i in range(NUM_INPUTS):
        dut.observations[i].value = float_to_fixed(test_obs[i])

    # Start inference
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    completed = await wait_for_done(dut, timeout_cycles=2000)
    assert completed, "Inference should complete within timeout"

    # done should be high
    assert dut.done.value == 1, "done should be 1 after inference completes"

    dut._log.info("Inference completion test passed")


@cocotb.test()
async def test_neural_network_selected_action_valid(dut):
    """Test that selected_action is valid after inference completes."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Provide observations
    test_obs = [0.5, -0.3, 0.2, 0.1]
    for i in range(NUM_INPUTS):
        dut.observations[i].value = float_to_fixed(test_obs[i])

    # Start inference
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    completed = await wait_for_done(dut)
    assert completed, "Inference should complete"

    # Read selected_action (should be 0 or 1, not X)
    action = int(dut.selected_action.value)

    dut._log.info(f"selected_action: {action}")
    assert action in [0, 1], f"Invalid action: {action}"

    # Just verify it's valid (not checking accuracy)
    # The value will depend on the test weights
    dut._log.info("Selected action valid test passed")


@cocotb.test()
async def test_neural_network_back_to_back(dut):
    """Test back-to-back inferences work correctly."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    for inference_num in range(3):
        dut._log.info(f"Starting inference {inference_num}")

        # Provide different observations each time
        test_obs = [0.1 * (inference_num + 1), -0.2, 0.3, -0.1 * inference_num]
        for i in range(NUM_INPUTS):
            dut.observations[i].value = float_to_fixed(test_obs[i])

        # Start inference
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0

        # Wait for done
        completed = await wait_for_done(dut)
        assert completed, f"Inference {inference_num} should complete"

        # Read selected action
        action = int(dut.selected_action.value)
        dut._log.info(f"Inference {inference_num} selected_action: {action}")

        # For back-to-back, start can be asserted while done is still high
        # (DONE_STATE handles this)

    dut._log.info("Back-to-back inference test passed")


@cocotb.test()
async def test_neural_network_done_stays_high(dut):
    """Test that done stays high until next start."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Provide observations and start
    for i in range(NUM_INPUTS):
        dut.observations[i].value = float_to_fixed(0.1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    completed = await wait_for_done(dut)
    assert completed, "Inference should complete"

    # done should stay high for multiple cycles
    for _ in range(10):
        await RisingEdge(dut.clk)
        assert dut.done.value == 1, "done should stay high until next start"

    dut._log.info("Done stays high test passed")


@cocotb.test()
async def test_neural_network_zero_input(dut):
    """Test inference with all-zero observations."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # All-zero observations
    for i in range(NUM_INPUTS):
        dut.observations[i].value = 0

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    completed = await wait_for_done(dut)
    assert completed, "Inference with zero input should complete"

    action = int(dut.selected_action.value)
    dut._log.info(f"Zero input selected_action: {action}")

    dut._log.info("Zero input test passed")


@cocotb.test()
async def test_neural_network_large_input(dut):
    """Test inference with large (saturating) observations."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Large positive and negative values
    test_obs = [1.9, -1.9, 1.5, -1.5]
    for i in range(NUM_INPUTS):
        dut.observations[i].value = float_to_fixed(test_obs[i])

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    completed = await wait_for_done(dut)
    assert completed, "Inference with large input should complete"

    action = int(dut.selected_action.value)
    dut._log.info(f"Large input selected_action: {action}")

    dut._log.info("Large input test passed")


@cocotb.test()
async def test_neural_network_timing_estimate(dut):
    """Measure inference timing in clock cycles."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Provide observations
    for i in range(NUM_INPUTS):
        dut.observations[i].value = float_to_fixed(0.5)

    # Count cycles
    cycle_count = 0

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    cycle_count += 1

    while dut.done.value == 0:
        await RisingEdge(dut.clk)
        cycle_count += 1
        if cycle_count > 5000:
            assert False, "Inference took too long"

    dut._log.info(f"Inference completed in {cycle_count} clock cycles")
    dut._log.info(f"At 100MHz, that's {cycle_count * 10} ns = {cycle_count * 0.01} us")

    dut._log.info("Timing estimate test passed")
