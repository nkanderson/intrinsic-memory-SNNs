"""
Cocotb tests for the neuron_membrane_buffer module.

The neuron_membrane_buffer is a per-neuron buffer that stores membrane potentials
across all timesteps. Each LIF neuron in the final hidden layer gets its own
instance of this buffer.

Interface:
  - write_en: Write enable, assert when neuron outputs valid membrane
  - write_timestep: Which timestep slot to write to
  - membrane_in: Membrane value to store
  - read_timestep: Which timestep to read
  - membrane_out: Membrane value for requested timestep (combinational)
  - full: Asserts when all timesteps have been written
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

# Test parameters (should match module instantiation)
NUM_TIMESTEPS = 8
MEMBRANE_WIDTH = 24


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


async def reset_dut(dut):
    """Apply reset sequence to the DUT."""
    dut.reset.value = 1
    dut.clear.value = 0
    dut.write_en.value = 0
    dut.write_timestep.value = 0
    dut.membrane_in.value = 0
    dut.read_timestep.value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_neuron_membrane_buffer_reset(dut):
    """Test that reset properly initializes the buffer."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Check reset state
    assert dut.full.value == 0, "full should be 0 after reset"

    dut._log.info("Reset test passed")


@cocotb.test()
async def test_neuron_membrane_buffer_single_write(dut):
    """Test writing a single value and reading it back."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write a value to timestep 3
    test_value = 12345
    dut.write_en.value = 1
    dut.write_timestep.value = 3
    dut.membrane_in.value = test_value

    await RisingEdge(dut.clk)

    dut.write_en.value = 0

    # Read it back (combinational)
    dut.read_timestep.value = 3
    await ClockCycles(dut.clk, 1)  # Allow combinational logic to settle

    read_value = int(dut.membrane_out.value)
    assert read_value == test_value, f"Expected {test_value}, got {read_value}"

    # Should not be full yet (only 1 of NUM_TIMESTEPS written)
    assert dut.full.value == 0, "Should not be full after single write"

    dut._log.info("Single write test passed")


@cocotb.test()
async def test_neuron_membrane_buffer_sequential_writes(dut):
    """Test writing values sequentially to all timesteps."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write unique values to each timestep
    test_values = [1000 * (t + 1) for t in range(NUM_TIMESTEPS)]

    for t, val in enumerate(test_values):
        dut.write_en.value = 1
        dut.write_timestep.value = t
        dut.membrane_in.value = val
        await RisingEdge(dut.clk)

    dut.write_en.value = 0
    await RisingEdge(dut.clk)

    # Should be full now
    assert dut.full.value == 1, "Should be full after writing all timesteps"

    # Verify all values by reading back
    for t, expected_val in enumerate(test_values):
        dut.read_timestep.value = t
        await ClockCycles(dut.clk, 1)

        read_value = int(dut.membrane_out.value)
        assert (
            read_value == expected_val
        ), f"Timestep {t}: expected {expected_val}, got {read_value}"

    dut._log.info("Sequential writes test passed")


@cocotb.test()
async def test_neuron_membrane_buffer_negative_values(dut):
    """Test writing and reading negative membrane potentials."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write negative values
    test_values = [
        -100,
        -50000,
        -(1 << (MEMBRANE_WIDTH - 2)),  # Large negative
        0,
        100,
        50000,
        (1 << (MEMBRANE_WIDTH - 2)) - 1,  # Large positive
        -1,
    ]

    for t, val in enumerate(test_values):
        dut.write_en.value = 1
        dut.write_timestep.value = t
        dut.membrane_in.value = to_unsigned(val, MEMBRANE_WIDTH)
        await RisingEdge(dut.clk)

    dut.write_en.value = 0
    await RisingEdge(dut.clk)

    # Verify all values
    for t, expected_val in enumerate(test_values):
        dut.read_timestep.value = t
        await ClockCycles(dut.clk, 1)

        raw_value = int(dut.membrane_out.value)
        read_value = to_signed(raw_value, MEMBRANE_WIDTH)
        assert (
            read_value == expected_val
        ), f"Timestep {t}: expected {expected_val}, got {read_value}"

    dut._log.info("Negative values test passed")


@cocotb.test()
async def test_neuron_membrane_buffer_clear(dut):
    """Test that clear resets the written tracking without clearing storage."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Fill buffer
    for t in range(NUM_TIMESTEPS):
        dut.write_en.value = 1
        dut.write_timestep.value = t
        dut.membrane_in.value = 1000 * (t + 1)
        await RisingEdge(dut.clk)

    dut.write_en.value = 0
    await RisingEdge(dut.clk)

    assert dut.full.value == 1, "Should be full"

    # Clear
    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0
    await RisingEdge(dut.clk)

    # Should no longer be full
    assert dut.full.value == 0, "Should not be full after clear"

    dut._log.info("Clear test passed")


@cocotb.test()
async def test_neuron_membrane_buffer_random_access(dut):
    """Test random-access reads after all writes complete."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write all timesteps
    test_values = [12345 + t * 111 for t in range(NUM_TIMESTEPS)]

    for t, val in enumerate(test_values):
        dut.write_en.value = 1
        dut.write_timestep.value = t
        dut.membrane_in.value = val
        await RisingEdge(dut.clk)

    dut.write_en.value = 0

    # Random access pattern
    access_order = [5, 0, 7, 2, 1, 6, 3, 4]

    for t in access_order:
        dut.read_timestep.value = t
        await ClockCycles(dut.clk, 1)

        read_value = int(dut.membrane_out.value)
        expected = test_values[t]
        assert (
            read_value == expected
        ), f"Random read timestep {t}: expected {expected}, got {read_value}"

    dut._log.info("Random access test passed")


@cocotb.test()
async def test_neuron_membrane_buffer_overwrite(dut):
    """Test that overwriting a timestep works correctly."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write initial value to timestep 2
    dut.write_en.value = 1
    dut.write_timestep.value = 2
    dut.membrane_in.value = 1111
    await RisingEdge(dut.clk)

    # Overwrite with new value
    dut.membrane_in.value = 2222
    await RisingEdge(dut.clk)

    dut.write_en.value = 0

    # Read back - should be the new value
    dut.read_timestep.value = 2
    await ClockCycles(dut.clk, 1)

    read_value = int(dut.membrane_out.value)
    assert read_value == 2222, f"Expected 2222 after overwrite, got {read_value}"

    dut._log.info("Overwrite test passed")
