"""
Cocotb tests for the spike_buffer module.

Tests a simple spike_buffer that assumes synchronized HL1
processing where all neurons produce a complete spike vector each cycle.
The neural_network module manages timestep synchronization.

For tests of the legacy staggered-input version, see test_spike_buffer_staggered.py.

Interface:
  - write_en + write_timestep: Store spike vector at specified timestep
  - read_timestep: Combinational read of spike vector at specified timestep
  - clear: Reset all storage for new inference
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import random

# Test parameters (should match module instantiation)
NUM_NEURONS = 8  # Small for testing
NUM_TIMESTEPS = 4  # Small for testing


async def reset_dut(dut):
    """Apply reset sequence to the DUT."""
    dut.reset.value = 1
    dut.clear.value = 0
    dut.write_en.value = 0
    dut.write_timestep.value = 0
    dut.spike_in.value = 0
    dut.read_timestep.value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_spike_buffer_reset(dut):
    """Test that reset properly initializes the spike buffer."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Check that all timesteps read as 0 after reset
    for t in range(NUM_TIMESTEPS):
        dut.read_timestep.value = t
        await RisingEdge(dut.clk)
        spikes = int(dut.spikes_out.value)
        assert spikes == 0, f"Timestep {t} should be 0 after reset, got 0x{spikes:x}"

    dut._log.info("Reset test passed")


@cocotb.test()
async def test_spike_buffer_write_read(dut):
    """Test basic write and read operations."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write a unique pattern to each timestep
    patterns = []
    for t in range(NUM_TIMESTEPS):
        pattern = (1 << t) | (0xAA if t % 2 == 0 else 0x55)
        pattern &= (1 << NUM_NEURONS) - 1  # Mask to NUM_NEURONS bits
        patterns.append(pattern)

        dut.write_timestep.value = t
        dut.spike_in.value = pattern
        dut.write_en.value = 1
        await RisingEdge(dut.clk)

    dut.write_en.value = 0

    # Read back and verify each timestep
    for t in range(NUM_TIMESTEPS):
        dut.read_timestep.value = t
        await RisingEdge(dut.clk)  # Wait for combinational read to settle
        spikes = int(dut.spikes_out.value)
        expected = patterns[t]
        assert spikes == expected, (
            f"Timestep {t}: expected 0b{expected:0{NUM_NEURONS}b}, "
            f"got 0b{spikes:0{NUM_NEURONS}b}"
        )
        dut._log.info(
            f"Timestep {t}: read 0b{spikes:0{NUM_NEURONS}b} (expected 0b{expected:0{NUM_NEURONS}b})"
        )

    dut._log.info("Write/read test passed")


@cocotb.test()
async def test_spike_buffer_all_spikes(dut):
    """Test storing all-ones spike vectors."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    all_ones = (1 << NUM_NEURONS) - 1

    # Write all-ones to each timestep
    for t in range(NUM_TIMESTEPS):
        dut.write_timestep.value = t
        dut.spike_in.value = all_ones
        dut.write_en.value = 1
        await RisingEdge(dut.clk)

    dut.write_en.value = 0

    # Verify all timesteps contain all-ones
    for t in range(NUM_TIMESTEPS):
        dut.read_timestep.value = t
        await RisingEdge(dut.clk)
        spikes = int(dut.spikes_out.value)
        assert spikes == all_ones, f"Timestep {t}: expected all ones, got 0x{spikes:x}"

    dut._log.info("All spikes test passed")


@cocotb.test()
async def test_spike_buffer_no_spikes(dut):
    """Test storing all-zeros spike vectors."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write all-zeros to each timestep
    for t in range(NUM_TIMESTEPS):
        dut.write_timestep.value = t
        dut.spike_in.value = 0
        dut.write_en.value = 1
        await RisingEdge(dut.clk)

    dut.write_en.value = 0

    # Verify all timesteps contain zeros
    for t in range(NUM_TIMESTEPS):
        dut.read_timestep.value = t
        await RisingEdge(dut.clk)
        spikes = int(dut.spikes_out.value)
        assert spikes == 0, f"Timestep {t}: expected 0, got 0x{spikes:x}"

    dut._log.info("No spikes test passed")


@cocotb.test()
async def test_spike_buffer_clear(dut):
    """Test that clear properly resets all storage."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write non-zero patterns to all timesteps
    for t in range(NUM_TIMESTEPS):
        dut.write_timestep.value = t
        dut.spike_in.value = 0xFF & ((1 << NUM_NEURONS) - 1)
        dut.write_en.value = 1
        await RisingEdge(dut.clk)

    dut.write_en.value = 0

    # Verify data was written
    dut.read_timestep.value = 0
    await RisingEdge(dut.clk)
    assert int(dut.spikes_out.value) != 0, "Data should be written before clear"

    # Assert clear
    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0
    await RisingEdge(dut.clk)

    # Verify all timesteps are cleared
    for t in range(NUM_TIMESTEPS):
        dut.read_timestep.value = t
        await RisingEdge(dut.clk)
        spikes = int(dut.spikes_out.value)
        assert spikes == 0, f"Timestep {t} should be 0 after clear, got 0x{spikes:x}"

    dut._log.info("Clear test passed")


@cocotb.test()
async def test_spike_buffer_overwrite(dut):
    """Test that writing to the same timestep overwrites previous data."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    timestep = 2
    first_pattern = 0xAA & ((1 << NUM_NEURONS) - 1)
    second_pattern = 0x55 & ((1 << NUM_NEURONS) - 1)

    # Write first pattern
    dut.write_timestep.value = timestep
    dut.spike_in.value = first_pattern
    dut.write_en.value = 1
    await RisingEdge(dut.clk)
    dut.write_en.value = 0

    # Verify first pattern
    dut.read_timestep.value = timestep
    await RisingEdge(dut.clk)
    assert int(dut.spikes_out.value) == first_pattern

    # Overwrite with second pattern
    dut.spike_in.value = second_pattern
    dut.write_en.value = 1
    await RisingEdge(dut.clk)
    dut.write_en.value = 0

    # Verify second pattern
    await RisingEdge(dut.clk)
    spikes = int(dut.spikes_out.value)
    assert spikes == second_pattern, (
        f"Expected 0b{second_pattern:0{NUM_NEURONS}b}, "
        f"got 0b{spikes:0{NUM_NEURONS}b}"
    )

    dut._log.info("Overwrite test passed")


@cocotb.test()
async def test_spike_buffer_combinational_read(dut):
    """Test that reads are combinational (no cycle delay)."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write patterns to all timesteps
    patterns = [random.randint(0, (1 << NUM_NEURONS) - 1) for _ in range(NUM_TIMESTEPS)]
    for t, pattern in enumerate(patterns):
        dut.write_timestep.value = t
        dut.spike_in.value = pattern
        dut.write_en.value = 1
        await RisingEdge(dut.clk)

    dut.write_en.value = 0

    # Rapidly switch read_timestep and check combinational output
    # (without waiting for clock edge after changing read_timestep)
    for _ in range(10):
        t = random.randint(0, NUM_TIMESTEPS - 1)
        dut.read_timestep.value = t
        # Small delay for combinational logic to settle (not a clock edge)
        await cocotb.triggers.Timer(1, unit="ns")
        spikes = int(dut.spikes_out.value)
        expected = patterns[t]
        assert spikes == expected, (
            f"Combinational read failed for timestep {t}: "
            f"expected 0x{expected:x}, got 0x{spikes:x}"
        )

    dut._log.info("Combinational read test passed")


@cocotb.test()
async def test_spike_buffer_multiple_inferences(dut):
    """Test that the buffer can be cleared and reused for multiple inferences."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    for inference in range(3):
        dut._log.info(f"Starting inference {inference}")

        # Use different pattern for each inference
        base_pattern = (inference + 1) * 0x11 & ((1 << NUM_NEURONS) - 1)

        # Write patterns
        for t in range(NUM_TIMESTEPS):
            pattern = (base_pattern + t) & ((1 << NUM_NEURONS) - 1)
            dut.write_timestep.value = t
            dut.spike_in.value = pattern
            dut.write_en.value = 1
            await RisingEdge(dut.clk)

        dut.write_en.value = 0

        # Verify patterns
        for t in range(NUM_TIMESTEPS):
            expected = (base_pattern + t) & ((1 << NUM_NEURONS) - 1)
            dut.read_timestep.value = t
            await RisingEdge(dut.clk)
            spikes = int(dut.spikes_out.value)
            assert spikes == expected, (
                f"Inference {inference}, timestep {t}: "
                f"expected 0x{expected:x}, got 0x{spikes:x}"
            )

        # Clear for next inference
        dut.clear.value = 1
        await RisingEdge(dut.clk)
        dut.clear.value = 0
        await RisingEdge(dut.clk)

        dut._log.info(f"Inference {inference} complete")

    dut._log.info("Multiple inferences test passed")


@cocotb.test()
async def test_spike_buffer_write_while_read(dut):
    """Test simultaneous write and read to different timesteps."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write initial data to timestep 0
    dut.write_timestep.value = 0
    dut.spike_in.value = 0xAA & ((1 << NUM_NEURONS) - 1)
    dut.write_en.value = 1
    await RisingEdge(dut.clk)
    dut.write_en.value = 0

    # Now write to timestep 1 while reading from timestep 0
    dut.write_timestep.value = 1
    dut.spike_in.value = 0x55 & ((1 << NUM_NEURONS) - 1)
    dut.write_en.value = 1
    dut.read_timestep.value = 0
    await RisingEdge(dut.clk)
    dut.write_en.value = 0

    # Verify read got the correct value
    spikes = int(dut.spikes_out.value)
    expected = 0xAA & ((1 << NUM_NEURONS) - 1)
    assert (
        spikes == expected
    ), f"Read during write failed: expected 0x{expected:x}, got 0x{spikes:x}"

    # Verify write succeeded
    dut.read_timestep.value = 1
    await RisingEdge(dut.clk)
    spikes = int(dut.spikes_out.value)
    expected = 0x55 & ((1 << NUM_NEURONS) - 1)
    assert (
        spikes == expected
    ), f"Write during read failed: expected 0x{expected:x}, got 0x{spikes:x}"

    dut._log.info("Write while read test passed")
