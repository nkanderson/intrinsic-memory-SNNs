"""
Cocotb tests for the membrane_buffer module.

The membrane_buffer collects membrane potentials from neurons that start at
staggered times (one per cycle from linear_layer) and outputs synchronized
membrane potential vectors.

Similar to spike_buffer but stores multi-bit membrane potentials instead of
single-bit spikes. Used between the final hidden layer and output linear layer.

Timing pattern (after start signal):
  Cycle 0: neuron 0 outputs timestep 0
  Cycle 1: neuron 0 outputs timestep 1, neuron 1 outputs timestep 0
  Cycle K: neuron N outputs timestep (K-N) for all N <= K where (K-N) < NUM_TIMESTEPS

Timestep T is complete at cycle (T + NUM_NEURONS - 1).
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

# Test parameters (should match module instantiation)
NUM_NEURONS = 4       # Small for testing
NUM_TIMESTEPS = 3     # Small for testing
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
    dut.start.value = 0
    for i in range(NUM_NEURONS):
        dut.membrane_in[i].value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


def get_neuron_timestep(cycle: int, neuron: int) -> int:
    """
    Calculate which timestep a neuron is at for a given cycle.
    Returns -1 if neuron hasn't started yet or has finished.
    """
    if cycle < neuron:
        return -1  # Neuron hasn't started
    timestep = cycle - neuron
    if timestep >= NUM_TIMESTEPS:
        return -1  # Neuron has finished
    return timestep


def generate_membrane_pattern(cycle: int, pattern: str = "identity") -> list:
    """
    Generate membrane_in values for a given cycle based on pattern.

    Patterns:
    - "identity": membrane = neuron_index * 1000 + timestep
    - "zero": all zeros
    - "negative": negative values based on neuron/timestep
    """
    membranes = [0] * NUM_NEURONS
    for n in range(NUM_NEURONS):
        ts = get_neuron_timestep(cycle, n)
        if ts < 0:
            membranes[n] = 0
            continue

        if pattern == "identity":
            # Unique value per neuron and timestep for easy verification
            membranes[n] = n * 1000 + ts
        elif pattern == "zero":
            membranes[n] = 0
        elif pattern == "negative":
            # Negative values
            membranes[n] = -(n * 100 + ts + 1)

    return membranes


async def set_membrane_inputs(dut, membranes: list):
    """Set membrane_in values on the DUT."""
    for i, val in enumerate(membranes):
        dut.membrane_in[i].value = to_unsigned(val, MEMBRANE_WIDTH)


@cocotb.test()
async def test_membrane_buffer_reset(dut):
    """Test that reset properly initializes the membrane buffer."""

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Apply reset
    dut.reset.value = 1
    dut.start.value = 0
    for i in range(NUM_NEURONS):
        dut.membrane_in[i].value = 0

    await ClockCycles(dut.clk, 5)

    # Check reset state
    assert dut.timestep_ready.value == 0, "timestep_ready should be 0 after reset"
    assert dut.done.value == 0, "done should be 0 after reset"

    dut._log.info("Reset test passed")


@cocotb.test()
async def test_membrane_buffer_identity_pattern(dut):
    """Test collection with identity pattern (unique value per neuron/timestep)."""

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Start collection
    membranes = generate_membrane_pattern(0, "identity")
    await set_membrane_inputs(dut, membranes)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Run through collection cycles
    total_cycles = NUM_NEURONS + NUM_TIMESTEPS - 1

    timesteps_seen = []
    for cycle in range(1, total_cycles + 5):
        membranes = generate_membrane_pattern(cycle, "identity")
        await set_membrane_inputs(dut, membranes)

        await RisingEdge(dut.clk)

        if dut.timestep_ready.value == 1:
            ts = int(dut.timestep_out.value)

            # Only record if this is a new timestep
            if len(timesteps_seen) == 0 or timesteps_seen[-1] != ts:
                timesteps_seen.append(ts)

                # Verify membrane values
                for n in range(NUM_NEURONS):
                    expected = n * 1000 + ts
                    actual = to_signed(int(dut.membranes_out[n].value), MEMBRANE_WIDTH)
                    assert actual == expected, \
                        f"Timestep {ts}, neuron {n}: expected {expected}, got {actual}"

                dut._log.info(f"Cycle {cycle}: timestep {ts} verified")

        if dut.done.value == 1:
            dut._log.info(f"Cycle {cycle}: done asserted")
            break

    # Verify we saw all timesteps
    assert len(timesteps_seen) == NUM_TIMESTEPS, \
        f"Expected {NUM_TIMESTEPS} timesteps, saw {len(timesteps_seen)}"
    assert timesteps_seen == list(range(NUM_TIMESTEPS)), \
        f"Timesteps out of order: {timesteps_seen}"
    assert dut.done.value == 1, "done should be asserted"

    dut._log.info("Identity pattern test passed")


@cocotb.test()
async def test_membrane_buffer_zero_pattern(dut):
    """Test collection with all zeros."""

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Start collection
    await set_membrane_inputs(dut, [0] * NUM_NEURONS)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    total_cycles = NUM_NEURONS + NUM_TIMESTEPS - 1

    for cycle in range(1, total_cycles + 5):
        await set_membrane_inputs(dut, [0] * NUM_NEURONS)
        await RisingEdge(dut.clk)

        if dut.timestep_ready.value == 1:
            ts = int(dut.timestep_out.value)
            for n in range(NUM_NEURONS):
                actual = to_signed(int(dut.membranes_out[n].value), MEMBRANE_WIDTH)
                assert actual == 0, f"Timestep {ts}, neuron {n}: expected 0, got {actual}"

        if dut.done.value == 1:
            break

    assert dut.done.value == 1, "done should be asserted"
    dut._log.info("Zero pattern test passed")


@cocotb.test()
async def test_membrane_buffer_negative_values(dut):
    """Test collection with negative membrane potentials."""

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Start collection
    membranes = generate_membrane_pattern(0, "negative")
    await set_membrane_inputs(dut, membranes)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    total_cycles = NUM_NEURONS + NUM_TIMESTEPS - 1

    timesteps_seen = []
    for cycle in range(1, total_cycles + 5):
        membranes = generate_membrane_pattern(cycle, "negative")
        await set_membrane_inputs(dut, membranes)

        await RisingEdge(dut.clk)

        if dut.timestep_ready.value == 1:
            ts = int(dut.timestep_out.value)

            if len(timesteps_seen) == 0 or timesteps_seen[-1] != ts:
                timesteps_seen.append(ts)

                for n in range(NUM_NEURONS):
                    expected = -(n * 100 + ts + 1)
                    actual = to_signed(int(dut.membranes_out[n].value), MEMBRANE_WIDTH)
                    assert actual == expected, \
                        f"Timestep {ts}, neuron {n}: expected {expected}, got {actual}"

                dut._log.info(f"Cycle {cycle}: timestep {ts} verified (negative values)")

        if dut.done.value == 1:
            break

    assert dut.done.value == 1, "done should be asserted"
    dut._log.info("Negative values test passed")


@cocotb.test()
async def test_membrane_buffer_timing(dut):
    """Test that timestep_ready asserts at the correct cycles."""

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Start collection
    membranes = generate_membrane_pattern(0, "identity")
    await set_membrane_inputs(dut, membranes)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    ready_cycles = {}
    total_cycles = NUM_NEURONS + NUM_TIMESTEPS + 5

    for cycle in range(1, total_cycles):
        membranes = generate_membrane_pattern(cycle, "identity")
        await set_membrane_inputs(dut, membranes)
        await RisingEdge(dut.clk)

        if dut.timestep_ready.value == 1:
            ts = int(dut.timestep_out.value)
            if ts not in ready_cycles:
                ready_cycles[ts] = cycle
                dut._log.info(f"Timestep {ts} first ready at cycle {cycle}")

        if dut.done.value == 1:
            break

    # Verify timing
    for ts in range(NUM_TIMESTEPS):
        expected_cycle = ts + NUM_NEURONS
        actual = ready_cycles.get(ts)
        assert actual is not None, f"Timestep {ts} was never ready"
        assert actual == expected_cycle, \
            f"Timestep {ts}: expected ready at cycle {expected_cycle}, actual cycle {actual}"
        dut._log.info(f"Timestep {ts}: ready at cycle {actual} (expected {expected_cycle})")

    dut._log.info("Timing test passed")


@cocotb.test()
async def test_membrane_buffer_multiple_inferences(dut):
    """Test that the buffer can be restarted for multiple inferences."""

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    for inference in range(2):
        dut._log.info(f"Starting inference {inference}")

        # Use different patterns for each inference
        pattern = "identity" if inference == 0 else "negative"

        # Start collection
        membranes = generate_membrane_pattern(0, pattern)
        await set_membrane_inputs(dut, membranes)
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0

        total_cycles = NUM_NEURONS + NUM_TIMESTEPS + 5
        timesteps_verified = 0

        for cycle in range(1, total_cycles):
            membranes = generate_membrane_pattern(cycle, pattern)
            await set_membrane_inputs(dut, membranes)
            await RisingEdge(dut.clk)

            if dut.timestep_ready.value == 1:
                ts = int(dut.timestep_out.value)
                if ts == timesteps_verified:
                    # Verify values
                    for n in range(NUM_NEURONS):
                        if pattern == "identity":
                            expected = n * 1000 + ts
                        else:
                            expected = -(n * 100 + ts + 1)
                        actual = to_signed(int(dut.membranes_out[n].value), MEMBRANE_WIDTH)
                        assert actual == expected, \
                            f"Inference {inference}, ts {ts}, neuron {n}: expected {expected}, got {actual}"
                    timesteps_verified += 1

            if dut.done.value == 1:
                dut._log.info(f"Inference {inference} complete at cycle {cycle}")
                break

        assert dut.done.value == 1, f"Inference {inference}: done should be asserted"
        assert timesteps_verified == NUM_TIMESTEPS, \
            f"Inference {inference}: expected {NUM_TIMESTEPS} timesteps, verified {timesteps_verified}"

        # Wait a cycle before starting next inference
        await RisingEdge(dut.clk)

    dut._log.info("Multiple inferences test passed")
