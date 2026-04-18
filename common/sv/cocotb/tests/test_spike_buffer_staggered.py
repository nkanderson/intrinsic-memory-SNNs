"""
Cocotb tests for the spike_buffer_staggered module.

This is the legacy staggered-input version that collects spikes from neurons
that start at staggered times (one per cycle from linear_layer) and outputs
synchronized spike vectors.

For the simplified version used with synchronized HL1 processing, see test_spike_buffer.py.

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
NUM_NEURONS = 8  # Small for testing
NUM_TIMESTEPS = 4  # Small for testing


async def reset_dut(dut):
    """Apply reset sequence to the DUT."""
    dut.reset.value = 1
    dut.start.value = 0
    dut.spike_in.value = 0
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


def generate_spike_pattern(cycle: int, pattern: str = "all") -> int:
    """
    Generate spike_in value for a given cycle based on pattern.

    Patterns:
    - "all": All active neurons spike
    - "none": No neurons spike
    - "even": Even-indexed neurons spike
    - "odd": Odd-indexed neurons spike
    - "timestep": Neurons spike on even timesteps only
    """
    spike_in = 0
    for n in range(NUM_NEURONS):
        ts = get_neuron_timestep(cycle, n)
        if ts < 0:
            continue  # Neuron not active

        spike = False
        if pattern == "all":
            spike = True
        elif pattern == "none":
            spike = False
        elif pattern == "even":
            spike = n % 2 == 0
        elif pattern == "odd":
            spike = n % 2 == 1
        elif pattern == "timestep":
            spike = ts % 2 == 0

        if spike:
            spike_in |= 1 << n

    return spike_in


@cocotb.test()
async def test_spike_buffer_staggered_reset(dut):
    """Test that reset properly initializes the spike buffer."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Apply reset
    dut.reset.value = 1
    dut.start.value = 0
    dut.spike_in.value = 0

    await ClockCycles(dut.clk, 5)

    # Check reset state
    assert dut.timestep_ready.value == 0, "timestep_ready should be 0 after reset"
    assert dut.done.value == 0, "done should be 0 after reset"

    dut._log.info("Reset test passed")


@cocotb.test()
async def test_spike_buffer_staggered_all_spikes(dut):
    """Test collection with all neurons spiking on all timesteps."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Start collection - on this cycle, provide spike for neuron 0 timestep 0
    dut.spike_in.value = generate_spike_pattern(0, "all")
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Run through collection cycles
    # Total cycles needed: NUM_NEURONS + NUM_TIMESTEPS - 1
    total_cycles = NUM_NEURONS + NUM_TIMESTEPS - 1

    timesteps_seen = []
    for cycle in range(1, total_cycles + 5):  # Extra cycles to verify done
        # Generate spike pattern for this cycle
        dut.spike_in.value = generate_spike_pattern(cycle, "all")

        await RisingEdge(dut.clk)

        # Check if timestep is ready (only count new timesteps)
        if dut.timestep_ready.value == 1:
            ts = int(dut.timestep_out.value)
            spikes = int(dut.spikes_out.value)

            # Only record if this is a new timestep
            if len(timesteps_seen) == 0 or timesteps_seen[-1] != ts:
                timesteps_seen.append(ts)
                # NOTE: cycle starts from 1 in this loop, so it is the actual count
                # and not the zero-indexed cycle number
                dut._log.info(
                    f"Cycle {cycle}: timestep {ts} ready, spikes=0b{spikes:0{NUM_NEURONS}b}"
                )

                # With "all" pattern, all neurons should have spiked
                expected = (1 << NUM_NEURONS) - 1
                assert (
                    spikes == expected
                ), f"Expected all spikes (0x{expected:x}), got 0x{spikes:x}"

        if dut.done.value == 1:
            dut._log.info(f"Cycle {cycle}: done asserted")
            break

    # Verify we saw all timesteps
    assert (
        len(timesteps_seen) == NUM_TIMESTEPS
    ), f"Expected {NUM_TIMESTEPS} timesteps, saw {len(timesteps_seen)}"
    assert timesteps_seen == list(
        range(NUM_TIMESTEPS)
    ), f"Timesteps out of order: {timesteps_seen}"
    assert dut.done.value == 1, "done should be asserted"

    dut._log.info("All spikes test passed")


@cocotb.test()
async def test_spike_buffer_staggered_no_spikes(dut):
    """Test collection with no neurons spiking."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Start collection
    dut.spike_in.value = 0
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    total_cycles = NUM_NEURONS + NUM_TIMESTEPS - 1

    for cycle in range(1, total_cycles + 5):
        dut.spike_in.value = 0
        await RisingEdge(dut.clk)

        if dut.timestep_ready.value == 1:
            spikes = int(dut.spikes_out.value)
            ts = int(dut.timestep_out.value)
            dut._log.info(f"Cycle {cycle}: timestep {ts} ready, spikes=0x{spikes:x}")
            assert spikes == 0, f"Expected no spikes, got 0x{spikes:x}"

        if dut.done.value == 1:
            break

    assert dut.done.value == 1, "done should be asserted"
    dut._log.info("No spikes test passed")


@cocotb.test()
async def test_spike_buffer_staggered_even_neurons(dut):
    """Test collection with only even-indexed neurons spiking."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Start collection
    dut.spike_in.value = generate_spike_pattern(0, "even")
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    total_cycles = NUM_NEURONS + NUM_TIMESTEPS - 1

    # Expected pattern: neurons 0, 2, 4, 6 spike (for 8 neurons)
    expected = 0
    for n in range(0, NUM_NEURONS, 2):
        expected |= 1 << n

    for cycle in range(1, total_cycles + 5):
        dut.spike_in.value = generate_spike_pattern(cycle, "even")
        await RisingEdge(dut.clk)

        if dut.timestep_ready.value == 1:
            spikes = int(dut.spikes_out.value)
            ts = int(dut.timestep_out.value)
            dut._log.info(
                f"Cycle {cycle}: timestep {ts} ready, spikes=0b{spikes:0{NUM_NEURONS}b}"
            )
            assert (
                spikes == expected
            ), f"Expected 0b{expected:0{NUM_NEURONS}b}, got 0b{spikes:0{NUM_NEURONS}b}"

        if dut.done.value == 1:
            break

    assert dut.done.value == 1, "done should be asserted"
    dut._log.info("Even neurons test passed")


@cocotb.test()
async def test_spike_buffer_staggered_timestep_pattern(dut):
    """Test collection where neurons spike only on even timesteps."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Start collection
    dut.spike_in.value = generate_spike_pattern(0, "timestep")
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    total_cycles = NUM_NEURONS + NUM_TIMESTEPS - 1

    for cycle in range(1, total_cycles + 5):
        dut.spike_in.value = generate_spike_pattern(cycle, "timestep")
        await RisingEdge(dut.clk)

        if dut.timestep_ready.value == 1:
            spikes = int(dut.spikes_out.value)
            ts = int(dut.timestep_out.value)
            dut._log.info(
                f"Cycle {cycle}: timestep {ts} ready, spikes=0b{spikes:0{NUM_NEURONS}b}"
            )

            # On even timesteps, all neurons spike; on odd, none
            if ts % 2 == 0:
                expected = (1 << NUM_NEURONS) - 1
            else:
                expected = 0

            assert (
                spikes == expected
            ), f"Timestep {ts}: expected 0b{expected:0{NUM_NEURONS}b}, got 0b{spikes:0{NUM_NEURONS}b}"

        if dut.done.value == 1:
            break

    assert dut.done.value == 1, "done should be asserted"
    dut._log.info("Timestep pattern test passed")


@cocotb.test()
async def test_spike_buffer_staggered_timing(dut):
    """Test that timestep_ready asserts at the correct cycles."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Start collection
    dut.spike_in.value = generate_spike_pattern(0, "all")
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Track when each timestep becomes ready
    # Timestep T should be ready at cycle (T + NUM_NEURONS - 1)
    # But we started counting from 0 when start was asserted, so:
    # - Cycle 0 (start): neuron 0 timestep 0
    # - Cycle NUM_NEURONS-1: all neurons have done timestep 0, so timestep 0 ready

    ready_cycles = {}
    total_cycles = NUM_NEURONS + NUM_TIMESTEPS + 5

    for cycle in range(1, total_cycles):
        dut.spike_in.value = generate_spike_pattern(cycle, "all")
        await RisingEdge(dut.clk)

        if dut.timestep_ready.value == 1:
            ts = int(dut.timestep_out.value)
            if ts not in ready_cycles:
                ready_cycles[ts] = cycle - 1
                dut._log.info(f"Timestep {ts} first ready at cycle {cycle - 1}")

        if dut.done.value == 1:
            break

    # Verify timing: timestep T ready at cycle T + NUM_NEURONS - 1
    # Note: cycle count starts after the start cycle, so actual ready cycle
    # is (T + NUM_NEURONS - 1) relative to start
    for ts in range(NUM_TIMESTEPS):
        expected_cycle = (ts + NUM_NEURONS) - 1
        actual = ready_cycles.get(ts)
        assert actual is not None, f"Timestep {ts} was never ready"
        assert (
            actual == expected_cycle
        ), f"Timestep {ts}: expected ready at cycle {expected_cycle}, actual cycle {actual}"
        dut._log.info(
            f"Timestep {ts}: ready at cycle {actual} (expected {expected_cycle})"
        )

    dut._log.info("Timing test passed")


@cocotb.test()
async def test_spike_buffer_staggered_multiple_inferences(dut):
    """Test that the buffer can be restarted for multiple inferences."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    for inference in range(2):
        dut._log.info(f"Starting inference {inference}")

        # Use different patterns for each inference
        pattern = "all" if inference == 0 else "none"
        expected = (1 << NUM_NEURONS) - 1 if inference == 0 else 0

        # Start collection
        dut.spike_in.value = generate_spike_pattern(0, pattern)
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0

        total_cycles = NUM_NEURONS + NUM_TIMESTEPS + 5

        for cycle in range(1, total_cycles):
            dut.spike_in.value = generate_spike_pattern(cycle, pattern)
            await RisingEdge(dut.clk)

            if dut.timestep_ready.value == 1:
                spikes = int(dut.spikes_out.value)
                ts = int(dut.timestep_out.value)
                assert (
                    spikes == expected
                ), f"Inference {inference}, timestep {ts}: expected 0x{expected:x}, got 0x{spikes:x}"

            if dut.done.value == 1:
                dut._log.info(f"Inference {inference} complete at cycle {cycle}")
                break

        assert dut.done.value == 1, f"Inference {inference}: done should be asserted"

        # Wait a cycle before starting next inference
        await RisingEdge(dut.clk)

    dut._log.info("Multiple inferences test passed")
