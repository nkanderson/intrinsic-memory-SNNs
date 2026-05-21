"""
Cocotb equivalence test for neural_network_fractional: parallel vs. archived serial.

Drives the neural_network_fractional_equivalence wrapper with identical
observations through both the new parallel `neural_network_fractional` and the
archived `neural_network_fractional_serial`, then asserts `selected_action`
matches exactly. Both networks share weights/biases and fractional_lif
configuration (including GL coefficient file), so identical inputs must
produce identical actions if the parallel rewrite preserves bit-exactness.
"""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

# Fixed-point format constants
TOTAL_BITS = 16
FRAC_BITS = 13  # fractional variant uses QS2.13
SCALE_FACTOR = 2**FRAC_BITS
MAX_SIGNED = 2 ** (TOTAL_BITS - 1) - 1
MIN_SIGNED = -(2 ** (TOTAL_BITS - 1))
UNSIGNED_RANGE = 2**TOTAL_BITS

NUM_INPUTS = 4
HL1_SIZE = 8
HL2_SIZE = 4
NUM_ACTIONS = 2
NUM_TIMESTEPS = 4


def float_to_fixed(value: float) -> int:
    scaled = int(round(value * SCALE_FACTOR))
    if scaled > MAX_SIGNED:
        scaled = MAX_SIGNED
    elif scaled < MIN_SIGNED:
        scaled = MIN_SIGNED
    if scaled < 0:
        scaled = scaled + UNSIGNED_RANGE
    return scaled


async def reset_dut(dut):
    dut.reset.value = 1
    dut.start.value = 0
    for i in range(NUM_INPUTS):
        dut.observations[i].value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


async def run_inference(dut, observations_fixed: list[int], timeout: int = 50000):
    """Drive one inference through both networks and return (par_action, ser_action, par_cyc, ser_cyc)."""
    for i, v in enumerate(observations_fixed):
        dut.observations[i].value = v

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    par_done_cycle: int | None = None
    par_action_at_done: int | None = None
    ser_done_cycle: int | None = None
    ser_action_at_done: int | None = None

    for cycle in range(timeout):
        await RisingEdge(dut.clk)

        if par_done_cycle is None and int(dut.par_done.value) == 1:
            par_done_cycle = cycle
            par_action_at_done = int(dut.par_selected_action.value)

        if ser_done_cycle is None and int(dut.ser_done.value) == 1:
            ser_done_cycle = cycle
            ser_action_at_done = int(dut.ser_selected_action.value)

        if par_done_cycle is not None and ser_done_cycle is not None:
            break
    else:
        assert False, (
            f"timeout: par_done_cycle={par_done_cycle}, ser_done_cycle={ser_done_cycle}"
        )

    assert par_done_cycle <= ser_done_cycle, (
        f"parallel should finish no later than serial; "
        f"par={par_done_cycle}, ser={ser_done_cycle}"
    )

    return par_action_at_done, ser_action_at_done, par_done_cycle, ser_done_cycle


@cocotb.test()
async def test_equivalence_zero_observations(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    par, ser, par_cyc, ser_cyc = await run_inference(dut, [0] * NUM_INPUTS)
    dut._log.info(f"par_action={par} (cycle {par_cyc}), ser_action={ser} (cycle {ser_cyc})")
    assert par == ser, f"action mismatch: parallel={par}, serial={ser}"


@cocotb.test()
async def test_equivalence_typical_observations(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    cases = [
        [0.1, 0.0, -0.05, 0.0],
        [-0.2, 0.5, 0.1, -0.3],
        [0.8, -0.4, -0.15, 0.6],
        [-0.5, -0.2, 0.05, 0.4],
    ]

    for ci, obs in enumerate(cases):
        obs_fixed = [float_to_fixed(v) for v in obs]
        par, ser, par_cyc, ser_cyc = await run_inference(dut, obs_fixed)
        dut._log.info(
            f"case {ci}: obs={obs} → par={par} (cyc {par_cyc}), ser={ser} (cyc {ser_cyc})"
        )
        assert par == ser, (
            f"case {ci} obs={obs}: action mismatch parallel={par}, serial={ser}"
        )
        await ClockCycles(dut.clk, 5)


@cocotb.test()
async def test_equivalence_randomized_sweep(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    rng = random.Random(0xFBAC)
    NUM_CASES = 8

    samples: list[tuple[int, int]] = []

    for case in range(NUM_CASES):
        obs = [rng.uniform(-1.0, 1.0) for _ in range(NUM_INPUTS)]
        obs_fixed = [float_to_fixed(v) for v in obs]

        par, ser, par_cyc, ser_cyc = await run_inference(dut, obs_fixed)
        assert par == ser, (
            f"randomized case {case} obs={obs}: action mismatch "
            f"parallel={par}, serial={ser}"
        )
        samples.append((par_cyc, ser_cyc))
        await ClockCycles(dut.clk, 5)

    avg_par = sum(p for p, _ in samples) / len(samples)
    avg_ser = sum(s for _, s in samples) / len(samples)
    dut._log.info(
        f"Randomized sweep: {NUM_CASES} cases agreed. "
        f"avg cycles parallel={avg_par:.1f}, serial={avg_ser:.1f}, "
        f"speedup ≈ {avg_ser / max(avg_par, 1):.2f}×"
    )
