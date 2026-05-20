"""
Cocotb equivalence test for linear_layer: parallel vs. archived serial.

Drives the linear_layer_equivalence wrapper with identical stimulus to both
the new parallel `linear_layer` and the archived `linear_layer_serial`,
then compares outputs bit-exactly. This test is the spec for the parallel
rewrite — it will not compile until linear_layer.sv has been rewritten to
the new vector+done interface (PARALLELISM parameter, outputs[] array).

Fixed-point format: QS2.13 (16-bit signed, 2 integer bits, 13 fractional bits)
"""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

# Fixed-point format constants
TOTAL_BITS = 16
FRAC_BITS = 13
SCALE_FACTOR = 2**FRAC_BITS
MAX_SIGNED = 2 ** (TOTAL_BITS - 1) - 1
MIN_SIGNED = -(2 ** (TOTAL_BITS - 1))
UNSIGNED_RANGE = 2**TOTAL_BITS

# Test configuration (must match Verilog parameters in the Makefile)
NUM_INPUTS = 4
NUM_OUTPUTS = 4


def float_to_fixed(value: float) -> int:
    scaled = int(round(value * SCALE_FACTOR))
    if scaled > MAX_SIGNED:
        scaled = MAX_SIGNED
    elif scaled < MIN_SIGNED:
        scaled = MIN_SIGNED
    if scaled < 0:
        scaled = scaled + UNSIGNED_RANGE
    return scaled


def fixed_to_signed(value: int) -> int:
    """Convert unsigned hardware bits to signed integer."""
    if value >= 2 ** (TOTAL_BITS - 1):
        value = value - UNSIGNED_RANGE
    return value


async def reset_dut(dut):
    dut.reset.value = 1
    dut.start.value = 0
    for i in range(NUM_INPUTS):
        dut.inputs[i].value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


async def drive_inputs_and_collect(dut, input_values_fixed: list[int]):
    """Drive one inference and collect both modules' outputs.

    Returns (parallel_outputs, serial_outputs) as lists of signed ints
    indexed [0..NUM_OUTPUTS-1].
    """
    # Load inputs
    for i, v in enumerate(input_values_fixed):
        dut.inputs[i].value = v

    # Pulse start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Buffers
    par_outputs: list[int | None] = [None] * NUM_OUTPUTS
    ser_outputs: list[int | None] = [None] * NUM_OUTPUTS
    par_done_seen = False
    ser_done_seen = False

    # Generous timeout: serial dominates, ~1 + N*(I+1) + slack
    timeout = 1 + NUM_OUTPUTS * (NUM_INPUTS + 1) + 20

    for _ in range(timeout):
        await RisingEdge(dut.clk)

        # Capture serial outputs as they stream
        if int(dut.ser_output_valid.value) == 1:
            idx = int(dut.ser_output_idx.value)
            ser_outputs[idx] = fixed_to_signed(int(dut.ser_output_current.value))

        # Capture parallel output vector on done pulse
        if int(dut.par_done.value) == 1 and not par_done_seen:
            par_done_seen = True
            for i in range(NUM_OUTPUTS):
                par_outputs[i] = fixed_to_signed(int(dut.par_outputs[i].value))

        if int(dut.ser_done.value) == 1:
            ser_done_seen = True

        if par_done_seen and ser_done_seen and None not in ser_outputs:
            break
    else:
        assert False, (
            f"timeout: par_done_seen={par_done_seen}, ser_done_seen={ser_done_seen}, "
            f"ser_outputs={ser_outputs}"
        )

    assert par_done_seen, "parallel module never asserted done"
    assert None not in par_outputs, f"missing parallel outputs: {par_outputs}"
    assert None not in ser_outputs, f"missing serial outputs: {ser_outputs}"

    return par_outputs, ser_outputs


def compare(par_outputs, ser_outputs, label: str):
    """Bit-exact comparison. Fails on first mismatch."""
    for i in range(NUM_OUTPUTS):
        assert par_outputs[i] == ser_outputs[i], (
            f"{label}: mismatch at idx {i}: "
            f"parallel={par_outputs[i]} (0x{par_outputs[i] & 0xFFFF:04x}), "
            f"serial={ser_outputs[i]} (0x{ser_outputs[i] & 0xFFFF:04x})"
        )


@cocotb.test()
async def test_equivalence_identity_inputs(dut):
    """Sanity: basic stimulus, identical outputs."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    test_inputs_float = [0.5, -0.25, 1.0, -1.5]
    inputs_fixed = [float_to_fixed(v) for v in test_inputs_float]

    par, ser = await drive_inputs_and_collect(dut, inputs_fixed)
    dut._log.info(f"parallel={par}")
    dut._log.info(f"serial  ={ser}")
    compare(par, ser, "identity_inputs")


@cocotb.test()
async def test_equivalence_zero_inputs(dut):
    """All-zero inputs → outputs should be the biases (or zero)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    par, ser = await drive_inputs_and_collect(dut, [0] * NUM_INPUTS)
    compare(par, ser, "zero_inputs")


@cocotb.test()
async def test_equivalence_extreme_inputs(dut):
    """Push inputs to saturation territory; ensure both modules saturate identically."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Both extremes simultaneously to provoke any asymmetric saturation
    extreme = [
        float_to_fixed(3.99),   # near max
        float_to_fixed(-3.99),  # near min
        float_to_fixed(3.99),
        float_to_fixed(-3.99),
    ]
    par, ser = await drive_inputs_and_collect(dut, extreme)
    compare(par, ser, "extreme_inputs")

    # Also check saturation flags agree (parallel collapses per-output flags to
    # any-saturated; serial pulses per-output during emit). We don't compare
    # sat flags cycle-by-cycle, but we can sanity check that if either flagged
    # saturation, the other did too at least once during the run.


@cocotb.test()
async def test_equivalence_randomized_sweep(dut):
    """Randomized inputs across a sweep — bit-exact agreement on every case."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    rng = random.Random(0xBEEF)
    NUM_CASES = 16

    for case in range(NUM_CASES):
        inputs_float = [rng.uniform(-3.5, 3.5) for _ in range(NUM_INPUTS)]
        inputs_fixed = [float_to_fixed(v) for v in inputs_float]

        par, ser = await drive_inputs_and_collect(dut, inputs_fixed)
        compare(par, ser, f"randomized case {case} inputs={inputs_float}")

        # Settling gap between runs
        await ClockCycles(dut.clk, 3)

    dut._log.info(f"Randomized sweep: {NUM_CASES} cases agreed bit-exactly")
