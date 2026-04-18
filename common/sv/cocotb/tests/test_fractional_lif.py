"""
Cocotb tests for fractional_lif.sv.

Covers:
- reset/clear behavior
- basic spiking behavior
- fractional-vs-standard trajectory differences
- exact fixed-point golden model match to RTL arithmetic
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
from pathlib import Path

# Fixed-point / module constants for this test configuration
DATA_WIDTH = 16
MEMBRANE_WIDTH = 24
FRAC_BITS = 13
SCALE = 1 << FRAC_BITS
THRESHOLD = 1 << FRAC_BITS

# Fractional parameters for the baseline unit test profile (H=8).
#
# Important:
# - `test_fractional_lif_matches_fixed_point_golden_baseline` is intentionally tied to this
#   baseline profile.
# - `test_fractional_lif_matches_fixed_point_golden_hist64` covers the
#   H=64 variant profile used by the corresponding hist64 target.
HISTORY_LENGTH = 8
COEFF_FRAC_BITS = 15  # QU1.15
C_SCALED = 256  # Q8.8, C=1.0
INV_DENOM = 58982  # Q0.16 for 1/(1+0.111...) ~ 0.9

# Coefficient magnitudes |g_k| for alpha=0.5, k=1..7, QU1.15 integers
COEFFS_MAG = [0x4000, 0x1000, 0x0800, 0x0500, 0x0380, 0x02A0, 0x0210]

# Standard LIF helper constants (for comparison test only)
BETA_Q1_7 = 115


# -----------------------------
# Numeric helpers
# -----------------------------
def wrap_signed(val: int, bits: int) -> int:
    """Wrap integer to signed two's-complement range for given bit width."""
    mask = (1 << bits) - 1
    val &= mask
    if val >= (1 << (bits - 1)):
        val -= 1 << bits
    return val


def float_to_fixed_qs2_13(x: float) -> int:
    """Convert float to signed QS2.13 integer (returned as signed int)."""
    raw = int(round(x * SCALE))
    return wrap_signed(raw, DATA_WIDTH)


def signal_to_signed(value: int, bits: int) -> int:
    """Interpret a cocotb integer value as signed 'bits'-wide."""
    return wrap_signed(int(value), bits)


def load_coeffs_mem(mem_path: Path, expected_count: int) -> list[int]:
    coeffs = []
    with mem_path.open("r") as f:
        for line in f:
            data = line.split("//", 1)[0].strip()
            if not data:
                continue
            coeffs.append(int(data, 16))
    assert (
        len(coeffs) >= expected_count
    ), f"Coefficient file {mem_path} has {len(coeffs)} entries, expected >= {expected_count}"
    return coeffs[:expected_count]


# -----------------------------
# Golden models
# -----------------------------
class FractionalGolden:
    """Parameterized bit-accurate model matching fractional_lif.sv arithmetic."""

    def __init__(
        self,
        history_length: int,
        coeff_frac_bits: int,
        c_scaled: int,
        inv_denom: int,
        coeffs_mag: list[int],
    ):
        self.history_length = history_length
        self.coeff_frac_bits = coeff_frac_bits
        self.c_scaled = c_scaled
        self.inv_denom = inv_denom
        self.coeffs_mag = coeffs_mag
        self.mem = 0
        self.spike_prev = 0
        self.history = [0] * history_length
        self.ptr = 0  # points to oldest (next write location)

    def step(self, current_qs2_13: int):
        # Extend current to membrane width
        current_ext = wrap_signed(current_qs2_13, MEMBRANE_WIDTH)

        # history_sum = Σ |g_k| * V[n-k], k=1..H-1
        history_sum = 0  # RTL uses signed [47:0]
        for k in range(self.history_length - 1):
            if self.ptr >= (k + 1):
                hist_idx = self.ptr - (k + 1)
            else:
                hist_idx = self.ptr + self.history_length - (k + 1)

            hist_val = self.history[hist_idx]  # signed 24-bit
            coeff_mag = self.coeffs_mag[k]  # unsigned 16-bit
            product = coeff_mag * hist_val  # signed math
            history_sum = wrap_signed(history_sum + product, 48)

        reset_subtract = THRESHOLD if self.spike_prev else 0

        # scaled_history = (c_scaled * history_sum) >>> (8 + coeff_frac_bits)
        scaled_history = (self.c_scaled * history_sum) >> (8 + self.coeff_frac_bits)

        # numerator = current_extended + MEMBRANE_WIDTH'(scaled_history)
        scaled_hist_narrow = wrap_signed(scaled_history, MEMBRANE_WIDTH)
        numerator = wrap_signed(current_ext + scaled_hist_narrow, MEMBRANE_WIDTH)

        # scaled_result = numerator * signed({1'b0, INV_DENOM}), then >>>16
        scaled_result = wrap_signed(numerator * self.inv_denom, MEMBRANE_WIDTH + 16)
        div_result = scaled_result >> 16

        next_mem = wrap_signed(
            wrap_signed(div_result, MEMBRANE_WIDTH) - reset_subtract, MEMBRANE_WIDTH
        )
        next_spike = 1 if next_mem >= THRESHOLD else 0

        # Sequential update order matches RTL
        self.history[self.ptr] = self.mem
        self.ptr = (self.ptr + 1) % self.history_length
        self.mem = next_mem
        self.spike_prev = next_spike

        return next_spike, next_mem


class FractionalGoldenBaseline(FractionalGolden):
    """Baseline H=8 golden profile used by the default fractional_lif unit target."""

    def __init__(self):
        super().__init__(
            history_length=HISTORY_LENGTH,
            coeff_frac_bits=COEFF_FRAC_BITS,
            c_scaled=C_SCALED,
            inv_denom=INV_DENOM,
            coeffs_mag=COEFFS_MAG,
        )


def standard_lif_step(mem: int, spike_prev: int, current_qs2_13: int):
    """Simple fixed-point standard LIF step for behavioral comparison."""
    current_ext = wrap_signed(current_qs2_13, MEMBRANE_WIDTH)
    decay = wrap_signed(mem * BETA_Q1_7, 32) >> 7
    decay = wrap_signed(decay, MEMBRANE_WIDTH)
    reset_subtract = THRESHOLD if spike_prev else 0
    next_mem = wrap_signed(decay + current_ext - reset_subtract, MEMBRANE_WIDTH)
    next_spike = 1 if next_mem >= THRESHOLD else 0
    return next_spike, next_mem


# -----------------------------
# DUT helpers
# -----------------------------
async def reset_dut(dut):
    dut.reset.value = 1
    dut.clear.value = 0
    dut.enable.value = 0
    dut.current.value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


async def step_dut(dut, current_signed: int):
    """Apply one enabled timestep and return (spike, membrane_signed)."""
    dut.current.value = current_signed & ((1 << DATA_WIDTH) - 1)
    dut.enable.value = 1
    await RisingEdge(dut.clk)
    dut.enable.value = 0

    if int(dut.output_valid.value) == 0:
        max_wait_cycles = 1024
        for _ in range(max_wait_cycles):
            await RisingEdge(dut.clk)
            if int(dut.output_valid.value) == 1:
                break
        else:
            raise AssertionError(
                f"Timed out waiting for output_valid after {max_wait_cycles} cycles"
            )

    spike = int(dut.spike_out.value)
    membrane = signal_to_signed(int(dut.membrane_out.value), MEMBRANE_WIDTH)
    return spike, membrane


@cocotb.test()
async def test_fractional_lif_reset(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    assert int(dut.spike_out.value) == 0
    assert signal_to_signed(int(dut.membrane_out.value), MEMBRANE_WIDTH) == 0


@cocotb.test()
async def test_fractional_lif_clear(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Build non-zero membrane
    for _ in range(4):
        await step_dut(dut, float_to_fixed_qs2_13(0.6))

    assert signal_to_signed(int(dut.membrane_out.value), MEMBRANE_WIDTH) != 0

    # Clear state
    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.spike_out.value) == 0
    assert signal_to_signed(int(dut.membrane_out.value), MEMBRANE_WIDTH) == 0


@cocotb.test()
async def test_fractional_lif_no_spike_small_input(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    small = float_to_fixed_qs2_13(0.03)
    for t in range(20):
        spike, _ = await step_dut(dut, small)
        assert spike == 0, f"Unexpected spike at timestep {t}"


@cocotb.test()
async def test_fractional_lif_spike_large_input(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    big = float_to_fixed_qs2_13(1.5)
    spike, _ = await step_dut(dut, big)
    assert spike == 1, "Should spike on first step for large input"


@cocotb.test()
async def test_fractional_vs_standard_lif_pulse_response(dut):
    """Behavioral difference test using a short pulse then zeros."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # 3-step pulse then decay window
    stimulus = [float_to_fixed_qs2_13(0.7)] * 3 + [0] * 20

    frac_mem = []
    for i_val in stimulus:
        _, m = await step_dut(dut, i_val)
        frac_mem.append(m)

    std_mem = []
    m_std = 0
    s_prev = 0
    for i_val in stimulus:
        s_prev, m_std = standard_lif_step(m_std, s_prev, i_val)
        std_mem.append(m_std)

    # We only need to verify trajectories differ meaningfully, not exact threshold behavior.
    assert (
        frac_mem != std_mem
    ), "Fractional and standard trajectories are unexpectedly identical"

    l1_diff = sum(abs(a - b) for a, b in zip(frac_mem, std_mem))
    assert l1_diff > 32, f"Trajectory difference too small: L1={l1_diff}"


@cocotb.test()
async def test_fractional_lif_matches_fixed_point_golden_baseline(dut):
    """
    Bit-accurate check against baseline local fixed-point golden model.

    This test is intentionally for the baseline H=8 profile only. The local
    `FractionalGoldenBaseline` model uses fixed H=8 coefficient assumptions
    (`COEFFS_MAG`) and baseline constants from this file.
    """
    history_len = int(dut.HISTORY_LENGTH.value)
    if history_len != 8:
        cocotb.log.info(
            "Skipping baseline golden test "
            f"(HISTORY_LENGTH={history_len}, expected 8)."
        )
        return

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    golden = FractionalGoldenBaseline()

    stimulus_f = [0.35, 0.0, 0.8, 0.0, 0.0, 0.2, 1.2, 0.0, 0.5, 0.5, -0.1, 0.0]
    stimulus = [float_to_fixed_qs2_13(x) for x in stimulus_f]

    for t, i_val in enumerate(stimulus):
        exp_spk, exp_mem = golden.step(i_val)
        got_spk, got_mem = await step_dut(dut, i_val)

        assert (
            got_spk == exp_spk
        ), f"Spike mismatch at t={t}: got={got_spk}, exp={exp_spk}"
        assert (
            got_mem == exp_mem
        ), f"Mem mismatch at t={t}: got={got_mem}, exp={exp_mem}"


@cocotb.test()
async def test_fractional_lif_matches_fixed_point_golden_hist64(dut):
    """
    Bit-accurate check for the H=64 variant parameter profile.

    This test is intentionally for H=64 runs (e.g., `fractional_lif_hist64`) and
    uses the corresponding H=64 coefficient file plus DUT-provided constants.
    """
    history_len = int(dut.HISTORY_LENGTH.value)
    if history_len != 64:
        cocotb.log.info(
            f"Skipping hist64 equivalence test (HISTORY_LENGTH={history_len}, expected 64)."
        )
        return

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    coeff_file = Path("weights/fractional_order/gl_coefficients.mem")
    coeffs_mag = load_coeffs_mem(coeff_file, history_len - 1)

    golden = FractionalGolden(
        history_length=history_len,
        coeff_frac_bits=int(dut.COEFF_FRAC_BITS.value),
        c_scaled=int(dut.C_SCALED.value),
        inv_denom=int(dut.INV_DENOM.value),
        coeffs_mag=coeffs_mag,
    )

    stimulus_f = [
        0.02,
        0.05,
        0.10,
        0.25,
        0.40,
        -0.05,
        0.00,
        0.80,
        0.10,
        -0.10,
        0.30,
        0.00,
        0.60,
        -0.20,
        0.15,
        0.45,
        0.00,
        0.05,
        -0.05,
        0.35,
    ]
    stimulus = [float_to_fixed_qs2_13(x) for x in stimulus_f]

    for t, i_val in enumerate(stimulus):
        exp_spk, exp_mem = golden.step(i_val)
        got_spk, got_mem = await step_dut(dut, i_val)

        assert (
            got_spk == exp_spk
        ), f"[H64] Spike mismatch at t={t}: got={got_spk}, exp={exp_spk}"
        assert (
            got_mem == exp_mem
        ), f"[H64] Mem mismatch at t={t}: got={got_mem}, exp={exp_mem}"
