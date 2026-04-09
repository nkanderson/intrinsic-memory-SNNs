"""
Cocotb tests for bitshift_lif.sv.

Covers:
- reset/clear behavior
- enable/hold timing behavior
- bit-accurate fixed-point golden equivalence for current DUT parameters
- compile-time SHIFT_MODE profile sanity checks (mode-gated)
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


# Fixed-point format constants
DATA_WIDTH = 16
MEMBRANE_WIDTH = 24
FRAC_BITS = 13
SCALE = 1 << FRAC_BITS
THRESHOLD = 1 << FRAC_BITS


def wrap_signed(value: int, bits: int) -> int:
    """Wrap integer to signed two's-complement range for bit width."""
    mask = (1 << bits) - 1
    value &= mask
    if value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


def float_to_fixed_qs2_13(x: float) -> int:
    """Convert float to signed QS2.13 integer."""
    raw = int(round(x * SCALE))
    return wrap_signed(raw, DATA_WIDTH)


def signal_to_signed(value: int, bits: int) -> int:
    """Interpret cocotb signal integer as signed two's-complement."""
    return wrap_signed(int(value), bits)


def get_shift_amount(
    idx: int, shift_mode: int, shift_width: int, custom_decay_rate: int
) -> int:
    """Mirror bitshift_lif.sv get_shift_amount_const behavior."""
    if shift_mode == 0:
        shift_amount = idx
    elif shift_mode == 1:
        shift_amount = 0 if idx == 0 else (idx + 1) // 2
    elif shift_mode == 2:
        if idx == 0:
            shift_amount = 0
        elif idx == 1:
            shift_amount = 1
        elif idx == 2:
            shift_amount = 3
        elif idx == 3:
            shift_amount = 4
        else:
            shift_amount = 5 + ((idx - 4) // custom_decay_rate)
    else:
        if idx == 0:
            shift_amount = 0
        elif idx == 1:
            shift_amount = 1
        elif idx == 2:
            shift_amount = 3
        elif idx == 3:
            shift_amount = 4
        else:
            rem = idx - 4
            shift_amount = 5
            found = False
            max_shift = (1 << shift_width) - 1
            for s in range(5, max_shift + 1):
                repeat_count = s - 2
                if (not found) and (rem < repeat_count):
                    shift_amount = s
                    found = True
                elif not found:
                    rem -= repeat_count

    max_shift = (1 << shift_width) - 1
    if shift_amount < 0:
        shift_amount = 0
    if shift_amount > max_shift:
        shift_amount = max_shift
    return shift_amount


class BitshiftGolden:
    """Bit-accurate golden model matching bitshift_lif.sv arithmetic."""

    def __init__(
        self,
        history_length: int,
        shift_mode: int,
        shift_width: int,
        custom_decay_rate: int,
        c_scaled: int,
        c_scaled_frac_bits: int,
        inv_denom: int,
        inv_denom_frac_bits: int,
    ):
        self.history_length = history_length
        self.shift_mode = shift_mode
        self.shift_width = shift_width
        self.custom_decay_rate = custom_decay_rate
        self.c_scaled = c_scaled
        self.c_scaled_frac_bits = c_scaled_frac_bits
        self.inv_denom = inv_denom
        self.inv_denom_frac_bits = inv_denom_frac_bits

        self.mem = 0
        self.spike_prev = 0
        self.history = [0] * history_length
        self.ptr = 0

    def step(self, current_qs2_13: int):
        current_ext = wrap_signed(current_qs2_13, MEMBRANE_WIDTH)

        history_sum = 0
        for k in range(self.history_length - 1):
            if self.ptr >= (k + 1):
                hist_idx = self.ptr - (k + 1)
            else:
                hist_idx = self.ptr + self.history_length - (k + 1)

            hist_val = self.history[hist_idx]
            shift_amt = get_shift_amount(
                idx=k + 1,
                shift_mode=self.shift_mode,
                shift_width=self.shift_width,
                custom_decay_rate=self.custom_decay_rate,
            )
            shifted_hist = hist_val >> shift_amt
            history_sum += shifted_hist

        reset_subtract = THRESHOLD if self.spike_prev else 0

        scaled_history = (self.c_scaled * history_sum) >> self.c_scaled_frac_bits
        numerator = current_ext - scaled_history

        scaled_result = numerator * self.inv_denom
        membrane_pre_reset = scaled_result >> self.inv_denom_frac_bits
        membrane_after_reset = membrane_pre_reset - reset_subtract

        next_mem = wrap_signed(membrane_after_reset, MEMBRANE_WIDTH)
        next_spike = 1 if next_mem >= THRESHOLD else 0

        self.history[self.ptr] = self.mem
        self.ptr = (self.ptr + 1) % self.history_length
        self.mem = next_mem
        self.spike_prev = next_spike

        return next_spike, next_mem


async def reset_dut(dut):
    dut.reset.value = 1
    dut.clear.value = 0
    dut.enable.value = 0
    dut.current.value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


async def step_dut(dut, current_signed: int):
    dut.current.value = current_signed & ((1 << DATA_WIDTH) - 1)
    dut.enable.value = 1
    await RisingEdge(dut.clk)
    dut.enable.value = 0
    await RisingEdge(dut.clk)

    spike = int(dut.spike_out.value)
    membrane = signal_to_signed(int(dut.membrane_out.value), MEMBRANE_WIDTH)
    return spike, membrane


@cocotb.test()
async def test_bitshift_lif_reset(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    assert int(dut.spike_out.value) == 0
    assert signal_to_signed(int(dut.membrane_out.value), MEMBRANE_WIDTH) == 0


@cocotb.test()
async def test_bitshift_lif_clear(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    for _ in range(4):
        await step_dut(dut, float_to_fixed_qs2_13(0.6))

    assert signal_to_signed(int(dut.membrane_out.value), MEMBRANE_WIDTH) != 0

    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.spike_out.value) == 0
    assert signal_to_signed(int(dut.membrane_out.value), MEMBRANE_WIDTH) == 0


@cocotb.test()
async def test_bitshift_lif_enable_hold_behavior(dut):
    """When enable=0, outputs/state should hold."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # One active step
    await step_dut(dut, float_to_fixed_qs2_13(0.8))
    mem_before = signal_to_signed(int(dut.membrane_out.value), MEMBRANE_WIDTH)
    spk_before = int(dut.spike_out.value)

    # Hold for several cycles
    dut.enable.value = 0
    dut.current.value = float_to_fixed_qs2_13(1.2) & ((1 << DATA_WIDTH) - 1)
    await ClockCycles(dut.clk, 4)

    mem_after = signal_to_signed(int(dut.membrane_out.value), MEMBRANE_WIDTH)
    spk_after = int(dut.spike_out.value)

    assert mem_after == mem_before, "membrane_out changed while enable=0"
    assert spk_after == spk_before, "spike_out changed while enable=0"


@cocotb.test()
async def test_bitshift_lif_matches_fixed_point_golden(dut):
    """Bit-accurate check against parameterized local fixed-point model."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    history_length = int(dut.HISTORY_LENGTH.value)
    shift_mode = int(dut.SHIFT_MODE.value)
    shift_width = int(dut.SHIFT_WIDTH.value)
    custom_decay_rate = int(dut.CUSTOM_DECAY_RATE.value)
    c_scaled = int(dut.C_SCALED.value)
    c_scaled_frac_bits = int(dut.C_SCALED_FRAC_BITS.value)
    inv_denom = int(dut.INV_DENOM.value)
    inv_denom_frac_bits = int(dut.INV_DENOM_FRAC_BITS.value)

    golden = BitshiftGolden(
        history_length=history_length,
        shift_mode=shift_mode,
        shift_width=shift_width,
        custom_decay_rate=custom_decay_rate,
        c_scaled=c_scaled,
        c_scaled_frac_bits=c_scaled_frac_bits,
        inv_denom=inv_denom,
        inv_denom_frac_bits=inv_denom_frac_bits,
    )

    stimulus_f = [
        0.30,
        0.10,
        0.00,
        0.75,
        0.20,
        -0.05,
        1.20,
        0.00,
        0.40,
        0.55,
        0.00,
        -0.10,
        0.95,
        0.15,
    ]
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
async def test_bitshift_lif_mode0_simple_profile(dut):
    """Mode-specific sanity check for SHIFT_MODE=0 (simple)."""
    shift_mode = int(dut.SHIFT_MODE.value)
    if shift_mode != 0:
        cocotb.log.info(
            f"Skipping mode0 profile check (SHIFT_MODE={shift_mode}, expected 0)."
        )
        return

    # For simple mode and idx>=1, sequence should increase by 1 each step.
    for idx in range(1, 12):
        a = get_shift_amount(idx, shift_mode=0, shift_width=8, custom_decay_rate=3)
        b = get_shift_amount(idx + 1, shift_mode=0, shift_width=8, custom_decay_rate=3)
        assert b == a + 1


@cocotb.test()
async def test_bitshift_lif_mode3_custom_slow_decay_profile(dut):
    """Mode-specific sanity check for SHIFT_MODE=3 (custom_slow_decay)."""
    shift_mode = int(dut.SHIFT_MODE.value)
    if shift_mode != 3:
        cocotb.log.info(
            f"Skipping mode3 profile check (SHIFT_MODE={shift_mode}, expected 3)."
        )
        return

    # Expected prefix: [0,1,3,4,5,5,5,6,6,6,6,7,7,7,7,7]
    expected = [0, 1, 3, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7]
    observed = [
        get_shift_amount(idx=i, shift_mode=3, shift_width=8, custom_decay_rate=3)
        for i in range(len(expected))
    ]
    assert (
        observed == expected
    ), f"Mode3 sequence mismatch: got={observed}, exp={expected}"
