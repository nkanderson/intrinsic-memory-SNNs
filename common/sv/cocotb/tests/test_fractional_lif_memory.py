import csv
from pathlib import Path
from statistics import mean

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge


DATA_WIDTH = 16
MEMBRANE_WIDTH = 24
FRAC_BITS = 13


def _wrap_signed(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    value &= mask
    if value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


def float_to_qs2_13(value: float) -> int:
    return _wrap_signed(int(round(value * (1 << FRAC_BITS))), DATA_WIDTH)


def signal_to_signed(value: int, bits: int) -> int:
    return _wrap_signed(int(value), bits)


class FractionalMemoryTester:
    def __init__(self, dut):
        self.dut = dut
        self.reset_traces()

    def reset_traces(self):
        self.spike_steps = []
        self.current_trace = []
        self.membrane_trace = []
        self.step_trace = []
        self.step_count = 0

    async def record_step(self, current_signed: int):
        self.dut.current.value = current_signed & ((1 << DATA_WIDTH) - 1)
        self.dut.enable.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.enable.value = 0

        if int(self.dut.output_valid.value) == 0:
            max_wait_cycles = 1024
            for _ in range(max_wait_cycles):
                await RisingEdge(self.dut.clk)
                if int(self.dut.output_valid.value) == 1:
                    break
            else:
                raise AssertionError(
                    f"Timed out waiting for output_valid after {max_wait_cycles} cycles"
                )

        spike = int(self.dut.spike_out.value)
        membrane = signal_to_signed(int(self.dut.membrane_out.value), MEMBRANE_WIDTH)

        if spike:
            self.spike_steps.append(self.step_count)

        self.current_trace.append(current_signed)
        self.membrane_trace.append(membrane)
        self.step_trace.append(self.step_count)
        self.step_count += 1

        return spike, membrane

    async def run_constant_current(self, current_signed: int, steps: int):
        for _ in range(steps):
            await self.record_step(current_signed)

    async def run_profile(self, profile):
        for current_signed, steps in profile:
            for _ in range(steps):
                await self.record_step(current_signed)

    def inter_spike_intervals(self):
        if len(self.spike_steps) < 2:
            return []
        return [
            self.spike_steps[i] - self.spike_steps[i - 1]
            for i in range(1, len(self.spike_steps))
        ]

    def inter_spike_intervals_in_window(self, start_step: int, end_step: int):
        spikes = [s for s in self.spike_steps if start_step <= s < end_step]
        if len(spikes) < 2:
            return []
        return [spikes[i] - spikes[i - 1] for i in range(1, len(spikes))]

    def export_spike_cycle_csv(self, filename: Path):
        if not self.spike_steps:
            return

        filename.parent.mkdir(parents=True, exist_ok=True)

        cycle_ranges = []
        first_nonzero_current_idx = 0
        for idx, current in enumerate(self.current_trace):
            if current != 0:
                first_nonzero_current_idx = idx
                break

        cycle_ranges.append((first_nonzero_current_idx, self.spike_steps[0]))
        for idx in range(1, len(self.spike_steps)):
            cycle_ranges.append((self.spike_steps[idx - 1] + 1, self.spike_steps[idx]))

        max_len = max((end - start + 1) for start, end in cycle_ranges)

        with filename.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            header = [f"membrane_potential_{i + 1}" for i in range(max_len)]
            writer.writerow(["cycle_num", "spike_step", "cycle_length"] + header)

            for cycle_idx, (start_idx, end_idx) in enumerate(cycle_ranges, start=1):
                values = self.membrane_trace[start_idx : end_idx + 1]
                cycle_len = len(values)
                padded = values + [""] * (max_len - cycle_len)
                writer.writerow([cycle_idx, end_idx, cycle_len] + padded)


async def reset_dut(dut):
    dut.reset.value = 1
    dut.clear.value = 0
    dut.enable.value = 0
    dut.current.value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


async def sweep_best_constant_current(
    dut,
    tester: FractionalMemoryTester,
    candidate_currents,
    run_steps=3000,
    min_spikes_required=14,
):
    best = None

    for current_float in candidate_currents:
        await reset_dut(dut)
        tester.reset_traces()

        current_fixed = float_to_qs2_13(current_float)
        await tester.run_constant_current(current_fixed, run_steps)

        intervals = tester.inter_spike_intervals()
        spike_count = len(tester.spike_steps)
        dut._log.info(
            f"I={current_float:.3f} -> spikes={spike_count}, intervals={len(intervals)}"
        )

        if spike_count < min_spikes_required or len(intervals) < 10:
            continue

        early = intervals[:6]
        late = intervals[-6:]
        if len(early) < 4 or len(late) < 4:
            continue

        early_mean = mean(early)
        late_mean = mean(late)
        if late_mean <= 0:
            continue

        gain = early_mean / late_mean
        acceleration = early_mean - late_mean
        dut._log.info(
            f"I={current_float:.3f}: early_mean={early_mean:.2f}, late_mean={late_mean:.2f}, "
            f"gain={gain:.3f}, acceleration={acceleration:.2f}"
        )

        if best is None or gain > best["gain"]:
            best = {
                "current_float": current_float,
                "current_fixed": current_fixed,
                "gain": gain,
                "acceleration": acceleration,
                "early_mean": early_mean,
                "late_mean": late_mean,
                "spike_count": spike_count,
                "spike_steps": list(tester.spike_steps),
                "membrane_trace": list(tester.membrane_trace),
                "current_trace": list(tester.current_trace),
                "intervals": list(intervals),
            }

    return best


@cocotb.test()
async def test_constant_current_memory_effect(dut):
    """Case 1: constant current should yield increasing spike rate over time."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    tester = FractionalMemoryTester(dut)

    candidate_currents = [
        0.160,
        0.170,
        0.180,
        0.190,
        0.200,
        0.210,
        0.220,
        0.230,
        0.240,
        0.250,
        0.260,
        0.280,
    ]

    best = await sweep_best_constant_current(
        dut=dut,
        tester=tester,
        candidate_currents=candidate_currents,
        run_steps=3000,
        min_spikes_required=14,
    )

    assert best is not None, (
        "No candidate current produced enough spikes for memory analysis. "
        "Try increasing HISTORY_LENGTH or widening current range."
    )

    dut._log.info(
        "Case 1 best: "
        f"I={best['current_float']:.3f} (fixed={best['current_fixed']}), "
        f"gain={best['gain']:.3f}, acceleration={best['acceleration']:.2f}, "
        f"early={best['early_mean']:.2f}, late={best['late_mean']:.2f}, "
        f"spikes={best['spike_count']}"
    )
    dut._log.info(f"Case 1 first 10 intervals: {best['intervals'][:10]}")
    dut._log.info(f"Case 1 last 10 intervals:  {best['intervals'][-10:]}")

    tester.spike_steps = best["spike_steps"]
    tester.membrane_trace = best["membrane_trace"]
    tester.current_trace = best["current_trace"]
    tester.export_spike_cycle_csv(
        Path("../results/fractional_lif_memory_spike_cycles.csv")
    )

    assert best["gain"] > 1.03, (
        "Expected increasing spike rate under constant current, "
        f"but best gain was {best['gain']:.3f} at I={best['current_float']:.3f}."
    )


@cocotb.test()
async def test_dropout_recovery_memory_effect(dut):
    """Case 2: spike rate after dropout/restart should exceed initial startup rate."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    tester = FractionalMemoryTester(dut)

    candidate_currents = [
        0.160,
        0.170,
        0.180,
        0.190,
        0.200,
        0.210,
        0.220,
        0.230,
        0.240,
        0.250,
        0.260,
        0.280,
    ]

    best = await sweep_best_constant_current(
        dut=dut,
        tester=tester,
        candidate_currents=candidate_currents,
        run_steps=2200,
        min_spikes_required=10,
    )

    assert best is not None, "Could not find operating point for dropout/recovery test."

    test_current_fixed = best["current_fixed"]
    test_current_float = best["current_float"]

    await reset_dut(dut)
    tester.reset_traces()

    startup_steps = 1200
    dropout_steps = 120
    recovery_steps = 1200
    recovery_start = startup_steps + dropout_steps

    profile = [
        (test_current_fixed, startup_steps),
        (0, dropout_steps),
        (test_current_fixed, recovery_steps),
    ]
    await tester.run_profile(profile)

    startup_intervals = tester.inter_spike_intervals_in_window(0, startup_steps)
    recovery_intervals = tester.inter_spike_intervals_in_window(
        recovery_start, recovery_start + recovery_steps
    )

    # Compare early startup against early recovery after current restarts.
    startup_early = startup_intervals[:6]
    recovery_early = recovery_intervals[:6]

    assert (
        len(startup_early) >= 4
    ), f"Not enough startup intervals for analysis: got {len(startup_early)}"
    assert (
        len(recovery_early) >= 4
    ), f"Not enough recovery intervals for analysis: got {len(recovery_early)}"

    startup_mean = mean(startup_early)
    recovery_mean = mean(recovery_early)
    recovery_gain = startup_mean / recovery_mean if recovery_mean > 0 else 0.0

    dut._log.info(
        "Case 2 summary: "
        f"I={test_current_float:.3f} (fixed={test_current_fixed}), "
        f"startup_mean={startup_mean:.2f}, recovery_mean={recovery_mean:.2f}, "
        f"recovery_gain={recovery_gain:.3f}"
    )
    dut._log.info(
        f"Case 2 phase boundaries (steps): startup=[0,{startup_steps}), "
        f"dropout=[{startup_steps},{recovery_start}), recovery=[{recovery_start},{recovery_start + recovery_steps})"
    )

    tester.export_spike_cycle_csv(
        Path("../results/fractional_lif_dropout_recovery_spike_cycles.csv")
    )

    assert recovery_gain > 1.03, (
        "Expected faster spiking after dropout recovery than initial startup, "
        f"but recovery_gain={recovery_gain:.3f} (startup_mean={startup_mean:.2f}, recovery_mean={recovery_mean:.2f})."
    )
