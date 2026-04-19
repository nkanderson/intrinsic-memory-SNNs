import csv
from pathlib import Path
from statistics import mean, median

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


class BitshiftMemoryTester:
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
            max_wait_cycles = 512
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

    def inter_spike_intervals(self):
        if len(self.spike_steps) < 2:
            return []
        return [
            self.spike_steps[i] - self.spike_steps[i - 1]
            for i in range(1, len(self.spike_steps))
        ]

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


@cocotb.test()
async def test_constant_current_bitshift_memory_effect(dut):
    """Explore constant-input spike-rate adaptation for bitshift_lif (mode3 first)."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    tester = BitshiftMemoryTester(dut)

    history_length = int(dut.HISTORY_LENGTH.value)
    shift_mode = int(dut.SHIFT_MODE.value)
    custom_decay_rate = int(dut.CUSTOM_DECAY_RATE.value)
    inv_denom = int(dut.INV_DENOM.value)
    dut._log.info(
        "Bitshift memory test config: "
        f"HISTORY_LENGTH={history_length}, SHIFT_MODE={shift_mode}, "
        f"CUSTOM_DECAY_RATE={custom_decay_rate}, INV_DENOM={inv_denom}"
    )

    if shift_mode != 3:
        cocotb.log.info(
            f"This exploratory test is tuned for SHIFT_MODE=3; got SHIFT_MODE={shift_mode}."
        )

    coarse_currents = [0.30, 0.40, 0.50, 0.65, 0.80, 1.00, 1.10, 1.20, 1.30, 1.40, 1.60]
    run_steps_coarse = 1200
    run_steps_fine = 4000
    min_spikes_required = 6

    best = None
    first_spiking_current = None
    last_silent_current = None
    first_dense_current = None
    coarse_results = []
    fine_results = []

    # Stage 1: coarse sweep to find where spiking begins.
    for current_float in coarse_currents:
        await reset_dut(dut)
        tester.reset_traces()

        current_fixed = float_to_qs2_13(current_float)
        await tester.run_constant_current(current_fixed, run_steps_coarse)

        spike_count = len(tester.spike_steps)
        intervals = tester.inter_spike_intervals()
        coarse_results.append(
            {
                "current": current_float,
                "spikes": spike_count,
                "intervals": len(intervals),
            }
        )
        dut._log.info(
            f"[coarse] I={current_float:.3f} -> spikes={spike_count}, intervals={len(intervals)}"
        )

        if spike_count == 0:
            last_silent_current = current_float

        if spike_count >= 2 and first_spiking_current is None:
            first_spiking_current = current_float

        if spike_count >= 40 and first_dense_current is None:
            first_dense_current = current_float

        if first_spiking_current is not None and first_dense_current is not None:
            break

    if first_spiking_current is None:
        dut._log.warning(
            "Bitshift neuron did not spike in coarse sweep; adaptation not observed in tested range."
        )
        dut._log.info(f"Coarse conditions tested: {coarse_results}")
        return

    # Stage 2: denser sweep near threshold transition (between silent and dense spiking).
    if last_silent_current is not None:
        fine_start = max(0.10, last_silent_current - 0.05)
    else:
        fine_start = max(0.10, first_spiking_current - 0.20)

    if first_dense_current is not None:
        fine_end = first_dense_current + 0.05
    else:
        fine_end = first_spiking_current + 0.25

    fine_step = 0.01

    candidate_currents = []
    current = fine_start
    while current <= fine_end + 1e-9:
        candidate_currents.append(round(current, 3))
        current += fine_step

    # Add stronger candidates for context/sanity checks.
    candidate_currents.append(round(first_spiking_current + 0.20, 3))
    candidate_currents.append(round(first_spiking_current + 0.40, 3))
    candidate_currents = sorted(set(candidate_currents))
    dut._log.info(
        f"Fine sweep around first spiking current {first_spiking_current:.3f}: {candidate_currents}"
    )

    for current_float in candidate_currents:
        await reset_dut(dut)
        tester.reset_traces()

        current_fixed = float_to_qs2_13(current_float)
        await tester.run_constant_current(current_fixed, run_steps_fine)

        intervals = tester.inter_spike_intervals()
        spike_count = len(tester.spike_steps)
        dut._log.info(
            f"I={current_float:.3f} -> spikes={spike_count}, intervals={len(intervals)}"
        )

        result = {
            "current": current_float,
            "spikes": spike_count,
            "interval_count": len(intervals),
        }

        if spike_count < min_spikes_required or len(intervals) < 6:
            result["status"] = "insufficient_spikes"
            fine_results.append(result)
            continue

        early = intervals[:4]
        late = intervals[-4:]
        if len(early) < 4 or len(late) < 4:
            result["status"] = "insufficient_windows"
            fine_results.append(result)
            continue

        early_mean = mean(early)
        late_mean = mean(late)
        if late_mean <= 0:
            result["status"] = "invalid_late_mean"
            fine_results.append(result)
            continue

        gain = early_mean / late_mean
        acceleration = early_mean - late_mean
        median_isi = median(intervals)
        if median_isi <= 3:
            dut._log.info(
                f"I={current_float:.3f}: skipping saturated regime (median ISI={median_isi:.2f})"
            )
            result["status"] = "saturated"
            result["median_isi"] = median_isi
            result["gain"] = gain
            fine_results.append(result)
            continue

        result["status"] = "analyzable"
        result["gain"] = gain
        result["acceleration"] = acceleration
        result["median_isi"] = median_isi
        fine_results.append(result)
        dut._log.info(
            f"I={current_float:.3f}: early_mean={early_mean:.2f}, late_mean={late_mean:.2f}, "
            f"gain={gain:.3f}, acceleration={acceleration:.2f}, median_isi={median_isi:.2f}"
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
                "median_isi": median_isi,
            }

    dut._log.info(f"Coarse conditions tested ({len(coarse_results)}): {coarse_results}")
    dut._log.info(f"Fine conditions tested ({len(fine_results)}): {fine_results}")

    if best is None:
        dut._log.warning(
            "No analyzable non-saturated regime found; adaptation not observed under tested conditions."
        )
        return

    dut._log.info(
        "Best operating point: "
        f"I={best['current_float']:.3f} (fixed={best['current_fixed']}), "
        f"gain={best['gain']:.3f}, acceleration={best['acceleration']:.2f}, "
        f"early={best['early_mean']:.2f}, late={best['late_mean']:.2f}, median_isi={best['median_isi']:.2f}, "
        f"spikes={best['spike_count']}"
    )
    dut._log.info(f"First 10 intervals: {best['intervals'][:10]}")
    dut._log.info(f"Last 10 intervals:  {best['intervals'][-10:]}")

    tester.spike_steps = best["spike_steps"]
    tester.membrane_trace = best["membrane_trace"]
    tester.current_trace = best["current_trace"]
    results_csv = Path("../results/bitshift_lif_memory_spike_cycles.csv")
    tester.export_spike_cycle_csv(results_csv)
    dut._log.info(f"Exported spike-cycle membrane traces to {results_csv}")

    if best["gain"] > 1.03:
        dut._log.info(
            "ADAPTATION OBSERVED: bitshift late-rate increase >3% in tested conditions."
        )
    else:
        dut._log.warning(
            "ADAPTATION NOT OBSERVED: no >3% late-rate increase in analyzable tested conditions."
        )
