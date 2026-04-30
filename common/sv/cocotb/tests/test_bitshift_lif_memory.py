import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
from cocotb.utils import get_sim_time


DATA_WIDTH = 16
MEMBRANE_WIDTH = 24
FRAC_BITS = 13

RESULTS_DIR = Path("../results")
# Operating-point JSONs live outside ../results so they survive `make clean`.
OP_DIR = Path("../operating_points")

# Coarse sweep starts at 0.20 (not 0.30) so near-threshold currents that show adaptation
# in the fractional model (around 0.16-0.28) are not missed.
COARSE_CURRENTS = [
    0.20,
    0.30,
    0.40,
    0.55,
    0.70,
    0.85,
    1.00,
    1.20,
    1.40,
    1.60,
    1.80,
    2.00,
]


def save_operating_point(path: Path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(fields, f, indent=2)


def load_operating_point(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Operating point file not found: {path}. Run the sweep testcase first."
        )
    with path.open() as f:
        return json.load(f)


def write_phase_metadata(path: Path, phases: list):
    """Write phase boundary metadata so the plot script can shade the dropout window
    against the actual current-off period rather than inferring it from spike gaps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({"phases": phases}, f, indent=2)


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
        self.spike_times_ns = []
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
            self.spike_times_ns.append(get_sim_time(unit="ns"))

        self.current_trace.append(current_signed)
        self.membrane_trace.append(membrane)
        self.step_trace.append(self.step_count)
        self.step_count += 1

        return spike, membrane

    async def run_constant_current(self, current_signed: int, steps: int):
        for _ in range(steps):
            await self.record_step(current_signed)

    async def run_profile(self, profile, labels=None):
        """Run a (current, steps) profile sequentially.

        If labels is provided (one label per profile entry), records sim_time at the
        start and end of each phase and returns a list of phase dicts with the actual
        boundary times (when current changes), suitable for a sidecar JSON.
        """
        phases = []
        for i, (current_signed, steps) in enumerate(profile):
            label = labels[i] if labels is not None else None
            t_start = float(get_sim_time(unit="ns"))
            step_start = self.step_count
            for _ in range(steps):
                await self.record_step(current_signed)
            if label is not None:
                phases.append(
                    {
                        "label": label,
                        "start_step": step_start,
                        "end_step": self.step_count,
                        "t_start_ns": t_start,
                        "t_end_ns": float(get_sim_time(unit="ns")),
                    }
                )
        return phases

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

    def export_spike_cycle_csv(self, filename: Path, phases=None):
        """Export spike-cycle membrane traces to CSV.

        phases: optional list of (start_step, end_step, label) tuples that tag each
        cycle with the phase it belongs to. When None, every row gets label "constant".
        """
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

        def _phase_label(spike_step):
            if phases is None:
                return "constant"
            for start_step, end_step, label in phases:
                if start_step <= spike_step < end_step:
                    return label
            return "constant"

        spike_step_to_ns = dict(zip(self.spike_steps, self.spike_times_ns))

        with filename.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            header = [f"membrane_potential_{i + 1}" for i in range(max_len)]
            writer.writerow(
                ["cycle_num", "spike_step", "spike_time_ns", "cycle_length", "phase"]
                + header
            )

            for cycle_idx, (start_idx, end_idx) in enumerate(cycle_ranges, start=1):
                values = self.membrane_trace[start_idx : end_idx + 1]
                cycle_len = len(values)
                padded = values + [""] * (max_len - cycle_len)
                phase = _phase_label(end_idx)
                spike_ns = spike_step_to_ns.get(end_idx, "")
                writer.writerow(
                    [cycle_idx, end_idx, spike_ns, cycle_len, phase] + padded
                )


async def reset_dut(dut):
    dut.reset.value = 1
    dut.clear.value = 0
    dut.enable.value = 0
    dut.current.value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


async def _run_coarse_fine_sweep(dut, tester, run_steps_fine=4000):
    """Run the two-stage coarse-then-fine sweep and return the best operating point dict.

    Returns None (not raises) when no analyzable regime is found, so callers can
    write a null operating point and the corresponding capture test can skip cleanly.
    """
    history_length = int(dut.HISTORY_LENGTH.value)
    shift_mode = int(dut.SHIFT_MODE.value)
    custom_decay_rate = int(dut.CUSTOM_DECAY_RATE.value)
    inv_denom = int(dut.INV_DENOM.value)
    dut._log.info(
        "Bitshift memory sweep config: "
        f"HISTORY_LENGTH={history_length}, SHIFT_MODE={shift_mode}, "
        f"CUSTOM_DECAY_RATE={custom_decay_rate}, INV_DENOM={inv_denom}"
    )

    run_steps_coarse = 1200
    # min_spikes_required=4 in fine stage so near-threshold currents aren't rejected
    # prematurely when the neuron barely crosses threshold.
    min_spikes_required_fine = 4

    first_spiking_current = None
    last_silent_current = None
    first_dense_current = None
    coarse_results = []

    # Stage 1: coarse sweep to locate the spiking threshold.
    for current_float in COARSE_CURRENTS:
        await reset_dut(dut)
        tester.reset_traces()

        current_fixed = float_to_qs2_13(current_float)
        await tester.run_constant_current(current_fixed, run_steps_coarse)

        spike_count = len(tester.spike_steps)
        intervals = tester.inter_spike_intervals()
        coarse_results.append({"current": current_float, "spikes": spike_count})
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
        return None

    # Stage 2: denser sweep near threshold.
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

    candidate_currents.append(round(first_spiking_current + 0.20, 3))
    candidate_currents.append(round(first_spiking_current + 0.40, 3))
    candidate_currents = sorted(set(candidate_currents))
    dut._log.info(
        f"Fine sweep around first spiking current {first_spiking_current:.3f}: {candidate_currents}"
    )

    best = None
    fine_results = []

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

        if spike_count < min_spikes_required_fine or len(intervals) < 6:
            result["status"] = "insufficient_spikes"
            fine_results.append(result)
            continue

        # Get coefficient of variation (CV) and range to confirm that the sweep is
        # capturing a variety of ISI patterns and not just noise.
        cv = pstdev(intervals) / mean(intervals) if mean(intervals) > 0 else 0.0
        dut._log.info(
            f"I={current_float:.3f}: ISI CV={cv:.2f}, range [{min(intervals)}, {max(intervals)}]"
        )

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
            result.update(
                {"status": "saturated", "median_isi": median_isi, "gain": gain}
            )
            fine_results.append(result)
            continue

        result.update(
            {
                "status": "analyzable",
                "gain": gain,
                "acceleration": acceleration,
                "median_isi": median_isi,
            }
        )
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
        return None

    dut._log.info(
        "Best operating point: "
        f"I={best['current_float']:.3f} (fixed={best['current_fixed']}), "
        f"gain={best['gain']:.3f}, acceleration={best['acceleration']:.2f}, "
        f"early={best['early_mean']:.2f}, late={best['late_mean']:.2f}, "
        f"median_isi={best['median_isi']:.2f}, spikes={best['spike_count']}"
    )
    dut._log.info(f"First 10 intervals: {best['intervals'][:10]}")
    dut._log.info(f"Last 10 intervals:  {best['intervals'][-10:]}")

    return best


@cocotb.test()
async def test_constant_current_bitshift_sweep(dut):
    """Sweep currents to find the best constant-current operating point for bitshift_lif.

    Does not hard-assert adaptation — logs ADAPTATION OBSERVED/NOT OBSERVED and writes
    the JSON regardless so the capture test can proceed (or skip) cleanly.
    """
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    tester = BitshiftMemoryTester(dut)
    best = await _run_coarse_fine_sweep(dut, tester, run_steps_fine=4000)

    op_path = OP_DIR / "bitshift_lif_memory_operating_point.json"
    if best is None:
        save_operating_point(
            op_path,
            variant="bitshift_lif_memory",
            scenario="constant",
            current_float=None,
            current_fixed=None,
            frac_bits=FRAC_BITS,
            notes="no analyzable operating point found in sweep",
        )
        dut._log.warning(f"No operating point found; null JSON written to {op_path}")
        return

    if best["gain"] > 1.03:
        dut._log.info(
            "ADAPTATION OBSERVED: bitshift late-rate increase >3% in tested conditions."
        )
    else:
        dut._log.warning(
            "ADAPTATION NOT OBSERVED: no >3% late-rate increase in analyzable tested conditions."
        )

    save_operating_point(
        op_path,
        variant="bitshift_lif_memory",
        scenario="constant",
        current_float=best["current_float"],
        current_fixed=best["current_fixed"],
        frac_bits=FRAC_BITS,
        sweep_run_steps=4000,
        best_gain=best["gain"],
        best_spike_count=best["spike_count"],
        notes="auto-selected from coarse+fine sweep",
    )
    dut._log.info(f"Operating point saved to {op_path}")


@cocotb.test()
async def test_constant_current_bitshift_capture(dut):
    """Run only the best operating-point current; export a focused FST and spike-cycle CSV.

    Skips cleanly when the sweep found no viable operating point.
    """
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    op_path = OP_DIR / "bitshift_lif_memory_operating_point.json"
    op = load_operating_point(op_path)

    if op.get("current_float") is None:
        dut._log.warning(
            "No viable operating point; skipping bitshift constant-current capture."
        )
        return

    current_fixed = op["current_fixed"]
    current_float = op["current_float"]
    capture_steps = op.get("sweep_run_steps", 4000)

    dut._log.info(
        f"Capture: I={current_float:.3f} (fixed={current_fixed}), steps={capture_steps}"
    )

    tester = BitshiftMemoryTester(dut)
    await reset_dut(dut)
    await tester.run_constant_current(current_fixed, capture_steps)

    csv_path = RESULTS_DIR / "bitshift_lif_memory_spike_cycles.csv"
    tester.export_spike_cycle_csv(csv_path)
    dut._log.info(f"Spike-cycle CSV written to {csv_path}")
    dut._log.info(
        "To plot (run fst2vcd inside Docker, then plot on host):\n"
        "  fst2vcd ../results/sim_build/bitshift_lif.fst -o ../results/bitshift_lif.vcd\n"
        "  python common/scripts/plot_membrane_potential.py "
        "common/sv/cocotb/results/bitshift_lif.vcd "
        "--membrane-signal bitshift_lif.membrane_out[23:0] "
        "--current-signal bitshift_lif.current[15:0] "
        "--spike-signal bitshift_lif.spike_out "
        "--output common/images/bitshift_lif_memory_constant.svg"
    )


@cocotb.test()
async def test_dropout_recovery_bitshift_sweep(dut):
    """Sweep currents to find an operating point for the bitshift dropout/recovery scenario.

    Does not hard-assert adaptation — bitshift not showing adaptation is a valid finding.
    """
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    tester = BitshiftMemoryTester(dut)
    best = await _run_coarse_fine_sweep(dut, tester, run_steps_fine=2200)

    op_path = OP_DIR / "bitshift_lif_dropout_operating_point.json"
    if best is None:
        save_operating_point(
            op_path,
            variant="bitshift_lif_memory",
            scenario="dropout",
            current_float=None,
            current_fixed=None,
            frac_bits=FRAC_BITS,
            notes="no analyzable operating point found in sweep",
        )
        dut._log.warning(f"No operating point found; null JSON written to {op_path}")
        return

    save_operating_point(
        op_path,
        variant="bitshift_lif_memory",
        scenario="dropout",
        current_float=best["current_float"],
        current_fixed=best["current_fixed"],
        frac_bits=FRAC_BITS,
        sweep_run_steps=2200,
        best_gain=best["gain"],
        best_spike_count=best["spike_count"],
        notes="auto-selected from coarse+fine sweep",
    )
    dut._log.info(f"Dropout operating point saved to {op_path}")


@cocotb.test()
async def test_dropout_recovery_bitshift_capture(dut):
    """Run the startup/dropout/recovery profile once; export a focused FST and phase-labeled CSV.

    Logs recovery_gain but does not assert — bitshift not showing memory is a valid result.
    Skips cleanly when the sweep found no viable operating point.
    """
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    op_path = OP_DIR / "bitshift_lif_dropout_operating_point.json"
    op = load_operating_point(op_path)

    if op.get("current_float") is None:
        dut._log.warning(
            "No viable operating point; skipping bitshift dropout capture."
        )
        return

    current_fixed = op["current_fixed"]
    current_float = op["current_float"]

    startup_steps = 300
    dropout_steps = 120
    recovery_steps = 300
    recovery_start = startup_steps + dropout_steps

    dut._log.info(
        f"Dropout capture: I={current_float:.3f} (fixed={current_fixed}), "
        f"profile=[{startup_steps}+{dropout_steps}+{recovery_steps}]"
    )

    tester = BitshiftMemoryTester(dut)
    await reset_dut(dut)

    profile = [
        (current_fixed, startup_steps),
        (0, dropout_steps),
        (current_fixed, recovery_steps),
    ]
    phase_meta = await tester.run_profile(
        profile, labels=["startup", "dropout", "recovery"]
    )

    phases_path = RESULTS_DIR / "bitshift_lif_dropout_recovery_phases.json"
    write_phase_metadata(phases_path, phase_meta)
    dut._log.info(f"Phase boundary metadata written to {phases_path}")

    startup_intervals = tester.inter_spike_intervals_in_window(0, startup_steps)
    recovery_intervals = tester.inter_spike_intervals_in_window(
        recovery_start, recovery_start + recovery_steps
    )

    startup_early = startup_intervals[:6]
    recovery_early = recovery_intervals[:6]

    if len(startup_early) >= 4 and len(recovery_early) >= 4:
        startup_mean = mean(startup_early)
        recovery_mean = mean(recovery_early)
        recovery_gain = startup_mean / recovery_mean if recovery_mean > 0 else 0.0
        dut._log.info(
            "Dropout capture summary: "
            f"I={current_float:.3f}, startup_mean={startup_mean:.2f}, "
            f"recovery_mean={recovery_mean:.2f}, recovery_gain={recovery_gain:.3f}"
        )
        if recovery_gain > 1.03:
            dut._log.info("MEMORY OBSERVED: recovery spiking faster than startup.")
        else:
            dut._log.warning("MEMORY NOT OBSERVED: recovery gain <= 1.03.")
    else:
        dut._log.warning(
            f"Insufficient intervals for gain analysis "
            f"(startup_early={len(startup_early)}, recovery_early={len(recovery_early)})"
        )

    phases = [
        (0, startup_steps, "startup"),
        (startup_steps, recovery_start, "dropout"),
        (recovery_start, recovery_start + recovery_steps, "recovery"),
    ]

    csv_path = RESULTS_DIR / "bitshift_lif_dropout_recovery_spike_cycles.csv"
    tester.export_spike_cycle_csv(csv_path, phases=phases)
    dut._log.info(f"Dropout spike-cycle CSV written to {csv_path}")
    dut._log.info(
        "To plot (run fst2vcd inside Docker, then plot on host):\n"
        "  fst2vcd ../results/sim_build/bitshift_lif.fst -o ../results/bitshift_lif.vcd\n"
        "  python common/scripts/plot_membrane_potential.py "
        "common/sv/cocotb/results/bitshift_lif.vcd "
        "--membrane-signal bitshift_lif.membrane_out[23:0] "
        "--current-signal bitshift_lif.current[15:0] "
        "--spike-signal bitshift_lif.spike_out "
        "--phase-csv common/sv/cocotb/results/bitshift_lif_dropout_recovery_spike_cycles.csv "
        "--output common/images/bitshift_lif_memory_dropout.svg"
    )
