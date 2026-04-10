import cocotb
from cocotb.triggers import Timer, RisingEdge, ClockCycles
from cocotb.clock import Clock
import matplotlib.pyplot as plt
import numpy as np


class FracOrderMemoryTester:
    """Helper class for fractional-order LIF memory effect testing."""

    def __init__(self, dut):
        self.dut = dut
        self.spike_times = []
        self.current_trace = []
        self.time_trace = []
        self.membrane_potential_trace = []
        self.cycle_count = 0

    def reset_traces(self):
        """Reset all recorded traces."""
        self.spike_times = []
        self.current_trace = []
        self.time_trace = []
        self.membrane_potential_trace = []
        self.cycle_count = 0

    async def record_cycle(self, current_input):
        """Record data for one clock cycle."""
        self.dut.current.value = current_input
        await RisingEdge(self.dut.clk)

        # Record spike if it occurred
        if self.dut.spike.value:
            self.spike_times.append(self.cycle_count)

        # Record traces
        self.current_trace.append(current_input)
        self.time_trace.append(self.cycle_count)
        self.membrane_potential_trace.append(int(self.dut.membrane_potential.value))
        self.cycle_count += 1

    async def wait_for_spikes(self, current_input, min_spikes=5, max_cycles=10000):
        """Wait until we get a minimum number of spikes or hit max cycles."""
        initial_spike_count = len(self.spike_times)
        cycles_waited = 0

        while (
            len(self.spike_times) - initial_spike_count
        ) < min_spikes and cycles_waited < max_cycles:
            await self.record_cycle(current_input)
            cycles_waited += 1

        return len(self.spike_times) - initial_spike_count, cycles_waited

    async def measure_spike_interval(self, current_input, max_cycles=8000):
        """Measure the time between two consecutive spikes."""
        initial_spikes = len(self.spike_times)
        cycles_waited = 0

        # Wait for first spike
        while len(self.spike_times) == initial_spikes and cycles_waited < max_cycles:
            await self.record_cycle(current_input)
            cycles_waited += 1

        if len(self.spike_times) == initial_spikes:
            return None, cycles_waited  # No spike found

        first_spike_time = self.spike_times[-1]

        # Wait for second spike
        while (
            len(self.spike_times) == (initial_spikes + 1) and cycles_waited < max_cycles
        ):
            await self.record_cycle(current_input)
            cycles_waited += 1

        if len(self.spike_times) == (initial_spikes + 1):
            return None, cycles_waited  # Only one spike found

        second_spike_time = self.spike_times[-1]
        interval = second_spike_time - first_spike_time

        return interval, cycles_waited

    def calculate_spike_frequency(self, window_start, window_end):
        """Calculate spike frequency in a given time window."""
        spikes_in_window = [
            t for t in self.spike_times if window_start <= t < window_end
        ]
        window_duration = window_end - window_start
        if window_duration > 0:
            return len(spikes_in_window) / window_duration
        return 0

    def get_inter_spike_intervals(self, window_start=None, window_end=None):
        """Calculate inter-spike intervals in a given window."""
        if window_start is None:
            window_start = 0
        if window_end is None:
            window_end = self.cycle_count

        spikes_in_window = [
            t for t in self.spike_times if window_start <= t < window_end
        ]
        if len(spikes_in_window) < 2:
            return []

        intervals = []
        for i in range(1, len(spikes_in_window)):
            intervals.append(spikes_in_window[i] - spikes_in_window[i - 1])
        return intervals

    def export_spike_cycle_csv(
        self, filename="spike_cycle_membrane_potential.csv", start_current_idx=None
    ):
        """Export CSV with membrane potential values for each spike cycle.

        Each row contains all membrane potential values from one spike to the next.
        The first row starts from when current becomes non-zero and goes to the first spike.

        Args:
            filename: CSV filename to write
            start_current_idx: Index where current first becomes non-zero. If None,
                              searches for first non-zero current automatically.
        """
        import csv

        if not self.spike_times:
            print(f"No spikes recorded, cannot generate {filename}")
            return

        # Find the starting point (when current first becomes non-zero)
        if start_current_idx is None:
            start_current_idx = 0
            for i, current in enumerate(self.current_trace):
                if current > 0:
                    start_current_idx = i
                    break

        # Create list of spike cycle ranges
        cycle_ranges = []

        # First cycle: from start_current_idx to first spike (inclusive)
        if self.spike_times:
            first_spike_idx = self.spike_times[0]
            cycle_ranges.append((start_current_idx, first_spike_idx))

        # Subsequent cycles: from previous spike+1 to current spike (inclusive)
        for i in range(1, len(self.spike_times)):
            prev_spike_idx = self.spike_times[i - 1]
            curr_spike_idx = self.spike_times[i]
            cycle_ranges.append((prev_spike_idx + 1, curr_spike_idx))

        # Write CSV file
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            max_cycle_length = max(end - start + 1 for start, end in cycle_ranges)
            header = [f"membrane_potential_{i+1}" for i in range(max_cycle_length)]
            writer.writerow(["cycle_num", "spike_time", "cycle_length"] + header)

            # Write data rows
            for cycle_num, (start_idx, end_idx) in enumerate(cycle_ranges, 1):
                membrane_values = self.membrane_potential_trace[start_idx : end_idx + 1]
                spike_time = self.time_trace[end_idx]
                cycle_length = len(membrane_values)

                # Pad row with empty values if this cycle is shorter than max
                padded_values = membrane_values + [""] * (
                    max_cycle_length - cycle_length
                )

                row = [cycle_num, spike_time, cycle_length] + padded_values
                writer.writerow(row)

        print(f"Exported {len(cycle_ranges)} spike cycles to {filename}")
        print(
            f"Cycles range from {min(end-start+1 for start, end in cycle_ranges)} to {max(end-start+1 for start, end in cycle_ranges)} time steps"
        )

    def plot_results(self, title="Fractional-Order LIF Test Results"):
        """Generate plots of the test results."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Input current and spike raster
        ax1.plot(
            self.time_trace,
            self.current_trace,
            "b-",
            label="Input Current",
            linewidth=2,
        )
        if self.spike_times:
            spike_y = [max(self.current_trace) * 1.1] * len(self.spike_times)
            ax1.scatter(
                self.spike_times,
                spike_y,
                color="red",
                marker="|",
                s=100,
                label="Spikes",
            )
        ax1.set_ylabel("Current / Spike Events")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(title)

        # Plot 2: Spike frequency over time (using sliding window)
        if len(self.spike_times) >= 2:
            window_size = 50  # cycles
            frequencies = []
            freq_times = []

            for i in range(window_size, len(self.time_trace), 10):
                freq = self.calculate_spike_frequency(i - window_size, i)
                frequencies.append(freq)
                freq_times.append(i)

            ax2.plot(freq_times, frequencies, "g-", linewidth=2)
            ax2.set_ylabel("Spike Frequency (spikes/cycle)")
            ax2.set_xlabel("Time (cycles)")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(
                0.5,
                0.5,
                "Insufficient spikes for frequency analysis",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

        plt.tight_layout()
        return fig


@cocotb.test(skip=True)
async def test_steady_input_frequency_increase(dut):
    """Test 1: Steady input current showing frequency increase over time."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    tester = FracOrderMemoryTester(dut)

    # Reset the DUT
    dut.rst.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 5)

    # Use a lower current to avoid saturation and see gradual changes
    steady_current = 35
    max_test_cycles = 8000  # Maximum cycles

    dut._log.info(
        f"Starting steady input test with current={steady_current}, max_cycles={max_test_cycles}"
    )

    # Measure early inter-spike intervals (first few spikes)
    early_intervals = []
    early_spike_times = []

    for i in range(5):  # Get first 5 inter-spike intervals
        interval, cycles_used = await tester.measure_spike_interval(
            steady_current, max_cycles=8000
        )
        if interval is None:
            dut._log.warning(
                f"Failed to get spike interval {i+1} after {cycles_used} cycles"
            )
            break
        early_intervals.append(interval)
        early_spike_times.append(tester.spike_times[-1])
        dut._log.info(f"Early interval {i+1}: {interval} cycles")

    if len(early_intervals) < 2:
        assert (
            False
        ), f"Could not measure enough early intervals: got {len(early_intervals)}"

    # Continue running to allow memory effects to build up
    # Run for a significant number of cycles to build history
    dut._log.info("Building up memory effects...")
    additional_spikes, cycles_used = await tester.wait_for_spikes(
        steady_current, min_spikes=30, max_cycles=2000
    )
    dut._log.info(
        f"Generated {additional_spikes} additional spikes in {cycles_used} cycles"
    )

    # Measure late inter-spike intervals (after memory has built up)
    late_intervals = []
    late_spike_times = []

    for i in range(5):  # Get last 5 inter-spike intervals
        interval, cycles_used = await tester.measure_spike_interval(
            steady_current, max_cycles=8000
        )
        if interval is None:
            dut._log.warning(
                f"Failed to get late spike interval {i+1} after {cycles_used} cycles"
            )
            break
        late_intervals.append(interval)
        late_spike_times.append(tester.spike_times[-1])
        dut._log.info(f"Late interval {i+1}: {interval} cycles")

    if len(late_intervals) < 2:
        assert (
            False
        ), f"Could not measure enough late intervals: got {len(late_intervals)}"

    # Analyze results
    total_spikes = len(tester.spike_times)
    total_cycles = tester.cycle_count
    overall_frequency = total_spikes / total_cycles if total_cycles > 0 else 0

    avg_early_interval = np.mean(early_intervals)
    avg_late_interval = np.mean(late_intervals)
    early_frequency = 1.0 / avg_early_interval if avg_early_interval > 0 else 0
    late_frequency = 1.0 / avg_late_interval if avg_late_interval > 0 else 0

    dut._log.info(f"Total spikes: {total_spikes} over {total_cycles} cycles")
    dut._log.info(f"Overall frequency: {overall_frequency:.4f} spikes/cycle")
    dut._log.info(f"Early intervals: {early_intervals}")
    dut._log.info(f"Late intervals: {late_intervals}")
    dut._log.info(
        f"Average early inter-spike interval: {avg_early_interval:.2f} cycles"
    )
    dut._log.info(f"Average late inter-spike interval: {avg_late_interval:.2f} cycles")
    dut._log.info(f"Early frequency: {early_frequency:.4f} spikes/cycle")
    dut._log.info(f"Late frequency: {late_frequency:.4f} spikes/cycle")

    if len(tester.spike_times) >= 10:
        dut._log.info(f"First spike times: {tester.spike_times[:5]}")
        dut._log.info(f"Last spike times: {tester.spike_times[-5:]}")

    # Generate plot
    # fig = tester.plot_results("Test 1: Steady Input Frequency Increase")
    # fig.savefig("test1_steady_input.png", dpi=150, bbox_inches="tight")
    # plt.close(fig)

    # Export membrane potential CSV data for spike cycles
    tester.export_spike_cycle_csv("test1_steady_input_membrane_potential.csv")

    # Basic assertions
    assert (
        total_spikes >= 10
    ), f"Insufficient spikes for analysis: got {total_spikes}, need at least 10"

    # Check for frequency increase (memory effect causes shorter intervals over time)
    if early_frequency > 0 and late_frequency > 0:
        frequency_ratio = late_frequency / early_frequency
        interval_ratio = (
            avg_early_interval / avg_late_interval
        )  # Should be > 1 if intervals are getting shorter

        dut._log.info(f"Frequency ratio (late/early): {frequency_ratio:.2f}")
        dut._log.info(f"Interval ratio (early/late): {interval_ratio:.2f}")

        # Fractional-order memory should cause frequency to increase over time
        # (intervals should get shorter as memory accumulates)
        assert (
            frequency_ratio > 1.05
        ), f"Expected frequency increase >5%, got ratio {frequency_ratio:.2f} (early={early_frequency:.4f}, late={late_frequency:.4f})"
    else:
        assert (
            False
        ), f"Could not calculate frequencies: early={early_frequency:.4f}, late={late_frequency:.4f}"


# TODO: This one is currently failing.
@cocotb.test()
async def test_dropout_recovery_memory_effect(dut):
    """Test 2: Current dropout and recovery showing memory effect."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    tester = FracOrderMemoryTester(dut)

    # Reset the DUT
    dut.rst.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 5)

    # Test parameters - use same current as test 1 for consistency
    steady_current = 35
    max_cycles_per_phase = 8000

    dut._log.info(f"Starting dropout/recovery test:")
    dut._log.info(f"  Buildup: wait for initial steady-state intervals")
    dut._log.info(f"  Dropout: apply zero current until membrane decays")
    dut._log.info(f"  Recovery: measure how quickly spiking resumes")

    # Phase 1: Measure initial steady-state intervals
    dut._log.info("Phase 1: Measuring initial steady-state intervals...")
    initial_intervals = []
    initial_spike_times = []
    num_intervals = 3

    # Get several intervals to establish baseline
    for i in range(num_intervals):
        interval, cycles_used = await tester.measure_spike_interval(
            steady_current, max_cycles=8000
        )
        if interval is None:
            dut._log.warning(
                f"Failed to get initial interval {i+1} after {cycles_used} cycles"
            )
            break
        initial_intervals.append(interval)
        initial_spike_times.append(tester.spike_times[-1])
        dut._log.info(f"Initial interval {i+1}: {interval} cycles")

    if len(initial_intervals) < 3:
        assert (
            False
        ), f"Could not establish baseline: got {len(initial_intervals)} intervals"

    baseline_interval = np.mean(initial_intervals)
    baseline_frequency = 1.0 / baseline_interval
    phase1_end_cycle = tester.cycle_count

    dut._log.info(
        f"Baseline established: avg interval = {baseline_interval:.2f} cycles, freq = {baseline_frequency:.4f}"
    )

    # Phase 2: Current dropout - wait long enough for membrane to decay significantly
    dut._log.info("Phase 2: Current dropout...")
    dropout_start_cycle = tester.cycle_count

    # Apply zero current for a period longer than several spike intervals
    # to ensure membrane potential decays significantly
    dropout_duration = int(baseline_interval)
    dut._log.info(f"Applying zero current for {dropout_duration} cycles...")

    for cycle in range(dropout_duration):
        await tester.record_cycle(0)

    dropout_end_cycle = tester.cycle_count
    dropout_spikes = len(
        [t for t in tester.spike_times if dropout_start_cycle <= t < dropout_end_cycle]
    )
    dut._log.info(
        f"Dropout phase: {dropout_spikes} spikes during {dropout_duration} cycles"
    )

    # Phase 3: Recovery - measure how quickly normal spiking resumes
    dut._log.info("Phase 3: Recovery measurement...")
    recovery_start_cycle = tester.cycle_count

    # Measure time to first spike after recovery starts
    cycles_to_first_spike = 0
    first_spike_found = False

    for cycle in range(max_cycles_per_phase):
        await tester.record_cycle(steady_current)
        cycles_to_first_spike += 1
        if tester.spike_times and tester.spike_times[-1] >= recovery_start_cycle:
            first_spike_found = True
            break

    if not first_spike_found:
        assert (
            False
        ), f"No spike found during recovery phase after {cycles_to_first_spike} cycles"

    dut._log.info(f"Time to first spike after recovery: {cycles_to_first_spike} cycles")

    # Now measure several recovery intervals
    recovery_intervals = []
    for i in range(num_intervals):
        interval, cycles_used = await tester.measure_spike_interval(
            steady_current, max_cycles=8000
        )
        if interval is None:
            dut._log.warning(
                f"Failed to get recovery interval {i+1} after {cycles_used} cycles"
            )
            break
        recovery_intervals.append(interval)
        dut._log.info(f"Recovery interval {i+1}: {interval} cycles")

    if len(recovery_intervals) < 2:
        assert (
            False
        ), f"Could not measure recovery intervals: got {len(recovery_intervals)}"

    # Analysis
    avg_recovery_interval = np.mean(recovery_intervals)
    recovery_frequency = 1.0 / avg_recovery_interval

    dut._log.info(
        f"Initial avg interval: {baseline_interval:.2f} cycles, freq = {baseline_frequency:.4f}"
    )
    dut._log.info(
        f"Recovery avg interval: {avg_recovery_interval:.2f} cycles, freq = {recovery_frequency:.4f}"
    )
    dut._log.info(f"Cycles to first recovery spike: {cycles_to_first_spike}")

    # Memory effect analysis
    interval_ratio = (
        baseline_interval / avg_recovery_interval
    )  # >1 means recovery is faster
    frequency_ratio = recovery_frequency / baseline_frequency
    recovery_delay_ratio = cycles_to_first_spike / baseline_interval

    dut._log.info(f"Interval ratio (baseline/recovery): {interval_ratio:.2f}")
    dut._log.info(f"Frequency ratio (recovery/baseline): {frequency_ratio:.2f}")
    dut._log.info(
        f"Recovery delay ratio (delay/baseline_interval): {recovery_delay_ratio:.2f}"
    )

    # Generate plot
    # fig = tester.plot_results("Test 2: Dropout and Recovery Memory Effect")

    # # Add phase markers to the plot
    # ax1 = fig.axes[0]
    # ax1.axvline(
    #     x=dropout_start_cycle,
    #     color="orange",
    #     linestyle="--",
    #     alpha=0.7,
    #     label="Dropout Start",
    # )
    # ax1.axvline(
    #     x=recovery_start_cycle,
    #     color="green",
    #     linestyle="--",
    #     alpha=0.7,
    #     label="Recovery Start",
    # )
    # ax1.legend()

    # fig.savefig("test2_dropout_recovery.png", dpi=150, bbox_inches="tight")
    # plt.close(fig)

    # Assertions
    assert (
        len(tester.spike_times) >= 10
    ), f"Insufficient total spikes: {len(tester.spike_times)}"

    # For fractional-order neurons, we expect:
    # Recovery should be faster than baseline due to memory effects
    # The fractional-order memory should cause faster recovery

    # Check for memory effect - recovery frequency should be higher than baseline
    if frequency_ratio > 1.1:
        dut._log.info(
            f"MEMORY EFFECT DETECTED: Recovery frequency {frequency_ratio:.2f}x baseline"
        )
    else:
        # This should fail until we tune the parameters properly
        assert (
            False
        ), f"Expected memory effect: recovery frequency should be >10% higher than baseline. Got freq_ratio={frequency_ratio:.2f} (recovery={recovery_frequency:.4f}, baseline={baseline_frequency:.4f}). This indicates fractional-order memory parameters need tuning."

    # Additional check: recovery delay should also be reasonable
    if recovery_delay_ratio > 3.0:
        dut._log.warning(
            f"Recovery delay seems long: {recovery_delay_ratio:.2f}x baseline interval"
        )


@cocotb.test(skip=True)
async def test_parameter_sensitivity(dut):
    """Test 3: Parameter sensitivity check - should fail if neuron is too sensitive."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Test different current levels to find good operating range
    # Based on hardware parameters, use a more focused range
    current_levels = [1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30]
    max_cycles_per_test = (
        10000  # Maximum cycles per current level - allow for wide dynamic range
    )
    target_spikes = 5  # Try to get at least this many spikes for each current

    dut._log.info("Testing parameter sensitivity:")
    dut._log.info("This test should FAIL if the neuron is oversensitive")

    results = []

    for current_level in current_levels:
        # Reset
        dut.rst.value = 1
        await ClockCycles(dut.clk, 5)
        dut.rst.value = 0
        await ClockCycles(dut.clk, 5)

        # Apply current and count spikes
        spike_count = 0
        cycles_used = 0

        for cycle in range(max_cycles_per_test):
            dut.current.value = current_level
            await RisingEdge(dut.clk)
            cycles_used += 1

            if dut.spike.value:
                spike_count += 1

            # Stop early if we have enough spikes for a good measurement
            if spike_count >= target_spikes and cycles_used >= 100:
                break

        frequency = spike_count / cycles_used if cycles_used > 0 else 0
        results.append((current_level, spike_count, frequency, cycles_used))

        dut._log.info(
            f"Current {current_level:2d}: {spike_count:3d} spikes in {cycles_used:4d} cycles, freq = {frequency:.4f}"
        )

    # Analysis and assertions for proper sensitivity
    dut._log.info("\nSensitivity Analysis:")

    # Check for problematic responses
    frequencies = [freq for curr, spikes, freq, cycles in results]
    max_freq = max(frequencies) if frequencies else 0
    min_nonzero_freq = (
        min([f for f in frequencies if f > 0]) if any(f > 0 for f in frequencies) else 0
    )

    # Check that very small currents don't cause excessive spiking
    for current_level, spike_count, frequency, cycles_used in results:
        if (
            current_level <= 5 and frequency > 0.10
        ):  # More than 10% spike rate for small currents
            dut._log.error(
                f"OVERSENSITIVE: Current {current_level} gives {frequency:.4f} spike rate (>10%)"
            )
            assert (
                False
            ), f"Neuron too sensitive: current {current_level} gives {frequency:.4f} spike rate. Need to increase threshold or reduce gain."
        elif current_level <= 5 and frequency > 0.05:  # 5-10% spike rate
            dut._log.warning(
                f"SENSITIVE: Current {current_level} gives {frequency:.4f} spike rate (5-10%)"
            )

    # Check for flat response indicating oversensitivity or saturation
    # Group frequencies by rounded values to detect flat responses
    frequency_groups = {}
    for curr, spikes, freq, cycles in results:
        if freq > 0:  # Only count non-zero frequencies
            freq_rounded = round(freq, 3)  # Round to 3 decimal places
            if freq_rounded not in frequency_groups:
                frequency_groups[freq_rounded] = []
            frequency_groups[freq_rounded].append(curr)

    # Check for the most common frequency
    if frequency_groups:
        largest_group_size = max(
            len(currents) for currents in frequency_groups.values()
        )
        total_nonzero_results = sum(
            len(currents) for currents in frequency_groups.values()
        )

        if largest_group_size >= max(
            4, total_nonzero_results * 0.6
        ):  # 60% or more give same frequency, or 4+ currents
            # Find which frequency has the most occurrences
            most_common_freq = None
            for freq, currents in frequency_groups.items():
                if len(currents) == largest_group_size:
                    most_common_freq = freq
                    break

            dut._log.error(
                f"FLAT RESPONSE: {largest_group_size}/{total_nonzero_results} current levels give same frequency {most_common_freq:.3f}"
            )
            # Show which currents give the same response
            same_freq_currents = frequency_groups[most_common_freq]
            dut._log.error(f"Currents giving same frequency: {same_freq_currents}")

            assert (
                False
            ), f"Neuron shows flat response: {largest_group_size}/{total_nonzero_results} different currents give frequency {most_common_freq:.3f}. This indicates oversensitivity - need to reduce gain or increase threshold."

    # Check maximum frequency isn't too high (saturation at high currents is OK)
    high_current_results = [
        (curr, freq) for curr, spikes, freq, cycles in results if curr >= 20
    ]
    if high_current_results:
        max_high_current_freq = max(freq for curr, freq in high_current_results)
        if max_high_current_freq > 0.50:  # More than 50% spike rate at high current
            dut._log.warning(
                f"High saturation frequency: {max_high_current_freq:.4f} at high currents"
            )
            # This is actually OK for high currents, but let's log it

    # Check that we have a reasonable dynamic range
    nonzero_freqs = [freq for curr, spikes, freq, cycles in results if freq > 0]
    if len(nonzero_freqs) >= 2:
        freq_range = max(nonzero_freqs) - min(nonzero_freqs)
        if freq_range < 0.02:  # Less than 2% range
            dut._log.error(
                f"Frequency range too narrow: {freq_range:.4f} (from {min(nonzero_freqs):.4f} to {max(nonzero_freqs):.4f})"
            )
            assert (
                False
            ), f"Neuron shows narrow frequency range {freq_range:.4f}. This indicates poor sensitivity tuning."

    # Check for reasonable threshold behavior (some currents should give zero response)
    zero_response_currents = [
        curr for curr, spikes, freq, cycles in results if freq == 0
    ]
    if len(zero_response_currents) == 0 and len(results) > 5:
        dut._log.warning(
            "No current levels gave zero response - threshold might be too low"
        )
    elif len(zero_response_currents) > len(results) * 0.8:
        dut._log.error(
            f"THRESHOLD TOO HIGH: {len(zero_response_currents)}/{len(results)} current levels give zero response"
        )
        assert (
            False
        ), f"Threshold too high: {len(zero_response_currents)}/{len(results)} currents give zero spikes. Need to reduce THRESHOLD parameter or increase gain."

    dut._log.info("\nIdeal behavior should show:")
    dut._log.info("- Small currents (1-5) should give 0-5% spike rates")
    dut._log.info("- Medium currents (8-15) should give 5-30% spike rates")
    dut._log.info("- Large currents (20-30) should give 20-50% spike rates")
    dut._log.info("- Should show gradual increase, not flat response")
    dut._log.info("- Should have some currents with zero response (below threshold)")

    # If we get here, the sensitivity is acceptable
    dut._log.info(
        "Parameter sensitivity test PASSED - neuron has reasonable sensitivity"
    )
