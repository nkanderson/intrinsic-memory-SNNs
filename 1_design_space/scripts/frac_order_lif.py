"""
Fractional-order Leaky Integrate-and-Fire (LIF) neuron model in Python.

This script mirrors the behavior of the SystemVerilog implementation in src/frac_order_lif.sv
but uses floating point arithmetic instead of fixed point.

The model implements the discrete-time fractional-order LIF equation:
V[n] = (1/(1+(tau/h^alpha))) * [R*I[n] - (tau/h^alpha) *
       sum(k=1 to history_size)(-1^k * (alpha choose k) * V[n-k])]

Usage:
    python frac_order_lif.py [--current INPUT_CURRENT] [--steps NUM_STEPS] [--plot] [--csv FILENAME]

Example:
    python frac_order_lif.py --current 20 --steps 1000 --plot --csv spike_cycles.csv
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import csv


class FractionalOrderLIF:
    """Fractional-order LIF neuron implementation in Python."""

    def __init__(
        self,
        alpha=8 / 15,  # Fractional order parameter
        tau=20.0,  # Time constant
        h=1.0,  # Time step
        R_scaled=9,  # Resistance scaling factor
        threshold=75,  # Spike threshold
        refractory_period=5,  # Refractory period in steps
        history_size=1024,
    ):

        self.alpha = alpha
        self.tau = tau
        self.h = h
        self.R_scaled = R_scaled
        self.threshold = threshold
        self.refractory_period = refractory_period
        self.history_size = history_size

        # Calculate derived parameters
        self.tau_over_h_alpha = tau / (h**alpha)
        self.norm_factor = 1.0 / (1.0 + self.tau_over_h_alpha)

        # Pre-compute binomial coefficients for k=1 to history_size
        self.coefficients = self._compute_binomial_coefficients()

        # State variables
        self.membrane_potential = 0.0
        self.history_buffer = deque([0.0] * history_size, maxlen=history_size)
        self.refractory_counter = 0
        self.spike_times = []
        self.membrane_history = []
        self.pre_spike_potentials = {}  # Store membrane potential just before spike
        self.spike_cycle_potentials = (
            []
        )  # Store membrane potentials for each spike cycle
        self.current_cycle_potentials = []  # Current cycle's membrane potentials
        self.time_step = 0

        print(f"Fractional-order LIF neuron initialized:")
        print(f"  alpha = {self.alpha:.4f}")
        print(f"  tau = {self.tau}")
        print(f"  R_scaled = {self.R_scaled}")
        print(f"  threshold = {self.threshold}")
        print(f"  history_size = {self.history_size}")
        print(f"  tau/h^alpha = {self.tau_over_h_alpha:.4f}")
        print(f"  norm_factor = {self.norm_factor:.4f}")
        # print(f"  Coefficients: {[f'{c:.8f}' for c in self.coefficients]}")
        print(f"  First 8 Coefficients: {[f'{c:.8f}' for c in self.coefficients[:8]]}")
        print(f"  Last 8 Coefficients: {[f'{c:.8f}' for c in self.coefficients[-8:]]}")

    def _compute_binomial_coefficients(self):
        """Compute generalized binomial coefficients (alpha choose k) for k=1 to history_size."""
        coefficients = []

        for k in range(1, self.history_size + 1):
            # Generalized binomial coefficient: (alpha choose k) = alpha*(alpha-1)*...*(alpha-k+1) / k!
            coeff = 1.0
            for i in range(k):
                coeff *= (self.alpha - i) / (i + 1)

            coefficients.append(coeff)

        return coefficients

    def _calculate_fractional_sum(self):
        """Calculate the fractional derivative sum using history buffer."""
        fractional_sum = 0.0

        # Sum over k=1 to min(time_step, history_size)
        # The SystemVerilog uses parallel multiplication; here we use a loop
        # Only iterate up to current time_step if it's less than the history size
        # to avoid unnecessary iterations when history is not yet full.
        for k in range(1, min(self.time_step + 1, self.history_size + 1)):
            # Get V[n-k] from history buffer (most recent is at index -1)
            history_idx = -k  # -1 is V[n-1], -2 is V[n-2], etc.
            history_value = self.history_buffer[history_idx]

            # Apply coefficient with proper sign: (-1)^k * (alpha choose k) * V[n-k]
            # The SystemVerilog stores magnitude only and applies the final negative effect
            # For 0 < alpha < 1: all final weights should be negative (memory decay)
            coeff_magnitude = self.coefficients[
                k - 1
            ]  # k-1 because coefficients array is 0-indexed

            # Apply alternating sign: (-1)^k where k starts at 1
            # k=1: (-1)^1 = -1, k=2: (-1)^2 = +1, k=3: (-1)^3 = -1, etc.
            alternating_sign = (-1) ** k

            # For fractional alpha in (0,1), the binomial coefficients alternate in sign
            # and the combined effect with (-1)^k typically results in negative weights
            term = alternating_sign * coeff_magnitude * history_value
            fractional_sum += term

        return fractional_sum

    def _calculate_membrane_potential(self, input_current):
        """Calculate the membrane potential using the fractional-order LIF equation."""
        current_term = self.R_scaled * input_current
        fractional_sum = self._calculate_fractional_sum()

        # In the SystemVerilog: history_term = (tau/h^alpha) * fractional_sum
        # The equation is: R*I[n] - (tau/h^alpha) * sum(...)
        # So we subtract the history term
        history_term = self.tau_over_h_alpha * fractional_sum

        unnormalized_potential = current_term - history_term  # Note the minus sign here
        updated_potential = self.norm_factor * unnormalized_potential

        # Apply saturation (8-bit in SystemVerilog)
        return min(255.0, max(0.0, updated_potential))

    def update(self, input_current):
        """Update the neuron for one time step with the given input current."""
        self.time_step += 1
        spike_occurred = False

        # Check if in refractory period
        in_refractory = self.refractory_counter > 0

        if in_refractory:
            self.refractory_counter -= 1

            # SystemVerilog behavior: update history and membrane potential during refractory
            if self.refractory_counter == 1:  # Last cycle of refractory
                # Add current membrane potential to history
                self.history_buffer.append(self.membrane_potential)

                # Calculate new membrane potential
                updated_potential = self._calculate_membrane_potential(input_current)

                # Update membrane potential
                self.membrane_potential = updated_potential

                # Track refractory potential for next cycle
                self.current_cycle_potentials.append(self.membrane_potential)

            elif self.refractory_counter == 0:  # Just exited refractory
                # Add current membrane potential to history
                self.history_buffer.append(self.membrane_potential)

                # Track refractory potential for next cycle
                self.current_cycle_potentials.append(self.membrane_potential)

        else:
            # Normal operation
            # Calculate membrane potential using fractional-order equation
            updated_potential = self._calculate_membrane_potential(input_current)

            # Check for spike BEFORE updating membrane potential
            spike_occurred = updated_potential >= self.threshold

            if spike_occurred:
                # Add the membrane potential that caused the spike to current cycle
                self.current_cycle_potentials.append(updated_potential)

                # Store the potential that caused the spike
                self.pre_spike_potentials[self.time_step] = updated_potential

                # Record spike at current time step
                self.spike_times.append(self.time_step)

                # Save this spike cycle's data
                self.spike_cycle_potentials.append(self.current_cycle_potentials.copy())
                self.current_cycle_potentials = []  # Reset for next cycle

                # Add current membrane potential to history before reset
                self.history_buffer.append(self.membrane_potential)

                # Reset membrane potential and enter refractory period
                self.membrane_potential = 0.0  # Reset to 0 as in SystemVerilog
                self.refractory_counter = self.refractory_period
            else:
                # Normal update - add to current cycle
                self.current_cycle_potentials.append(updated_potential)

                # Add current membrane potential to history
                self.history_buffer.append(self.membrane_potential)

                # Update membrane potential
                self.membrane_potential = updated_potential

        # Record membrane potential for plotting (AFTER potential spike reset)
        self.membrane_history.append(self.membrane_potential)

        return self.membrane_potential, spike_occurred

    def simulate(self, input_currents, num_steps=None):
        """Simulate the neuron for multiple time steps."""
        if num_steps is None:
            num_steps = len(input_currents)

        results = []
        spikes = []

        for step in range(num_steps):
            # Get input current (cycle through if input_currents is shorter)
            if isinstance(input_currents, (list, np.ndarray)):
                current = input_currents[step % len(input_currents)]
            else:
                current = input_currents  # Constant current

            membrane_potential, spike_occurred = self.update(current)
            results.append(membrane_potential)
            spikes.append(spike_occurred)

        return np.array(results), np.array(spikes)

    def plot_results(self, input_currents=None, title="Fractional-Order LIF Neuron"):
        """Plot membrane potential and spikes."""
        if not self.membrane_history:
            print("No simulation data to plot. Run simulate() first.")
            return

        time_axis = np.arange(len(self.membrane_history))

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

        # Create a modified membrane potential trace that shows spikes correctly
        # For spike visualization, we want to show the membrane potential reaching threshold
        # before dropping to 0, rather than just showing the post-spike reset value
        display_membrane = self.membrane_history.copy()

        # For each spike time, modify the display to show the actual potential that caused the spike
        for spike_time in self.spike_times:
            if (
                spike_time - 1 < len(display_membrane)
                and spike_time in self.pre_spike_potentials
            ):
                # Set the membrane potential at the time step just before the reset
                # to show the actual value that triggered the spike
                display_membrane[spike_time - 1] = self.pre_spike_potentials[spike_time]

        # Plot membrane potential on left y-axis
        ax1.plot(
            time_axis,
            display_membrane,
            "b-",
            linewidth=1.5,
            label="Membrane Potential",
        )
        ax1.axhline(
            y=self.threshold,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Threshold ({self.threshold})",
        )

        # Mark spike times
        if self.spike_times:
            # Show spikes at threshold level at the correct time
            spike_y = [self.threshold] * len(self.spike_times)
            # Adjust spike time by -1 to show at the moment threshold was reached
            adjusted_spike_times = [max(0, t - 1) for t in self.spike_times]
            ax1.scatter(
                adjusted_spike_times,
                spike_y,
                color="red",
                marker="v",
                s=50,
                label="Spikes",
                zorder=5,
            )

        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Membrane Potential", color="b")
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(255, max(self.membrane_history) * 1.1))
        ax1.tick_params(axis="y", labelcolor="b")

        # Create second y-axis for input current
        ax2 = ax1.twinx()

        # Plot input current on right y-axis
        if input_currents is not None:
            if isinstance(input_currents, (list, np.ndarray)):
                # Extend input_currents to match simulation length
                extended_currents = []
                for i in range(len(self.membrane_history)):
                    extended_currents.append(input_currents[i % len(input_currents)])
                ax2.plot(
                    time_axis,
                    extended_currents,
                    "g-",
                    linewidth=1.5,
                    label="Input Current",
                    alpha=0.7,
                )
            else:
                # Constant current
                ax2.axhline(
                    y=input_currents,
                    color="g",
                    linewidth=1.5,
                    label=f"Input Current ({input_currents})",
                    alpha=0.7,
                )

        ax2.set_ylabel("Input Current", color="g")
        ax2.tick_params(axis="y", labelcolor="g")

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # Add statistics
        num_spikes = len(self.spike_times)
        firing_rate = (
            num_spikes / len(self.membrane_history) if self.membrane_history else 0
        )

        stats_text = f"Spikes: {num_spikes}\nFiring Rate: {firing_rate:.4f} spikes/step"
        ax1.text(
            0.02,
            0.98,
            stats_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()

    def export_spike_cycles_to_csv(self, filename="spike_cycles.csv"):
        """
        Export membrane potential values for each spike cycle to a CSV file.

        Each row represents one spike cycle, containing all membrane potential values
        calculated leading up to that spike.

        Args:
            filename (str): Output CSV filename
        """
        if not self.spike_cycle_potentials:
            print(
                "No spike cycle data to export. Run simulate() first and ensure spikes occurred."
            )
            return

        # Find the maximum cycle length to determine number of columns
        max_cycle_length = (
            max(len(cycle) for cycle in self.spike_cycle_potentials)
            if self.spike_cycle_potentials
            else 0
        )

        # Create headers
        headers = [f"step_{i+1}" for i in range(max_cycle_length)]

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(["spike_number"] + headers)

            # Write data for each spike cycle
            for spike_num, cycle_potentials in enumerate(
                self.spike_cycle_potentials, 1
            ):
                # Pad shorter cycles with empty values
                padded_cycle = cycle_potentials + [""] * (
                    max_cycle_length - len(cycle_potentials)
                )
                writer.writerow([spike_num] + padded_cycle)

        print(f"Spike cycle data exported to {filename}")
        print(f"  Number of spike cycles: {len(self.spike_cycle_potentials)}")
        print(f"  Maximum cycle length: {max_cycle_length} steps")

        # Add summary statistics
        if self.spike_cycle_potentials:
            cycle_lengths = [len(cycle) for cycle in self.spike_cycle_potentials]
            print(f"  Average cycle length: {np.mean(cycle_lengths):.2f} steps")
            print(
                f"  Cycle length range: {min(cycle_lengths)} - {max(cycle_lengths)} steps"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Fractional-order LIF neuron simulation"
    )
    parser.add_argument(
        "--current",
        type=float,
        default=35.0,
        help="Input current value (default: 35.0)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1500,
        help="Number of simulation steps",
    )
    parser.add_argument("--plot", action="store_true", help="Show plot of results")
    parser.add_argument(
        "--alpha",
        type=float,
        default=5 / 15,
        help="Fractional order parameter (0 < alpha < 1)",
    )
    parser.add_argument(
        "--tau", type=float, default=25.0, help="Time constant (default: 5.0)"
    )
    parser.add_argument(
        "--threshold", type=float, default=60, help="Spike threshold (default: 75)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Export spike cycle data to CSV file (e.g., --csv spike_cycles.csv)",
    )

    args = parser.parse_args()

    # Create neuron
    neuron = FractionalOrderLIF(
        alpha=args.alpha, tau=args.tau, threshold=args.threshold
    )

    # Run simulation
    print(
        f"\nRunning simulation with constant current {args.current} for {args.steps} steps..."
    )
    membrane_potentials, spikes = neuron.simulate(args.current, args.steps)

    # Print results
    num_spikes = np.sum(spikes)
    firing_rate = num_spikes / args.steps

    print("\nSimulation Results:")
    print(f"  Total spikes: {num_spikes}")
    print(f"  Spikes from spike_times list: {len(neuron.spike_times)}")
    print(f"  Spike times: {neuron.spike_times[:10]}...")  # Show first 10
    print(f"  Firing rate: {firing_rate:.4f} spikes/step")
    print(f"  Final membrane potential: {membrane_potentials[-1]:.2f}")

    if args.plot:
        neuron.plot_results(input_currents=args.current)

    if args.csv:
        neuron.export_spike_cycles_to_csv(args.csv)


if __name__ == "__main__":
    main()
