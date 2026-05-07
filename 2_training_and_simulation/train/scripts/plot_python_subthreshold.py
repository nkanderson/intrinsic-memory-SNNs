import argparse
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Also add the train directory so we can import fractional_lif
TRAIN_DIR = ROOT_DIR / "2_training_and_simulation" / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from fractional_lif import FractionalLIF
import snntorch as snn
from common.scripts.plot_styles import (
    OKABE_ITO,
    COLOR_RAW,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    get_latex_figsize,
)


def plot_subthreshold_dynamics(
    output_file,
    show_plot=False,
    include_baseline=False,
    use_log=False,
    steps_discharge=None,
):
    torch.set_default_dtype(torch.float64)
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    dt = 1.0
    lam = 0.1

    if steps_discharge is None:
        steps_discharge = 2000 if use_log else 150

    # The history buffer must be larger than the total simulation time (charge + discharge)
    # to ensure the initial charge pulse doesn't fall off the back of the memory buffer.
    history_length = steps_discharge + 500
    threshold = 1000.0  # High to prevent spiking

    # 50 steps of charge (current=1.0), dynamic discharge to see tail
    steps_charge = 50
    total_steps = steps_charge + steps_discharge

    inputs = torch.zeros(total_steps, 1)
    inputs[:steps_charge, 0] = 1.0

    traces = {}

    for alpha in alphas:
        neuron = FractionalLIF(
            alpha=alpha,
            lam=lam,
            history_length=history_length,
            dt=dt,
            threshold=threshold,
            init_hidden=True,
        )
        FractionalLIF.reset_hidden()

        mem_trace = []
        for t in range(total_steps):
            inp = inputs[t : t + 1]
            neuron(inp)
            mem_trace.append(neuron.mem.item())

        traces[alpha] = mem_trace

    if include_baseline:
        beta = 1.0 / (1.0 + lam)
        leaky = snn.Leaky(beta=beta, threshold=threshold, init_hidden=True)

        leaky_trace = []
        for t in range(total_steps):
            # FractionalLIF divides the entire right-hand side (including the input) by (C + lam).
            # Because C = 1.0 (dt=1.0), the denominator is (1.0 + lam).
            # snnTorch.Leaky adds the input unscaled, so we must manually divide the input
            # by (1.0 + lam) to ensure an apples-to-apples comparison.
            scaled_input = inputs[t : t + 1] / (1.0 + lam)
            leaky(scaled_input)
            leaky_trace.append(leaky.mem.item())

        traces["Leaky"] = leaky_trace

    figsize = get_latex_figsize(width_scale=1.6, height_scale=0.85)
    fig, ax = plt.subplots(figsize=(figsize["width"], figsize["height"]))

    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")

        if include_baseline:
            ax.plot(
                range(1, steps_discharge + 1),
                traces["Leaky"][steps_charge:],
                label="snnTorch.Leaky",
                color=COLOR_RAW,
                linewidth=2.5,
                linestyle="--",
                zorder=5,
            )

        for i, alpha in enumerate(alphas):
            color = OKABE_ITO[i % len(OKABE_ITO)]
            ax.plot(
                range(1, steps_discharge + 1),
                traces[alpha][steps_charge:],
                label=f"$\\alpha$ = {alpha}",
                color=color,
                linewidth=1.5,
            )

        ax.set_title(
            "FractionalLIF Discharge Dynamics (Log-Log Scale)",
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        ax.set_xlabel("Time Step (Since Discharge Start)", fontsize=AXIS_LABEL_FONTSIZE)

    else:
        if include_baseline:
            ax.plot(
                range(total_steps),
                traces["Leaky"],
                label="snnTorch.Leaky",
                color=COLOR_RAW,
                linewidth=2.5,
                linestyle="--",
                zorder=5,
            )

        for i, alpha in enumerate(alphas):
            color = OKABE_ITO[i % len(OKABE_ITO)]
            ax.plot(
                range(total_steps),
                traces[alpha],
                label=f"$\\alpha$ = {alpha}",
                color=color,
                linewidth=1.5,
            )

        # Draw a dashed vertical line to indicate when current turns off
        ax.axvline(
            steps_charge, color="gray", linestyle="--", alpha=0.7, label="Current off"
        )

        ax.set_title(
            "FractionalLIF Sub-threshold Dynamics (Charge and Discharge)",
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        ax.set_xlabel("Time Step", fontsize=AXIS_LABEL_FONTSIZE)

    ax.set_ylabel("Membrane Potential", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="best")

    plt.tight_layout()
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path, format=output_path.suffix.lstrip(".") or "svg", bbox_inches="tight"
    )
    print(f"Saved plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot sub-threshold dynamics for FractionalLIF with varying alpha."
    )
    parser.add_argument(
        "--output",
        default="common/images/plot_python_subthreshold.svg",
        help="Output plot path",
    )
    parser.add_argument("--show", action="store_true", help="Show interactive window")
    parser.add_argument(
        "--baseline", action="store_true", help="Include snnTorch.Leaky baseline"
    )
    parser.add_argument(
        "--log", action="store_true", help="Plot discharge phase on log-log scale"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of discharge steps (default: 2000 if --log, 150 otherwise)",
    )
    args = parser.parse_args()

    output_path = args.output
    if args.log and output_path == "common/images/plot_python_subthreshold.svg":
        output_path = "common/images/plot_python_subthreshold_log.svg"

    plot_subthreshold_dynamics(
        output_path, args.show, args.baseline, args.log, args.steps
    )
