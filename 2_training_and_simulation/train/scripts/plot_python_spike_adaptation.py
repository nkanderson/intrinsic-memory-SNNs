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


def plot_spike_adaptation(output_file, show_plot=False):
    torch.set_default_dtype(torch.float64)
    alphas = [1.0, 0.9, 0.7, 0.5, 0.3]
    dt = 1.0
    history_length = 250
    threshold = 1.0
    total_steps = 300

    input_current = 0.3
    inputs = torch.ones(total_steps, 1) * input_current

    isi_traces = {}

    # 1. First run the standard snnTorch.Leaky for baseline comparison
    # FractionalLIF effectively scales input by 1 / (1 + lam) and beta = 1 / (1 + lam)
    lam = 0.1
    beta = 1.0 / (1.0 + lam)
    leaky = snn.Leaky(beta=beta, threshold=threshold, init_hidden=True)

    spikes = []
    for t in range(total_steps):
        # Scale input to match FractionalLIF denominator logic
        scaled_input = inputs[t : t + 1] / (1.0 + lam)
        spk = leaky(scaled_input)
        if spk.item() > 0:
            spikes.append(t)
    isis = [spikes[i] - spikes[i - 1] for i in range(1, len(spikes))]
    isi_traces["Leaky"] = isis

    # 2. Run FractionalLIF for various alphas
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

        spikes = []
        for t in range(total_steps):
            inp = inputs[t : t + 1]
            spk = neuron(inp)
            if spk.item() > 0:
                spikes.append(t)

        isis = [spikes[i] - spikes[i - 1] for i in range(1, len(spikes))]
        isi_traces[alpha] = isis

    plot_keys = ["Leaky"] + alphas
    figsize = get_latex_figsize(width_scale=1.6, height_scale=0.85 * len(plot_keys))
    fig, axes = plt.subplots(
        len(plot_keys), 1, figsize=(figsize["width"], figsize["height"]), sharex=True
    )

    if len(plot_keys) == 1:
        axes = [axes]

    for i, key in enumerate(plot_keys):
        ax = axes[i]

        if key == "Leaky":
            color = COLOR_RAW
            label = "snnTorch.Leaky Baseline"
        else:
            color = OKABE_ITO[i % len(OKABE_ITO)]
            label = f"$\\alpha$ = {key}"

        isis = isi_traces[key]
        if len(isis) > 0:
            ax.plot(
                range(1, len(isis) + 1),
                isis,
                marker="o",
                markersize=4,
                label=label,
                color=color,
                linewidth=1.5,
            )
        else:
            ax.plot(
                [],
                [],
                marker="o",
                markersize=4,
                label=label + " (No spikes)",
                color=color,
                linewidth=1.5,
            )

        ax.set_ylabel("ISI", fontsize=TICK_LABEL_FONTSIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")

        # Only set y-axis integer ticks if max ISI is small
        if len(isis) > 0 and max(isis) <= 10:
            ax.set_yticks(range(0, max(isis) + 2, 2))

    axes[0].set_title(
        "FractionalLIF Spike Frequency Adaptation", fontsize=AXIS_LABEL_FONTSIZE
    )
    axes[-1].set_xlabel("Spike Index", fontsize=AXIS_LABEL_FONTSIZE)

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
        description="Plot spike frequency adaptation (ISI) for FractionalLIF."
    )
    parser.add_argument(
        "--output",
        default="common/images/plot_python_spike_adaptation.svg",
        help="Output plot path",
    )
    parser.add_argument("--show", action="store_true", help="Show interactive window")
    args = parser.parse_args()

    plot_spike_adaptation(args.output, args.show)
