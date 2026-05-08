import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from pathlib import Path

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent
train_dir = script_dir.parent
project_root = train_dir.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(train_dir))

from common.scripts.plot_styles import (
    OKABE_ITO,
    get_latex_figsize,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
)

from fractional_lif import FractionalLIF

def plot_sv_subthreshold(
    output_file,
    show_plot=False,
    include_python_05=False,
    use_log=False
):
    torch.set_default_dtype(torch.float64)

    results_dir = project_root / "common" / "sv" / "cocotb" / "tests" / "results"
    
    frac_csv = results_dir / "subthreshold_fractional_lif.csv"
    bitshift_csv = results_dir / "subthreshold_bitshift_lif.csv"
    
    has_frac = frac_csv.exists()
    has_bitshift = bitshift_csv.exists()
    
    if not has_frac and not has_bitshift:
        print("Error: No SV subthreshold CSVs found in common/sv/cocotb/tests/results/")
        return

    # Load SV data
    sv_frac_df = pd.read_csv(frac_csv) if has_frac else None
    sv_bitshift_df = pd.read_csv(bitshift_csv) if has_bitshift else None
    
    figsize = get_latex_figsize(width_scale=1.6, height_scale=0.85)
    fig, ax = plt.subplots(figsize=(figsize["width"], figsize["height"]))
    
    max_t = 0
    steps_charge = 50
    
    if has_frac:
        max_t = max(max_t, sv_frac_df["timestep"].max())
    if has_bitshift:
        max_t = max(max_t, sv_bitshift_df["timestep"].max())
    
    if include_python_05:
        total_steps = int(max_t) + 1
        
        inputs = torch.zeros(total_steps, 1)
        inputs[:steps_charge, 0] = 0.1
        
        lam = 0.1
        history_length = 250
        
        alpha_neuron = FractionalLIF(
            alpha=0.5,
            lam=lam,
            history_length=history_length,
            threshold=1.0,
            init_hidden=True,
        )
        FractionalLIF.reset_hidden()
        
        trace_05 = []
        for t in range(total_steps):
            inp = inputs[t : t + 1]
            alpha_neuron(inp)
            trace_05.append(alpha_neuron.mem.item())
            
        ax.plot(
            range(total_steps),
            trace_05,
            label=r"Python $\alpha=0.5$",
            color=OKABE_ITO[0],  # Okabe-Ito Blue
            linewidth=2.5,
            alpha=0.6,
        )
        
    # Plot SV Fractional
    if has_frac:
        ax.plot(
            sv_frac_df["timestep"],
            sv_frac_df["membrane_potential"],
            label=r"SV Fractional ($\alpha=0.5$)",
            color=OKABE_ITO[1],  # Orange
            linewidth=1.5,
            linestyle="--",
        )
        
    # Plot SV Bitshift
    if has_bitshift:
        ax.plot(
            sv_bitshift_df["timestep"],
            sv_bitshift_df["membrane_potential"],
            label="SV Bitshift (Custom Slow Decay)",
            color=OKABE_ITO[7],  # Black
            linewidth=1.5,
            linestyle=":",
        )
        
    if not use_log:
        ax.axvline(
            steps_charge, color="gray", linestyle="--", alpha=0.7, label="Current off"
        )
        
    ax.set_xlabel("Time Step", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Membrane Potential", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Start axis limits slightly above 0 for log scale
        ax.set_xlim(left=0.5)
        ax.set_ylim(bottom=1e-4)
    else:
        ax.set_xlim(0, max_t)
        ax.set_ylim(bottom=0)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, format="svg", bbox_inches="tight")
        print(f"Saved plot to {output_file}")
    
    if show_plot:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot subthreshold dynamics from SystemVerilog simulations."
    )
    parser.add_argument(
        "--output", type=str, default="common/images/plot_sv_subthreshold.svg", help="Output file path"
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the plot interactively"
    )
    parser.add_argument(
        "--python-baseline",
        action="store_true",
        help="Include Python FractionalLIF (alpha=0.5) baseline for comparison",
    )
    parser.add_argument(
        "--log", action="store_true", help="Use log-log scale for axes"
    )
    
    args = parser.parse_args()
    
    plot_sv_subthreshold(
        output_file=args.output,
        show_plot=args.show,
        include_python_05=args.python_baseline,
        use_log=args.log
    )
