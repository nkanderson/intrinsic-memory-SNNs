import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from scipy.optimize import curve_fit

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


def _power_law(t, a, p):
    """Power-law decay V(t) = a * t^(-p)."""
    return a * np.power(t, -p)


def _exponential(t, a, tau):
    """Exponential decay V(t) = a * exp(-t / tau)."""
    return a * np.exp(-t / tau)


def _fit_decay_params(df, steps_charge, func, p0, fit_from=None):
    """Fit `func` to the decay phase and return its parameters.

    Elapsed time is always measured from the first step after the current turns
    off (tau = timestep - steps_charge + 1), so a reference line built from these
    params lines up with the data regardless of where the fit started. When
    `fit_from` is given the fit is restricted to timestep >= fit_from; this skips
    the current-off transient so the power law captures the asymptotic tail
    rather than being dragged shallow by the initial drop. Zero-valued samples
    (the fixed-point floor the bitshift hits) are dropped so the fit stays valid
    on a log scale.
    """
    start = steps_charge if fit_from is None else fit_from
    decay = df[df["timestep"] >= start]
    timesteps = decay["timestep"].values
    y = decay["membrane_potential"].values
    tau = timesteps - steps_charge + 1

    mask = y > 0
    popt, _ = curve_fit(func, tau[mask], y[mask], p0=p0, maxfev=10000)
    return popt


def _reference_line(steps_charge, draw_from, max_t, func, popt):
    """Evaluate a fitted reference form over [draw_from, max_t].

    tau stays anchored to current-off (steps_charge) so the curve is consistent
    with the data; `draw_from` only controls where the drawn line begins (e.g.
    the power-law tail reference starts after the transient instead of blowing up
    near tau=1).
    """
    timesteps = np.arange(draw_from, int(max_t) + 1)
    tau = timesteps - steps_charge + 1
    return timesteps, func(tau, *popt)


def _fit_shared_references(dfs, steps_charge):
    """Fit ONE exponential and ONE power law jointly to the pooled decay of all
    traces. Used for the short-window comparison, where the fractional and
    bitshift are not yet separable: a single basic exponential and power law sit
    close to both curves, rather than each curve getting its own best fit.
    Returns (exp_params, power_law_params).
    """
    taus, ys = [], []
    for df in dfs:
        decay = df[df["timestep"] >= steps_charge]
        tau = decay["timestep"].values - steps_charge + 1
        y = decay["membrane_potential"].values
        mask = y > 0
        taus.append(tau[mask])
        ys.append(y[mask])
    tau = np.concatenate(taus)
    y = np.concatenate(ys)
    exp_p, _ = curve_fit(_exponential, tau, y, p0=[0.4, 30.0], maxfev=20000)
    pl_p, _ = curve_fit(_power_law, tau, y, p0=[0.4, 0.5], maxfev=20000)
    return exp_p, pl_p


def plot_sv_subthreshold(
    output_file,
    show_plot=False,
    include_python_05=False,
    use_log=False,
    fits=True,
    fit_mode="per_curve",
    power_law_tail_skip=100,
    max_timestep=None,
):
    torch.set_default_dtype(torch.float64)

    # Read the tracked, canonical copies under common/metrics (the cocotb tests
    # write to common/sv/cocotb/tests/results/, which is git-ignored; data is
    # promoted here once finalized so the figure regenerates without re-running).
    data_dir = project_root / "common" / "metrics"

    frac_csv = data_dir / "subthreshold_fractional_lif.csv"
    bitshift_csv = data_dir / "subthreshold_bitshift_lif.csv"

    has_frac = frac_csv.exists()
    has_bitshift = bitshift_csv.exists()

    if not has_frac and not has_bitshift:
        print("Error: No SV subthreshold CSVs found in common/metrics/")
        return

    # Load SV data
    sv_frac_df = pd.read_csv(frac_csv) if has_frac else None
    sv_bitshift_df = pd.read_csv(bitshift_csv) if has_bitshift else None

    # Optionally truncate to a shorter window (e.g. to reproduce the original
    # 200-step figure from the current longer run -- the early steps are
    # essentially identical across history-length/precision configs).
    if max_timestep is not None:
        if has_frac:
            sv_frac_df = sv_frac_df[sv_frac_df["timestep"] <= max_timestep]
        if has_bitshift:
            sv_bitshift_df = sv_bitshift_df[sv_bitshift_df["timestep"] <= max_timestep]

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

    # Reference forms overlaid for comparison. Two modes:
    #  - "per_curve" (long-history figure): the power law is fit to the fractional
    #    tail and the exponential to the bitshift, so each trace rides the form it
    #    obeys and the two clearly separate.
    #  - "shared" (short-window figure): a single basic power law and exponential,
    #    fit to both traces pooled, to show that at short history the two neurons
    #    -- and the two decay forms -- are nearly indistinguishable (they only
    #    separate on a log axis or at longer history).
    if fits and fit_mode == "shared":
        dfs = [d for d in (sv_frac_df, sv_bitshift_df) if d is not None]
        exp_p, pl_p = _fit_shared_references(dfs, steps_charge)
        # Skip the first couple of post-current-off steps so the power law's
        # tau^-p spike at tau=1 doesn't overshoot the shared curve.
        draw_from = steps_charge + 2
        t_pl, y_pl = _reference_line(steps_charge, draw_from, max_t, _power_law, pl_p)
        ax.plot(
            t_pl,
            y_pl,
            label=rf"Power law ($\propto t^{{-{pl_p[1]:.2f}}}$)",
            color=OKABE_ITO[2],  # Bluish green
            linewidth=2.0,
            linestyle="--",
            alpha=0.9,
        )
        t_exp, y_exp = _reference_line(
            steps_charge, draw_from, max_t, _exponential, exp_p
        )
        ax.plot(
            t_exp,
            y_exp,
            label=rf"Exponential ($\tau={exp_p[1]:.0f}$)",
            color=OKABE_ITO[4],  # Sky blue
            linewidth=2.0,
            linestyle="-",
            alpha=0.9,
        )
    elif fits:
        # per_curve: power law fit to (and drawn over) the fractional tail only --
        # the leaky dt=1 GL neuron is power-law in its tail but not Mittag-Leffler
        # over the whole decay -- and the exponential fit to the bitshift, spanning
        # the full window so it visibly plummets below the persisting fractional.
        tail_start = steps_charge + power_law_tail_skip
        have_tail = has_frac and int((sv_frac_df["timestep"] >= tail_start).sum()) > 5
        if have_tail:
            a_pl, p = _fit_decay_params(
                sv_frac_df, steps_charge, _power_law, p0=[0.4, 1.0], fit_from=tail_start
            )
            t_pl, y_pl = _reference_line(
                steps_charge, tail_start, max_t, _power_law, (a_pl, p)
            )
            ax.plot(
                t_pl,
                y_pl,
                label=rf"Power law ($\propto t^{{-{p:.2f}}}$)",
                color=OKABE_ITO[2],  # Bluish green
                linewidth=2.0,
                linestyle="-",
                alpha=0.9,
            )

        if has_bitshift:
            a_exp, tau = _fit_decay_params(
                sv_bitshift_df, steps_charge, _exponential, p0=[0.4, 20.0]
            )
            t_exp, y_exp = _reference_line(
                steps_charge, steps_charge, max_t, _exponential, (a_exp, tau)
            )
            ax.plot(
                t_exp,
                y_exp,
                label=rf"Exponential ($\tau={tau:.0f}$)",
                color=OKABE_ITO[4],  # Sky blue
                linewidth=2.0,
                linestyle="-",
                alpha=0.9,
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
        "--output",
        type=str,
        default="common/images/plot_sv_subthreshold-1024.svg",
        help="Output file path (default is the long-history HISTORY_LENGTH=1024 "
        "figure; the base name plot_sv_subthreshold.svg holds the original "
        "short-window plot)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the plot interactively"
    )
    parser.add_argument(
        "--python-baseline",
        action="store_true",
        help="Include Python FractionalLIF (alpha=0.5) baseline for comparison",
    )
    parser.add_argument("--log", action="store_true", help="Use log-log scale for axes")
    parser.add_argument(
        "--fits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay power-law (fractional) and exponential (bitshift) fits",
    )
    parser.add_argument(
        "--fit-mode",
        choices=["per_curve", "shared"],
        default="per_curve",
        help="per_curve: power law fit to the fractional tail + exponential to "
        "the bitshift (long-history figure). shared: one power law + one "
        "exponential fit to both traces pooled (short-window figure).",
    )
    parser.add_argument(
        "--power-law-tail-skip",
        type=int,
        default=100,
        help="Discharge steps after current-off to skip before fitting the "
        "power-law tail (excludes the current-off transient)",
    )
    parser.add_argument(
        "--max-timestep",
        type=int,
        default=None,
        help="Truncate the plot to timesteps <= this value (e.g. 199 to "
        "reproduce the original short-window figure). Combine with --no-fits.",
    )

    args = parser.parse_args()

    plot_sv_subthreshold(
        output_file=args.output,
        show_plot=args.show,
        include_python_05=args.python_baseline,
        use_log=args.log,
        fits=args.fits,
        fit_mode=args.fit_mode,
        power_law_tail_skip=args.power_law_tail_skip,
        max_timestep=args.max_timestep,
    )
