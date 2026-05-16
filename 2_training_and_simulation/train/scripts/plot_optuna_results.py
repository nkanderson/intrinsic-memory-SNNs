"""
Plot Optuna tail-IQM against total network size or history length.

Usage examples:
    python scripts/plot_optuna_results.py \
        --study leaky-iqm@sqlite:///optuna_studies/leaky-v3.db \
        --study fractional-iqm@sqlite:///optuna_studies/fractional-v3.db \
        --study bitshift-custom-slow-iqm@sqlite:///optuna_studies/bitshift-v3.db \
        --threshold 350

    python scripts/plot_optuna_results.py --study-name leaky-iqm \
        --storage sqlite:///optuna_studies/leaky-v3.db --format svg

    python scripts/plot_optuna_results.py --study-name fractional-iqm \
        --storage sqlite:///optuna_studies/fractional-v3.db --x-axis history_length
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.trial import TrialState

# ── Path setup for shared modules ──────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[3]
TRAIN_DIR = Path(__file__).resolve().parents[1]
for _p in (ROOT_DIR, TRAIN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.scripts.plot_styles import (  # noqa: E402
    OKABE_ITO,
    COLOR_RAW,
    COMPARISON_MARKERS,
    DEFAULT_FIGSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
)
from optuna_analysis import HL1_KEYS, HL2_KEYS, HIST_KEYS, _find_param  # noqa: E402

THRESHOLD_COLOR = OKABE_ITO[1]

NEURON_STYLE = {
    "leaky": {"color": OKABE_ITO[0], "marker": COMPARISON_MARKERS[0]},
    "fractional": {"color": OKABE_ITO[2], "marker": COMPARISON_MARKERS[1]},
    "bitshift": {"color": OKABE_ITO[5], "marker": COMPARISON_MARKERS[2]},
}


def _parse_study_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    if args.study:
        if len(args.study) > 3:
            raise SystemExit("Error: maximum of 3 --study entries is supported.")
        for item in args.study:
            if "@" in item:
                name, storage = item.split("@", 1)
                specs.append({"name": name, "storage": storage})
            else:
                name = item
                storage = f"sqlite:///optuna_studies/{name}.db"
                specs.append({"name": name, "storage": storage})
        return specs

    if not args.study_name:
        raise SystemExit("Error: provide --study-name or one or more --study entries.")

    storage = args.storage or f"sqlite:///optuna_studies/{args.study_name}.db"
    specs.append({"name": args.study_name, "storage": storage})
    return specs


def _label_from_study(study_name: str) -> str:
    base = study_name
    lower = base.lower()
    for suffix in ("-iqm", "_iqm", "iqm"):
        if lower.endswith(suffix):
            base = base[: -len(suffix)].rstrip("-_")
            break
    for sep in ("-", "_"):
        if sep in base:
            base = base.split(sep, 1)[0]
            break
    return base or study_name


# Three-way classification of trials by the selection gate from
# select_top_candidates.py. The gate has two parts:
#   (1) convergence_episode is not None (the convergence definition itself
#       guarantees the trailing-100 avg never drops below 475 from that
#       episode onward), and
#   (2) total_episodes - convergence_episode >= K (the sustained-stability
#       interval is at least K episodes long).
# A trial that meets (1) but not (2) is "converged_marginal" — the trial did
# converge in the formal sense, but the demonstrated sustained interval was
# shorter than K and we don't trust it.
GATE_PASSING = "gate_passing"
CONVERGED_MARGINAL = "converged_marginal"
NEVER_CONVERGED = "never_converged"


def _gate_status(
    trial: optuna.trial.FrozenTrial,
    min_sustained_episodes: int,
    num_episodes_fallback: int,
) -> str:
    conv = trial.user_attrs.get("convergence_episode")
    if conv is None:
        return NEVER_CONVERGED
    total = trial.user_attrs.get("total_episodes") or num_episodes_fallback
    if conv > total - min_sustained_episodes:
        return CONVERGED_MARGINAL
    return GATE_PASSING


def _load_points(
    study: optuna.study.Study,
    only_converged: bool,
    x_axis: str,
    min_sustained_episodes: int,
    num_episodes_fallback: int,
) -> list[tuple[int, float, int, str]]:
    """Return (x, tail_iqm, trial_number, gate_status) tuples."""
    points: list[tuple[int, float, int, str]] = []
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE:
            continue
        status = _gate_status(trial, min_sustained_episodes, num_episodes_fallback)
        if only_converged and status == NEVER_CONVERGED:
            continue
        tail_iqm = trial.user_attrs.get("tail_iqm_avg_reward")
        if tail_iqm is None:
            continue
        params = trial.params or {}
        if x_axis == "history_length":
            hist = _find_param(params, HIST_KEYS)
            if hist is None:
                continue
            points.append((hist, float(tail_iqm), trial.number, status))
        else:
            hl1 = _find_param(params, HL1_KEYS)
            hl2 = _find_param(params, HL2_KEYS)
            if hl1 is None or hl2 is None:
                continue
            total = hl1 + hl2
            points.append((total, float(tail_iqm), trial.number, status))
    return points


def _savefig(fig: plt.Figure, out_dir: Path, stem: str, fmt: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.{fmt}"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_scatter(args, study_specs, out_dir: Path) -> Path:
    """Tail-IQM vs (total_neurons | history_length) scatter.

    Gate status is encoded as marker fill:
      - gate-passing       : filled, full alpha, black edge
      - converged_marginal : filled, low alpha, black edge
      - never_converged    : hollow (no facecolor), colored edge

    Alternative encoding considered: shade a horizontal band on the y-axis
    over the tail-IQM range achievable by clean-late-converging trials
    (math-driven, depends on K), to make the gate's effect on tail-IQM
    explicit. Not implemented here because the band depends on assumptions
    about pre-convergence average and has to be computed per study; the
    marker-fill encoding tells the same story without those assumptions.
    """
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, constrained_layout=True)

    for idx, spec in enumerate(study_specs):
        study_name = spec["name"]
        storage = spec["storage"]
        study = optuna.load_study(study_name=study_name, storage=storage)

        points = _load_points(
            study,
            args.only_converged,
            args.x_axis,
            args.min_sustained_episodes,
            args.num_episodes_fallback,
        )
        if not points:
            print(f"No eligible trials found for {study_name}.")
            continue

        label = _label_from_study(study_name)
        style = NEURON_STYLE.get(label, {})
        color = style.get("color", OKABE_ITO[idx % len(OKABE_ITO)])
        marker = style.get("marker", COMPARISON_MARKERS[idx % len(COMPARISON_MARKERS)])

        by_status = {GATE_PASSING: [], CONVERGED_MARGINAL: [], NEVER_CONVERGED: []}
        for x, y, _tn, status in points:
            by_status[status].append((x, y))

        n_total = len(points)
        n_gate = len(by_status[GATE_PASSING])
        legend_label = f"{label} [gate: {n_gate}/{n_total}]"

        # Plot in z-order: never_converged (back), marginal, gate-passing (front).
        if by_status[NEVER_CONVERGED]:
            xs, ys = zip(*by_status[NEVER_CONVERGED])
            ax.scatter(
                xs, ys,
                s=38, marker=marker,
                facecolors="none", edgecolors=color, linewidths=1.0,
                alpha=0.7,
            )
        if by_status[CONVERGED_MARGINAL]:
            xs, ys = zip(*by_status[CONVERGED_MARGINAL])
            ax.scatter(
                xs, ys,
                s=42, marker=marker,
                color=color, edgecolors="black", linewidths=0.4,
                alpha=0.35,
            )
        # The gate-passing scatter carries the per-study legend label so the
        # legend uses the same marker style as the headline points.
        gp_xs, gp_ys = zip(*by_status[GATE_PASSING]) if by_status[GATE_PASSING] else ([], [])
        ax.scatter(
            gp_xs, gp_ys,
            s=48, marker=marker,
            color=color, edgecolors="black", linewidths=0.5,
            alpha=0.95,
            label=legend_label,
        )

    # Status-key entries (neutral, one per gate state). Drawn as a single
    # extra scatter call each so they appear in the legend.
    from matplotlib.lines import Line2D
    status_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="0.4", markeredgecolor="black",
               markersize=7, markeredgewidth=0.5,
               label="gate-passing"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="0.4", markeredgecolor="black",
               markersize=7, markeredgewidth=0.5, alpha=0.35,
               label="converged, sub-K-buffer"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="none", markeredgecolor="0.4",
               markersize=7, markeredgewidth=1.0,
               label="never converged"),
    ]
    # Combine the auto-collected study handles with the status key.
    auto_handles, auto_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=auto_handles + status_handles,
        labels=auto_labels + [h.get_label() for h in status_handles],
        fontsize=LEGEND_FONTSIZE, frameon=False,
    )

    if args.x_axis == "history_length":
        ax.set_xlabel("History length", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xticks([0, 4, 8, 16, 32, 64, 128])
    else:
        ax.set_xlabel("Total neurons (hl1 + hl2)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Tail IQM avg reward", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.6, color=COLOR_RAW, alpha=0.35)

    stem = args.output_stem
    if args.x_axis == "history_length" and stem == "optuna_tail_iqm_vs_size":
        stem = "optuna_tail_iqm_vs_history"
    if args.only_converged:
        stem = f"{stem}-converged"

    return _savefig(fig, out_dir, stem, args.format)


def _plot_convergence_strip(args, study_specs, out_dir: Path) -> Path:
    """Per-study strip plot of convergence_episode.

    For each study, one column on the x-axis; converged trials show as
    dots at y=convergence_episode. Marker fill encodes gate status. The
    K-buffer cutoff for each study is drawn as a horizontal segment at
    y = total_episodes - K (gate-passing trials are at or below this
    line; sub-buffer trials are above it). Never-converged trials are
    annotated as a count below the x-tick (they have no y-value).
    """
    fig_w = max(DEFAULT_FIGSIZE[0], 1.6 * len(study_specs))
    fig, ax = plt.subplots(figsize=(fig_w, DEFAULT_FIGSIZE[1]), constrained_layout=True)
    rng = np.random.default_rng(0)

    column_labels = []
    K = args.min_sustained_episodes
    all_totals: set[int] = set()

    for x_idx, spec in enumerate(study_specs):
        study_name = spec["name"]
        storage = spec["storage"]
        study = optuna.load_study(study_name=study_name, storage=storage)

        label = _label_from_study(study_name)
        column_labels.append(label)
        style = NEURON_STYLE.get(label, {})
        color = style.get("color", OKABE_ITO[x_idx % len(OKABE_ITO)])
        marker = style.get("marker", COMPARISON_MARKERS[x_idx % len(COMPARISON_MARKERS)])

        gate_ys = []
        marg_ys = []
        n_never = 0
        totals = set()
        for trial in study.trials:
            if trial.state != TrialState.COMPLETE:
                continue
            status = _gate_status(
                trial, args.min_sustained_episodes, args.num_episodes_fallback
            )
            if status == NEVER_CONVERGED:
                n_never += 1
                continue
            conv = trial.user_attrs.get("convergence_episode")
            total = trial.user_attrs.get("total_episodes") or args.num_episodes_fallback
            totals.add(total)
            all_totals.add(total)
            if status == GATE_PASSING:
                gate_ys.append(conv)
            else:
                marg_ys.append(conv)

        # K-buffer cutoff lines, one per distinct total_episodes value present
        # in this study's converged trials (usually one value).
        for total in totals:
            ax.plot(
                [x_idx - 0.3, x_idx + 0.3],
                [total - K, total - K],
                color=color, linewidth=1.4, alpha=0.6,
            )
            # Top-of-training reference (thin dotted)
            ax.plot(
                [x_idx - 0.3, x_idx + 0.3],
                [total, total],
                color=color, linewidth=0.8, linestyle=":", alpha=0.5,
            )

        if gate_ys:
            xs = x_idx + rng.uniform(-0.1, 0.1, size=len(gate_ys))
            ax.scatter(
                xs, gate_ys,
                s=52, marker=marker,
                color=color, edgecolors="black", linewidths=0.5,
                alpha=0.95,
            )
        if marg_ys:
            xs = x_idx + rng.uniform(-0.1, 0.1, size=len(marg_ys))
            ax.scatter(
                xs, marg_ys,
                s=46, marker=marker,
                color=color, edgecolors="black", linewidths=0.4,
                alpha=0.35,
            )

        n_total = len(gate_ys) + len(marg_ys) + n_never
        ax.text(
            x_idx, 0.02,
            f"gate {len(gate_ys)}/{n_total}\nno-conv {n_never}",
            ha="center", va="bottom",
            fontsize=LEGEND_FONTSIZE,
            transform=ax.get_xaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="none", alpha=0.9),
        )

    ax.set_xticks(range(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=20, ha="right",
                       fontsize=TICK_LABEL_FONTSIZE)
    ax.set_ylabel("Convergence episode", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3, axis="y")
    # Force y-range to span [0, max total_episodes] so the K-buffer cutoff
    # line and "all converged trials cluster in the last few hundred ep"
    # are both visible; without this matplotlib auto-scales tightly to the
    # trial cluster and the gate cutoff visually overlaps with the dots.
    if all_totals:
        ax.set_ylim(0, max(all_totals) * 1.04)

    # Status key — same marker semantics as the scatter plot.
    from matplotlib.lines import Line2D
    status_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="0.4", markeredgecolor="black",
               markersize=7, markeredgewidth=0.5,
               label="gate-passing"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="0.4", markeredgecolor="black",
               markersize=7, markeredgewidth=0.4, alpha=0.35,
               label="converged, sub-K-buffer"),
        Line2D([0], [0], color="0.4", linewidth=1.4, alpha=0.6,
               label=f"K-buffer cutoff (K={K})"),
        Line2D([0], [0], color="0.4", linewidth=0.8, linestyle=":", alpha=0.5,
               label="total episodes"),
    ]
    ax.legend(handles=status_handles, fontsize=LEGEND_FONTSIZE,
              frameon=False, loc="center left")

    return _savefig(fig, out_dir, "optuna_convergence_strip", args.format)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Optuna tail-IQM scatter or convergence-episode strip."
    )
    parser.add_argument(
        "--study",
        action="append",
        default=[],
        help=(
            "Study spec in the form <study_name>@<storage>. Repeat up to 3 times. "
            "If storage is omitted, defaults to sqlite:///optuna_studies/<study_name>.db."
        ),
    )
    parser.add_argument("--study-name", help="Optuna study name")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL for --study-name",
    )
    parser.add_argument(
        "--plot",
        choices=["scatter", "convergence_strip"],
        default="scatter",
        help="Plot type. 'scatter' (default) is the tail-IQM vs size/history "
        "scatter with gate-status encoding. 'convergence_strip' shows per-"
        "study convergence_episode distributions with the K-buffer cutoff.",
    )
    parser.add_argument(
        "--min-sustained-episodes",
        type=int,
        default=200,
        help="K: matches the gate in select_top_candidates.py (default: 200). "
        "Drives marker-fill classification and the cutoff line.",
    )
    parser.add_argument(
        "--num-episodes-fallback",
        type=int,
        default=1500,
        help="Fallback num_episodes for older trials lacking the "
        "'total_episodes' user_attr (default: 1500). Pass 2500 for older "
        "v3 trials run under the longer episode budget.",
    )
    parser.add_argument(
        "--only-converged",
        action="store_true",
        help="Scatter only: hide never-converged trials.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["total_neurons", "history_length"],
        default="total_neurons",
        help="Scatter only: x-axis metric (default: total_neurons).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="images/optuna_analysis",
        help="Output directory for plots (default: images/optuna_analysis).",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="optuna_tail_iqm_vs_size",
        help="Filename stem for the scatter plot. Ignored for the "
        "convergence_strip plot, which always writes to "
        "optuna_convergence_strip.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "svg"],
        default="png",
        help="Output format (default: png).",
    )
    args = parser.parse_args()

    study_specs = _parse_study_specs(args)
    out_dir = Path(args.output_dir)

    if args.plot == "convergence_strip":
        out_path = _plot_convergence_strip(args, study_specs, out_dir)
    else:
        out_path = _plot_scatter(args, study_specs, out_dir)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
