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


def _load_points(
    study: optuna.study.Study,
    only_converged: bool,
    x_axis: str,
) -> list[tuple[int, float, int]]:
    points: list[tuple[int, float, int]] = []
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE:
            continue
        if only_converged and trial.user_attrs.get("convergence_episode") is None:
            continue
        tail_iqm = trial.user_attrs.get("tail_iqm_avg_reward")
        if tail_iqm is None:
            continue
        params = trial.params or {}
        if x_axis == "history_length":
            hist = _find_param(params, HIST_KEYS)
            if hist is None:
                continue
            points.append((hist, float(tail_iqm), trial.number))
        else:
            hl1 = _find_param(params, HL1_KEYS)
            hl2 = _find_param(params, HL2_KEYS)
            if hl1 is None or hl2 is None:
                continue
            total = hl1 + hl2
            points.append((total, float(tail_iqm), trial.number))
    return points


def _savefig(fig: plt.Figure, out_dir: Path, stem: str, fmt: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.{fmt}"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Optuna tail-IQM vs total neurons or history length."
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
        "--threshold",
        type=float,
        default=450.0,
        help="Threshold for tail_iqm_avg_reward reference line (default: 450.0)",
    )
    parser.add_argument(
        "--only-converged",
        action="store_true",
        help="Include only trials with convergence_episode set.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["total_neurons", "history_length"],
        default="total_neurons",
        help="X-axis metric (default: total_neurons).",
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
        help="Filename stem for the output plot.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "svg"],
        default="png",
        help="Output format (default: png).",
    )
    args = parser.parse_args()

    study_specs = _parse_study_specs(args)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, constrained_layout=True)

    for idx, spec in enumerate(study_specs):
        study_name = spec["name"]
        storage = spec["storage"]
        study = optuna.load_study(study_name=study_name, storage=storage)

        points = _load_points(study, args.only_converged, args.x_axis)
        if not points:
            print(f"No eligible trials found for {study_name}.")
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        label = _label_from_study(study_name)
        style = NEURON_STYLE.get(label, {})
        color = style.get("color", OKABE_ITO[idx % len(OKABE_ITO)])
        marker = style.get("marker", COMPARISON_MARKERS[idx % len(COMPARISON_MARKERS)])

        ax.scatter(
            xs,
            ys,
            s=42,
            marker=marker,
            color=color,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.85,
            label=label,
        )

    ax.axhline(
        args.threshold,
        color=THRESHOLD_COLOR,
        linestyle="--",
        linewidth=1.6,
        label=f"threshold = {args.threshold:g}",
    )

    if args.x_axis == "history_length":
        ax.set_xlabel("History length", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xticks([0, 4, 8, 16, 32, 64, 128])
    else:
        ax.set_xlabel("Total neurons (hl1 + hl2)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Tail IQM avg reward", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.6, color=COLOR_RAW, alpha=0.35)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False)

    out_dir = Path(args.output_dir)
    stem = args.output_stem
    if args.x_axis == "history_length" and stem == "optuna_tail_iqm_vs_size":
        stem = "optuna_tail_iqm_vs_history"
    if args.only_converged:
        stem = f"{stem}-converged"

    out_path = _savefig(fig, out_dir, stem, args.format)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
