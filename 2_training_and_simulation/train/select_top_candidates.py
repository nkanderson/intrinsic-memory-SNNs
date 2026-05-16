"""
Select the top-N Optuna trials that demonstrably converged stably, and
export each as a standard nested-YAML config usable by main.py /
multi_seed_train.py.

Why top-N (not just study.best_trial)? The single best trial might have
gotten lucky on its random seed. Retraining the top-N with multiple seeds
each (via multi_seed_train.py) is more robust.

Selection structure (two-tier):

  Gate (hard filter — a trial must pass ALL of these):
    - state == COMPLETE                                  (skips PRUNED / FAILED)
    - convergence_episode is not None                    (reached and held >=475)
    - convergence_episode <= num_episodes - K            (sustained for >= K ep)

  Rank survivors by --rank-by (default tail_iqm):
    - tail_iqm  -> tail_iqm_avg_reward (IQM of trailing-100 averages over
                   the last 25% of training; outlier-robust per Agarwal
                   et al. 2021)
    - final_avg -> final_avg_reward
    - objective -> trial.value (penalized objective)

The convergence gate (Bellemare et al. 2013 / Machado et al. 2018 ALE
convention, strengthened with a K-episode stability buffer) does the
'did it converge?' work; the rank metric measures 'how well did it sit
once converged?'. A separate metric floor is intentionally not used: it
would mechanically penalize clean late-converging trials whose tail-IQM
is dragged down by the still-learning prefix.

Usage:
    # In-flight v3 study (defaults: --storage matches <neuron-type>-v3.db,
    # --study-name matches <neuron-type>):
    python select_top_candidates.py --neuron-type leaky --top-n 3

    # Stricter stability requirement (must converge 500 ep before end):
    python select_top_candidates.py --neuron-type leaky \\
        --min-sustained-episodes 500 --top-n 3

    # Older finished study (override storage):
    python select_top_candidates.py --neuron-type leaky \\
        --storage sqlite:///optuna_studies/leaky.db --top-n 3

    # Dry run — print the matching trials but do not write configs:
    python select_top_candidates.py --neuron-type leaky --top-n 3 --dry-run
"""

import argparse
from datetime import datetime
from pathlib import Path

import optuna
import yaml

from optimize import params_to_config_yaml

DEFAULT_NUM_EPISODES = 1500
DEFAULT_SURROGATE_SLOPE = 25
DEFAULT_BETA = 0.9
DEFAULT_ALPHA = 0.5
DEFAULT_DT = 1.0

NEURON_TYPE_MAP = {
    "bitshift-custom_slow_decay": "bitshift",
}

SHIFT_FUNC_BY_NEURON_TYPE = {
    "bitshift-custom_slow_decay": "custom_slow_decay",
}


def storage_uri_for(neuron_type: str) -> str:
    """Default: the in-flight -v3 SQLite file."""
    return f"sqlite:///optuna_studies/{neuron_type}-v3.db"


def storage_filename_stem(storage_uri: str) -> str:
    """Extract 'leaky-v3' from 'sqlite:///optuna_studies/leaky-v3.db'.

    The exported config filename is derived from this so that v3 studies
    produce optimized-leaky-v3-topN.yaml (avoiding overwrite of older
    optimized-leaky-topN.yaml configs).
    """
    after_scheme = (
        storage_uri.split("///", 1)[-1] if "///" in storage_uri else storage_uri
    )
    return Path(after_scheme).stem


def _rank_metric(trial, metric: str):
    """Return the metric to rank survivors by.

    'tail_iqm'  -> tail_iqm_avg_reward (preferred — IQM over the tail,
                   robust to single lucky/unlucky windows)
    'final_avg' -> final_avg_reward
    'objective' -> trial.value (penalized objective)
    """
    if metric == "tail_iqm":
        return trial.user_attrs.get("tail_iqm_avg_reward")
    if metric == "final_avg":
        return trial.user_attrs.get("final_avg_reward")
    if metric == "objective":
        return trial.value
    raise ValueError(f"Unknown metric: {metric}")


def filter_and_rank(
    study,
    rank_by: str,
    min_sustained_episodes: int,
    num_episodes_fallback: int,
):
    """Apply the convergence gate and return survivors sorted by rank_by.

    Gate: COMPLETE AND convergence_episode is not None
          AND convergence_episode <= num_episodes - min_sustained_episodes

    num_episodes is read from trial.user_attrs["total_episodes"] for trials
    instrumented by current optimize.py. Older trials (before that
    instrumentation) don't carry it — for those, num_episodes_fallback is
    used. 'num_episodes' is a 'fixed' search-space value and so isn't in
    trial.params.
    """
    complete = study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
    )
    eligible = []
    skipped_no_metric = 0
    rejected_no_conv = 0
    rejected_unsustained = 0

    for t in complete:
        conv_ep = t.user_attrs.get("convergence_episode")
        if conv_ep is None:
            rejected_no_conv += 1
            continue

        num_eps = t.user_attrs.get("total_episodes") or num_episodes_fallback
        if conv_ep > num_eps - min_sustained_episodes:
            rejected_unsustained += 1
            continue

        rank_value = _rank_metric(t, rank_by)
        if rank_value is None:
            skipped_no_metric += 1
            continue

        eligible.append(t)

    print(
        f"  gate summary: {len(complete)} COMPLETE -> "
        f"{rejected_no_conv} no convergence, "
        f"{rejected_unsustained} converged but sustained < "
        f"{min_sustained_episodes} ep, "
        f"{skipped_no_metric} missing rank metric, "
        f"{len(eligible)} survived"
    )

    eligible.sort(key=lambda t: _rank_metric(t, rank_by), reverse=True)
    return eligible


def apply_required_defaults(config: dict, neuron_type: str) -> None:
    training = config.setdefault("training", {})
    snn = config.setdefault("snn", {})

    if "num_episodes" not in training:
        training["num_episodes"] = DEFAULT_NUM_EPISODES

    if "surrogate_gradient_slope" not in snn:
        snn["surrogate_gradient_slope"] = DEFAULT_SURROGATE_SLOPE

    if "neuron_type" not in snn:
        snn["neuron_type"] = NEURON_TYPE_MAP.get(neuron_type, neuron_type)

    if "beta" not in snn and snn["neuron_type"] in ("fractional", "bitshift"):
        snn["beta"] = DEFAULT_BETA

    if "alpha" not in snn and snn["neuron_type"] in ("fractional", "bitshift"):
        snn["alpha"] = DEFAULT_ALPHA

    if "dt" not in snn and snn["neuron_type"] in ("fractional", "bitshift"):
        snn["dt"] = DEFAULT_DT

    if snn["neuron_type"] == "bitshift" and "shift_func" not in snn:
        default_shift = SHIFT_FUNC_BY_NEURON_TYPE.get(neuron_type)
        if default_shift is not None:
            snn["shift_func"] = default_shift


def export_trial_config(
    trial,
    output_path: Path,
    study_name: str,
    rank: int,
    neuron_type: str,
):
    config = params_to_config_yaml(trial.params)
    apply_required_defaults(config, neuron_type)
    final_avg = trial.user_attrs.get("final_avg_reward")
    best_avg = trial.user_attrs.get("best_avg_reward")
    conv_ep = trial.user_attrs.get("convergence_episode")

    header_lines = [
        "# Auto-generated by select_top_candidates.py",
        f"# Study: {study_name}",
        f"# Trial: {trial.number} (rank {rank})",
        f"# Objective value (penalized): {trial.value:.2f}",
    ]
    if final_avg is not None:
        header_lines.append(f"# final_avg_reward: {final_avg:.2f}")
    if best_avg is not None:
        header_lines.append(f"# best_avg_reward:  {best_avg:.2f}")
    if conv_ep is not None:
        header_lines.append(f"# convergence_episode: {conv_ep}")
    header_lines.append(f"# Generated: {datetime.now().isoformat()}")
    header_lines.append("#")
    header_lines.append("# Multi-seed retrain:")
    header_lines.append(f"#   python multi_seed_train.py --config {output_path}")
    header = "\n".join(header_lines) + "\n\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description="Filter Optuna trials by threshold and export top-N configs."
    )
    parser.add_argument(
        "--neuron-type",
        type=str,
        required=True,
        choices=["leaky", "fractional", "bitshift", "bitshift-custom_slow_decay"],
        help="Neuron type. Drives default --storage and --study-name.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name inside the SQLite DB. If omitted and the DB "
        "contains exactly one study, it is auto-selected; otherwise the "
        "available study names are listed and the script exits.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="SQLite URI. Defaults to sqlite:///optuna_studies/<neuron-type>-v3.db.",
    )
    parser.add_argument(
        "--min-sustained-episodes",
        type=int,
        default=200,
        help="K: a trial must have convergence_episode <= num_episodes - K "
        "to pass the stability gate (default: 200 — about 13%% of a 1500-ep "
        "run). The CartPole convergence rule already requires the trailing-"
        "100 avg to never drop below 475 after convergence, so K episodes "
        "of demonstrated stability above threshold are guaranteed.",
    )
    parser.add_argument(
        "--num-episodes-fallback",
        type=int,
        default=1500,
        help="Fallback num_episodes for older trials lacking the "
        "'total_episodes' user_attr (default: 1500). Trials run after the "
        "total_episodes instrumentation read it from user_attrs. Pass 2500 "
        "(or whatever the v2-era setting was) when re-analyzing older DBs.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top trials to export (default: 3).",
    )
    parser.add_argument(
        "--rank-by",
        choices=["tail_iqm", "final_avg", "objective"],
        default="tail_iqm",
        help="Metric to rank gate-passing trials by. 'tail_iqm' (default, "
        "recommended) uses tail_iqm_avg_reward — the Interquartile Mean of "
        "trailing-100 averages over the final 25%% of training. "
        "'final_avg' is the trailing-100 avg at the last episode. "
        "'objective' uses trial.value with size/history penalties.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override output directory. By default, configs are written to "
            "configs/optimized-<label>/ where <label> is --study-name if provided, "
            "otherwise the storage DB name (e.g., leaky-v3)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matching trials but do not write any config files.",
    )
    args = parser.parse_args()

    storage = args.storage or storage_uri_for(args.neuron_type)
    stem = storage_filename_stem(storage)

    available = optuna.get_all_study_names(storage=storage)
    if not available:
        print(f"\nNo studies found in storage: {storage}")
        return

    if args.study_name is None:
        if len(available) == 1:
            study_name = available[0]
            print(f"  (auto-selected the single study in this DB: {study_name!r})")
        else:
            print(f"\nMultiple studies in {storage}:")
            for n in available:
                print(f"  - {n}")
            print(
                "\nPick one with --study-name <name>. "
                "(There is no fixed convention across DBs; older DBs use "
                "timestamped names.)"
            )
            return
    else:
        study_name = args.study_name
        if study_name not in available:
            print(f"\nStudy {study_name!r} not in {storage}. Available studies:")
            for n in available:
                print(f"  - {n}")
            return

    output_label = args.study_name if args.study_name else stem
    if args.output_dir is None:
        output_dir = Path("configs") / f"optimized-{output_label}"
    else:
        output_dir = Path(args.output_dir)

    print(f"Loading Optuna study:")
    print(f"  study_name             : {study_name}")
    print(f"  storage                : {storage}")
    print(f"  rank_by                : {args.rank_by}")
    print(f"  min_sustained_episodes : {args.min_sustained_episodes}")
    print(f"  top_n                  : {args.top_n}")

    study = optuna.load_study(study_name=study_name, storage=storage)
    eligible = filter_and_rank(
        study,
        rank_by=args.rank_by,
        min_sustained_episodes=args.min_sustained_episodes,
        num_episodes_fallback=args.num_episodes_fallback,
    )

    if not eligible:
        print(
            "\nNo trials survived the convergence gate "
            f"(min_sustained_episodes={args.min_sustained_episodes})."
        )
        return

    print(f"\nMatched {len(eligible)} trials. Top {min(args.top_n, len(eligible))}:")
    print(
        f"{'rank':>4}  {'trial':>5}  {'objective':>10}  "
        f"{'tail_iqm':>9}  {'final_avg':>10}  {'best_avg':>9}  {'conv_ep':>8}"
    )
    for i, t in enumerate(eligible[: args.top_n], start=1):
        final_avg = t.user_attrs.get("final_avg_reward")
        best_avg = t.user_attrs.get("best_avg_reward")
        tail_iqm = t.user_attrs.get("tail_iqm_avg_reward")
        conv = t.user_attrs.get("convergence_episode")
        conv_str = f"{conv}" if conv is not None else "-"
        print(
            f"{i:>4}  {t.number:>5}  {t.value:>10.2f}  "
            f"{(tail_iqm if tail_iqm is not None else 0):>9.2f}  "
            f"{(final_avg if final_avg is not None else 0):>10.2f}  "
            f"{(best_avg if best_avg is not None else 0):>9.2f}  "
            f"{conv_str:>8}"
        )

    if args.dry_run:
        print("\n(dry-run: no configs written)")
        return

    print(f"  output_dir : {output_dir}")
    print()
    for rank, trial in enumerate(eligible[: args.top_n], start=1):
        out_path = output_dir / f"optimized-{output_label}-top{rank}.yaml"
        export_trial_config(trial, out_path, study_name, rank, args.neuron_type)
        print(f"  wrote: {out_path}")


if __name__ == "__main__":
    main()
