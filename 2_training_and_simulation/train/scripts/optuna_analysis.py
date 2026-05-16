"""
Analyze Optuna studies for trials meeting a metric threshold.

Usage examples:
    python optuna_analysis.py --study-name leaky-iqm --storage sqlite:///optuna_studies/leaky-v3.db
    python optuna_analysis.py --study-name fractional-iqm --storage sqlite:///optuna_studies/fractional-v3.db \
            --threshold 350 --print-trials
    python optuna_analysis.py --study leaky-iqm@sqlite:///optuna_studies/leaky-v3.db \
            --study fractional-iqm@sqlite:///optuna_studies/fractional-v3.db \
            --study bitshift-custom-slow-iqm@sqlite:///optuna_studies/bitshift-v3.db
"""

from __future__ import annotations

import argparse
import re
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional
import optuna
from optuna.trial import TrialState

HL1_KEYS = [
    "hl1",
    "hidden1",
    "hidden_1",
    "hl_1",
    "n_hidden_1",
    "hidden_size_1",
    "hidden_layer1",
    "hidden_layer_1",
    "layer1",
    "layer_1",
    "hidden1_size",
]
HL2_KEYS = [
    "hl2",
    "hidden2",
    "hidden_2",
    "hl_2",
    "n_hidden_2",
    "hidden_size_2",
    "hidden_layer2",
    "hidden_layer_2",
    "layer2",
    "layer_2",
    "hidden2_size",
]
HIST_KEYS = ["hist", "history", "history_length", "history_len"]

METRIC_ATTRS = {
    "tail_iqm": "tail_iqm_avg_reward",
    "final_avg": "final_avg_reward",
}


def _extract_int_from_value(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v)
    m = re.search(r"(-?\d+)", s)
    if m:
        return int(m.group(1))
    return None


def _find_param(params: Dict[str, Any], candidates: Iterable[str]) -> Optional[int]:
    for k in candidates:
        for key in params.keys():
            if key.lower() == k.lower():
                val = _extract_int_from_value(params[key])
                if val is not None:
                    return val
    # try conservative fuzzy match by token overlap (avoids accidental matches
    # like 'h' hitting unrelated params such as batch_size/hidden_size)
    for key in params.keys():
        lk = key.lower()
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", lk) if tok]
        for cand in candidates:
            cl = cand.lower()
            if cl in tokens:
                val = _extract_int_from_value(params[key])
                if val is not None:
                    return val
    return None


def _trial_metric_value(
    trial: optuna.trial.FrozenTrial, metric: str
) -> Optional[float]:
    if metric == "objective":
        return float(trial.value) if trial.value is not None else None
    attr = METRIC_ATTRS.get(metric)
    if attr is None:
        return None
    value = trial.user_attrs.get(attr)
    return float(value) if value is not None else None


def analyze(
    study: optuna.study.Study,
    threshold: float,
    metric: str,
) -> Dict[str, Any]:
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    converged_trials = [
        t
        for t in complete_trials
        if t.user_attrs.get("convergence_episode") is not None
    ]
    success_trials = []
    missing_metric = 0
    for t in complete_trials:
        metric_value = _trial_metric_value(t, metric)
        if metric_value is None:
            missing_metric += 1
            continue
        if metric_value >= threshold:
            success_trials.append(t)

    converged_success = [
        t for t in success_trials if t.user_attrs.get("convergence_episode") is not None
    ]

    results: Dict[str, Any] = {
        "metric": metric,
        "metric_attr": METRIC_ATTRS.get(metric, "objective"),
        "n_complete": len(complete_trials),
        "converged_trials": converged_trials,
        "n_converged": len(converged_trials),
        "converged_success": converged_success,
        "n_converged_success": len(converged_success),
        "n_success": len(success_trials),
        "missing_metric": missing_metric,
    }
    if not success_trials:
        return results

    hl1_vals: List[int] = []
    hl2_vals: List[int] = []
    hist_vals: List[int] = []
    total_neurons: List[int] = []
    tail_iqm_vals: List[float] = []
    final_avg_vals: List[float] = []
    best_avg_vals: List[float] = []
    conv_ep_vals: List[float] = []
    metric_vals: List[float] = []
    trial_summaries: List[Dict[str, Any]] = []

    for t in success_trials:
        p = t.params or {}
        hl1 = _find_param(p, HL1_KEYS)
        hl2 = _find_param(p, HL2_KEYS)
        hist = _find_param(p, HIST_KEYS)
        tot = None
        if hl1 is not None and hl2 is not None:
            tot = hl1 + hl2
            total_neurons.append(tot)
        if hl1 is not None:
            hl1_vals.append(hl1)
        if hl2 is not None:
            hl2_vals.append(hl2)
        if hist is not None:
            hist_vals.append(hist)
        tail_iqm = t.user_attrs.get("tail_iqm_avg_reward")
        final_avg = t.user_attrs.get("final_avg_reward")
        best_avg = t.user_attrs.get("best_avg_reward")
        conv_ep = t.user_attrs.get("convergence_episode")
        metric_value = _trial_metric_value(t, metric)

        if tail_iqm is not None:
            tail_iqm_vals.append(float(tail_iqm))
        if final_avg is not None:
            final_avg_vals.append(float(final_avg))
        if best_avg is not None:
            best_avg_vals.append(float(best_avg))
        if conv_ep is not None:
            conv_ep_vals.append(float(conv_ep))
        if metric_value is not None:
            metric_vals.append(float(metric_value))

        trial_summaries.append(
            {
                "trial_number": t.number,
                "objective": t.value,
                "metric_value": metric_value,
                "tail_iqm": tail_iqm,
                "final_avg": final_avg,
                "best_avg": best_avg,
                "convergence_episode": conv_ep,
                "hl1": hl1,
                "hl2": hl2,
                "history": hist,
                "total_neurons": tot,
                "params": p,
            }
        )

    def _avg_or_none(xs: List[float]) -> Optional[float]:
        return float(mean(xs)) if xs else None

    results.update(
        {
            "avg_hl1": _avg_or_none(hl1_vals),
            "avg_hl2": _avg_or_none(hl2_vals),
            "avg_total_neurons": _avg_or_none(total_neurons),
            "avg_history": _avg_or_none(hist_vals),
            "avg_metric": _avg_or_none(metric_vals),
            "avg_tail_iqm": _avg_or_none(tail_iqm_vals),
            "avg_final_avg": _avg_or_none(final_avg_vals),
            "avg_best_avg": _avg_or_none(best_avg_vals),
            "avg_convergence_episode": _avg_or_none(conv_ep_vals),
            "trial_summaries": trial_summaries,
        }
    )
    # For studies that reference "leaky", also include the smallest-neuron
    # successful trial(s) in the returned results so callers can consume them
    # programmatically.
    try:
        study_name = (study.study_name or "").lower()
    except Exception:
        study_name = ""
    if "leaky" in study_name:
        total_candidates = [
            t for t in trial_summaries if t["total_neurons"] is not None
        ]
        if total_candidates:
            best_smallest_total = min(
                total_candidates,
                key=lambda t: (
                    t["total_neurons"],
                    -(t["metric_value"] or float("-inf")),
                    t["history"] if t["history"] is not None else 10**9,
                    t["trial_number"],
                ),
            )
            best_smallest_total_history_first = min(
                total_candidates,
                key=lambda t: (
                    t["total_neurons"],
                    t["history"] if t["history"] is not None else 10**9,
                    -(t["metric_value"] or float("-inf")),
                    t["trial_number"],
                ),
            )
            results["best_smallest_total"] = best_smallest_total
            results["best_smallest_total_history_first"] = (
                best_smallest_total_history_first
            )
    return results


def _parse_study_specs(args: argparse.Namespace) -> List[Dict[str, str]]:
    specs: List[Dict[str, str]] = []
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Optuna study successes")
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
        "--metric",
        choices=["tail_iqm", "final_avg", "objective"],
        default="tail_iqm",
        help="Metric used for thresholding and ranking (default: tail_iqm).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=450.0,
        help="Success threshold for the chosen metric (default: 450.0)",
    )
    parser.add_argument(
        "--print-trials",
        action="store_true",
        help="Print summary of individual successful trials",
    )
    args = parser.parse_args()

    metric_label = METRIC_ATTRS.get(args.metric, "objective")
    study_specs = _parse_study_specs(args)

    for idx, spec in enumerate(study_specs, start=1):
        study_name = spec["name"]
        storage = spec["storage"]

        if idx > 1:
            print("\n" + "=" * 72 + "\n")

        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Study: {study_name}")
        print(f"Storage: {storage}")
        print(f"Metric: {args.metric} ({metric_label})")
        print(f"Threshold: {args.threshold}")
        print(f"Total trials in study: {len(study.trials)}")

        out = analyze(study, args.threshold, args.metric)
        if out.get("missing_metric", 0):
            print(
                "Missing metric for COMPLETE trials: "
                f"{out['missing_metric']} (skipped)"
            )
        print(f"Converged trials (any metric): {out.get('n_converged', 0)}")
        converged_trials = out.get("converged_trials", [])
        if converged_trials:
            converged_ids = sorted(t.number for t in converged_trials)
            print(f"  trials: {converged_ids}")
        print(
            "Converged trials meeting threshold: "
            f"{out.get('n_converged_success', 0)}"
        )
        converged_success = out.get("converged_success", [])
        if converged_success:
            converged_success_ids = sorted(t.number for t in converged_success)
            print(f"  trials: {converged_success_ids}")
        print(f"Successful trials: {out.get('n_success', 0)}")
        if out.get("n_success", 0) == 0:
            continue

        print("Averages over successful trials:")
        if out["avg_metric"] is not None:
            print(f"  Avg {metric_label}: {out['avg_metric']:.2f}")
        if out["avg_tail_iqm"] is not None:
            print(f"  Avg tail_iqm_avg_reward: {out['avg_tail_iqm']:.2f}")
        if out["avg_final_avg"] is not None:
            print(f"  Avg final_avg_reward: {out['avg_final_avg']:.2f}")
        if out["avg_best_avg"] is not None:
            print(f"  Avg best_avg_reward: {out['avg_best_avg']:.2f}")
        if out["avg_convergence_episode"] is not None:
            print("  Avg convergence episode: " f"{out['avg_convergence_episode']:.1f}")
        else:
            print("  Avg convergence episode: N/A")
        if out["avg_hl1"] is not None:
            print(f"  Avg hidden layer 1 size: {out['avg_hl1']:.1f}")
        else:
            print("  Avg hidden layer 1 size: N/A")
        if out["avg_hl2"] is not None:
            print(f"  Avg hidden layer 2 size: {out['avg_hl2']:.1f}")
        else:
            print("  Avg hidden layer 2 size: N/A")
        if out["avg_total_neurons"] is not None:
            print(f"  Avg total neurons (hl1+hl2): {out['avg_total_neurons']:.1f}")
        else:
            print("  Avg total neurons (hl1+hl2): N/A")
        if out["avg_history"] is not None:
            print(f"  Avg history length: {out['avg_history']:.1f}")
        else:
            print("  Avg history length: N/A")

        if args.print_trials:
            print("\nSuccessful trial details:")
            for t in out["trial_summaries"]:
                metric_val = t["metric_value"]
                tail_iqm = t["tail_iqm"]
                final_avg = t["final_avg"]
                obj_val = t["objective"]
                conv_ep = t["convergence_episode"]
                print(f"  Trial #{t['trial_number']}")
                print(
                    "    metric: " f"{metric_val if metric_val is not None else 'N/A'}"
                )
                print(
                    "    tail_iqm_avg_reward: "
                    f"{tail_iqm if tail_iqm is not None else 'N/A'}"
                )
                print(
                    "    final_avg_reward: "
                    f"{final_avg if final_avg is not None else 'N/A'}"
                )
                print("    objective: " f"{obj_val if obj_val is not None else 'N/A'}")
                print("    hl1: " f"{t['hl1']}")
                print("    hl2: " f"{t['hl2']}")
                print("    total_neurons: " f"{t['total_neurons']}")
                print("    history: " f"{t['history']}")
                print(
                    "    convergence_episode: "
                    f"{conv_ep if conv_ep is not None else 'N/A'}"
                )

        summaries = out["trial_summaries"]

        # Trial with smallest total neurons (if hl1/hl2 are available)
        total_candidates = [t for t in summaries if t["total_neurons"] is not None]
        if total_candidates:
            best_smallest_total = min(
                total_candidates,
                key=lambda t: (
                    t["total_neurons"],
                    -(t["metric_value"] or float("-inf")),
                    t["history"] if t["history"] is not None else 10**9,
                    t["trial_number"],
                ),
            )
            best_smallest_total_history_first = min(
                total_candidates,
                key=lambda t: (
                    t["total_neurons"],
                    t["history"] if t["history"] is not None else 10**9,
                    -(t["metric_value"] or float("-inf")),
                    t["trial_number"],
                ),
            )

            print(
                "\nSmallest-neuron successful trial "
                "(priority: neurons -> metric -> history):"
            )
            print(f"  Trial #{best_smallest_total['trial_number']}")
            print(f"    metric: {best_smallest_total['metric_value']}")
            print(f"    tail_iqm_avg_reward: {best_smallest_total['tail_iqm']}")
            print(f"    final_avg_reward: {best_smallest_total['final_avg']}")
            print(f"    hl1: {best_smallest_total['hl1']}")
            print(f"    hl2: {best_smallest_total['hl2']}")
            print(f"    total_neurons: {best_smallest_total['total_neurons']}")
            print(f"    history: {best_smallest_total['history']}")
            conv_ep = best_smallest_total["convergence_episode"]
            print(
                "    convergence_episode: "
                f"{conv_ep if conv_ep is not None else 'N/A'}"
            )
            print(
                "\nSmallest-neuron successful trial "
                "(priority: neurons -> history -> metric):"
            )
            print(f"  Trial #{best_smallest_total_history_first['trial_number']}")
            print(f"    metric: {best_smallest_total_history_first['metric_value']}")
            print(
                "    tail_iqm_avg_reward: "
                f"{best_smallest_total_history_first['tail_iqm']}"
            )
            print(
                "    final_avg_reward: "
                f"{best_smallest_total_history_first['final_avg']}"
            )
            print(f"    hl1: {best_smallest_total_history_first['hl1']}")
            print(f"    hl2: {best_smallest_total_history_first['hl2']}")
            print(
                "    total_neurons: "
                f"{best_smallest_total_history_first['total_neurons']}"
            )
            print(f"    history: {best_smallest_total_history_first['history']}")
            conv_ep = best_smallest_total_history_first["convergence_episode"]
            print(
                "    convergence_episode: "
                f"{conv_ep if conv_ep is not None else 'N/A'}"
            )

        # Trial with shortest history (if history is present)
        # Priority A: history -> metric -> total neurons -> trial number
        # Priority B: history -> total neurons -> metric -> trial number
        history_candidates = [t for t in summaries if t["history"] is not None]
        if history_candidates:
            best_shortest_history_metric_first = min(
                history_candidates,
                key=lambda t: (
                    t["history"],
                    -(t["metric_value"] or float("-inf")),
                    t["total_neurons"] if t["total_neurons"] is not None else 10**9,
                    t["trial_number"],
                ),
            )
            best_shortest_history_neurons_first = min(
                history_candidates,
                key=lambda t: (
                    t["history"],
                    t["total_neurons"] if t["total_neurons"] is not None else 10**9,
                    -(t["metric_value"] or float("-inf")),
                    t["trial_number"],
                ),
            )

            print(
                "\nShortest-history successful trial "
                "(priority: history -> metric -> neurons):"
            )
            print(f"  Trial #{best_shortest_history_metric_first['trial_number']}")
            print(f"    metric: {best_shortest_history_metric_first['metric_value']}")
            print(
                "    tail_iqm_avg_reward: "
                f"{best_shortest_history_metric_first['tail_iqm']}"
            )
            print(
                "    final_avg_reward: "
                f"{best_shortest_history_metric_first['final_avg']}"
            )
            print(f"    hl1: {best_shortest_history_metric_first['hl1']}")
            print(f"    hl2: {best_shortest_history_metric_first['hl2']}")
            print(
                "    total_neurons: "
                f"{best_shortest_history_metric_first['total_neurons']}"
            )
            print(f"    history: {best_shortest_history_metric_first['history']}")
            conv_ep = best_shortest_history_metric_first["convergence_episode"]
            print(
                "    convergence_episode: "
                f"{conv_ep if conv_ep is not None else 'N/A'}"
            )
            print(
                "\nShortest-history successful trial "
                "(priority: history -> neurons -> metric):"
            )
            print(f"  Trial #{best_shortest_history_neurons_first['trial_number']}")
            print(f"    metric: {best_shortest_history_neurons_first['metric_value']}")
            print(
                "    tail_iqm_avg_reward: "
                f"{best_shortest_history_neurons_first['tail_iqm']}"
            )
            print(
                "    final_avg_reward: "
                f"{best_shortest_history_neurons_first['final_avg']}"
            )
            print(f"    hl1: {best_shortest_history_neurons_first['hl1']}")
            print(f"    hl2: {best_shortest_history_neurons_first['hl2']}")
            print(
                "    total_neurons: "
                f"{best_shortest_history_neurons_first['total_neurons']}"
            )
            print(f"    history: {best_shortest_history_neurons_first['history']}")
            conv_ep = best_shortest_history_neurons_first["convergence_episode"]
            print(
                "    convergence_episode: "
                f"{conv_ep if conv_ep is not None else 'N/A'}"
            )

            compact_history_candidates = [
                t
                for t in summaries
                if t["history"] is not None
                and t["total_neurons"] is not None
                and t["history"] < 32
                and t["total_neurons"] < 48
            ]
            print(
                "\nSuccessful trials with history < 32 and total neurons < 48: "
                f"{len(compact_history_candidates)}"
            )
            if compact_history_candidates:
                for t in sorted(
                    compact_history_candidates,
                    key=lambda x: (
                        -(x["metric_value"] or float("-inf")),
                        x["total_neurons"],
                        x["history"],
                        x["trial_number"],
                    ),
                ):
                    conv_ep = t["convergence_episode"]
                    print(f"  Trial #{t['trial_number']}")
                    print(f"    metric: {t['metric_value']}")
                    print(f"    tail_iqm_avg_reward: {t['tail_iqm']}")
                    print(f"    final_avg_reward: {t['final_avg']}")
                    print(f"    hl1: {t['hl1']}")
                    print(f"    hl2: {t['hl2']}")
                    print(f"    total_neurons: {t['total_neurons']}")
                    print(f"    history: {t['history']}")
                    print(
                        "    convergence_episode: "
                        f"{conv_ep if conv_ep is not None else 'N/A'}"
                    )


if __name__ == "__main__":
    main()
