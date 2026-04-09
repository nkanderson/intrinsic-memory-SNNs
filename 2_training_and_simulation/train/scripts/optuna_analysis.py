"""
Analyze Optuna study for trials meeting a reward threshold.

Usage examples:
  python analyze_optuna_study.py --study-name fractional-1000ep
  python analyze_optuna_study.py --study-name fractional-1000ep --storage sqlite:///optuna_studies/fractional.db --threshold 450 --print-trials
"""

from __future__ import annotations

import argparse
import re
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional
import optuna


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
]
HIST_KEYS = ["hist", "history", "history_length", "history_len"]


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


def analyze(study: optuna.study.Study, threshold: float) -> Dict[str, Any]:
    success_trials = [
        t for t in study.trials if t.value is not None and t.value >= threshold
    ]
    results: Dict[str, Any] = {}
    results["n_success"] = len(success_trials)
    if not success_trials:
        return results

    hl1_vals: List[int] = []
    hl2_vals: List[int] = []
    hist_vals: List[int] = []
    total_neurons: List[int] = []
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
        trial_summaries.append(
            {
                "trial_number": t.number,
                "value": t.value,
                "hl1": hl1,
                "hl2": hl2,
                "history": hist,
                "total_neurons": tot,
                "params": p,
            }
        )

    def _avg_or_none(xs: List[int]) -> Optional[float]:
        return float(mean(xs)) if xs else None

    results.update(
        {
            "avg_hl1": _avg_or_none(hl1_vals),
            "avg_hl2": _avg_or_none(hl2_vals),
            "avg_total_neurons": _avg_or_none(total_neurons),
            "avg_history": _avg_or_none(hist_vals),
            "trial_summaries": trial_summaries,
        }
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Optuna study successes")
    parser.add_argument("--study-name", required=True, help="Optuna study name")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL (default: sqlite:///optuna_studies/<study>.db)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=450.0,
        help="Success threshold for trial objective (default: 450.0)",
    )
    parser.add_argument(
        "--print-trials",
        action="store_true",
        help="Print summary of individual successful trials",
    )
    args = parser.parse_args()

    storage = args.storage
    if storage is None:
        storage = f"sqlite:///optuna_studies/{args.study_name}.db"

    study = optuna.load_study(study_name=args.study_name, storage=storage)
    print(f"Study: {args.study_name}")
    print(f"Storage: {storage}")
    print(f"Threshold: {args.threshold}")
    print(f"Total trials in study: {len(study.trials)}")

    out = analyze(study, args.threshold)
    print(f"Successful trials: {out.get('n_success', 0)}")
    if out.get("n_success", 0) == 0:
        return

    print("Averages over successful trials:")
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
            print(
                f"  trial#{t['trial_number']:3d} value={t['value']:.3f} hl1={t['hl1']} hl2={t['hl2']} history={t['history']} total={t['total_neurons']}"
            )

    summaries = out["trial_summaries"]

    # Trial with smallest total neurons (if hl1/hl2 are available)
    total_candidates = [t for t in summaries if t["total_neurons"] is not None]
    if total_candidates:
        best_smallest_total = min(
            total_candidates,
            key=lambda t: (
                t["total_neurons"],
                -float(t["value"]),
                t["history"] if t["history"] is not None else 10**9,
                t["trial_number"],
            ),
        )
        best_smallest_total_history_first = min(
            total_candidates,
            key=lambda t: (
                t["total_neurons"],
                t["history"] if t["history"] is not None else 10**9,
                -float(t["value"]),
                t["trial_number"],
            ),
        )

        print(
            "\nSmallest-neuron successful trial (priority: neurons -> reward -> history):"
        )
        print(
            f"  Trial #{best_smallest_total['trial_number']} | "
            f"Reward: {best_smallest_total['value']:.3f} | "
            f"HL1: {best_smallest_total['hl1']} | "
            f"HL2: {best_smallest_total['hl2']} | "
            f"Total neurons: {best_smallest_total['total_neurons']} | "
            f"History length: {best_smallest_total['history']}"
        )
        print(
            "\nSmallest-neuron successful trial "
            "(priority: neurons -> history -> reward):"
        )
        print(
            f"  Trial #{best_smallest_total_history_first['trial_number']} | "
            f"Reward: {best_smallest_total_history_first['value']:.3f} | "
            f"HL1: {best_smallest_total_history_first['hl1']} | "
            f"HL2: {best_smallest_total_history_first['hl2']} | "
            f"Total neurons: {best_smallest_total_history_first['total_neurons']} | "
            f"History length: {best_smallest_total_history_first['history']}"
        )

    # Trial with shortest history (if history is present)
    # Priority A (current): history -> reward -> total neurons -> trial number
    # Priority B (alternate): history -> total neurons -> reward -> trial number
    history_candidates = [t for t in summaries if t["history"] is not None]
    if history_candidates:
        best_shortest_history_reward_first = min(
            history_candidates,
            key=lambda t: (
                t["history"],
                -float(t["value"]),
                t["total_neurons"] if t["total_neurons"] is not None else 10**9,
                t["trial_number"],
            ),
        )
        best_shortest_history_neurons_first = min(
            history_candidates,
            key=lambda t: (
                t["history"],
                t["total_neurons"] if t["total_neurons"] is not None else 10**9,
                -float(t["value"]),
                t["trial_number"],
            ),
        )

        print(
            "\nShortest-history successful trial (priority: history -> reward -> neurons):"
        )
        print(
            f"  Trial #{best_shortest_history_reward_first['trial_number']} | "
            f"Reward: {best_shortest_history_reward_first['value']:.3f} | "
            f"HL1: {best_shortest_history_reward_first['hl1']} | "
            f"HL2: {best_shortest_history_reward_first['hl2']} | "
            f"Total neurons: {best_shortest_history_reward_first['total_neurons']} | "
            f"History length: {best_shortest_history_reward_first['history']}"
        )
        print(
            "\nShortest-history successful trial "
            "(priority: history -> neurons -> reward):"
        )
        print(
            f"  Trial #{best_shortest_history_neurons_first['trial_number']} | "
            f"Reward: {best_shortest_history_neurons_first['value']:.3f} | "
            f"HL1: {best_shortest_history_neurons_first['hl1']} | "
            f"HL2: {best_shortest_history_neurons_first['hl2']} | "
            f"Total neurons: {best_shortest_history_neurons_first['total_neurons']} | "
            f"History length: {best_shortest_history_neurons_first['history']}"
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
                    -float(x["value"]),
                    x["total_neurons"],
                    x["history"],
                    x["trial_number"],
                ),
            ):
                print(
                    f"  Trial #{t['trial_number']} | "
                    f"Reward: {t['value']:.3f} | "
                    f"HL1: {t['hl1']} | "
                    f"HL2: {t['hl2']} | "
                    f"Total neurons: {t['total_neurons']} | "
                    f"History length: {t['history']}"
                )


if __name__ == "__main__":
    main()
