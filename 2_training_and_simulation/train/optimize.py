"""
Optuna hyperparameter optimization for SNN-DQN on CartPole.

This script searches for optimal hyperparameters for a given neuron type
using Optuna. Search spaces are defined in YAML config files (one per
neuron type) under configs/search_spaces/.

Studies are persisted to SQLite so they can be resumed, compared, and
visualized with `optuna-dashboard`.

Usage examples:
    # Optimize leaky neuron with 50 trials
    python optimize.py --neuron-type leaky --n-trials 50

    # Optimize fractional neuron, custom study name, 100 trials
    python optimize.py --neuron-type fractional --n-trials 100 --study-name flif-v2

    # Optimize bitshift neuron, shorter episodes for fast screening
    python optimize.py --neuron-type bitshift --n-trials 30 --num-episodes 300

    # Resume an existing study (same study name + storage)
    python optimize.py --neuron-type leaky --n-trials 20 --study-name leaky-v1

    # Export best config to a standard YAML (usable with main.py --config)
    python optimize.py --neuron-type leaky --study-name leaky-v1 --export-best

    # Run post-optimization importance analysis (fANOVA)
    python optimize.py --neuron-type leaky --n-trials 50 --get-importance

    # Run importance analysis on an existing study by adding 0 trials
    python optimize.py --neuron-type leaky --study-name leaky-v1 --n-trials 0 --get-importance

    # Use a specific search space file
    python optimize.py --neuron-type leaky --search-space configs/search_spaces/my_custom.yaml

Importance analysis artifacts (when --get-importance is set):
    optuna_studies/importance/<study_name>/importance.json
    optuna_studies/importance/<study_name>/importance_plot.html
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import optuna

import yaml

# Ensure we can import from the train directory
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_search_space(path: str) -> dict:
    """Load a search space YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def sample_params(trial, search_space: dict) -> dict:
    """
    Use an Optuna trial to sample hyperparameters from a search space config.

    The search space YAML has a nested structure (training/snn sections),
    but this function returns a flat dict suitable for train_fn.train().

    Supported param types:
        fixed:       trial returns the value unchanged
        int:         trial.suggest_int(name, low, high)
        float:       trial.suggest_float(name, low, high, log=...)
        categorical: trial.suggest_categorical(name, choices)

    Args:
        trial: optuna.Trial object
        search_space: Parsed YAML search space dict

    Returns:
        Flat dict of sampled hyperparameters
    """
    params = {}

    for section_name, section in search_space.items():
        for param_name, spec in section.items():
            ptype = spec["type"]

            if ptype == "fixed":
                params[param_name] = spec["value"]
            elif ptype == "int":
                params[param_name] = trial.suggest_int(
                    param_name, spec["low"], spec["high"]
                )
            elif ptype == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )
            elif ptype == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, spec["choices"]
                )
            else:
                raise ValueError(
                    f"Unknown search space type '{ptype}' for param '{param_name}'"
                )

    return params


def apply_constraints(params: dict) -> dict:
    """
    Apply inter-parameter constraints after sampling.

    For example, bitshift 'simple' shift_func needs small history_length
    to avoid numerical instability.

    Args:
        params: Flat dict of sampled hyperparameters

    Returns:
        Modified params dict with constraints applied
    """
    # Bitshift constraint: 'simple' shift_func needs history_length <= 20
    if params.get("neuron_type") == "bitshift" and params.get("shift_func") == "simple":
        if params.get("history_length", 0) > 20:
            params["history_length"] = 16

    # Ensure hidden2_size <= hidden1_size (typical funnel architecture)
    if params.get("hidden2_size", 0) > params.get("hidden1_size", 0):
        params["hidden2_size"] = params["hidden1_size"]

    return params


def params_to_config_yaml(params: dict) -> dict:
    """
    Convert a flat params dict back to the nested YAML config format
    used by main.py (training + snn sections).

    Args:
        params: Flat dict of hyperparameters

    Returns:
        Nested dict matching the standard config YAML structure
    """
    training_keys = {
        "batch_size",
        "gamma",
        "eps_start",
        "eps_end",
        "eps_decay",
        "tau",
        "lr",
        "num_episodes",
    }
    snn_keys = {
        "num_steps",
        "beta",
        "surrogate_gradient_slope",
        "neuron_type",
        "hidden1_size",
        "hidden2_size",
        # Fractional
        "alpha",
        "lam",
        "history_length",
        "dt",
        # Bitshift
        "shift_func",
    }

    config = {"training": {}, "snn": {}}
    for key, value in params.items():
        if key in training_keys:
            config["training"][key] = value
        elif key in snn_keys:
            config["snn"][key] = value
        else:
            # Put unknown keys in snn section as a fallback
            config["snn"][key] = value

    return config


def get_device(hw_acceleration: bool) -> str:
    """Determine the best available torch device."""
    import torch

    if hw_acceleration and torch.cuda.is_available():
        return "cuda"
    elif hw_acceleration and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_objective(
    search_space: dict, device: str, num_episodes_override: int = None
):
    """
    Create an Optuna objective function (closure) for the given search space.

    Args:
        search_space: Parsed search space YAML
        device: Torch device string
        num_episodes_override: If set, override num_episodes in the search space
            (useful for faster screening runs)

    Returns:
        Callable objective(trial) -> float
    """
    from train_fn import train

    def objective(trial):
        # Sample hyperparameters
        params = sample_params(trial, search_space)
        params = apply_constraints(params)

        # Override num_episodes if requested
        if num_episodes_override is not None:
            params["num_episodes"] = num_episodes_override

        # Log the trial params
        neuron_type = params.get("neuron_type", "unknown")
        h1 = params.get("hidden1_size", "?")
        h2 = params.get("hidden2_size", "?")
        lr = params.get("lr", "?")
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}: {neuron_type} [{h1}-{h2}] lr={lr}")
        print(f"{'='*60}")

        # Run training
        result = train(
            config=params,
            device=device,
            verbose=True,
            save_models=False,
            optuna_trial=trial,
        )

        best_avg = result["best_avg_reward"]
        final_avg = result["final_avg_reward"]
        print(
            f"Trial {trial.number} done: best_avg={best_avg:.1f}, final_avg={final_avg:.1f}"
        )

        return best_avg

    return objective


def export_best_config(study, output_path: str):
    """
    Export the best trial's parameters as a standard config YAML
    that can be used directly with main.py --config.

    Args:
        study: Optuna study object
        output_path: Path to write the YAML file
    """
    best = study.best_trial
    config = params_to_config_yaml(best.params)

    # Add metadata as comments
    header = (
        f"# Auto-generated by optimize.py\n"
        f"# Study: {study.study_name}\n"
        f"# Best trial: {best.number} (value: {best.value:.2f})\n"
        f"# Generated: {datetime.now().isoformat()}\n"
        f"#\n"
        f"# To train with this config:\n"
        f"#   python main.py --config {output_path}\n\n"
    )

    with open(output_path, "w") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Best config exported to: {output_path}")


def _build_complete_trials_study(source_study):
    """Create an in-memory study containing only COMPLETE trials."""
    complete_trials = source_study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
    )
    complete_study = optuna.create_study(direction=source_study.direction)
    for trial in complete_trials:
        complete_study.add_trial(trial)
    return complete_study, complete_trials


def run_importance_analysis(
    study,
    output_dir: Path,
    min_complete_trials: int = 30,
    drift_step: int = 10,
):
    """
    Compute fANOVA-based parameter importances and save artifacts.

    Analysis is performed only on COMPLETE trials. If there are fewer than
    `min_complete_trials`, a warning is printed and analysis is skipped.

    Artifacts:
      - JSON summary with importances + drift snapshots
      - Optuna importance plot (HTML)
    """
    complete_study, complete_trials = _build_complete_trials_study(study)
    n_complete = len(complete_trials)

    if n_complete < min_complete_trials:
        print(
            f"WARNING: importance analysis skipped: only {n_complete} complete trials "
            f"(minimum recommended: {min_complete_trials})."
        )
        return None

    evaluator = optuna.importance.FanovaImportanceEvaluator()
    importances = optuna.importance.get_param_importances(
        complete_study,
        evaluator=evaluator,
    )

    checkpoints = []
    if n_complete >= min_complete_trials:
        checkpoint_sizes = list(range(min_complete_trials, n_complete + 1, drift_step))
        if checkpoint_sizes[-1] != n_complete:
            checkpoint_sizes.append(n_complete)

        for size in checkpoint_sizes:
            partial = optuna.create_study(direction=study.direction)
            for trial in complete_trials[:size]:
                partial.add_trial(trial)

            partial_importances = optuna.importance.get_param_importances(
                partial,
                evaluator=evaluator,
            )
            checkpoints.append(
                {
                    "n_complete_trials": size,
                    "importances": partial_importances,
                }
            )

    top_param = next(iter(importances.keys()), None)
    top_param_drift = []
    if top_param is not None:
        for checkpoint in checkpoints:
            top_param_drift.append(
                {
                    "n_complete_trials": checkpoint["n_complete_trials"],
                    "importance": checkpoint["importances"].get(top_param, 0.0),
                }
            )

    analysis_payload = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "timestamp": datetime.now().isoformat(),
        "evaluator": "fanova",
        "minimum_complete_trials": min_complete_trials,
        "complete_trials_used": n_complete,
        "importances": importances,
        "importance_drift": {
            "step": drift_step,
            "checkpoints": checkpoints,
            "top_param": top_param,
            "top_param_drift": top_param_drift,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "importance.json"
    with open(json_path, "w") as file:
        json.dump(analysis_payload, file, indent=2)

    plot_path = output_dir / "importance_plot.html"
    try:
        fig = optuna.visualization.plot_param_importances(
            complete_study,
            evaluator=evaluator,
        )
        fig.write_html(str(plot_path))
    except Exception as exc:
        print(
            "WARNING: failed to export Optuna importance plot "
            f"to {plot_path}: {exc}"
        )
        plot_path = None

    print(f"Importance JSON saved to: {json_path}")
    if plot_path is not None:
        print(f"Importance plot saved to: {plot_path}")
    print(
        f"Top parameter by fANOVA: {top_param} "
        f"(importance={importances.get(top_param, 0.0):.4f})"
    )

    return {
        "json_path": str(json_path),
        "plot_path": str(plot_path) if plot_path is not None else None,
        "complete_trials": n_complete,
        "top_param": top_param,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for SNN-DQN on CartPole",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize.py --neuron-type leaky --n-trials 50
  python optimize.py --neuron-type fractional --n-trials 100 --study-name flif-v2
  python optimize.py --neuron-type bitshift --n-trials 30 --num-episodes 300
  python optimize.py --neuron-type leaky --study-name leaky-v1 --export-best
        """,
    )

    parser.add_argument(
        "--neuron-type",
        type=str,
        required=True,
        choices=["leaky", "fractional", "bitshift"],
        help="Neuron type to optimize",
    )
    parser.add_argument(
        "--search-space",
        type=str,
        default=None,
        help="Path to search space YAML config. Defaults to "
        "configs/search_spaces/<neuron-type>.yaml",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run (default: 50)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name. Defaults to '<neuron-type>-<timestamp>'",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URI. Defaults to "
        "sqlite:///optuna_studies/<neuron-type>.db",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Override num_episodes per trial (useful for fast screening)",
    )
    parser.add_argument(
        "--no-hw-acceleration",
        dest="hw_acceleration",
        action="store_false",
        help="Disable hardware acceleration (CUDA/MPS)",
    )
    parser.set_defaults(hw_acceleration=True)
    parser.add_argument(
        "--export-best",
        action="store_true",
        help="Export the best trial's config as a standard YAML and exit "
        "(requires an existing study)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for Optuna sampler reproducibility",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="none",
        choices=["median", "hyperband", "none"],
        help="Optuna pruner strategy (default: none). 'none' disables pruning.",
    )
    parser.add_argument(
        "--get-importance",
        action="store_true",
        help="Run post-optimization fANOVA importance analysis and export artifacts.",
    )

    args = parser.parse_args()

    # ── Resolve defaults ──
    neuron_type = args.neuron_type

    # Search space file
    if args.search_space is None:
        args.search_space = f"configs/search_spaces/{neuron_type}.yaml"

    # Study name
    if args.study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.study_name = f"{neuron_type}-{timestamp}"

    # Storage
    if args.storage is None:
        studies_dir = Path("optuna_studies")
        studies_dir.mkdir(exist_ok=True)
        args.storage = f"sqlite:///optuna_studies/{neuron_type}.db"

    # ── Export-only mode ──
    if args.export_best:
        study = optuna.load_study(
            study_name=args.study_name,
            storage=args.storage,
        )
        output_path = f"configs/optimized-{args.study_name}.yaml"
        export_best_config(study, output_path)
        return

    # ── Load search space ──
    if not Path(args.search_space).exists():
        print(f"ERROR: Search space file not found: {args.search_space}")
        sys.exit(1)

    search_space = load_search_space(args.search_space)
    print(f"Search space loaded from: {args.search_space}")

    # ── Set up device ──
    device = get_device(args.hw_acceleration)
    print(f"Using device: {device}")

    # ── Configure pruner ──
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=100,  # Don't prune before 100 episodes
        )
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=50,
            max_resource=args.num_episodes or 600,
        )
    else:
        pruner = optuna.pruners.NopPruner()

    # ── Configure sampler ──
    # Following Optuna recommendation to use TPE sampler since there are fewer than 1000 trials
    # and the parameters are not correlated in a simple way.
    sampler = optuna.samplers.TPESampler(seed=args.seed)

    # ── Create or load study ──
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",  # Maximize average reward
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True,  # Resume if study already exists
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"Resuming study '{args.study_name}' with {n_existing} existing trials")

    # ── Run optimization ──
    print(f"\nStudy: {args.study_name}")
    print(f"Neuron type: {neuron_type}")
    print(f"Trials: {args.n_trials}")
    print(f"Storage: {args.storage}")
    if args.num_episodes:
        print(f"Episodes per trial: {args.num_episodes} (override)")
    print()

    objective = create_objective(
        search_space=search_space,
        device=device,
        num_episodes_override=args.num_episodes,
    )

    study.optimize(objective, n_trials=args.n_trials)

    # ── Print results ──
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    best = study.best_trial
    print(f"\nBest trial: {best.number}")
    print(f"Best value (avg reward): {best.value:.2f}")
    print("\nBest hyperparameters:")
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    # Auto-export best config
    output_path = f"configs/optimized-{args.study_name}.yaml"
    export_best_config(study, output_path)

    if args.get_importance:
        print("\nRunning post-optimization importance analysis (fANOVA)...")
        importance_dir = Path("optuna_studies") / "importance" / args.study_name
        run_importance_analysis(
            study=study,
            output_dir=importance_dir,
            min_complete_trials=30,
            drift_step=10,
        )

    # Summary
    print(f"\nStudy saved to: {args.storage}")
    print(f"To see all trials: optuna-dashboard {args.storage}")
    print(f"To train with best config: python main.py --config {output_path}")
    print(
        f"To resume this study: python optimize.py --neuron-type {neuron_type} "
        f"--study-name {args.study_name} --n-trials <N>"
    )
    if args.get_importance:
        print(
            f"Importance artifacts: optuna_studies/importance/{args.study_name}/"
        )


if __name__ == "__main__":
    main()
