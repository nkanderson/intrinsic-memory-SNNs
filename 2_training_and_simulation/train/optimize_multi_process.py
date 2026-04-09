"""
Multi-process launcher for Optuna optimization on CPU or GPU.

This script launches multiple worker processes, all sharing the same
Optuna study/storage so trials are coordinated.

It accepts most of the same CLI options as optimize.py and adds
process/device controls.

Examples:
    # 4 GPU workers on GPUs 0-3
    python optimize_multi_process.py --device-mode gpu --gpu-ids 0,1,2,3 --neuron-type leaky --n-trials 120

    # 8 CPU workers
    python optimize_multi_process.py --device-mode cpu --workers 8 --neuron-type fractional --n-trials 200

    # Auto mode (prefer GPU if available)
    python optimize_multi_process.py --device-mode auto --neuron-type bitshift --n-trials 80
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_SQLITE_MAX_WORKERS = 8


def split_trials(total_trials: int, workers: int) -> list[int]:
    """Split total trials as evenly as possible across workers."""
    base = total_trials // workers
    remainder = total_trials % workers
    parts = [base] * workers
    for index in range(remainder):
        parts[index] += 1
    return parts


def parse_gpu_ids(gpu_ids_raw: str) -> list[int]:
    """Parse --gpu-ids input like '0,1,2,3' into a list of unique ints."""
    ids = []
    for token in gpu_ids_raw.split(","):
        token = token.strip()
        if not token:
            continue
        gpu_id = int(token)
        if gpu_id < 0:
            raise ValueError("GPU ids must be non-negative")
        if gpu_id not in ids:
            ids.append(gpu_id)

    if not ids:
        raise ValueError("--gpu-ids must contain at least one GPU id")

    return ids


def detect_device_mode(requested_mode: str) -> str:
    """Resolve runtime device mode from requested --device-mode."""
    if requested_mode in {"cpu", "gpu"}:
        return requested_mode

    try:
        import torch

        if torch.cuda.is_available():
            return "gpu"
    except Exception:
        pass

    return "cpu"


def is_sqlite_storage(storage_uri: str) -> bool:
    """Return True when Optuna storage URI points to SQLite."""
    return storage_uri.startswith("sqlite:///")


def build_worker_command(
    args, study_name: str, storage: str, worker_trials: int, force_cpu: bool
) -> list[str]:
    """Build one optimize.py command for a worker."""
    command = [
        sys.executable,
        "optimize.py",
        "--neuron-type",
        args.neuron_type,
        "--study-name",
        study_name,
        "--storage",
        storage,
        "--n-trials",
        str(worker_trials),
    ]

    if force_cpu:
        command.append("--no-hw-acceleration")

    if args.search_space:
        command.extend(["--search-space", args.search_space])
    if args.num_episodes is not None:
        command.extend(["--num-episodes", str(args.num_episodes)])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.pruner:
        command.extend(["--pruner", args.pruner])

    return command


def run_final_importance(
    args,
    study_name: str,
    storage: str,
    force_cpu: bool,
    env: dict | None = None,
) -> int:
    """Run a final single-process importance/export pass after workers complete."""
    command = [
        sys.executable,
        "optimize.py",
        "--neuron-type",
        args.neuron_type,
        "--study-name",
        study_name,
        "--storage",
        storage,
        "--n-trials",
        "0",
        "--get-importance",
    ]

    if force_cpu:
        command.append("--no-hw-acceleration")

    if args.search_space:
        command.extend(["--search-space", args.search_space])
    if args.num_episodes is not None:
        command.extend(["--num-episodes", str(args.num_episodes)])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.pruner:
        command.extend(["--pruner", args.pruner])

    print("\nRunning final post-optimization importance pass...")
    print(" ".join(command))
    return subprocess.call(command, env=env)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Launch optimize.py across multiple worker processes on CPU or GPU"
        )
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
        help="Path to search space YAML config",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Total number of trials across all workers",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Study name shared by all workers",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URI shared by all workers",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Override num_episodes per trial",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed. Worker index is added for per-worker seeds.",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="none",
        choices=["median", "hyperband", "none"],
        help="Optuna pruner strategy",
    )
    parser.add_argument(
        "--get-importance",
        action="store_true",
        help="Run one final --get-importance pass after all workers finish",
    )
    parser.add_argument(
        "--device-mode",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Execution mode: auto prefers GPU when CUDA is available",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to GPU count in gpu mode, else min(cpu_count, 8).",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0,1,2,3",
        help="Comma-separated GPU ids used in gpu mode (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--sqlite-max-workers",
        type=int,
        default=DEFAULT_SQLITE_MAX_WORKERS,
        help=(
            "Safety cap for worker count when using SQLite storage "
            f"(default: {DEFAULT_SQLITE_MAX_WORKERS})"
        ),
    )
    parser.add_argument(
        "--allow-oversubscribe-sqlite",
        action="store_true",
        help="Allow workers > --sqlite-max-workers for SQLite storage (higher lock contention risk)",
    )
    parser.add_argument(
        "--worker-start-delay",
        type=float,
        default=0.35,
        help="Seconds to wait between launching workers (reduces SQLite lock storms)",
    )

    args = parser.parse_args()

    if args.n_trials < 0:
        raise ValueError("--n-trials must be >= 0")

    effective_mode = detect_device_mode(args.device_mode)

    if args.study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.study_name = f"{args.neuron_type}-mproc-{effective_mode}-{timestamp}"

    if args.storage is None:
        studies_dir = Path("optuna_studies")
        studies_dir.mkdir(exist_ok=True)
        args.storage = f"sqlite:///optuna_studies/{args.neuron_type}.db"

    if effective_mode == "gpu":
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        default_workers = len(gpu_ids)
    else:
        gpu_ids = []
        # Default max of 4 to avoid oversubscription on CPU, especially with
        # SQLite storage. User can override with --workers.
        default_workers = min(os.cpu_count() or 1, 4)

    workers = args.workers or default_workers
    if workers <= 0:
        raise ValueError("--workers must be >= 1")

    if effective_mode == "gpu" and workers > len(gpu_ids):
        raise ValueError(
            "In gpu mode, --workers cannot exceed number of --gpu-ids "
            f"(workers={workers}, gpu_ids={len(gpu_ids)})"
        )

    if args.worker_start_delay < 0:
        raise ValueError("--worker-start-delay must be >= 0")

    if is_sqlite_storage(args.storage) and workers > args.sqlite_max_workers:
        if not args.allow_oversubscribe_sqlite:
            raise ValueError(
                "SQLite storage detected and worker count exceeds safe cap: "
                f"workers={workers}, sqlite_max_workers={args.sqlite_max_workers}. "
                "Use --allow-oversubscribe-sqlite to override."
            )
        print(
            "WARNING: SQLite oversubscription enabled. "
            "Expect higher lock contention and lower throughput at high worker counts."
        )

    log_dir = Path("optuna_studies") / "logs" / args.study_name
    log_dir.mkdir(parents=True, exist_ok=True)

    trial_splits = split_trials(args.n_trials, workers)

    print(f"Study: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Requested mode: {args.device_mode}")
    print(f"Effective mode: {effective_mode}")
    print(f"Workers: {workers}")
    print(f"Total trials: {args.n_trials}")
    print(f"Trial split: {trial_splits}")
    if effective_mode == "gpu":
        print(f"GPU ids: {gpu_ids[:workers]}")
        print("CUDA device order: PCI_BUS_ID (matches nvidia-smi indexing)")
    if is_sqlite_storage(args.storage):
        print(
            "SQLite storage detected: recommended workers <= "
            f"{args.sqlite_max_workers}"
        )

    processes = []

    for worker_index in range(workers):
        worker_trials = trial_splits[worker_index]

        # Skip idle workers when total trials < number of workers
        if worker_trials == 0:
            continue

        worker_seed = None
        if args.seed is not None:
            worker_seed = args.seed + worker_index

        worker_args = argparse.Namespace(**vars(args))
        worker_args.seed = worker_seed

        force_cpu = effective_mode == "cpu"
        command = build_worker_command(
            worker_args,
            study_name=args.study_name,
            storage=args.storage,
            worker_trials=worker_trials,
            force_cpu=force_cpu,
        )

        env = os.environ.copy()
        if effective_mode == "gpu":
            # Make CUDA ordinal mapping match nvidia-smi GPU indices.
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            gpu_id = gpu_ids[worker_index]
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            worker_label = f"gpu{gpu_id}"
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
            worker_label = f"cpu{worker_index}"

        log_file = log_dir / f"worker_{worker_label}.log"

        print(
            f"\nLaunching worker {worker_index} ({worker_label}, {worker_trials} trials)"
        )
        print(" ".join(command))
        print(f"Log: {log_file}")

        handle = open(log_file, "w")
        proc = subprocess.Popen(
            command, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
        processes.append((proc, handle, worker_label, worker_trials))
        if args.worker_start_delay > 0:
            time.sleep(args.worker_start_delay)

    exit_codes = []
    for proc, handle, worker_label, worker_trials in processes:
        code = proc.wait()
        handle.close()
        exit_codes.append(code)
        status = "OK" if code == 0 else "FAILED"
        print(
            f"Worker {worker_label} ({worker_trials} trials): {status} (exit={code})"
        )

    if any(code != 0 for code in exit_codes):
        print("\nOne or more workers failed. See logs for details:")
        print(log_dir)
        sys.exit(1)

    if args.get_importance:
        importance_env = os.environ.copy()
        if effective_mode == "gpu":
            # Keep device index mapping consistent with worker launches.
            importance_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            importance_env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        else:
            importance_env["CUDA_VISIBLE_DEVICES"] = ""

        rc = run_final_importance(
            args,
            study_name=args.study_name,
            storage=args.storage,
            force_cpu=(effective_mode == "cpu"),
            env=importance_env,
        )
        if rc != 0:
            print("Final importance pass failed.")
            sys.exit(rc)

    print("\nAll workers completed successfully.")
    print(f"Study: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Logs: {log_dir}")


if __name__ == "__main__":
    main()
