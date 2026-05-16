"""
Parallel wrapper around multi_seed_train.py.

Runs N seeds for a single configuration concurrently by spawning one
subprocess per seed (each subprocess calls multi_seed_train.py with
--num-seeds 1 --base-seed <i>). After all subprocesses complete, the
per-seed CSVs are moved into the canonical output directory and the
per-seed summary CSVs are concatenated into one combined summary CSV.
Aggregate statistics (K/N solved, IQM with bootstrap 95% CI, mean ± std)
are then printed using the same helpers used by multi_seed_train.py.

Why a wrapper rather than modifying multi_seed_train.py:
    The sequential script is untouched and continues to work standalone.
    The wrapper only orchestrates subprocess lifecycle and post-run
    aggregation, so there is no risk of breaking the sequential path.

CPU oversubscription:
    Each subprocess inherits OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, and
    OPENBLAS_NUM_THREADS=1 to limit PyTorch / NumPy intra-op threading
    to one thread per worker. Without this, 10 subprocesses x N cores
    of intra-op threading produces severe context-switching overhead
    and is often slower than running them sequentially.

Hardware acceleration:
    Parallel mode forces --no-hw-acceleration by default. On the
    target Linux Xeon workstations there is no GPU available for this
    workload; on a Mac, MPS/CUDA have known multi-process issues.
    Override with --enable-hw-acceleration only if you know what
    you are doing.

Usage:
    python run_multi_seed_parallel.py --config configs/optimized-leaky-v2-top1.yaml
    python run_multi_seed_parallel.py --config configs/optimized-fractional.yaml \\
        --num-seeds 10 --max-parallel 10
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Reuse helpers from the sequential script so the aggregate stats are
# computed identically. Both files live in the same directory.
from multi_seed_train import _iqm, bootstrap_ci  # noqa: E402


SUCCESS_THRESHOLD = 475.0  # CartPole-v1 standard


def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel wrapper around multi_seed_train.py — runs one "
                    "subprocess per seed for a single config."
    )
    p.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to the YAML config (passed through to multi_seed_train.py).",
    )
    p.add_argument(
        "--num-seeds", type=int, default=10,
        help="Number of seeds to run (default: 10).",
    )
    p.add_argument(
        "--base-seed", type=int, default=42,
        help="First seed; seeds run from base..base+num_seeds-1 (default: 42).",
    )
    p.add_argument(
        "--max-parallel", type=int, default=None,
        help="Maximum concurrent subprocesses (default: min(num_seeds, 10)). "
             "Limit if you need to keep CPU headroom for other work.",
    )
    p.add_argument(
        "--threads-per-worker", type=int, default=1,
        help="OMP/MKL/OPENBLAS thread count per worker (default: 1). "
             "Anything >1 with many parallel workers risks oversubscription.",
    )
    p.add_argument(
        "--num-episodes", type=int, default=None,
        help="Override num_episodes from config (passed through).",
    )
    p.add_argument(
        "--save-best", action="store_true",
        help="Save -seed<N>-best.pth per seed (passed through).",
    )
    p.add_argument(
        "--no-save-final", action="store_true",
        help="Do NOT save -seed<N>-final.pth (passed through).",
    )
    p.add_argument(
        "--enable-hw-acceleration", action="store_true",
        help="Allow PyTorch to use CUDA/MPS in workers. Off by default in "
             "parallel mode (multiple workers contending for one GPU is "
             "usually counter-productive; MPS is known to multi-process "
             "poorly on macOS).",
    )
    p.add_argument(
        "--output-dir", type=str, default="metrics/multi-seed",
        help="Canonical output directory for per-seed and summary CSVs.",
    )
    p.add_argument(
        "--keep-worker-dirs", action="store_true", default=True,
        help="Keep worker subdirs (and their stdout logs) after aggregation "
             "for debugging. Default: keep.",
    )
    p.add_argument(
        "--clean-worker-dirs", dest="keep_worker_dirs", action="store_false",
        help="Remove worker subdirs after successful aggregation.",
    )
    return p.parse_args()


def build_subprocess_cmd(args, seed: int, worker_dir: Path) -> list[str]:
    """Construct the multi_seed_train.py invocation for a single seed.

    We pass --no-config-subdir so the worker writes its CSVs directly into
    worker_dir rather than creating an extra <config_name>/ subdirectory
    inside it; the wrapper is already managing the per-worker structure.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "multi_seed_train.py"),
        "--config", args.config,
        "--num-seeds", "1",
        "--base-seed", str(seed),
        "--output-dir", str(worker_dir),
        "--no-config-subdir",
    ]
    if args.num_episodes is not None:
        cmd += ["--num-episodes", str(args.num_episodes)]
    if args.save_best:
        cmd += ["--save-best"]
    if args.no_save_final:
        cmd += ["--no-save-final"]
    if not args.enable_hw_acceleration:
        cmd += ["--no-hw-acceleration"]
    return cmd


def run_one_seed(seed: int, cmd: list[str], worker_dir: Path,
                 threads_per_worker: int) -> dict:
    """Launch a single seed as a subprocess; return result metadata."""
    worker_dir.mkdir(parents=True, exist_ok=True)
    log_path = worker_dir / "stdout.log"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads_per_worker)
    env["MKL_NUM_THREADS"] = str(threads_per_worker)
    env["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
    # PyTorch intra-op limit — belt and suspenders on top of OMP.
    env["PYTORCH_NUM_THREADS"] = str(threads_per_worker)
    # Unbuffer Python stdout/stderr so the per-seed log file shows progress
    # live rather than waiting for the block buffer (~4-8 KB) to fill or the
    # process to exit. Without this the log appears empty for hours.
    env["PYTHONUNBUFFERED"] = "1"

    start = time.monotonic()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        rc = proc.wait()
    elapsed = time.monotonic() - start
    return {
        "seed": seed,
        "returncode": rc,
        "elapsed_sec": elapsed,
        "log_path": log_path,
        "worker_dir": worker_dir,
    }


def collect_per_seed_summary(worker_dir: Path, config_stem: str) -> dict | None:
    """Read the single-row summary CSV that multi_seed_train.py wrote."""
    summary_csv = worker_dir / f"{config_stem}-summary.csv"
    if not summary_csv.exists():
        return None
    with open(summary_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    # multi_seed_train.py wrote exactly one row since --num-seeds 1.
    row = rows[0]
    return {
        "seed": int(row["seed"]),
        "best_avg_reward": float(row["best_avg_reward"]),
        "final_avg_reward": float(row["final_avg_reward"]),
        "tail_iqm_avg_reward": (
            float(row["tail_iqm_avg_reward"])
            if row["tail_iqm_avg_reward"] not in ("", "None")
            else None
        ),
        "convergence_episode": (
            int(row["convergence_episode"])
            if row["convergence_episode"] not in ("", "None")
            else None
        ),
        "total_episodes": int(row["total_episodes"]),
    }


def move_per_seed_csv(worker_dir: Path, output_dir: Path, config_stem: str,
                      seed: int) -> Path | None:
    """Move <worker_dir>/<config_stem>-seed<N>.csv → <output_dir>/."""
    src = worker_dir / f"{config_stem}-seed{seed}.csv"
    if not src.exists():
        return None
    dst = output_dir / src.name
    shutil.move(str(src), str(dst))
    return dst


def write_combined_summary(output_dir: Path, config_stem: str,
                           rows: list[dict]) -> Path:
    """Write the combined summary CSV (one row per seed) to the canonical path."""
    path = output_dir / f"{config_stem}-summary.csv"
    fieldnames = [
        "seed",
        "best_avg_reward",
        "final_avg_reward",
        "tail_iqm_avg_reward",
        "convergence_episode",
        "total_episodes",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["seed"]):
            writer.writerow(row)
    return path


def print_aggregate_stats(rows: list[dict]):
    """Match the reporting format used by multi_seed_train.py."""
    finals = [r["final_avg_reward"] for r in rows]
    bests = [r["best_avg_reward"] for r in rows]
    iqms = [r["tail_iqm_avg_reward"] for r in rows if r["tail_iqm_avg_reward"] is not None]
    convs = [r["convergence_episode"] for r in rows if r["convergence_episode"] is not None]
    n = len(rows)

    def _msd(values):
        clean = [v for v in values if v is not None]
        if not clean:
            return None, None
        if len(clean) == 1:
            return clean[0], 0.0
        return statistics.mean(clean), statistics.stdev(clean)

    final_mean, final_std = _msd(finals)
    best_mean, best_std = _msd(bests)
    iqm_mean, iqm_std = _msd(iqms)
    conv_mean, conv_std = _msd(convs)

    solved_count = sum(1 for v in finals if v >= SUCCESS_THRESHOLD)

    print(f"\nResults across {n} seeds:")
    print(f"  solved (final_avg >= {SUCCESS_THRESHOLD:.0f}) : {solved_count}/{n} seeds")
    print(f"  final_avg_reward  : {final_mean:.2f} ± {final_std:.2f}  (primary metric)")
    print(f"  best_avg_reward   : {best_mean:.2f} ± {best_std:.2f}")
    if iqm_mean is not None:
        print(f"  tail_iqm_reward   : {iqm_mean:.2f} ± {iqm_std:.2f}  (diagnostic)")
    if not convs:
        print("  convergence_ep    : never reached threshold in any seed")
    else:
        print(
            f"  convergence_ep    : {conv_mean:.1f} ± {conv_std:.1f}  "
            f"({len(convs)}/{n} seeds converged)"
        )

    # IQM + bootstrap 95% CI on final_avg across seeds (Agarwal et al., 2021)
    iqm_point, iqm_lo, iqm_hi = bootstrap_ci(finals, statistic=_iqm)
    mean_point, mean_lo, mean_hi = bootstrap_ci(finals, statistic=lambda v: sum(v) / len(v))
    print("\nRobust reporting (final_avg_reward across seeds, 95% bootstrap CI):")
    if iqm_point is not None:
        print(f"  IQM   : {iqm_point:.2f}  [95% CI: {iqm_lo:.2f}, {iqm_hi:.2f}]")
        print(f"  Mean  : {mean_point:.2f}  [95% CI: {mean_lo:.2f}, {mean_hi:.2f}]")
    if n < 10:
        print(
            "  (note: with N < 10, bootstrap CIs and IQM are necessarily wide; "
            "per Colas et al. 2018 consider increasing --num-seeds.)"
        )


def main():
    args = parse_args()

    config_stem = Path(args.config).stem
    seeds = list(range(args.base_seed, args.base_seed + args.num_seeds))
    max_parallel = args.max_parallel or min(args.num_seeds, 10)
    max_parallel = max(1, min(max_parallel, args.num_seeds))

    # All outputs for this config now live under <output_dir>/<config_stem>/
    # so that runs for different configs don't share a flat namespace.
    # Worker subdirs are nested inside the same config directory under
    # _workers/ so everything related to one config is co-located.
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    config_output_dir = base_output_dir / config_stem
    config_output_dir.mkdir(parents=True, exist_ok=True)
    worker_root = config_output_dir / "_workers"
    worker_root.mkdir(parents=True, exist_ok=True)

    print(f"Parallel multi-seed run for: {config_stem}")
    print(f"  config            : {args.config}")
    print(f"  seeds             : {seeds[0]}..{seeds[-1]}  (N = {args.num_seeds})")
    print(f"  max_parallel      : {max_parallel}")
    print(f"  threads_per_worker: {args.threads_per_worker}")
    print(f"  hw_acceleration   : {'enabled' if args.enable_hw_acceleration else 'disabled (CPU)'}")
    print(f"  config_output_dir : {config_output_dir}")
    print(f"  worker_dirs       : {worker_root}/seed<N>/  (logs preserved here)")
    print()

    # Submit one job per seed; ThreadPoolExecutor caps concurrency to
    # max_parallel by letting only that many threads wait on subprocesses
    # simultaneously. Each thread does almost no Python work — it just
    # waits on subprocess.Popen — so the GIL is not a concern.
    results: dict[int, dict] = {}
    overall_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_seed = {}
        for seed in seeds:
            worker_dir = worker_root / f"seed{seed}"
            cmd = build_subprocess_cmd(args, seed, worker_dir)
            fut = executor.submit(
                run_one_seed, seed, cmd, worker_dir, args.threads_per_worker
            )
            future_to_seed[fut] = seed
            # Stagger submission tiny bit to avoid hammering torch import
            # exactly simultaneously across N workers.
            time.sleep(0.05)

        for fut in as_completed(future_to_seed):
            seed = future_to_seed[fut]
            result = fut.result()
            results[seed] = result
            mm = int(result["elapsed_sec"] // 60)
            ss = int(result["elapsed_sec"] % 60)
            stamp = time.strftime("%H:%M:%S")
            if result["returncode"] == 0:
                print(f"[{stamp}] seed {seed}: done   ({mm}m{ss:02d}s)")
            else:
                print(
                    f"[{stamp}] seed {seed}: FAILED (rc={result['returncode']}, "
                    f"{mm}m{ss:02d}s) — see {result['log_path']}"
                )

    overall_elapsed = time.monotonic() - overall_start
    om = int(overall_elapsed // 60)
    print(f"\nAll subprocesses complete in {om} min total wall-clock.")

    # Collect, move, aggregate.
    summary_rows = []
    failed_seeds = []
    for seed in seeds:
        result = results[seed]
        if result["returncode"] != 0:
            failed_seeds.append(seed)
            continue
        row = collect_per_seed_summary(result["worker_dir"], config_stem)
        if row is None:
            print(
                f"  warning: seed {seed} subprocess succeeded but summary CSV "
                f"missing in {result['worker_dir']}"
            )
            failed_seeds.append(seed)
            continue
        summary_rows.append(row)
        move_per_seed_csv(result["worker_dir"], config_output_dir, config_stem, seed)

    if failed_seeds:
        print(f"\nFAILED seeds ({len(failed_seeds)}): {failed_seeds}")
        print("  Inspect the corresponding stdout.log files under "
              f"{worker_root} for diagnostics.")
        if not summary_rows:
            print("No successful seeds — skipping aggregation.")
            sys.exit(1)
        print(f"Aggregating the {len(summary_rows)} successful seeds anyway.")

    combined_path = write_combined_summary(config_output_dir, config_stem, summary_rows)
    print(f"\nCombined summary CSV: {combined_path}")
    print_aggregate_stats(summary_rows)

    # Optional cleanup. Default: keep worker dirs (logs are tiny, debugging
    # value when something goes wrong is large).
    if not args.keep_worker_dirs:
        if failed_seeds:
            print("\nKeeping worker dirs because at least one seed failed.")
        else:
            shutil.rmtree(worker_root)
            print(f"\nRemoved worker dirs at {worker_root}.")


if __name__ == "__main__":
    main()
