"""Hardware-in-the-loop CartPole-v1 evaluation using the FPGA SNN accelerator.

Runs CartPole-v1 episodes with actions selected by the FPGA over UART and
writes per-episode results to:
    3_benchmarking_on_FPGA/results/<config>/hw_eval_seed<N>_ep<M>.csv

Usage:
    python eval_cartpole_hw.py --config lif-64-16 --episodes 100 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import gymnasium as gym
import torch

from fpga_interface import FpgaInterface
from snn_policy_hardware import SNNPolicyHardware


def run_episodes(
    policy: SNNPolicyHardware,
    env: gym.Env,
    n_episodes: int,
    seed: int,
) -> list[tuple[float, int, float]]:
    """Run n_episodes. Returns list of (total_reward, steps, elapsed_s)."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        total_reward = 0.0
        steps = 0
        t0 = time.monotonic()
        done = False
        while not done:
            with torch.no_grad():
                action = int(policy(state).max(1).indices.item())
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            total_reward += reward
            steps += 1
        elapsed = time.monotonic() - t0
        results.append((total_reward, steps, elapsed))
        print(
            f"  ep {ep + 1:4d}: reward={total_reward:6.0f}  "
            f"steps={steps:3d}  {elapsed * 1000:.0f} ms"
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hardware-in-the-loop CartPole evaluation"
    )
    parser.add_argument(
        "--config", default="lif-64-16",
        help="Model config name (e.g. lif-64-16, frac-32-4-16); used for results subdir",
    )
    parser.add_argument("--port", default="/dev/ttyUSB1")
    parser.add_argument("--baud", type=int, default=921_600)
    parser.add_argument("--timeout", type=float, default=1.0,
                        help="Per-operation serial timeout (s)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results_dir = (
        Path(__file__).resolve().parents[1] / "results" / args.config
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = (
        results_dir / f"hw_eval_seed{args.seed}_ep{args.episodes}.csv"
    )

    env = gym.make("CartPole-v1")

    with FpgaInterface(args.port, args.baud, timeout=args.timeout) as fpga:
        fpga.ping()
        print(
            f"PING OK — config={args.config}  "
            f"episodes={args.episodes}  seed={args.seed}  port={args.port}"
        )
        policy = SNNPolicyHardware(n_observations=4, n_actions=2, fpga_interface=fpga)
        results = run_episodes(policy, env, args.episodes, args.seed)

    env.close()

    rewards = [r[0] for r in results]
    mean_reward = sum(rewards) / len(rewards)
    print(f"\nMean reward: {mean_reward:.1f}  ({args.episodes} episodes)")

    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "elapsed_s"])
        for i, (reward, steps, elapsed) in enumerate(results):
            writer.writerow([i + 1, reward, steps, f"{elapsed:.4f}"])
    print(f"Results written to {results_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
