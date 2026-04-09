import argparse
import csv
from pathlib import Path
import sys

# Ensure train/ is on sys.path when running this script from train/scripts/
TRAIN_DIR = Path(__file__).resolve().parents[1]
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import gymnasium as gym
import torch
from snntorch import surrogate

from snn_policy import SNNPolicy


def parse_seeds(seeds_str: str):
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def run_trace(model_path: str, seeds: list[int], output_csv: str, max_steps: int = 500):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    spike_grad = surrogate.fast_sigmoid(slope=25)
    model = SNNPolicy(
        n_observations=cfg["n_observations"],
        n_actions=cfg["n_actions"],
        num_steps=cfg["num_steps"],
        beta=cfg["beta"],
        spike_grad=spike_grad,
        neuron_type=cfg["neuron_type"],
        hidden1_size=cfg.get("hidden1_size", 32),
        hidden2_size=cfg.get("hidden2_size", 16),
        alpha=cfg.get("alpha", 0.5),
        lam=cfg.get("lam", 0.111),
        history_length=cfg.get("history_length", 64),
        dt=cfg.get("dt", 1.0),
    )
    model.load_state_dict(checkpoint["policy_net_state_dict"])
    model.eval()

    env = gym.make("CartPole-v1")
    rows = []

    with torch.no_grad():
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            for step in range(max_steps):
                state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                q_values = model(state)
                action = int(q_values.max(1).indices.item())

                hl2_mem = getattr(model.lif2, "mem", None)
                if hl2_mem is not None and torch.is_tensor(hl2_mem) and hl2_mem.numel() > 0:
                    hl2_mem_mean = float(hl2_mem.mean().item())
                    hl2_mem_max = float(hl2_mem.max().item())
                    hl2_mem_min = float(hl2_mem.min().item())
                else:
                    hl2_mem_mean = ""
                    hl2_mem_max = ""
                    hl2_mem_min = ""

                rows.append(
                    {
                        "seed": seed,
                        "step": step,
                        "action": action,
                        "q0": float(q_values[0, 0].item()),
                        "q1": float(q_values[0, 1].item()),
                        "hl2_mem_mean": hl2_mem_mean,
                        "hl2_mem_max": hl2_mem_max,
                        "hl2_mem_min": hl2_mem_min,
                        "obs0": float(obs[0]),
                        "obs1": float(obs[1]),
                        "obs2": float(obs[2]),
                        "obs3": float(obs[3]),
                    }
                )

                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

    env.close()

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "step",
                "action",
                "q0",
                "q1",
                "hl2_mem_mean",
                "hl2_mem_max",
                "hl2_mem_min",
                "obs0",
                "obs1",
                "obs2",
                "obs3",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote software action trace: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export software policy action traces for fixed seeds"
    )
    parser.add_argument("--model", required=True, help="Path to .pth model checkpoint")
    parser.add_argument("--seeds", default="42,49", help="Comma-separated seed list")
    parser.add_argument(
        "--output",
        default="../metrics/cartpole_action_trace_sw.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--max-steps", type=int, default=500, help="Maximum steps per episode"
    )
    args = parser.parse_args()

    run_trace(
        args.model, parse_seeds(args.seeds), args.output, max_steps=args.max_steps
    )
