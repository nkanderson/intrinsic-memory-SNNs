"""Generate golden vectors for fixed-point reference models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import gymnasium as gym

from qs213_reference import DATA_WIDTH, load_model_config, run_inference, wrap_signed


def float_to_fixed(value: float, frac_bits: int) -> int:
    return wrap_signed(int(round(value * (1 << frac_bits))), DATA_WIDTH)


def obs_to_fixed(obs: List[float], frac_bits: int) -> List[int]:
    if len(obs) != 4:
        raise ValueError("CartPole observation must have 4 elements")
    return [float_to_fixed(v, frac_bits) for v in obs]


def default_output_path(model_name: str) -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "common"
        / "sv"
        / "cocotb"
        / "tests"
        / "golden_vectors"
        / f"{model_name}.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate golden vectors")
    parser.add_argument("--model", default="lif-64-16")
    parser.add_argument("--out", default="")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_model_config(args.model)

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)

    vectors = []
    for _ in range(args.steps):
        obs_q = obs_to_fixed(list(obs), cfg.frac_bits)
        action, q_accum = run_inference(obs_q, cfg)
        vectors.append(
            {
                "obs_qs213": obs_q,
                "expected_action": int(action),
                "expected_q_accum": [int(q) for q in q_accum],
            }
        )
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            obs, _ = env.reset()

    out_path = Path(args.out) if args.out else default_output_path(cfg.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": cfg.name, "vectors": vectors}, indent=2))
    print(f"Wrote {len(vectors)} vectors to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
