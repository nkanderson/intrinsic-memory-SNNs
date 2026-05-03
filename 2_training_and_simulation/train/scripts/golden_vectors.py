"""Generate golden vectors for QS2.13 reference model (lif-64-16).

Runs a deterministic CartPole-v1 rollout, converts observations to QS2.13,
computes expected actions/Q-accum via qs213_reference, and writes JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import gymnasium as gym

from qs213_reference import (
    FRAC_BITS,
    DATA_WIDTH,
    load_lif_64_16_config,
    run_inference,
    wrap_signed,
)


def float_to_qs213(value: float) -> int:
    return wrap_signed(int(round(value * (1 << FRAC_BITS))), DATA_WIDTH)


def obs_to_qs213(obs: List[float]) -> List[int]:
    if len(obs) != 4:
        raise ValueError("CartPole observation must have 4 elements")
    return [float_to_qs213(v) for v in obs]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate golden vectors")
    parser.add_argument("--out", default="golden_vectors.json")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_lif_64_16_config()

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)

    vectors = []
    for _ in range(args.steps):
        obs_q = obs_to_qs213(list(obs))
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

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"model": cfg.name, "vectors": vectors}, indent=2))
    print(f"Wrote {len(vectors)} vectors to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
