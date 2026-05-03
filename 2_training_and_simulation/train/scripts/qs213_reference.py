"""Bit-exact QS2.13 reference model for SNN inference (LIF only).

Implements the same fixed-point arithmetic as common/sv/neural_network.sv
for the lif-64-16 configuration. Fractional/bitshift variants can be
added later by extending the model matrix and neuron step functions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


FRAC_BITS = 13
DATA_WIDTH = 16
MEMBRANE_WIDTH = 24
THRESHOLD = 8192
BETA = 115


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_inputs: int
    hl1_size: int
    hl2_size: int
    num_actions: int
    num_timesteps: int
    fc2_output_width: int
    weights_dir: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def wrap_signed(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    value &= mask
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


def saturate_signed(value: int, bits: int) -> int:
    max_val = (1 << (bits - 1)) - 1
    min_val = -(1 << (bits - 1))
    return max(min(value, max_val), min_val)


def load_mem_file(path: Path, bits: int) -> List[int]:
    values: List[int] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        if "//" in line:
            line = line.split("//", 1)[0].strip()
        if not line:
            continue
        value = int(line, 16)
        values.append(wrap_signed(value, bits))
    return values


def reshape(values: List[int], rows: int, cols: int) -> List[List[int]]:
    if len(values) != rows * cols:
        raise ValueError(f"Expected {rows*cols} values, got {len(values)}")
    return [values[r * cols : (r + 1) * cols] for r in range(rows)]


def linear_layer(
    inputs: List[int],
    weights: List[List[int]],
    biases: List[int],
    output_width: int,
    frac_bits: int = FRAC_BITS,
) -> List[int]:
    outputs: List[int] = []
    for row, bias in zip(weights, biases):
        accum = 0
        for inp, w in zip(inputs, row):
            accum += inp * w
        scaled = (accum >> frac_bits) + bias
        outputs.append(saturate_signed(scaled, output_width))
    return outputs


def lif_step(
    membrane: int,
    spike_prev: int,
    current: int,
    data_width: int,
    membrane_width: int = MEMBRANE_WIDTH,
    threshold: int = THRESHOLD,
    beta: int = BETA,
) -> Tuple[int, int]:
    membrane = wrap_signed(membrane, membrane_width)
    current = wrap_signed(current, data_width)
    current_ext = wrap_signed(current, membrane_width)

    decay_temp = membrane * beta
    decay_potential = wrap_signed(decay_temp >> 7, membrane_width)
    reset_subtract = threshold if spike_prev else 0
    next_membrane = wrap_signed(
        decay_potential + current_ext - reset_subtract, membrane_width
    )
    next_spike = 1 if next_membrane >= threshold else 0
    return next_membrane, next_spike


def q_accumulate(
    membranes: List[List[int]],
    weights: List[List[int]],
    biases: List[int],
    frac_bits: int = FRAC_BITS,
) -> List[int]:
    num_timesteps = len(membranes)
    num_actions = len(weights)
    q_accum = [0 for _ in range(num_actions)]
    for t in range(num_timesteps):
        mem_t = membranes[t]
        for a in range(num_actions):
            accum = 0
            for mem, w in zip(mem_t, weights[a]):
                accum += mem * w
            q_t = (accum >> frac_bits) + biases[a]
            q_accum[a] += q_t
    return q_accum


def select_action(q_accum: List[int]) -> int:
    if len(q_accum) != 2:
        return int(max(range(len(q_accum)), key=lambda i: q_accum[i]))
    return 0 if q_accum[0] >= q_accum[1] else 1


def load_lif_64_16_config() -> ModelConfig:
    return ModelConfig(
        name="lif-64-16",
        num_inputs=4,
        hl1_size=64,
        hl2_size=16,
        num_actions=2,
        num_timesteps=10,
        fc2_output_width=24,
        weights_dir=repo_root()
        / "common"
        / "sv"
        / "cocotb"
        / "tests"
        / "weights"
        / "lif-64-16",
    )


def run_inference(obs_qs213: List[int], cfg: ModelConfig) -> Tuple[int, List[int]]:
    if len(obs_qs213) != cfg.num_inputs:
        raise ValueError(f"Expected {cfg.num_inputs} observations, got {len(obs_qs213)}")

    fc1_weights = reshape(
        load_mem_file(cfg.weights_dir / "fc1_weights.mem", DATA_WIDTH),
        cfg.hl1_size,
        cfg.num_inputs,
    )
    fc1_biases = load_mem_file(cfg.weights_dir / "fc1_bias.mem", DATA_WIDTH)
    fc2_weights = reshape(
        load_mem_file(cfg.weights_dir / "fc2_weights.mem", DATA_WIDTH),
        cfg.hl2_size,
        cfg.hl1_size,
    )
    fc2_biases = load_mem_file(cfg.weights_dir / "fc2_bias.mem", DATA_WIDTH)
    fc_out_weights = reshape(
        load_mem_file(cfg.weights_dir / "fc_out_weights.mem", DATA_WIDTH),
        cfg.num_actions,
        cfg.hl2_size,
    )
    fc_out_biases = load_mem_file(cfg.weights_dir / "fc_out_bias.mem", DATA_WIDTH)

    hl1_currents = linear_layer(obs_qs213, fc1_weights, fc1_biases, DATA_WIDTH)

    hl1_mem = [0 for _ in range(cfg.hl1_size)]
    hl1_spike_prev = [0 for _ in range(cfg.hl1_size)]
    hl2_mem = [0 for _ in range(cfg.hl2_size)]
    hl2_spike_prev = [0 for _ in range(cfg.hl2_size)]

    membranes_by_timestep: List[List[int]] = []

    for _t in range(cfg.num_timesteps):
        hl1_spikes: List[int] = []
        for i in range(cfg.hl1_size):
            hl1_mem[i], hl1_spike_prev[i] = lif_step(
                hl1_mem[i],
                hl1_spike_prev[i],
                hl1_currents[i],
                data_width=DATA_WIDTH,
            )
            hl1_spikes.append(hl1_spike_prev[i])

        fc2_inputs = [THRESHOLD if s else 0 for s in hl1_spikes]
        hl2_currents = linear_layer(
            fc2_inputs,
            fc2_weights,
            fc2_biases,
            cfg.fc2_output_width,
        )

        mem_t: List[int] = []
        for i in range(cfg.hl2_size):
            hl2_mem[i], hl2_spike_prev[i] = lif_step(
                hl2_mem[i],
                hl2_spike_prev[i],
                hl2_currents[i],
                data_width=cfg.fc2_output_width,
            )
            mem_t.append(hl2_mem[i])
        membranes_by_timestep.append(mem_t)

    q_accum = q_accumulate(membranes_by_timestep, fc_out_weights, fc_out_biases)
    action = select_action(q_accum)
    return action, q_accum


def parse_obs(values: str) -> List[int]:
    parts = [p.strip() for p in values.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("Expected 4 comma-separated values")
    obs = []
    for p in parts:
        if p.startswith("0x"):
            obs.append(wrap_signed(int(p, 16), DATA_WIDTH))
        else:
            obs.append(int(round(float(p) * (1 << FRAC_BITS))))
    return [wrap_signed(v, DATA_WIDTH) for v in obs]


def main() -> int:
    parser = argparse.ArgumentParser(description="QS2.13 reference model (lif-64-16)")
    parser.add_argument(
        "--obs",
        default="0.0,0.0,0.0,0.0",
        help="Comma-separated observations (float) or 0xHEX per value",
    )
    args = parser.parse_args()

    cfg = load_lif_64_16_config()
    action, q_accum = run_inference(parse_obs(args.obs), cfg)
    print(f"model={cfg.name}")
    print(f"selected_action={action}")
    print(f"q_accum={q_accum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
