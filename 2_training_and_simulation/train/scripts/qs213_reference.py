"""Bit-exact fixed-point reference model for SNN inference."""
from __future__ import annotations

import argparse
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from history_coefficients import (
    custom_bitshift,
    custom_slow_decay_bitshift,
    simple_bitshift,
    slow_decay_bitshift,
)


DATA_WIDTH = 16
MEMBRANE_WIDTH = 24
THRESHOLD = 8192
BETA = 115


@dataclass(frozen=True)
class ModelConfig:
    name: str
    neuron_type: str  # lif | fractional | bitshift
    num_inputs: int
    hl1_size: int
    hl2_size: int
    num_actions: int
    num_timesteps: int
    threshold: int
    fc2_output_width: int
    frac_bits: int
    weights_dir: Path
    # Fractional params
    history_length: int = 8
    coeff_width: int = 16
    coeff_frac_bits: int = 16
    inv_denom: int = 58982
    inv_denom_frac_bits: int = 16
    gl_coeff_file: str | None = None
    # Bitshift params
    shift_width: int = 8
    shift_mode: int = 3
    custom_decay_rate: int = 3


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


def load_mem_file_unsigned(path: Path, bits: int) -> List[int]:
    values: List[int] = []
    mask = (1 << bits) - 1
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        if "//" in line:
            line = line.split("//", 1)[0].strip()
        if not line:
            continue
        value = int(line, 16) & mask
        values.append(value)
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
    frac_bits: int,
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
    membrane_width: int,
    threshold: int,
    beta: int,
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


def fractional_step(
    membrane: int,
    spike_prev: int,
    current: int,
    history: List[int],
    history_ptr: int,
    cfg: ModelConfig,
) -> Tuple[int, int, List[int], int]:
    history_length = cfg.history_length
    coeffs_mag = load_mem_file_unsigned(
        cfg.weights_dir / cfg.gl_coeff_file, cfg.coeff_width
    )

    product_width = MEMBRANE_WIDTH + cfg.coeff_width + 1
    history_sum_width = product_width + 3
    numerator_input_width = max(history_sum_width, MEMBRANE_WIDTH)
    numerator_width = numerator_input_width + 1
    inv_denom_width = 16 + 1
    scaled_result_width = numerator_width + inv_denom_width

    history_sum_acc = 0
    for k in range(1, history_length):
        if history_ptr >= k:
            hist_idx = history_ptr - k
        else:
            hist_idx = history_ptr + history_length - k
        hist_val = wrap_signed(history[hist_idx], MEMBRANE_WIDTH)
        coeff_mag = coeffs_mag[k - 1]
        mac_product = wrap_signed(coeff_mag * hist_val, product_width)
        mac_product_ext = wrap_signed(mac_product, history_sum_width)
        history_sum_acc = wrap_signed(history_sum_acc + mac_product_ext, history_sum_width)

    prep_scaled_history = wrap_signed(history_sum_acc >> cfg.coeff_frac_bits, history_sum_width)
    current_latched = wrap_signed(current, MEMBRANE_WIDTH)
    numerator = wrap_signed(current_latched + prep_scaled_history, numerator_width)
    mul_scaled = wrap_signed(numerator * cfg.inv_denom, scaled_result_width)
    div_membrane = wrap_signed(mul_scaled >> cfg.inv_denom_frac_bits, scaled_result_width)

    reset_subtract = cfg.threshold if spike_prev else 0
    membrane_after_reset = div_membrane - reset_subtract

    membrane_max = (1 << (MEMBRANE_WIDTH - 1)) - 1
    membrane_min = -(1 << (MEMBRANE_WIDTH - 1))
    if membrane_after_reset > membrane_max:
        finalize_membrane = membrane_max
    elif membrane_after_reset < membrane_min:
        finalize_membrane = membrane_min
    else:
        finalize_membrane = wrap_signed(membrane_after_reset, MEMBRANE_WIDTH)

    finalize_spike = 1 if finalize_membrane >= cfg.threshold else 0

    history[history_ptr] = wrap_signed(membrane, MEMBRANE_WIDTH)
    history_ptr = 0 if history_ptr == history_length - 1 else history_ptr + 1
    return finalize_membrane, finalize_spike, history, history_ptr


@lru_cache(maxsize=None)
def _bitshift_sequence(
    history_length: int, shift_mode: int, custom_decay_rate: int
) -> Tuple[int, ...]:
    if shift_mode == 0:
        seq = simple_bitshift(history_length)
    elif shift_mode == 1:
        seq = slow_decay_bitshift(history_length)
    elif shift_mode == 2:
        seq = custom_bitshift(history_length, custom_decay_rate)
    else:
        seq = custom_slow_decay_bitshift(history_length)
    return tuple(seq)


def bitshift_shift_amount(idx: int, cfg: ModelConfig) -> int:
    seq = _bitshift_sequence(cfg.history_length, cfg.shift_mode, cfg.custom_decay_rate)
    shift = seq[idx] if idx < len(seq) else seq[-1]
    max_shift = (1 << cfg.shift_width) - 1
    if shift < 0:
        return 0
    if shift > max_shift:
        return max_shift
    return shift


def bitshift_step(
    membrane: int,
    spike_prev: int,
    current: int,
    history: List[int],
    history_ptr: int,
    cfg: ModelConfig,
) -> Tuple[int, int, List[int], int]:
    history_length = cfg.history_length
    history_sum_width = MEMBRANE_WIDTH + max(0, (history_length - 1).bit_length())
    numerator_input_width = max(history_sum_width, MEMBRANE_WIDTH)
    numerator_width = numerator_input_width + 1
    inv_denom_width = 16 + 1
    scaled_result_width = numerator_width + inv_denom_width

    history_sum_acc = 0
    for k in range(1, history_length):
        if history_ptr >= k:
            hist_idx = history_ptr - k
        else:
            hist_idx = history_ptr + history_length - k
        hist_val = wrap_signed(history[hist_idx], MEMBRANE_WIDTH)
        shift_amt = bitshift_shift_amount(k, cfg)
        shifted = wrap_signed(hist_val >> shift_amt, MEMBRANE_WIDTH)
        history_sum_acc = wrap_signed(history_sum_acc + shifted, history_sum_width)

    current_latched = wrap_signed(current, MEMBRANE_WIDTH)
    numerator = wrap_signed(
        wrap_signed(current_latched, numerator_width)
        + wrap_signed(history_sum_acc, numerator_width),
        numerator_width,
    )
    mul_scaled = wrap_signed(numerator * cfg.inv_denom, scaled_result_width)
    membrane_pre_reset = wrap_signed(mul_scaled >> cfg.inv_denom_frac_bits, scaled_result_width)

    reset_subtract = cfg.threshold if spike_prev else 0
    membrane_after_reset = membrane_pre_reset - reset_subtract

    membrane_max = (1 << (MEMBRANE_WIDTH - 1)) - 1
    membrane_min = -(1 << (MEMBRANE_WIDTH - 1))
    if membrane_after_reset > membrane_max:
        finalize_membrane = membrane_max
    elif membrane_after_reset < membrane_min:
        finalize_membrane = membrane_min
    else:
        finalize_membrane = wrap_signed(membrane_after_reset, MEMBRANE_WIDTH)

    finalize_spike = 1 if finalize_membrane >= cfg.threshold else 0

    history[history_ptr] = wrap_signed(membrane, MEMBRANE_WIDTH)
    history_ptr = 0 if history_ptr == history_length - 1 else history_ptr + 1
    return finalize_membrane, finalize_spike, history, history_ptr


def q_accumulate(
    membranes: List[List[int]],
    weights: List[List[int]],
    biases: List[int],
    frac_bits: int,
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


def model_matrix() -> Dict[str, ModelConfig]:
    base = repo_root() / "common" / "sv" / "cocotb" / "tests" / "weights"
    return {
        "lif-64-16": ModelConfig(
            name="lif-64-16",
            neuron_type="lif",
            num_inputs=4,
            hl1_size=64,
            hl2_size=16,
            num_actions=2,
            num_timesteps=10,
            threshold=THRESHOLD,
            fc2_output_width=24,
            frac_bits=13,
            weights_dir=base / "lif-64-16",
        ),
        "lif-32-16": ModelConfig(
            name="lif-32-16",
            neuron_type="lif",
            num_inputs=4,
            hl1_size=32,
            hl2_size=16,
            num_actions=2,
            num_timesteps=10,
            threshold=THRESHOLD,
            fc2_output_width=24,
            frac_bits=13,
            weights_dir=base / "lif-32-16",
        ),
        "fractional-16-4-32": ModelConfig(
            name="fractional-16-4-32",
            neuron_type="fractional",
            num_inputs=4,
            hl1_size=16,
            hl2_size=4,
            num_actions=2,
            num_timesteps=20,
            threshold=THRESHOLD,
            fc2_output_width=24,
            frac_bits=13,
            weights_dir=base / "fractional-16-4-32",
            history_length=32,
            coeff_width=16,
            coeff_frac_bits=16,
            inv_denom=55727,
            inv_denom_frac_bits=16,
            gl_coeff_file="gl_coefficients.mem",
        ),
        "fractional-32-4-16": ModelConfig(
            name="fractional-32-4-16",
            neuron_type="fractional",
            num_inputs=4,
            hl1_size=32,
            hl2_size=4,
            num_actions=2,
            num_timesteps=10,
            threshold=THRESHOLD,
            fc2_output_width=24,
            frac_bits=13,
            weights_dir=base / "fractional-32-4-16",
            history_length=16,
            coeff_width=16,
            coeff_frac_bits=16,
            inv_denom=62259,
            inv_denom_frac_bits=16,
            gl_coeff_file="gl_coefficients.mem",
        ),
        "bitshift-custom_slow_decay": ModelConfig(
            name="bitshift-custom_slow_decay",
            neuron_type="bitshift",
            num_inputs=4,
            hl1_size=32,
            hl2_size=8,
            num_actions=2,
            num_timesteps=10,
            threshold=1 << 12,
            fc2_output_width=24,
            frac_bits=12,
            weights_dir=base / "bitshift-custom_slow_decay",
            history_length=8,
            shift_width=8,
            shift_mode=3,
            custom_decay_rate=3,
            inv_denom=59823,
            inv_denom_frac_bits=16,
        ),
    }


def load_model_config(name: str) -> ModelConfig:
    configs = model_matrix()
    if name not in configs:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(configs)}")
    return configs[name]


def run_inference(obs_fixed: List[int], cfg: ModelConfig) -> Tuple[int, List[int]]:
    if len(obs_fixed) != cfg.num_inputs:
        raise ValueError(f"Expected {cfg.num_inputs} observations, got {len(obs_fixed)}")

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

    hl1_currents = linear_layer(obs_fixed, fc1_weights, fc1_biases, DATA_WIDTH, cfg.frac_bits)

    hl1_mem = [0 for _ in range(cfg.hl1_size)]
    hl1_spike_prev = [0 for _ in range(cfg.hl1_size)]
    hl2_mem = [0 for _ in range(cfg.hl2_size)]
    hl2_spike_prev = [0 for _ in range(cfg.hl2_size)]

    hl1_history = [[0 for _ in range(cfg.history_length)] for _ in range(cfg.hl1_size)]
    hl1_history_ptr = [0 for _ in range(cfg.hl1_size)]
    hl2_history = [[0 for _ in range(cfg.history_length)] for _ in range(cfg.hl2_size)]
    hl2_history_ptr = [0 for _ in range(cfg.hl2_size)]

    membranes_by_timestep: List[List[int]] = []
    hl1_spikes_for_fc2 = [0 for _ in range(cfg.hl1_size)]
    use_delayed_hl1 = cfg.neuron_type in ("fractional", "bitshift")

    for _t in range(cfg.num_timesteps):
        hl1_spikes: List[int] = []
        for i in range(cfg.hl1_size):
            if cfg.neuron_type == "lif":
                hl1_mem[i], hl1_spike_prev[i] = lif_step(
                    hl1_mem[i],
                    hl1_spike_prev[i],
                    hl1_currents[i],
                    data_width=DATA_WIDTH,
                    membrane_width=MEMBRANE_WIDTH,
                    threshold=THRESHOLD,
                    beta=BETA,
                )
            elif cfg.neuron_type == "fractional":
                hl1_mem[i], hl1_spike_prev[i], hl1_history[i], hl1_history_ptr[i] = fractional_step(
                    hl1_mem[i],
                    hl1_spike_prev[i],
                    hl1_currents[i],
                    hl1_history[i],
                    hl1_history_ptr[i],
                    cfg,
                )
            else:
                hl1_mem[i], hl1_spike_prev[i], hl1_history[i], hl1_history_ptr[i] = bitshift_step(
                    hl1_mem[i],
                    hl1_spike_prev[i],
                    hl1_currents[i],
                    hl1_history[i],
                    hl1_history_ptr[i],
                    cfg,
                )
            hl1_spikes.append(hl1_spike_prev[i])

        if use_delayed_hl1:
            fc2_inputs = [cfg.threshold if s else 0 for s in hl1_spikes_for_fc2]
            hl1_spikes_for_fc2 = hl1_spikes
        else:
            fc2_inputs = [cfg.threshold if s else 0 for s in hl1_spikes]
        hl2_currents = linear_layer(
            fc2_inputs,
            fc2_weights,
            fc2_biases,
            cfg.fc2_output_width,
            cfg.frac_bits,
        )

        mem_t: List[int] = []
        for i in range(cfg.hl2_size):
            if cfg.neuron_type == "lif":
                hl2_mem[i], hl2_spike_prev[i] = lif_step(
                    hl2_mem[i],
                    hl2_spike_prev[i],
                    hl2_currents[i],
                    data_width=cfg.fc2_output_width,
                    membrane_width=MEMBRANE_WIDTH,
                    threshold=THRESHOLD,
                    beta=BETA,
                )
            elif cfg.neuron_type == "fractional":
                hl2_mem[i], hl2_spike_prev[i], hl2_history[i], hl2_history_ptr[i] = fractional_step(
                    hl2_mem[i],
                    hl2_spike_prev[i],
                    hl2_currents[i],
                    hl2_history[i],
                    hl2_history_ptr[i],
                    cfg,
                )
            else:
                hl2_mem[i], hl2_spike_prev[i], hl2_history[i], hl2_history_ptr[i] = bitshift_step(
                    hl2_mem[i],
                    hl2_spike_prev[i],
                    hl2_currents[i],
                    hl2_history[i],
                    hl2_history_ptr[i],
                    cfg,
                )

            mem_t.append(hl2_mem[i])
        membranes_by_timestep.append(mem_t)

    q_accum = q_accumulate(membranes_by_timestep, fc_out_weights, fc_out_biases, cfg.frac_bits)
    action = select_action(q_accum)
    return action, q_accum


def parse_obs(values: str, frac_bits: int) -> List[int]:
    parts = [p.strip() for p in values.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("Expected 4 comma-separated values")
    obs = []
    for p in parts:
        if p.startswith("0x"):
            obs.append(wrap_signed(int(p, 16), DATA_WIDTH))
        else:
            obs.append(int(round(float(p) * (1 << frac_bits))))
    return [wrap_signed(v, DATA_WIDTH) for v in obs]


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed-point reference model")
    parser.add_argument(
        "--model",
        default="lif-64-16",
        help="Model name (lif-64-16, lif-32-16, fractional-16-4-32, fractional-32-4-16, bitshift-custom_slow_decay)",
    )
    parser.add_argument(
        "--obs",
        default="0.0,0.0,0.0,0.0",
        help="Comma-separated observations (float) or 0xHEX per value",
    )
    args = parser.parse_args()

    cfg = load_model_config(args.model)
    action, q_accum = run_inference(parse_obs(args.obs, cfg.frac_bits), cfg)
    print(f"model={cfg.name}")
    print(f"selected_action={action}")
    print(f"q_accum={q_accum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
