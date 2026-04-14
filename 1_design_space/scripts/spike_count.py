"""Compute expected spike-count outputs for Week 2 standalone neuron demos.

Models implemented here mirror the RTL arithmetic in:
  - sv/lif.sv
  - sv/fractional_lif.sv
  - sv/bitshift_lif.sv
"""

from __future__ import annotations

import argparse
from pathlib import Path


# Shared defaults (match top_*_lif_demo.sv)
RUN_STEPS_DEFAULT = 256
THRESHOLD_DEFAULT = 8192
DATA_WIDTH = 16
MEMBRANE_WIDTH = 24

# lif defaults
BETA_DEFAULT = 115

# fractional / bitshift defaults
HISTORY_LENGTH_DEFAULT = 32
BITSHIFT_HISTORY_LENGTH_DEFAULT = 32
C_SCALED_DEFAULT = 256
C_SCALED_FRAC_BITS_DEFAULT = 8
INV_DENOM_DEFAULT = 58982
INV_DENOM_FRAC_BITS_DEFAULT = 16

# bitshift defaults
SHIFT_MODE_DEFAULT = 3
CUSTOM_DECAY_RATE_DEFAULT = 3


def sat_signed(value: int, width: int) -> int:
    max_v = (1 << (width - 1)) - 1
    min_v = -(1 << (width - 1))
    if value > max_v:
        return max_v
    if value < min_v:
        return min_v
    return value


def parse_mem_file(path: Path) -> list[int]:
    coeffs: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("//", 1)[0].strip()
            if not line:
                continue
            coeffs.append(int(line, 16))
    return coeffs


def run_lif(current: int, run_steps: int, threshold: int, beta: int) -> int:
    mem = 0
    spike_prev = 0
    count = 0
    beta_u8 = beta & 0xFF

    for _ in range(run_steps):
        decay = (mem * beta_u8) >> 7
        reset_sub = threshold if spike_prev else 0
        nxt = decay + current - reset_sub
        spike = 1 if nxt >= threshold else 0

        mem = nxt
        spike_prev = spike
        count += spike

    return count


def run_fractional_lif(
    current: int,
    run_steps: int,
    threshold: int,
    history_length: int,
    c_scaled: int,
    c_scaled_frac_bits: int,
    inv_denom: int,
    inv_denom_frac_bits: int,
    gl_coeffs_mag: list[int],
) -> int:
    if len(gl_coeffs_mag) < history_length - 1:
        raise ValueError("Not enough GL coefficients for requested history length")

    history = [0] * history_length
    history_ptr = 0
    mem = 0
    spike_prev = 0
    count = 0

    for _ in range(run_steps):
        history_sum = 0
        for k in range(history_length - 1):
            k_plus_1 = k + 1
            if history_ptr >= k_plus_1:
                hist_idx = history_ptr - k_plus_1
            else:
                hist_idx = history_ptr + history_length - k_plus_1

            hist_val = history[hist_idx]
            coeff = gl_coeffs_mag[k]
            history_sum += coeff * hist_val

        reset_sub = threshold if spike_prev else 0

        scaled_history = (c_scaled * history_sum) >> (c_scaled_frac_bits + 15)
        numerator = current + scaled_history
        scaled_result = numerator * inv_denom
        membrane_pre_reset = scaled_result >> inv_denom_frac_bits
        membrane_after_reset = membrane_pre_reset - reset_sub

        nxt = sat_signed(membrane_after_reset, MEMBRANE_WIDTH)
        spike = 1 if nxt >= threshold else 0

        history[history_ptr] = mem
        history_ptr = 0 if history_ptr == history_length - 1 else history_ptr + 1
        mem = nxt
        spike_prev = spike
        count += spike

    return count


def get_shift_amount(idx: int, shift_mode: int, custom_decay_rate: int, shift_width: int = 8) -> int:
    if shift_mode == 0:
        shift = idx
    elif shift_mode == 1:
        shift = 0 if idx == 0 else (idx + 1) // 2
    elif shift_mode == 2:
        if idx == 0:
            shift = 0
        elif idx == 1:
            shift = 1
        elif idx == 2:
            shift = 3
        elif idx == 3:
            shift = 4
        else:
            shift = 5 + ((idx - 4) // custom_decay_rate)
    else:
        if idx == 0:
            shift = 0
        elif idx == 1:
            shift = 1
        elif idx == 2:
            shift = 3
        elif idx == 3:
            shift = 4
        else:
            rem = idx - 4
            shift = 5
            for s in range(5, (1 << shift_width)):
                repeat_count = s - 2
                if rem < repeat_count:
                    shift = s
                    break
                rem -= repeat_count

    shift = max(0, shift)
    shift = min((1 << shift_width) - 1, shift)
    return shift


def run_bitshift_lif(
    current: int,
    run_steps: int,
    threshold: int,
    history_length: int,
    shift_mode: int,
    custom_decay_rate: int,
    c_scaled: int,
    c_scaled_frac_bits: int,
    inv_denom: int,
    inv_denom_frac_bits: int,
) -> int:
    history = [0] * history_length
    history_ptr = 0
    mem = 0
    spike_prev = 0
    count = 0

    shifts = [get_shift_amount(i + 1, shift_mode, custom_decay_rate) for i in range(history_length - 1)]

    for _ in range(run_steps):
        history_sum = 0
        for k in range(history_length - 1):
            k_plus_1 = k + 1
            if history_ptr >= k_plus_1:
                hist_idx = history_ptr - k_plus_1
            else:
                hist_idx = history_ptr + history_length - k_plus_1
            history_sum += history[hist_idx] >> shifts[k]

        reset_sub = threshold if spike_prev else 0

        scaled_history = (c_scaled * history_sum) >> c_scaled_frac_bits
        numerator = current - scaled_history
        scaled_result = numerator * inv_denom
        membrane_pre_reset = scaled_result >> inv_denom_frac_bits
        membrane_after_reset = membrane_pre_reset - reset_sub

        nxt = sat_signed(membrane_after_reset, MEMBRANE_WIDTH)
        spike = 1 if nxt >= threshold else 0

        history[history_ptr] = mem
        history_ptr = 0 if history_ptr == history_length - 1 else history_ptr + 1
        mem = nxt
        spike_prev = spike
        count += spike

    return count


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_coeff_path = script_dir.parent / "sv" / "gl_coefficients.mem"

    parser = argparse.ArgumentParser(description="Expected spike counts for standalone neuron demos")
    parser.add_argument("--steps", type=int, default=RUN_STEPS_DEFAULT)
    parser.add_argument("--threshold", type=int, default=THRESHOLD_DEFAULT)
    parser.add_argument(
        "--currents",
        type=int,
        nargs="+",
        default=[5000, 9000, 10000, 20000],
        help="Input currents to evaluate (QS2.13 integer units)",
    )
    parser.add_argument("--coeff-file", type=Path, default=default_coeff_path)
    parser.add_argument(
        "--bitshift-history-length",
        type=int,
        default=BITSHIFT_HISTORY_LENGTH_DEFAULT,
        help="History length used for bitshift_lif expected counts",
    )
    args = parser.parse_args()

    gl_coeffs = parse_mem_file(args.coeff_file)

    print(f"RUN_STEPS={args.steps}, THRESHOLD={args.threshold}")
    print(f"GL_COEFF_FILE={args.coeff_file}")
    print(f"BITSHIFT_HISTORY_LENGTH={args.bitshift_history_length}")
    print()

    header = f"{'current':>8} | {'lif':>10} | {'fractional_lif':>16} | {'bitshift_lif':>14}"
    print(header)
    print("-" * len(header))

    for cur in args.currents:
        lif_count = run_lif(cur, args.steps, args.threshold, BETA_DEFAULT)
        frac_count = run_fractional_lif(
            cur,
            args.steps,
            args.threshold,
            HISTORY_LENGTH_DEFAULT,
            C_SCALED_DEFAULT,
            C_SCALED_FRAC_BITS_DEFAULT,
            INV_DENOM_DEFAULT,
            INV_DENOM_FRAC_BITS_DEFAULT,
            gl_coeffs,
        )
        bitshift_count = run_bitshift_lif(
            cur,
            args.steps,
            args.threshold,
            args.bitshift_history_length,
            SHIFT_MODE_DEFAULT,
            CUSTOM_DECAY_RATE_DEFAULT,
            C_SCALED_DEFAULT,
            C_SCALED_FRAC_BITS_DEFAULT,
            INV_DENOM_DEFAULT,
            INV_DENOM_FRAC_BITS_DEFAULT,
        )

        print(
            f"{cur:8d} | "
            f"{lif_count:4d} ({lif_count:#06x}) | "
            f"{frac_count:4d} ({frac_count:#06x}) | "
            f"{bitshift_count:4d} ({bitshift_count:#06x})"
        )


if __name__ == "__main__":
    main()
