import argparse
import csv
from collections import defaultdict


FRAC_BITS = 13
SCALE = 1 << FRAC_BITS


def parse_float_or_none(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def hw_fixed_to_float(value):
    parsed = parse_float_or_none(value)
    if parsed is None:
        return None
    return parsed / SCALE


def fmt_num(value):
    if value is None:
        return ""
    return f"{value:.6f}"


def load_trace(path):
    by_seed = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            seed = int(row["seed"])
            step = int(row["step"])
            by_seed[seed][step] = row
    return by_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare software vs hardware action traces"
    )
    parser.add_argument("--sw", required=True, help="Software trace CSV")
    parser.add_argument("--hw", required=True, help="Hardware trace CSV")
    args = parser.parse_args()

    sw = load_trace(args.sw)
    hw = load_trace(args.hw)

    seeds = sorted(set(sw.keys()) & set(hw.keys()))
    if not seeds:
        raise SystemExit("No overlapping seeds between traces")

    for seed in seeds:
        max_step = min(max(sw[seed].keys()), max(hw[seed].keys()))
        first_div = None
        for step in range(max_step + 1):
            sw_row = sw[seed].get(step)
            hw_row = hw[seed].get(step)
            if int(sw_row["action"]) != int(hw_row["action"]):
                first_div = step
                break

        if first_div is None:
            print(f"seed={seed}: actions identical through step {max_step}")
        else:
            sw_row = sw[seed][first_div]
            hw_row = hw[seed][first_div]
            sw_q0 = parse_float_or_none(sw_row.get("q0", ""))
            sw_q1 = parse_float_or_none(sw_row.get("q1", ""))
            sw_mem_mean = parse_float_or_none(sw_row.get("hl2_mem_mean", ""))

            hw_q0_raw = parse_float_or_none(hw_row.get("q0", ""))
            hw_q1_raw = parse_float_or_none(hw_row.get("q1", ""))
            hw_mem_mean_raw = parse_float_or_none(hw_row.get("hl2_mem_mean", ""))

            hw_q0 = hw_fixed_to_float(hw_row.get("q0", ""))
            hw_q1 = hw_fixed_to_float(hw_row.get("q1", ""))
            hw_mem_mean = hw_fixed_to_float(hw_row.get("hl2_mem_mean", ""))

            print(
                f"seed={seed}: first divergence at step {first_div} "
                f"(sw={sw_row['action']}, hw={hw_row['action']})"
            )
            print(
                f"  sw q=({fmt_num(sw_q0)}, {fmt_num(sw_q1)}) "
                f"hl2_mem_mean={fmt_num(sw_mem_mean)}"
            )
            print(
                f"  hw q_raw=({fmt_num(hw_q0_raw)}, {fmt_num(hw_q1_raw)}) "
                f"hl2_mem_mean_raw={fmt_num(hw_mem_mean_raw)}"
            )
            print(
                f"  hw q_decoded=({fmt_num(hw_q0)}, {fmt_num(hw_q1)}) "
                f"hl2_mem_mean_decoded={fmt_num(hw_mem_mean)}"
            )
