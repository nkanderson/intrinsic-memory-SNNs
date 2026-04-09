import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def to_int(value: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def to_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def preview_tuples(
    rows: list[dict[str, str]], keys: list[str], limit: int
) -> list[tuple[Any, ...]]:
    out = []
    for row in rows[:limit]:
        out.append(tuple(row.get(k, "") for k in keys))
    return out


def analyze(csv_path: Path, max_inferences: int, hl2_size: int, preview: int) -> int:
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"ERROR: CSV is empty: {csv_path}")
        return 1

    by_inference: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        inf = to_int(row.get("inference", ""))
        if inf is None:
            continue
        by_inference[inf].append(row)

    print(f"CSV: {csv_path}")
    print(f"Rows: {len(rows)}")
    print(f"Inferences captured: {len(by_inference)}")
    print()

    for inf in sorted(by_inference)[:max_inferences]:
        inf_rows = by_inference[inf]
        fc2_rows = [r for r in inf_rows if r.get("stage") == "fc2_stream"]
        tsum_rows = [r for r in inf_rows if r.get("stage") == "timestep_summary"]
        q_rows = [r for r in inf_rows if r.get("stage") == "q_progress"]
        final_rows = [r for r in inf_rows if r.get("stage") == "final"]
        final_action = final_rows[-1].get("selected_action", "") if final_rows else ""

        print(
            f"INF {inf}: action={final_action} fc2_rows={len(fc2_rows)} "
            f"timestep_summary={len(tsum_rows)} q_progress={len(q_rows)}"
        )

        if inf_rows and "obs0" in inf_rows[0]:
            obs_row = inf_rows[0]
            obs_float = (
                obs_row.get("obs0", ""),
                obs_row.get("obs1", ""),
                obs_row.get("obs2", ""),
                obs_row.get("obs3", ""),
            )
            obs_fixed = (
                obs_row.get("obs0_fixed", ""),
                obs_row.get("obs1_fixed", ""),
                obs_row.get("obs2_fixed", ""),
                obs_row.get("obs3_fixed", ""),
            )
            print(f"  obs_float: {obs_float}")
            print(f"  obs_fixed: {obs_fixed}")

            hl1_curr_sample = obs_row.get("hl1_t0_curr_sample", "")
            hl1_spike_sample = obs_row.get("hl1_t0_spike_sample", "")
            if hl1_curr_sample or hl1_spike_sample:
                print(f"  hl1_t0_curr_sample: {hl1_curr_sample}")
                print(f"  hl1_t0_spike_sample: {hl1_spike_sample}")

        print("  first_fc2:")
        for tup in preview_tuples(
            fc2_rows, ["timestep", "fc2_idx", "fc2_signed"], preview
        ):
            print(f"    {tup}")

        if fc2_rows and "fc2_sat_pos" in fc2_rows[0]:
            sat_pos_events = sum(
                to_int(r.get("fc2_sat_pos", "")) or 0 for r in fc2_rows
            )
            sat_neg_events = sum(
                to_int(r.get("fc2_sat_neg", "")) or 0 for r in fc2_rows
            )
            sat_pos_count = max(
                (to_int(r.get("fc2_sat_pos_count", "")) or 0) for r in inf_rows
            )
            sat_neg_count = max(
                (to_int(r.get("fc2_sat_neg_count", "")) or 0) for r in inf_rows
            )
            print(
                f"  fc2_saturation: pos_events={sat_pos_events} neg_events={sat_neg_events} "
                f"pos_count={sat_pos_count} neg_count={sat_neg_count}"
            )

        print("  first_timestep_summary:")
        for tup in preview_tuples(
            tsum_rows,
            ["timestep", "hl2_mem_mean", "hl2_mem_max", "hl2_mem_min"],
            preview,
        ):
            print(f"    {tup}")

        print("  first_q_progress:")
        for tup in preview_tuples(
            q_rows,
            ["q_read_timestep", "q_state", "q_accum0", "q_accum1", "q_div0", "q_div1"],
            preview,
        ):
            print(f"    {tup}")

        if fc2_rows and "hl1_spike_count" in fc2_rows[0]:
            hl1_counts = [to_int(r.get("hl1_spike_count", "")) for r in fc2_rows]
            hl1_counts = [x for x in hl1_counts if x is not None]
            if hl1_counts:
                print(
                    f"  hl1_spike_count (fc2_stream): min={min(hl1_counts)} "
                    f"max={max(hl1_counts)} unique={len(set(hl1_counts))}"
                )

        per_timestep_count: dict[int, int] = defaultdict(int)
        per_timestep_idx: dict[int, list[int]] = defaultdict(list)

        for row in fc2_rows:
            timestep = to_int(row.get("timestep", ""))
            idx = to_int(row.get("fc2_idx", ""))
            if timestep is None:
                continue
            per_timestep_count[timestep] += 1
            if idx is not None:
                per_timestep_idx[timestep].append(idx)

        bad_count = [
            (t, c) for t, c in sorted(per_timestep_count.items()) if c != hl2_size
        ]
        print(
            f"  fc2_count_mismatches (expected {hl2_size} per timestep): {len(bad_count)}"
        )
        if bad_count:
            print(f"    sample: {bad_count[:5]}")

        idx_issues = []
        expected = list(range(hl2_size))
        for t, idxs in sorted(per_timestep_idx.items()):
            sorted_idxs = sorted(idxs)
            if sorted_idxs != expected:
                idx_issues.append((t, sorted_idxs[:8], len(sorted_idxs)))
        print(f"  fc2_index_coverage_issues: {len(idx_issues)}")
        if idx_issues:
            print(f"    sample: {idx_issues[:5]}")

        print()

    ts0 = []
    for inf in sorted(by_inference)[:max_inferences]:
        rows0 = [
            r
            for r in by_inference[inf]
            if r.get("stage") == "timestep_summary" and r.get("timestep") == "0"
        ]
        if not rows0:
            continue
        r = rows0[0]
        ts0.append(
            (
                inf,
                to_float(r.get("hl2_mem_mean", "")),
                to_int(r.get("hl2_mem_max", "")),
                to_int(r.get("hl2_mem_min", "")),
            )
        )

    print("Timestep-0 HL2 summary across inferences:")
    for item in ts0:
        print(f"  inf={item[0]} mean={item[1]} max={item[2]} min={item[3]}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze cartpole timestep snapshot CSV exported by cocotb."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("../results/cartpole_timestep_snapshots_hw.csv"),
        help="Path to snapshot CSV",
    )
    parser.add_argument(
        "--inferences",
        type=int,
        default=5,
        help="How many inferences to summarize",
    )
    parser.add_argument(
        "--hl2-size",
        type=int,
        default=16,
        help="Expected HL2 neuron count / fc2 outputs per timestep",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=6,
        help="Rows to preview for each section",
    )
    args = parser.parse_args()

    return analyze(args.csv, args.inferences, args.hl2_size, args.preview)


if __name__ == "__main__":
    raise SystemExit(main())
