"""Join per-(config, profile) Vivado PPA metrics with cocotb-measured inference cycles.

Per-(config, profile) inputs (any may be missing — corresponding columns stay blank):
  - Vivado reports under 3_benchmarking_on_FPGA/results/<config>/<profile>/
  - Cycle CSV at one of:
      3_benchmarking_on_FPGA/results/<config>/<profile>/cycles.csv  (canonical post-copy)
      common/sv/cocotb/cycle_results/<config>/cycles.csv            (preserved cocotb output,
                                                                     baseline only)
      common/sv/cocotb/results/<config>/cycles.csv                  (legacy, baseline only)

Profile discovery: every subdir of results/<config>/ that contains
impl_utilization.rpt is treated as a profile. If none exist but a legacy
flat layout is present, the synth_metrics fallback handles it under the
default profile "baseline".

Derived columns (when both fmax_est_mhz and total_cycles are present):
  - latency_us               = total_cycles / fmax_est_mhz
  - throughput_hz            = 1e6 / latency_us
  - energy_per_inference_uj  = power_total_w * latency_us   (only if power_total_w present)

Output: 3_benchmarking_on_FPGA/results/summary/ppa_cycles_combined.csv

Usage:
    python 3_benchmarking_on_FPGA/scripts/aggregate_ppa.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

from configs import CONFIGS, REPO_ROOT, RESULTS_ROOT, Config
from synth_metrics import (
    DEFAULT_PROFILE,
    discover_profiles,
    parse_config as parse_synth,
)

COCOTB_CYCLE_RESULTS_ROOT = REPO_ROOT / "common" / "sv" / "cocotb" / "cycle_results"
COCOTB_LEGACY_RESULTS_ROOT = REPO_ROOT / "common" / "sv" / "cocotb" / "results"

CYCLE_FIELDS = [
    "total_cycles",
    "cycles_idle",
    "cycles_load_hl1",
    "cycles_run_timesteps",
    "cycles_finish_q",
    "cycles_done_state",
    "cycles_ts_hl1_step",
    "cycles_ts_fc2_start",
    "cycles_ts_fc2_wait",
    "cycles_ts_hl2_step",
    "cycles_ts_hl2_wait",
    "cycles_ts_next",
    "num_timesteps",
    "cycles_per_timestep",
    "hl1_size",
    "hl2_size",
    "history_length",
]

DERIVED_FIELDS = [
    "latency_us",
    "throughput_hz",
    "energy_per_inference_uj",
]


def find_cycles_csv(cfg: Config, profile: str) -> Optional[Path]:
    """Locate cycles.csv for (cfg, profile), preferring the canonical profile dir.

    Falls back to the preserved cocotb output and finally the legacy mount,
    but only for the default profile — non-baseline profiles must have their
    cycles.csv copied into the profile dir explicitly.
    """
    profile_csv = RESULTS_ROOT / cfg.name / profile / "cycles.csv"
    if profile_csv.exists():
        return profile_csv
    if profile == DEFAULT_PROFILE:
        for candidate in (
            COCOTB_CYCLE_RESULTS_ROOT / cfg.name / "cycles.csv",
            COCOTB_LEGACY_RESULTS_ROOT / cfg.name / "cycles.csv",
        ):
            if candidate.exists():
                return candidate
    return None


def read_cycles(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return dict(row)
    return {}


def _as_float(x) -> Optional[float]:
    if x is None or x == "" or x == "None":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _as_int(x) -> Optional[int]:
    f = _as_float(x)
    return int(f) if f is not None else None


def merge(cfg: Config, profile: str) -> dict[str, object]:
    row: dict[str, object] = parse_synth(cfg, profile)

    csv_path = find_cycles_csv(cfg, profile)
    if csv_path is not None:
        cycles = read_cycles(csv_path)
        row["cycles_csv_path"] = str(csv_path.relative_to(REPO_ROOT))
        for k in CYCLE_FIELDS:
            row[k] = cycles.get(k)
    else:
        row["cycles_csv_path"] = ""
        for k in CYCLE_FIELDS:
            row[k] = None

    fmax = _as_float(row.get("fmax_est_mhz"))
    cyc = _as_int(row.get("total_cycles"))
    pwr = _as_float(row.get("power_total_w"))

    latency_us: Optional[float] = None
    throughput_hz: Optional[float] = None
    energy_uj: Optional[float] = None
    if fmax and fmax > 0 and cyc and cyc > 0:
        latency_us = cyc / fmax
        throughput_hz = 1e6 / latency_us
        if pwr is not None and pwr > 0:
            energy_uj = pwr * latency_us
    row["latency_us"] = latency_us
    row["throughput_hz"] = throughput_hz
    row["energy_per_inference_uj"] = energy_uj
    return row


CSV_FIELDS = [
    "config",
    "profile",
    "neuron_type",
    "display_label",
    "slice_luts",
    "lut_as_logic",
    "lut_as_memory",
    "slice_registers",
    "dsp",
    "bram_tiles",
    "carry4",
    "f7_muxes",
    "f8_muxes",
    "wns_ns",
    "tns_ns",
    "whs_ns",
    "clock_period_ns",
    "fmax_est_mhz",
    "power_total_w",
    "power_dynamic_w",
    "power_static_w",
    "junction_temp_c",
    "power_confidence",
    *CYCLE_FIELDS,
    *DERIVED_FIELDS,
    "cycles_csv_path",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "summary" / "ppa_cycles_combined.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for c in CONFIGS:
        for prof in discover_profiles(c):
            rows.append(merge(c, prof))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_FIELDS})

    print(f"Wrote {args.output}")
    print(f"  rows (config x profile): {len(rows)}")
    populated = sum(1 for r in rows if r.get("slice_luts") is not None)
    cycled = sum(1 for r in rows if r.get("total_cycles") not in (None, ""))
    derived = sum(1 for r in rows if r.get("latency_us") is not None)
    print(f"  with synth metrics: {populated}")
    print(f"  with cycle data: {cycled}")
    print(f"  with derived figures (latency/throughput/energy): {derived}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
