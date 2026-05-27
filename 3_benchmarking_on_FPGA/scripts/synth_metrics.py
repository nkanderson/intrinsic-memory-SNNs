"""Parse Vivado synth/impl reports for one stage-3 config (profile-aware).

Expects the following canonical filenames under
3_benchmarking_on_FPGA/results/<config>/<profile>/ (produced by
build_config.tcl, or copied manually from a GUI Vivado project):

  impl_utilization.rpt
  impl_timing_summary.rpt
  impl_power.rpt

The default profile is "baseline". Additional profiles (e.g. alternative
FSM encodings) live alongside it; any subdir of results/<config>/ that
contains impl_utilization.rpt is treated as a profile. Reports missing
any of these are tolerated — the missing fields stay None.

Usage:
    python 3_benchmarking_on_FPGA/scripts/synth_metrics.py --config lif-64-16
    python 3_benchmarking_on_FPGA/scripts/synth_metrics.py --config lif-64-16 --profile onehot_top_fsm
    python 3_benchmarking_on_FPGA/scripts/synth_metrics.py            # all configs, all profiles
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

from configs import CONFIGS, RESULTS_ROOT, Config

REPORT_FILES = {
    "utilization": "impl_utilization.rpt",
    "timing": "impl_timing_summary.rpt",
    "power": "impl_power.rpt",
}

DEFAULT_PROFILE = "baseline"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _find_int_in_table(text: str, row_label: str) -> Optional[int]:
    pattern = re.compile(
        rf"\|\s*{re.escape(row_label)}\s*\|\s*([0-9,]+)\s*\|",
        re.IGNORECASE,
    )
    m = pattern.search(text)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def _find_float_summary(text: str, row_label: str) -> Optional[float]:
    pattern = re.compile(
        rf"\|\s*{re.escape(row_label)}\s*\|\s*([<>]?[0-9]+(?:\.[0-9]+)?)\s*\|",
        re.IGNORECASE,
    )
    m = pattern.search(text)
    if not m:
        return None
    v = m.group(1)
    if v.startswith(("<", ">")):
        v = v[1:]
    try:
        return float(v)
    except ValueError:
        return None


def _find_text_summary(text: str, row_label: str) -> Optional[str]:
    pattern = re.compile(
        rf"\|\s*{re.escape(row_label)}\s*\|\s*([^|]+?)\s*\|",
        re.IGNORECASE,
    )
    m = pattern.search(text)
    if not m:
        return None
    return m.group(1).strip()


def parse_utilization(path: Path) -> dict[str, Optional[int]]:
    text = _read(path)
    if not text:
        return {
            k: None
            for k in (
                "slice_luts",
                "lut_as_logic",
                "lut_as_memory",
                "slice_registers",
                "dsp",
                "bram_tiles",
                "carry4",
                "f7_muxes",
                "f8_muxes",
            )
        }
    return {
        "slice_luts": _find_int_in_table(text, "Slice LUTs*")
        or _find_int_in_table(text, "Slice LUTs"),
        "lut_as_logic": _find_int_in_table(text, "LUT as Logic"),
        "lut_as_memory": _find_int_in_table(text, "LUT as Memory"),
        "slice_registers": _find_int_in_table(text, "Slice Registers"),
        "dsp": _find_int_in_table(text, "DSPs"),
        "bram_tiles": _find_int_in_table(text, "Block RAM Tile"),
        "carry4": _find_int_in_table(text, "CARRY4"),
        "f7_muxes": _find_int_in_table(text, "F7 Muxes"),
        "f8_muxes": _find_int_in_table(text, "F8 Muxes"),
    }


def parse_timing(path: Path) -> dict[str, Optional[float]]:
    text = _read(path)
    out: dict[str, Optional[float]] = {
        "wns_ns": None,
        "tns_ns": None,
        "whs_ns": None,
        "clock_period_ns": None,
        "fmax_est_mhz": None,
    }
    if not text:
        return out

    # Design Timing Summary table row (WNS TNS FE WTN WHS THS FE WTH ...)
    row = re.search(
        r"\n\s*(-?[0-9.]+)\s+(-?[0-9.]+)\s+\d+\s+\d+\s+(-?[0-9.]+)\s+-?[0-9.]+\s+\d+\s+\d+",
        text,
    )
    if row:
        out["wns_ns"] = float(row.group(1))
        out["tns_ns"] = float(row.group(2))
        out["whs_ns"] = float(row.group(3))

    # Sys clock period from Clock Summary
    clk = re.search(r"sys_clk_pin\s+\{[^}]+\}\s+([0-9.]+)\s+[0-9.]+", text)
    if clk:
        period = float(clk.group(1))
        out["clock_period_ns"] = period
        if out["wns_ns"] is not None:
            arrival = period - out["wns_ns"]
            if arrival > 0:
                out["fmax_est_mhz"] = 1000.0 / arrival
    return out


def parse_power(path: Path) -> dict[str, object]:
    text = _read(path)
    if not text:
        return {
            "power_total_w": None,
            "power_dynamic_w": None,
            "power_static_w": None,
            "junction_temp_c": None,
            "power_confidence": None,
        }
    return {
        "power_total_w": _find_float_summary(text, "Total On-Chip Power (W)"),
        "power_dynamic_w": _find_float_summary(text, "Dynamic (W)"),
        "power_static_w": _find_float_summary(text, "Device Static (W)"),
        "junction_temp_c": _find_float_summary(text, "Junction Temperature (C)"),
        "power_confidence": _find_text_summary(text, "Confidence Level"),
    }


def profile_dir(cfg: Config, profile: str) -> Path:
    return RESULTS_ROOT / cfg.name / profile


def discover_profiles(cfg: Config) -> list[str]:
    """Return sorted profile names under results/<cfg.name>/ (dirs containing impl_utilization.rpt).

    If no profile subdirs exist but the legacy flat layout (impl_utilization.rpt
    directly under results/<cfg.name>/) is present, return [DEFAULT_PROFILE] and
    let parse_config fall back to the flat path. If neither exists, return
    [DEFAULT_PROFILE] anyway so the config still appears as an empty row.
    """
    base = RESULTS_ROOT / cfg.name
    if not base.is_dir():
        return [DEFAULT_PROFILE]
    profiles = sorted(
        p.name
        for p in base.iterdir()
        if p.is_dir() and (p / REPORT_FILES["utilization"]).exists()
    )
    if profiles:
        return profiles
    return [DEFAULT_PROFILE]


def parse_config(cfg: Config, profile: str = DEFAULT_PROFILE) -> dict[str, object]:
    cfg_dir = profile_dir(cfg, profile)
    # Fall back to the legacy flat layout if the profile dir does not exist.
    if not (cfg_dir / REPORT_FILES["utilization"]).exists():
        flat_dir = RESULTS_ROOT / cfg.name
        if (flat_dir / REPORT_FILES["utilization"]).exists():
            cfg_dir = flat_dir
    row: dict[str, object] = {
        "config": cfg.name,
        "profile": profile,
        "neuron_type": cfg.neuron_type,
        "display_label": cfg.display_label,
    }
    row.update(parse_utilization(cfg_dir / REPORT_FILES["utilization"]))
    row.update(parse_timing(cfg_dir / REPORT_FILES["timing"]))
    row.update(parse_power(cfg_dir / REPORT_FILES["power"]))
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
]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", help="Single config name; default: all with synth artifacts"
    )
    parser.add_argument(
        "--profile",
        help="Single profile name (e.g. 'baseline'). Default: every discovered profile per config.",
    )
    parser.add_argument("--csv", type=Path, help="Write results to CSV file")
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON to stdout instead of table"
    )
    args = parser.parse_args()

    if args.config:
        targets = [c for c in CONFIGS if c.name == args.config]
        if not targets:
            print(f"Unknown config {args.config!r}", file=sys.stderr)
            return 2
    else:
        targets = [c for c in CONFIGS if c.has_synth_artifacts]

    rows: list[dict[str, object]] = []
    for c in targets:
        profiles = [args.profile] if args.profile else discover_profiles(c)
        for prof in profiles:
            rows.append(parse_config(c, prof))

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        for r in rows:
            print(f"\n=== {r['config']} / {r['profile']} ({r['neuron_type']}) ===")
            for k in CSV_FIELDS[4:]:
                print(f"  {k:22s} {r.get(k)}")

    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nCSV written: {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
