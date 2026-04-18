"""Aggregate synthesis/implementation metrics for neuron demo variants.

Reads Vivado report files from 1_design_space/results for each variant and
prints a compact table covering area, timing, and power.
Also writes a CSV file to 1_design_space/metrics/synthesis_summary.csv.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class VariantConfig:
    name: str
    results_dir_name: str
    top_name: str


VARIANTS = [
    VariantConfig(name="lif", results_dir_name="lif", top_name="top_lif_demo"),
    VariantConfig(
        name="fractional_lif",
        results_dir_name="fractional_lif",
        top_name="top_fractional_lif_demo",
    ),
    VariantConfig(
        name="bitshift_lif",
        results_dir_name="bitshift_lif",
        top_name="top_bitshift_lif_demo",
    ),
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _find_int_in_table(text: str, row_label: str) -> Optional[int]:
    # Matches rows like: | Slice LUTs* |  960 | ...
    pattern = re.compile(rf"\|\s*{re.escape(row_label)}\s*\|\s*([0-9,]+)\s*\|", re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def _find_float_summary(text: str, row_label: str) -> Optional[float]:
    # Matches rows like: | Dynamic (W) | 0.004 |
    pattern = re.compile(rf"\|\s*{re.escape(row_label)}\s*\|\s*([<>]?[0-9]+(?:\.[0-9]+)?)\s*\|", re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        return None
    value = m.group(1)
    if value.startswith("<"):
        value = value[1:]
    try:
        return float(value)
    except ValueError:
        return None


def _find_text_summary(text: str, row_label: str) -> Optional[str]:
    pattern = re.compile(rf"\|\s*{re.escape(row_label)}\s*\|\s*([^|]+?)\s*\|", re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        return None
    return m.group(1).strip()


def parse_utilization_report(path: Path) -> Dict[str, Optional[int]]:
    text = _read_text(path)
    if not text:
        return {}

    metrics: Dict[str, Optional[int]] = {
        "slice_luts": _find_int_in_table(text, "Slice LUTs*"),
        "slice_registers": _find_int_in_table(text, "Slice Registers"),
        "dsp": _find_int_in_table(text, "DSPs"),
        "bram_tiles": _find_int_in_table(text, "Block RAM Tile"),
        "f7_muxes": _find_int_in_table(text, "F7 Muxes"),
        "f8_muxes": _find_int_in_table(text, "F8 Muxes"),
        "carry4": _find_int_in_table(text, "CARRY4"),
    }

    # Approximate total cells from primitives table (sum of "Used" column entries).
    total_cells = 0
    primitive_row_pattern = re.compile(r"\|\s*[A-Za-z0-9_]+\s*\|\s*([0-9,]+)\s*\|")
    in_primitives = False
    for line in text.splitlines():
        if line.strip().startswith("7. Primitives"):
            in_primitives = True
            continue
        if in_primitives and line.strip().startswith("8. Black Boxes"):
            break
        if in_primitives:
            m = primitive_row_pattern.search(line)
            if m:
                total_cells += int(m.group(1).replace(",", ""))
    metrics["total_cells_approx"] = total_cells if total_cells > 0 else None
    return metrics


def parse_timing_report(path: Path) -> Dict[str, Optional[float]]:
    text = _read_text(path)
    if not text:
        return {}

    metrics: Dict[str, Optional[float]] = {
        "wns_ns": None,
        "tns_ns": None,
        "whs_ns": None,
        "clock_period_ns": None,
        "fmax_est_mhz": None,
    }

    # Parse design timing summary data row
    summary_row = re.search(
        r"\n\s*([\-0-9.]+)\s+([\-0-9.]+)\s+\d+\s+\d+\s+([\-0-9.]+)\s+[\-0-9.]+\s+\d+\s+\d+",
        text,
    )
    if summary_row:
        metrics["wns_ns"] = float(summary_row.group(1))
        metrics["tns_ns"] = float(summary_row.group(2))
        metrics["whs_ns"] = float(summary_row.group(3))

    # Parse sys clock period from Clock Summary section
    clk = re.search(r"sys_clk_pin\s+\{[^}]+\}\s+([0-9.]+)\s+[0-9.]+", text)
    if clk:
        period = float(clk.group(1))
        metrics["clock_period_ns"] = period
        if metrics["wns_ns"] is not None:
            # Setup slack = required - arrival -> arrival ~= period - WNS
            worst_path_delay = period - metrics["wns_ns"]
            if worst_path_delay > 0:
                metrics["fmax_est_mhz"] = 1000.0 / worst_path_delay
    return metrics


def parse_power_report(path: Path) -> Dict[str, Optional[float | str]]:
    text = _read_text(path)
    if not text:
        return {}

    return {
        "power_total_w": _find_float_summary(text, "Total On-Chip Power (W)"),
        "power_dynamic_w": _find_float_summary(text, "Dynamic (W)"),
        "power_static_w": _find_float_summary(text, "Device Static (W)"),
        "junction_temp_c": _find_float_summary(text, "Junction Temperature (C)"),
        "power_confidence": _find_text_summary(text, "Confidence Level"),
    }


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_table(rows: list[dict]) -> None:
    columns = [
        ("variant", "Variant"),
        ("total_cells_approx", "TotalCells~"),
        ("slice_luts", "LUTs"),
        ("slice_registers", "FFs"),
        ("carry4", "CARRY4"),
        ("f7_muxes", "F7"),
        ("f8_muxes", "F8"),
        ("dsp", "DSP"),
        ("bram_tiles", "BRAM"),
        ("wns_ns", "WNS(ns)"),
        ("tns_ns", "TNS(ns)"),
        ("whs_ns", "WHS(ns)"),
        ("clock_period_ns", "Clk(ns)"),
        ("fmax_est_mhz", "FmaxEst(MHz)"),
        ("power_total_w", "Ptot(W)"),
        ("power_dynamic_w", "Pdyn(W)"),
        ("power_static_w", "Pstatic(W)"),
        ("junction_temp_c", "Tj(C)"),
        ("power_confidence", "PwrConf"),
    ]

    header = [title for _, title in columns]
    table = [[_fmt(row.get(key)) for key, _ in columns] for row in rows]
    widths = [max(len(header[i]), *(len(r[i]) for r in table)) for i in range(len(header))]

    def line(parts: list[str]) -> str:
        return " | ".join(parts[i].ljust(widths[i]) for i in range(len(parts)))

    print(line(header))
    print("-+-".join("-" * w for w in widths))
    for r in table:
        print(line(r))


def write_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "total_cells_approx",
        "slice_luts",
        "slice_registers",
        "carry4",
        "f7_muxes",
        "f8_muxes",
        "dsp",
        "bram_tiles",
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
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    results_root = base / "results"
    rows: list[dict] = []

    for variant in VARIANTS:
        variant_dir = results_root / variant.results_dir_name
        util_path = variant_dir / "synth_utilization.rpt"
        timing_path = variant_dir / "impl_timing_summary.rpt"
        power_path = (
            variant_dir
            / "vivado_project"
            / f"{variant.top_name}.runs"
            / "impl_1"
            / f"{variant.top_name}_power_routed.rpt"
        )

        row: dict = {"variant": variant.name}
        row.update(parse_utilization_report(util_path))
        row.update(parse_timing_report(timing_path))
        row.update(parse_power_report(power_path))
        rows.append(row)

    print("Synthesis/Implementation Summary\n")
    print_table(rows)

    csv_path = base / "metrics" / "synthesis_summary.csv"
    write_csv(rows, csv_path)
    print(f"\nCSV written: {csv_path}")


if __name__ == "__main__":
    main()

