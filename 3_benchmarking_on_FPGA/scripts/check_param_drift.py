"""Verify per-config simulation params (cocotb Makefile) match synthesis params (board_top SV).

For each config in configs.py that has both a Makefile cp_integration target and a
board_top SV file, parse the parameter values from each side and report any
mismatches. Exit nonzero if any drift is found.

Usage:
    python 3_benchmarking_on_FPGA/scripts/check_param_drift.py
    python 3_benchmarking_on_FPGA/scripts/check_param_drift.py --config lif-64-16
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from configs import CONFIGS, REPO_ROOT, SV_ROOT, Config

MAKEFILE = REPO_ROOT / "common" / "sv" / "cocotb" / "tests" / "Makefile"

PARAMS_NUMERIC = {
    "HL1_SIZE", "HL2_SIZE", "NUM_INPUTS", "NUM_ACTIONS", "NUM_TIMESTEPS",
    "Q_BATCH_SIZE", "FC2_OUTPUT_WIDTH", "FRAC_BITS",
    "HISTORY_LENGTH", "COEFF_WIDTH", "COEFF_FRAC_BITS", "INV_DENOM",
    "SHIFT_WIDTH", "SHIFT_MODE", "CUSTOM_DECAY_RATE",
    "DATA_WIDTH", "THRESHOLD",
}
PARAMS_FILE_BASENAME = {
    "FC1_WEIGHTS_FILE", "FC1_BIAS_FILE",
    "FC2_WEIGHTS_FILE", "FC2_BIAS_FILE",
    "FC_OUT_WEIGHTS_FILE", "FC_OUT_BIAS_FILE",
    "GL_COEFF_FILE",
}


def _strip_sized_literal(s: str) -> str:
    """Normalize Verilog sized literals: '16'd59823' -> '59823', '2'd3' -> '3'."""
    m = re.fullmatch(r"\d+'[dDhHbBoO]([0-9a-fA-F_]+)", s.strip())
    if m:
        return m.group(1).lstrip("0") or "0"
    return s.strip().lstrip("0") or "0"


def _basename(path_or_str: str) -> str:
    s = path_or_str.strip().strip('"')
    return Path(s).name


def parse_makefile_block(target: str) -> dict[str, str]:
    """Extract COMPILE_ARGS parameter overrides for a given TEST target.

    The Makefile has multiple blocks per target (sources, icarus COMPILE_ARGS,
    verilator COMPILE_ARGS). Scan all blocks and use the one that actually
    contains COMPILE_ARGS lines (the simulator branches).
    """
    text = MAKEFILE.read_text(encoding="utf-8")
    block_re = re.compile(
        rf"ifeq \(\$\(TEST\),{re.escape(target)}\)(.*?)endif",
        re.DOTALL,
    )

    param_re = re.compile(
        r"-P[A-Za-z_][A-Za-z_0-9]*\.([A-Z_][A-Z_0-9]*)=(\\\".*?\\\"|[^ \n\t\\]+)"
    )

    params: dict[str, str] = {}
    matched_any_block = False
    for m in block_re.finditer(text):
        body = m.group(1)
        if "COMPILE_ARGS" not in body:
            continue
        matched_any_block = True
        for pm in param_re.finditer(body):
            name = pm.group(1)
            raw = pm.group(2)
            if raw.startswith("\\\""):
                raw = raw[2:-2]
            # Only set on first sim-branch encounter; the icarus and verilator
            # branches should agree, but if they ever diverge we'd want to flag
            # that separately. Take icarus (the first one) as canonical.
            if name not in params:
                params[name] = raw
    if not matched_any_block:
        raise KeyError(f"No COMPILE_ARGS block found for TEST={target}")
    return params


def parse_board_top(sv_path: Path) -> dict[str, str]:
    """Extract `.PARAM(value)` overrides from the top_uart_accel_wrapper instantiation."""
    text = sv_path.read_text(encoding="utf-8")
    inst_re = re.compile(
        r"top_uart_accel_wrapper\s*#\s*\((.*?)\)\s*[A-Za-z_]",
        re.DOTALL,
    )
    m = inst_re.search(text)
    if not m:
        raise ValueError(f"No top_uart_accel_wrapper instantiation found in {sv_path}")
    body = m.group(1)

    params: dict[str, str] = {}
    param_re = re.compile(r"\.([A-Z_][A-Z_0-9]*)\s*\(\s*([^,)]+?)\s*\)")
    for pm in param_re.finditer(body):
        params[pm.group(1)] = pm.group(2)
    return params


def normalize(name: str, value: str) -> str:
    if name in PARAMS_FILE_BASENAME:
        return _basename(value)
    if name in PARAMS_NUMERIC:
        v = value.strip()
        v = v.replace("_", "")
        return _strip_sized_literal(v)
    return value.strip()


def compare_config(cfg: Config) -> tuple[list[tuple[str, str, str, bool]], int]:
    if not cfg.has_synth_artifacts:
        return [], 0

    mk = parse_makefile_block(cfg.cocotb_test_target)
    sv = parse_board_top(SV_ROOT / f"{cfg.board_top}.sv")

    common = sorted(set(mk) & set(sv))
    rows: list[tuple[str, str, str, bool]] = []
    mismatches = 0
    for name in common:
        mk_norm = normalize(name, mk[name])
        sv_norm = normalize(name, sv[name])
        ok = mk_norm == sv_norm
        if not ok:
            mismatches += 1
        rows.append((name, mk[name], sv[name], ok))
    return rows, mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Limit to a single config name")
    args = parser.parse_args()

    targets = [c for c in CONFIGS if c.has_synth_artifacts]
    if args.config:
        targets = [c for c in targets if c.name == args.config]
        if not targets:
            print(f"No synth-eligible config named {args.config!r}", file=sys.stderr)
            return 2

    total_mismatches = 0
    for cfg in targets:
        rows, n_mm = compare_config(cfg)
        total_mismatches += n_mm
        status = "OK" if n_mm == 0 else f"{n_mm} MISMATCH(es)"
        print(f"\n=== {cfg.name}  [{status}] ===")
        print(f"  Makefile : TEST={cfg.cocotb_test_target}")
        print(f"  Board top: {cfg.board_top}.sv")
        if not rows:
            print("  (no common parameters)")
            continue
        widths = [max(len(r[0]) for r in rows), max(len(r[1]) for r in rows),
                  max(len(r[2]) for r in rows)]
        header = f"  {'PARAM'.ljust(widths[0])}  {'MAKEFILE'.ljust(widths[1])}  {'BOARD_TOP'.ljust(widths[2])}  STATUS"
        print(header)
        for name, mk_v, sv_v, ok in rows:
            tag = "ok" if ok else "MISMATCH"
            print(f"  {name.ljust(widths[0])}  {mk_v.ljust(widths[1])}  {sv_v.ljust(widths[2])}  {tag}")

    print()
    if total_mismatches:
        print(f"FAIL: {total_mismatches} parameter mismatch(es) across configs.")
        return 1
    print("OK: all per-config parameters consistent between Makefile and board_top.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
