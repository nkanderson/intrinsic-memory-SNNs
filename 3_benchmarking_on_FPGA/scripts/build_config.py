"""Wrapper that invokes Vivado batch-mode build_config.tcl with paths from configs.py.

Run from repo root:
    python 3_benchmarking_on_FPGA/scripts/build_config.py <config_name>
    python 3_benchmarking_on_FPGA/scripts/build_config.py --all

Requires `vivado` on PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from configs import CONFIGS, REPO_ROOT, Config


TCL = REPO_ROOT / "3_benchmarking_on_FPGA" / "scripts" / "build_config.tcl"


def build(cfg: Config) -> int:
    if not cfg.has_synth_artifacts:
        print(f"SKIP {cfg.name}: missing board_top or xdc", file=sys.stderr)
        return 0
    if shutil.which("vivado") is None:
        print("ERROR: vivado not on PATH", file=sys.stderr)
        return 2

    cmd = [
        "vivado", "-mode", "batch", "-source", str(TCL),
        "-tclargs", cfg.name, cfg.board_top, cfg.xdc, cfg.weights_dir,
    ]
    print(f"+ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    if rc != 0:
        return rc

    # Post-synth sanity: scan the synth log for memory-init failures that
    # Vivado treats as non-fatal CRITICAL WARNINGs. A bitstream with
    # uninitialized memories programs cleanly but produces garbage results.
    results_dir = REPO_ROOT / "3_benchmarking_on_FPGA" / "results" / cfg.name
    synth_log = next(
        (p for p in results_dir.rglob("synth_1/runme.log")), None
    )
    if synth_log is not None:
        bad_lines = [
            line for line in synth_log.read_text(errors="ignore").splitlines()
            if "could not open $readmem data file" in line
        ]
        if bad_lines:
            print(
                f"\nERROR: synthesis log shows unresolved $readmemh files "
                f"({len(bad_lines)} occurrence(s)). The bitstream will run but "
                "produce garbage. First line:",
                file=sys.stderr,
            )
            print(f"  {bad_lines[0]}", file=sys.stderr)
            print(
                "Fix: ensure every .mem file referenced in the design is "
                "added to the Vivado project (see build_config.tcl).",
                file=sys.stderr,
            )
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("config", nargs="?", help="Config name to build")
    g.add_argument("--all", action="store_true", help="Build every config with synth artifacts")
    args = parser.parse_args()

    if args.all:
        targets = [c for c in CONFIGS if c.has_synth_artifacts]
    else:
        try:
            targets = [next(c for c in CONFIGS if c.name == args.config)]
        except StopIteration:
            print(f"Unknown config {args.config!r}", file=sys.stderr)
            return 2

    rc = 0
    for cfg in targets:
        r = build(cfg)
        if r != 0:
            print(f"FAIL {cfg.name} (exit {r})", file=sys.stderr)
            rc = rc or r
    return rc


if __name__ == "__main__":
    sys.exit(main())
