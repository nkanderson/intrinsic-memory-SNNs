"""Wrapper that invokes Vivado batch-mode build_config.tcl with paths from configs.py.

Run from repo root:
    python 3_benchmarking_on_FPGA/scripts/build_config.py <config_name>
    python 3_benchmarking_on_FPGA/scripts/build_config.py <config_name> --profile onehot_top_fsm
    python 3_benchmarking_on_FPGA/scripts/build_config.py --all
    python 3_benchmarking_on_FPGA/scripts/build_config.py --all --profile onehot_top_fsm

The Vivado project lives at results/<config>/vivado_project/ (shared across
profiles); reports + bitstream land in results/<config>/<profile>/. Default
profile is "baseline".

Requires `vivado` on PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import re

from configs import CONFIGS, REPO_ROOT, Config

TCL = REPO_ROOT / "3_benchmarking_on_FPGA" / "scripts" / "build_config.tcl"
DEFAULT_PROFILE = "baseline"

FSM_ENCODE_RE = re.compile(
    r"INFO: \[Synth 8-3354\] encoded FSM with state register '([^']+)' "
    r"using encoding '([^']+)' in module '([^']+)'"
)


def _is_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) == {"-"}


def parse_fsm_encodings(log_path: Path) -> list[dict[str, object]]:
    try:
        lines = log_path.read_text(errors="ignore").splitlines()
    except OSError:
        return []

    entries: list[dict[str, object]] = []
    last_table: list[tuple[str, str, str]] | None = None
    in_table = False
    saw_data = False
    rows: list[tuple[str, str, str]] = []

    for line in lines:
        if "State" in line and "New Encoding" in line and "Previous Encoding" in line:
            in_table = True
            saw_data = False
            rows = []
            continue

        if in_table:
            if _is_separator(line):
                if saw_data:
                    last_table = rows
                    in_table = False
                continue
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    state, new_enc, prev_enc = parts[0], parts[1], parts[2]
                    if state and state != "State":
                        rows.append((state, new_enc, prev_enc))
                        saw_data = True
                continue

        match = FSM_ENCODE_RE.search(line)
        if match:
            state_reg, encoding, module = match.groups()
            entries.append(
                {
                    "module": module,
                    "state_reg": state_reg,
                    "encoding": encoding,
                    "states": last_table,
                }
            )
            last_table = None

    return entries


def write_fsm_encodings(log_path: Path, profile_dir: Path) -> None:
    entries = parse_fsm_encodings(log_path)
    profile_dir.mkdir(parents=True, exist_ok=True)
    out_path = profile_dir / "fsm_encodings.txt"

    if not entries:
        out_path.write_text(
            f"No FSM encoding entries found in {log_path}\n", encoding="utf-8"
        )
        return

    lines: list[str] = [
        f"FSM encodings extracted from: {log_path}",
        "",
    ]
    for entry in entries:
        module = entry["module"]
        state_reg = entry["state_reg"]
        encoding = entry["encoding"]
        states = entry.get("states")
        lines.append(f"Module: {module}")
        lines.append(f"State register: {state_reg}")
        lines.append(f"Encoding: {encoding}")
        lines.append("States:")
        if states:
            lines.append("  State | New Encoding (Vivado) | Previous Encoding (RTL)")
            for state, new_enc, prev_enc in states:
                lines.append(f"  {state} | {new_enc} | {prev_enc}")
        else:
            lines.append("  (state table not found in log)")
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build(cfg: Config, profile: str) -> int:
    if not cfg.has_synth_artifacts:
        print(f"SKIP {cfg.name}: missing board_top or xdc", file=sys.stderr)
        return 0
    if shutil.which("vivado") is None:
        print("ERROR: vivado not on PATH", file=sys.stderr)
        return 2

    cmd = [
        "vivado",
        "-mode",
        "batch",
        "-source",
        str(TCL),
        "-tclargs",
        cfg.name,
        cfg.board_top,
        cfg.xdc,
        cfg.weights_dir,
        profile,
    ]
    print(f"+ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    if rc != 0:
        return rc

    # Post-synth sanity: scan the synth log for memory-init failures that
    # Vivado treats as non-fatal CRITICAL WARNINGs. A bitstream with
    # uninitialized memories programs cleanly but produces garbage results.
    # The synth log lives under the shared (config-level) project dir, not
    # the profile-scoped reports dir.
    results_dir = REPO_ROOT / "3_benchmarking_on_FPGA" / "results" / cfg.name
    profile_dir = results_dir / profile
    synth_log = next((p for p in results_dir.rglob("synth_1/runme.log")), None)
    if synth_log is not None:
        bad_lines = [
            line
            for line in synth_log.read_text(errors="ignore").splitlines()
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
        write_fsm_encodings(synth_log, profile_dir)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("config", nargs="?", help="Config name to build")
    g.add_argument(
        "--all", action="store_true", help="Build every config with synth artifacts"
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=f"Profile subdir for reports + bitstream (default: {DEFAULT_PROFILE!r}). "
        "The Vivado project dir is shared across profiles.",
    )
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
        r = build(cfg, args.profile)
        if r != 0:
            print(f"FAIL {cfg.name} (exit {r})", file=sys.stderr)
            rc = rc or r
    return rc


if __name__ == "__main__":
    sys.exit(main())
