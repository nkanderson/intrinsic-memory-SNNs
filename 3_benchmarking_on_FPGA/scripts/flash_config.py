"""Program a built bitstream onto the Nexys A7-100T via Vivado batch mode.

Looks up the bitstream from configs.py:
    results/<config>/bitstream.bit

Usage:
    python 3_benchmarking_on_FPGA/scripts/flash_config.py lif-64-16

Requires `vivado` on PATH. The board must be connected via the FT2232 USB-JTAG
bridge (Digilent driver / hw_server reachable at localhost:3121).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from configs import CONFIGS, RESULTS_ROOT, Config


# FT2232H on the Nexys A7-100T (Digilent JTAG interface)
NEXYS_FT2232_USB_ID = "0403:6010"

# Markers we expect to see in successful Vivado output. program_hw_devices
# doesn't reliably set a non-zero exit on JTAG failure, so we grep stdout.
SUCCESS_MARKERS = ("End of startup status: HIGH", "Programming finished")
FAILURE_MARKERS = (
    "ERROR: [Labtoolstcl",
    "ERROR: [Labtools",
    "Problem in startup",
    "Cannot find hardware target",
)

PROGRAM_TCL = """\
open_hw_manager
connect_hw_server -url localhost:3121 -allow_non_jtag
open_hw_target
current_hw_device [lindex [get_hw_devices xc7a100t_0] 0]
set_property PROGRAM.FILE {{{bit}}} [current_hw_device]
program_hw_devices [current_hw_device]
close_hw_manager
"""


def find_bitstream(cfg: Config) -> Path | None:
    path = RESULTS_ROOT / cfg.name / "bitstream.bit"
    return path if path.exists() else None


def board_attached() -> bool | None:
    """Return True if the Nexys A7's FT2232 shows up in lsusb, False if not,
    or None if we can't tell (lsusb missing)."""
    if shutil.which("lsusb") is None:
        return None
    result = subprocess.run(["lsusb"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    return NEXYS_FT2232_USB_ID in result.stdout


def flash(cfg: Config, force: bool = False) -> int:
    if shutil.which("vivado") is None:
        print("ERROR: vivado not on PATH", file=sys.stderr)
        return 2

    bit = find_bitstream(cfg)
    if bit is None:
        print(
            f"ERROR: bitstream not found at "
            f"{RESULTS_ROOT / cfg.name / 'bitstream.bit'}. "
            f"Run build_config.py {cfg.name} first.",
            file=sys.stderr,
        )
        return 2

    attached = board_attached()
    if attached is False and not force:
        print(
            f"ERROR: Nexys A7 FT2232 ({NEXYS_FT2232_USB_ID}) not found in lsusb. "
            "Check the USB cable / power. Pass --force to flash anyway.",
            file=sys.stderr,
        )
        return 2
    if attached is None:
        print("WARN: could not run lsusb to verify board attachment; proceeding.")

    tcl = PROGRAM_TCL.format(bit=bit)
    print(f"Flashing {bit.relative_to(RESULTS_ROOT.parent.parent)}")

    # Vivado's `-source /dev/stdin` does not interoperate cleanly with
    # subprocess piped input — the source command races with the writer and
    # Vivado often exits before the TCL is consumed. Use a real tempfile.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".tcl", delete=False, encoding="utf-8"
    ) as f:
        f.write(tcl)
        tcl_path = f.name

    try:
        proc = subprocess.run(
            ["vivado", "-mode", "batch", "-nojournal", "-nolog", "-source", tcl_path],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        os.unlink(tcl_path)

    # Always show Vivado output (it's useful for debugging JTAG problems)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    failure_hit = next(
        (m for m in FAILURE_MARKERS if m in proc.stdout or m in proc.stderr), None
    )
    success_hit = any(m in proc.stdout for m in SUCCESS_MARKERS)

    if proc.returncode != 0 or failure_hit or not success_hit:
        reason = []
        if proc.returncode != 0:
            reason.append(f"vivado exit code {proc.returncode}")
        if failure_hit:
            reason.append(f"saw {failure_hit!r}")
        if not success_hit:
            reason.append(
                f"did not see any of {SUCCESS_MARKERS} in Vivado output"
            )
        print(f"FAIL: {cfg.name} not flashed ({'; '.join(reason)})", file=sys.stderr)
        return proc.returncode or 1

    print(f"OK: {cfg.name} flashed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Config name to flash")
    parser.add_argument("--force", action="store_true",
                        help="Flash even if lsusb does not see the FT2232")
    args = parser.parse_args()

    try:
        cfg = next(c for c in CONFIGS if c.name == args.config)
    except StopIteration:
        print(f"Unknown config {args.config!r}", file=sys.stderr)
        return 2

    return flash(cfg, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
