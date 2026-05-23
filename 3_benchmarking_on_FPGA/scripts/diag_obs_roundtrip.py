"""Diagnose observation register write/read round-trip on the FPGA.

If validate_fpga.py shows exactly 50% match against a 250/250 balanced golden
vector set, the FPGA is likely returning a constant action — most often
because observations never landed in the obs registers (or always read as 0).

This script:
  1. Writes a distinctive pattern to REG_OBS0..3 via OPCODE_WRITE
  2. Reads each observation back via OPCODE_READ
  3. Reports any mismatch byte-by-byte
  4. Repeats with a second pattern to rule out lucky alignment
  5. Runs one inference with non-trivial obs and reports the action

Usage:
    python diag_obs_roundtrip.py [--port /dev/ttyUSB1]
"""

from __future__ import annotations

import argparse
import struct
import sys

from fpga_interface import (
    FpgaInterface,
    REG_OBS0,
    build_frame,
    build_read_frame,
    OPCODE_WRITE,
    ST_OK,
)


def write_obs_raw(fpga: FpgaInterface, obs: list[int]) -> None:
    payload = struct.pack("<4h", *obs)
    status, _ = fpga._transact(build_frame(OPCODE_WRITE, REG_OBS0, payload))
    if status != ST_OK:
        raise RuntimeError(f"WRITE OBS status=0x{status:02X}")


def read_obs_raw(fpga: FpgaInterface) -> list[int]:
    status, payload = fpga._transact(build_read_frame(REG_OBS0, 8))
    if status != ST_OK:
        raise RuntimeError(f"READ OBS status=0x{status:02X}")
    if len(payload) != 8:
        raise RuntimeError(f"READ OBS returned {len(payload)} bytes (expected 8)")
    return list(struct.unpack("<4h", payload))


def check_roundtrip(fpga: FpgaInterface, pattern: list[int], label: str) -> bool:
    print(f"\n=== {label}: writing obs = {pattern} ===")
    write_obs_raw(fpga, pattern)
    readback = read_obs_raw(fpga)
    print(f"  readback                = {readback}")
    ok = readback == pattern
    if ok:
        print("  MATCH")
    else:
        print("  MISMATCH (per-element):")
        for i, (w, r) in enumerate(zip(pattern, readback)):
            mark = "  ok" if w == r else f"  DIFF (wrote 0x{w & 0xFFFF:04X}, read 0x{r & 0xFFFF:04X})"
            print(f"    obs[{i}]: wrote {w:6d}  read {r:6d}{mark}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB1")
    parser.add_argument("--baud", type=int, default=921_600)
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args()

    with FpgaInterface(args.port, args.baud, timeout=args.timeout) as fpga:
        fpga.ping()
        print(f"PING OK — port={args.port}")

        # Distinctive bytes per slot so any swap shows up immediately.
        pattern1 = [0x0102, 0x0304, 0x0506, 0x0708]   # ascending bytes
        pattern2 = [-1, -2, -3, -4]                   # signed negatives
        pattern3 = [0x7FFF, -0x8000, 0x0001, -1]      # extremes

        ok = True
        ok &= check_roundtrip(fpga, pattern1, "Pattern 1 (ascending)")
        ok &= check_roundtrip(fpga, pattern2, "Pattern 2 (small negatives)")
        ok &= check_roundtrip(fpga, pattern3, "Pattern 3 (extremes)")

        # Quick inference sanity: write the first golden-vector obs and the
        # all-zero obs, compare actions. They should differ if the network
        # is actually using the observation values.
        print("\n=== Inference sanity ===")
        fpga.write_obs([0, 0, 0, 0])
        a0 = fpga.exec_and_read_action()
        print(f"  obs all zero       -> action {a0}")

        # First obs from golden_vectors_frac-32-8-8-q2_13.json
        nonzero = [112, -189, -376, -396]
        fpga.write_obs(nonzero)
        a1 = fpga.exec_and_read_action()
        print(f"  obs {nonzero} -> action {a1}")

        # An extreme obs
        extreme = [16000, -16000, 16000, -16000]
        fpga.write_obs(extreme)
        a2 = fpga.exec_and_read_action()
        print(f"  obs {extreme} -> action {a2}")

        if a0 == a1 == a2:
            print(f"  WARN: all three inferences returned action {a0} — "
                  "FPGA may be stuck (check obs roundtrip above)")
        else:
            print(f"  Actions differ across inputs (good)")

        if not ok:
            print("\nFAIL: observation roundtrip mismatch — bug is on the WRITE path")
            return 1
        print("\nOK: observation roundtrip clean")
        return 0


if __name__ == "__main__":
    sys.exit(main())
