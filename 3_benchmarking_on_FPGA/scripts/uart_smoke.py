"""UART smoke test — verifies both the classic 4-transaction path and the
new EXEC_ACTION 2-transaction path on a live board.

Run after programming any new bitstream to confirm the UART stack is alive
and that OPCODE_EXEC_ACTION (0x04) is recognised by the firmware.
"""
import argparse

from fpga_interface import (
    FpgaInterface, ProtocolError,
    build_frame, build_read_frame, parse_response,
    OPCODE_EXEC_ACTION,
    REG_OBS0, REG_STATUS, REG_ACTION,
    ST_OK,
    SOF_FPGA,
)


def hexdump(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


# Deterministic zero observations (all four int16 LE = 0x0000)
_ZERO_OBS = [0, 0, 0, 0]


def smoke_classic(fpga: FpgaInterface, poll_limit: int = 200) -> int:
    """4-transaction path: WRITE OBS → EXEC → STATUS poll → READ ACTION."""
    fpga.write_obs(_ZERO_OBS)
    print("  WRITE OBS  : OK")

    fpga.start_inference()
    print("  EXEC       : OK")

    for i in range(poll_limit):
        status, payload = fpga._transact(build_read_frame(REG_STATUS, 1))
        if status != ST_OK:
            print(f"  STATUS poll: ST=0x{status:02X} (unexpected)")
            return -1
        if payload[0] & 0x01:
            print(f"  STATUS poll: done after {i + 1} poll(s)")
            break
    else:
        print(f"  STATUS poll: done bit not set after {poll_limit} polls")
        return -1

    action = fpga.read_action()
    print(f"  READ ACTION: {action}")
    return action


def smoke_exec_action(fpga: FpgaInterface) -> int:
    """2-transaction path: WRITE OBS → EXEC_ACTION (deferred response)."""
    fpga.write_obs(_ZERO_OBS)
    print("  WRITE OBS  : OK")

    # Send EXEC_ACTION and capture raw response bytes for diagnostics.
    frame = build_frame(OPCODE_EXEC_ACTION, 0x00)
    print(f"  EXEC_ACTION TX: {hexdump(frame)}")
    fpga._ser.write(frame)
    fpga._ser.flush()

    # Read response header (SOF + STATUS + LEN).
    header = fpga._ser.read(3)
    print(f"  EXEC_ACTION RX header: {hexdump(header)}")
    if len(header) < 3:
        print("  ERROR: short header — check bitstream has OPCODE_EXEC_ACTION")
        return -1
    if header[0] != SOF_FPGA:
        print(f"  ERROR: bad SOF 0x{header[0]:02X}")
        return -1

    status_byte = header[1]
    length = header[2]
    rest = fpga._ser.read(length + 1)
    full_frame = header + rest
    print(f"  EXEC_ACTION RX full : {hexdump(full_frame)}")

    if status_byte != ST_OK:
        from fpga_interface import _ST_NAMES
        name = _ST_NAMES.get(status_byte, f"0x{status_byte:02X}")
        print(f"  ERROR: status={name} — bitstream may not have OPCODE_EXEC_ACTION")
        return -1

    if length != 1:
        print(f"  ERROR: expected 1 payload byte, got {length}")
        return -1

    action = rest[0] & 0x01
    print(f"  EXEC_ACTION action : {action}")
    return action


def main() -> int:
    parser = argparse.ArgumentParser(description="UART smoke test")
    parser.add_argument("--port", default="/dev/ttyUSB1")
    parser.add_argument("--baud", type=int, default=921_600)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()

    print(f"PORT: {args.port} @ {args.baud}\n")

    with FpgaInterface(args.port, args.baud, timeout=args.timeout) as fpga:
        # 1. PING
        try:
            fpga.ping()
            print("PING: OK\n")
        except ProtocolError as e:
            print(f"PING FAILED: {e}")
            return 1

        # 2. Classic 4-transaction path
        print("--- Classic path (EXEC + STATUS poll + READ ACTION) ---")
        action_classic = smoke_classic(fpga)
        print()

        # 3. New 2-transaction path
        print("--- New path (EXEC_ACTION) ---")
        action_exec_action = smoke_exec_action(fpga)
        print()

        # 4. Compare
        if action_classic >= 0 and action_exec_action >= 0:
            match = action_classic == action_exec_action
            print(f"Actions agree: {match}  (classic={action_classic}, exec_action={action_exec_action})")
            return 0 if match else 1
        else:
            print("One or both paths failed — check output above")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
