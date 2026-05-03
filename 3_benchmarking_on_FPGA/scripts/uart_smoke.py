"""UART smoke test for lif-64-16 board bring-up."""
import argparse
import time
import serial


def hexdump(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


def send_frame(ser, hex_bytes: str, read_len: int) -> bytes:
    ser.write(bytes.fromhex(hex_bytes))
    ser.flush()
    return ser.read(read_len)


def main() -> int:
    parser = argparse.ArgumentParser(description="UART smoke test for lif-64-16")
    parser.add_argument("--port", default="/dev/ttyUSB1")
    parser.add_argument("--baud", type=int, default=921_600)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--poll-ms", type=int, default=50)
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
    try:
        print(f"PORT: {args.port} @ {args.baud}")

        # PING
        resp = send_frame(ser, "A5 7F 00 00 DA", 5)
        print("PING RX:", hexdump(resp))

        # Write observations (all zero)
        resp = send_frame(ser, "A5 01 10 08 00 00 00 00 00 00 00 00 BC", 4)
        print("WR   RX:", hexdump(resp))

        # EXEC
        resp = send_frame(ser, "A5 03 00 00 A6", 4)
        print("EXEC RX:", hexdump(resp))

        # Poll STATUS until done
        while True:
            resp = send_frame(ser, "A5 02 04 01 A2", 5)
            print("STAT RX:", hexdump(resp))
            if len(resp) == 5 and (resp[3] & 0x01):
                break
            time.sleep(args.poll_ms / 1000.0)

        # Read ACTION
        resp = send_frame(ser, "A5 02 08 01 AE", 5)
        print("ACT  RX:", hexdump(resp))
    finally:
        ser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
