"""UART transport driver for the SNN accelerator (accel_uart_pkg.sv protocol).

Callers are responsible for QS2.13 encoding — no float math here.

Usage:
    with FpgaInterface("/dev/ttyUSB1") as fpga:
        fpga.ping()
        fpga.write_obs([obs0, obs1, obs2, obs3])  # int16 QS2.13
        fpga.start_inference()
        fpga.wait_done()
        action = fpga.read_action()
"""
from __future__ import annotations

import struct
import time
from typing import List

import serial

# Protocol constants (accel_uart_pkg.sv)
SOF_HOST = 0xA5
SOF_FPGA = 0x5A

OPCODE_WRITE = 0x01
OPCODE_READ  = 0x02
OPCODE_EXEC  = 0x03
OPCODE_PING  = 0x7F

ST_OK       = 0x00
ST_BAD_CSUM = 0x01
ST_BAD_CMD  = 0x02
ST_BAD_ADDR = 0x03
ST_BUSY     = 0x04
ST_BAD_LEN  = 0x05

_ST_NAMES = {
    ST_OK:       "OK",
    ST_BAD_CSUM: "BAD_CSUM",
    ST_BAD_CMD:  "BAD_CMD",
    ST_BAD_ADDR: "BAD_ADDR",
    ST_BUSY:     "BUSY",
    ST_BAD_LEN:  "BAD_LEN",
}

REG_CONTROL = 0x00
REG_STATUS  = 0x04  # bit0=done, bit1=busy
REG_ACTION  = 0x08  # bits[1:0]=selected_action
REG_OBS0    = 0x10  # 8 contiguous bytes covering all 4 int16 LE observations


class ProtocolError(Exception):
    """Non-OK status byte or malformed response frame from the FPGA."""


# --- Frame builders (stateless; mirrors test_uart_top_wrapper.py) ----------


def _xor(data: bytes) -> int:
    acc = 0
    for b in data:
        acc ^= b
    return acc & 0xFF


def build_frame(opcode: int, addr: int, payload: bytes = b"") -> bytes:
    """Build a host→FPGA frame for WRITE, EXEC, or PING."""
    header = bytes([SOF_HOST, opcode & 0xFF, addr & 0xFF, len(payload) & 0xFF])
    body = header + payload
    return body + bytes([_xor(body)])


def build_read_frame(addr: int, read_len: int) -> bytes:
    """Build a READ frame. READ carries no payload bytes on the wire;
    LEN encodes the number of bytes to read back (see host_if/README.md)."""
    header = bytes([SOF_HOST, OPCODE_READ, addr & 0xFF, read_len & 0xFF])
    return header + bytes([_xor(header)])


def parse_response(frame: bytes) -> tuple[int, bytes]:
    """Parse an FPGA→host frame. Returns (status, payload). Raises on bad frame."""
    if len(frame) < 4 or frame[0] != SOF_FPGA:
        raise ProtocolError(f"bad response frame: {frame.hex()}")
    status = frame[1]
    length = frame[2]
    if len(frame) != 4 + length:
        raise ProtocolError(f"frame length mismatch: got {len(frame)}, expected {4 + length}")
    payload = frame[3 : 3 + length]
    if frame[-1] != _xor(frame[:-1]):
        raise ProtocolError(f"response CSUM mismatch in frame: {frame.hex()}")
    return status, bytes(payload)


# --- Driver ----------------------------------------------------------------


class FpgaInterface:
    """Thin UART transport to the SNN accelerator. No float math."""

    def __init__(self, port: str, baud: int = 115_200, timeout: float = 1.0):
        self._ser = serial.Serial(port, baud, timeout=timeout)

    def close(self) -> None:
        self._ser.close()

    def __enter__(self) -> FpgaInterface:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _transact(self, frame: bytes, response_payload_len: int) -> tuple[int, bytes]:
        """Send frame, read response. Returns (status, payload)."""
        self._ser.write(frame)
        self._ser.flush()
        # Header: SOF + STATUS + LEN = 3 bytes
        header = self._ser.read(3)
        if len(header) < 3:
            raise ProtocolError(f"short header ({len(header)} bytes); check port/baud")
        if header[0] != SOF_FPGA:
            raise ProtocolError(f"bad SOF: 0x{header[0]:02X}")
        payload_len = header[2]
        rest = self._ser.read(payload_len + 1)  # payload + CSUM
        if len(rest) < payload_len + 1:
            raise ProtocolError(f"short body ({len(rest)} of {payload_len + 1} bytes)")
        return parse_response(header + rest)

    def _check(self, status: int, context: str) -> None:
        if status != ST_OK:
            raise ProtocolError(f"{context}: {_ST_NAMES.get(status, f'0x{status:02X}')}")

    def ping(self) -> None:
        """Send PING, verify 'P' payload. Raises ProtocolError on failure."""
        status, payload = self._transact(build_frame(OPCODE_PING, 0x00), 1)
        self._check(status, "PING")
        if payload != b"P":
            raise ProtocolError(f"PING: unexpected payload {payload!r}")

    def write_obs(self, obs: List[int]) -> None:
        """Write 4 int16 QS2.13 observations to REG_OBS0 in one 8-byte frame."""
        if len(obs) != 4:
            raise ValueError(f"expected 4 observations, got {len(obs)}")
        payload = struct.pack("<4h", *obs)
        status, _ = self._transact(build_frame(OPCODE_WRITE, REG_OBS0, payload), 0)
        self._check(status, "WRITE OBS")

    def start_inference(self) -> None:
        """Send EXEC to trigger inference. Returns after the FPGA ACKs the command
        (not after inference completes — call wait_done() for that)."""
        status, _ = self._transact(build_frame(OPCODE_EXEC, 0x00), 0)
        self._check(status, "EXEC")

    def wait_done(self, timeout_s: float = 1.0, poll_interval_s: float = 0.0) -> None:
        """Poll REG_STATUS until done bit (bit 0) is set. Raises TimeoutError."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status, payload = self._transact(build_read_frame(REG_STATUS, 1), 1)
            self._check(status, "READ STATUS")
            if payload[0] & 0x01:
                return
            if poll_interval_s > 0:
                time.sleep(poll_interval_s)
        raise TimeoutError(f"inference did not complete within {timeout_s}s")

    def read_action(self) -> int:
        """Read and return the selected action (0 or 1)."""
        status, payload = self._transact(build_read_frame(REG_ACTION, 1), 1)
        self._check(status, "READ ACTION")
        return payload[0] & 0x01
