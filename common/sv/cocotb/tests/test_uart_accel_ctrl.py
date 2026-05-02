"""
Cocotb test for uart_accel_ctrl. Drives bytes directly into rx_data/rx_valid
(skipping the UART serdes — that path is covered by test_uart_loopback) and
watches tx_data/tx_start to capture response frames.

Coverage:
- PING, WRITE OBS (single 8-byte frame), READ OBS, READ STATUS, READ ACTION,
  READ VERSION, EXEC, EXEC-while-busy
- Error responses: bad CSUM, bad CMD, bad ADDR, bad LEN
- start_pulse generation on EXEC

Frame protocol (see common/sv/host_if/README.md):
  Host  -> FPGA: SOF=0xA5, OPCODE, ADDR, LEN, PAYLOAD..., CSUM
  FPGA  -> Host: SOF=0x5A, STATUS, LEN, PAYLOAD..., CSUM
  CSUM is XOR of every byte in the frame except CSUM itself.
"""

from dataclasses import dataclass

import cocotb
from cocotb.clock import Clock
from cocotb.queue import Queue
from cocotb.triggers import ClockCycles, RisingEdge

# Protocol constants (must match accel_uart_pkg.sv)
SOF_HOST_TO_FPGA = 0xA5
SOF_FPGA_TO_HOST = 0x5A

OPCODE_WRITE = 0x01
OPCODE_READ = 0x02
OPCODE_EXEC = 0x03
OPCODE_PING = 0x7F

ST_OK = 0x00
ST_BAD_CSUM = 0x01
ST_BAD_CMD = 0x02
ST_BAD_ADDR = 0x03
ST_BUSY = 0x04
ST_BAD_LEN = 0x05

REG_CONTROL = 0x00
REG_STATUS = 0x04
REG_ACTION = 0x08
REG_OBS0 = 0x10
REG_OBS1 = 0x12
REG_OBS2 = 0x14
REG_OBS3 = 0x16
REG_VERSION = 0xFC

MAX_PAYLOAD = 64


def to_signed16(raw: int) -> int:
    raw &= 0xFFFF
    return raw - 0x10000 if raw & 0x8000 else raw


def xor_bytes(data) -> int:
    acc = 0
    for b in data:
        acc ^= b & 0xFF
    return acc & 0xFF


def build_host_frame(opcode: int, addr: int, payload: bytes = b"") -> bytes:
    """Build a host->FPGA frame with valid CSUM."""
    header = bytes([SOF_HOST_TO_FPGA, opcode & 0xFF, addr & 0xFF, len(payload) & 0xFF])
    body = header + payload
    return body + bytes([xor_bytes(body)])


def build_read_frame(addr: int, read_len: int) -> bytes:
    """READ frame helper.

    READ frames carry no payload on the wire — LEN is the number of bytes to
    return from the FPGA. On-wire layout: SOF | OPCODE | ADDR | LEN | CSUM.
    """
    header = bytes([SOF_HOST_TO_FPGA, OPCODE_READ, addr & 0xFF, read_len & 0xFF])
    return header + bytes([xor_bytes(header)])


@dataclass
class Response:
    status: int
    payload: bytes
    csum_ok: bool


def parse_response_frame(frame: bytes) -> Response:
    assert len(frame) >= 4, f"response frame too short ({len(frame)} bytes)"
    assert frame[0] == SOF_FPGA_TO_HOST, (
        f"bad response SOF: 0x{frame[0]:02X} != 0x{SOF_FPGA_TO_HOST:02X}"
    )
    status = frame[1]
    length = frame[2]
    expected_total = 4 + length
    assert len(frame) == expected_total, (
        f"response length mismatch: got {len(frame)} bytes, header says {expected_total}"
    )
    payload = frame[3 : 3 + length]
    csum_actual = frame[-1]
    csum_expected = xor_bytes(frame[:-1])
    return Response(status=status, payload=bytes(payload), csum_ok=(csum_actual == csum_expected))


# --- DUT plumbing -----------------------------------------------------------


async def reset_dut(dut):
    dut.reset.value = 1
    dut.rx_data.value = 0
    dut.rx_valid.value = 0
    dut.tx_busy.value = 0
    dut.accel_done.value = 0
    dut.accel_busy.value = 0
    dut.accel_action.value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


async def send_bytes(dut, payload: bytes):
    """Push each byte into the controller's RX port for one clock each."""
    for b in payload:
        dut.rx_data.value = b & 0xFF
        dut.rx_valid.value = 1
        await RisingEdge(dut.clk)
    dut.rx_valid.value = 0
    dut.rx_data.value = 0


async def tx_monitor(dut, captured: Queue):
    """Capture every byte the controller emits (one byte per tx_start pulse).

    tx_busy is held low by the test so the controller pumps a byte per cycle
    once it has a queued response.
    """
    while True:
        await RisingEdge(dut.clk)
        if int(dut.tx_start.value) == 1:
            captured.put_nowait(int(dut.tx_data.value) & 0xFF)


async def collect_response(dut, captured: Queue, timeout_cycles: int = 200) -> bytes:
    """Drain a complete FPGA->Host frame from the monitor queue.

    Reads SOF, STATUS, LEN, then LEN payload bytes, then CSUM.
    """
    bytes_out: list[int] = []

    async def get_byte():
        for _ in range(timeout_cycles):
            if not captured.empty():
                return await captured.get()
            await RisingEdge(dut.clk)
        raise AssertionError(
            f"timed out after {timeout_cycles} cycles waiting for response byte "
            f"(captured so far: {bytes_out!r})"
        )

    bytes_out.append(await get_byte())  # SOF
    bytes_out.append(await get_byte())  # STATUS
    bytes_out.append(await get_byte())  # LEN
    length = bytes_out[2]
    for _ in range(length):
        bytes_out.append(await get_byte())
    bytes_out.append(await get_byte())  # CSUM
    return bytes(bytes_out)


async def setup_test(dut):
    """Common harness: clock, reset, tx monitor."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    captured: Queue = Queue()
    cocotb.start_soon(tx_monitor(dut, captured))
    return captured


# --- Tests ------------------------------------------------------------------


@cocotb.test()
async def test_ping(dut):
    """PING returns ST_OK with payload 'P'."""
    captured = await setup_test(dut)

    await send_bytes(dut, build_host_frame(OPCODE_PING, addr=0x00))
    frame = await collect_response(dut, captured)
    resp = parse_response_frame(frame)

    assert resp.csum_ok, "response CSUM mismatch"
    assert resp.status == ST_OK, f"PING status: 0x{resp.status:02X}"
    assert resp.payload == b"P", f"PING payload: {resp.payload!r}"


@cocotb.test()
async def test_write_obs_block(dut):
    """A single 8-byte WRITE at REG_OBS0 lands all four observations.

    Verifies the contiguous int16 register layout (0x10/0x12/0x14/0x16).
    """
    captured = await setup_test(dut)

    obs_values = [0x1234, -0x0010, 0x4000, -0x4000]  # int16 range
    payload = b""
    for v in obs_values:
        payload += int(v & 0xFFFF).to_bytes(2, "little", signed=False)

    await send_bytes(dut, build_host_frame(OPCODE_WRITE, REG_OBS0, payload))
    resp = parse_response_frame(await collect_response(dut, captured))

    assert resp.csum_ok and resp.status == ST_OK, (
        f"WRITE OBS status: 0x{resp.status:02X}"
    )

    # Allow one cycle for the register writes to settle past the response.
    await ClockCycles(dut.clk, 2)

    for i, expected in enumerate(obs_values):
        actual = to_signed16(int(dut.observations[i].value))
        assert actual == expected, (
            f"observations[{i}] = {actual}, expected {expected}"
        )


@cocotb.test()
async def test_read_obs_roundtrip(dut):
    """WRITE then READ each OBS register independently."""
    captured = await setup_test(dut)

    written = [0x0001, 0x7FFF, -0x8000, -0x0001]
    addrs = [REG_OBS0, REG_OBS1, REG_OBS2, REG_OBS3]
    for addr, value in zip(addrs, written):
        payload = int(value & 0xFFFF).to_bytes(2, "little", signed=False)
        await send_bytes(dut, build_host_frame(OPCODE_WRITE, addr, payload))
        resp = parse_response_frame(await collect_response(dut, captured))
        assert resp.status == ST_OK, f"WRITE 0x{addr:02X} status: 0x{resp.status:02X}"

    for addr, value in zip(addrs, written):
        await send_bytes(dut, build_read_frame(addr, 2))
        resp = parse_response_frame(await collect_response(dut, captured))
        assert resp.status == ST_OK and len(resp.payload) == 2, (
            f"READ 0x{addr:02X} status: 0x{resp.status:02X} len: {len(resp.payload)}"
        )
        got = int.from_bytes(resp.payload, "little", signed=True)
        assert got == value, f"OBS @0x{addr:02X}: read {got}, wrote {value}"


@cocotb.test()
async def test_read_version(dut):
    """REG_VERSION returns 0x0001 (interface version)."""
    captured = await setup_test(dut)

    await send_bytes(dut, build_read_frame(REG_VERSION, 2))
    resp = parse_response_frame(await collect_response(dut, captured))

    assert resp.csum_ok and resp.status == ST_OK
    assert int.from_bytes(resp.payload, "little") == 0x0001, (
        f"VERSION = 0x{int.from_bytes(resp.payload, 'little'):04X}"
    )


@cocotb.test()
async def test_read_status_idle(dut):
    """While accel is idle, STATUS reads back done=0, busy=0."""
    captured = await setup_test(dut)
    dut.accel_done.value = 0
    dut.accel_busy.value = 0

    await send_bytes(dut, build_read_frame(REG_STATUS, 1))
    resp = parse_response_frame(await collect_response(dut, captured))
    assert resp.status == ST_OK
    assert resp.payload[0] == 0x00, f"STATUS idle: 0x{resp.payload[0]:02X}"


@cocotb.test()
async def test_read_status_done(dut):
    """When accel signals done, STATUS bit0 reads 1."""
    captured = await setup_test(dut)
    dut.accel_done.value = 1
    dut.accel_busy.value = 0

    await send_bytes(dut, build_read_frame(REG_STATUS, 1))
    resp = parse_response_frame(await collect_response(dut, captured))
    assert resp.status == ST_OK
    assert resp.payload[0] & 0x01 == 0x01, f"STATUS.done: 0x{resp.payload[0]:02X}"


@cocotb.test()
async def test_read_action(dut):
    """REG_ACTION reflects the accel_action input bits."""
    captured = await setup_test(dut)
    dut.accel_action.value = 0b01

    await send_bytes(dut, build_read_frame(REG_ACTION, 1))
    resp = parse_response_frame(await collect_response(dut, captured))
    assert resp.status == ST_OK
    assert resp.payload[0] & 0x03 == 0b01, f"ACTION = 0x{resp.payload[0]:02X}"


@cocotb.test()
async def test_exec_pulses_start(dut):
    """EXEC fires start_pulse for exactly one cycle and returns ST_OK."""
    captured = await setup_test(dut)
    dut.accel_busy.value = 0

    saw_pulse = []

    async def watch_start_pulse():
        # Capture the first rising edge on which start_pulse is observed high.
        for _ in range(200):
            await RisingEdge(dut.clk)
            if int(dut.start_pulse.value) == 1:
                saw_pulse.append(True)
                # Confirm it goes back low next cycle.
                await RisingEdge(dut.clk)
                if int(dut.start_pulse.value) == 0:
                    saw_pulse.append("one_cycle")
                return

    cocotb.start_soon(watch_start_pulse())
    await send_bytes(dut, build_host_frame(OPCODE_EXEC, 0x00))
    resp = parse_response_frame(await collect_response(dut, captured))

    assert resp.status == ST_OK, f"EXEC status: 0x{resp.status:02X}"
    await ClockCycles(dut.clk, 5)  # let watcher conclude
    assert saw_pulse and saw_pulse[0] is True, "start_pulse never went high"
    assert "one_cycle" in saw_pulse, "start_pulse stayed high more than one cycle"


@cocotb.test()
async def test_exec_while_busy_returns_st_busy(dut):
    """EXEC while accel_busy=1 must respond ST_BUSY and NOT pulse start."""
    captured = await setup_test(dut)
    dut.accel_busy.value = 1

    start_pulses: list[int] = []

    async def watch():
        for _ in range(200):
            await RisingEdge(dut.clk)
            if int(dut.start_pulse.value) == 1:
                start_pulses.append(1)

    cocotb.start_soon(watch())
    await send_bytes(dut, build_host_frame(OPCODE_EXEC, 0x00))
    resp = parse_response_frame(await collect_response(dut, captured))

    assert resp.status == ST_BUSY, f"expected ST_BUSY, got 0x{resp.status:02X}"
    await ClockCycles(dut.clk, 5)
    assert not start_pulses, "start_pulse must not fire when accel_busy=1"


@cocotb.test()
async def test_bad_csum(dut):
    """Frame with wrong CSUM is rejected with ST_BAD_CSUM."""
    captured = await setup_test(dut)

    good = build_host_frame(OPCODE_PING, 0x00)
    bad = good[:-1] + bytes([(good[-1] ^ 0xFF) & 0xFF])  # corrupt last byte
    await send_bytes(dut, bad)

    resp = parse_response_frame(await collect_response(dut, captured))
    assert resp.status == ST_BAD_CSUM, f"got status 0x{resp.status:02X}"


@cocotb.test()
async def test_bad_cmd(dut):
    """Unknown opcode -> ST_BAD_CMD."""
    captured = await setup_test(dut)

    await send_bytes(dut, build_host_frame(0x55, 0x00))  # not a valid opcode
    resp = parse_response_frame(await collect_response(dut, captured))
    assert resp.status == ST_BAD_CMD, f"got status 0x{resp.status:02X}"


@cocotb.test()
async def test_bad_addr_write(dut):
    """WRITE to an unmapped address returns ST_BAD_ADDR."""
    captured = await setup_test(dut)

    # 0x40 is between REG_OBS3 (0x16) and REG_VERSION (0xFC) — not mapped.
    await send_bytes(dut, build_host_frame(OPCODE_WRITE, 0x40, bytes([0x00, 0x00])))
    resp = parse_response_frame(await collect_response(dut, captured))
    assert resp.status == ST_BAD_ADDR, f"got status 0x{resp.status:02X}"


@cocotb.test()
async def test_bad_len(dut):
    """LEN exceeding MAX_PAYLOAD (64) returns ST_BAD_LEN."""
    captured = await setup_test(dut)

    # Build a frame manually with an oversized declared LEN. We must still
    # send LEN bytes of payload + CSUM so the parser progresses.
    over_len = MAX_PAYLOAD + 1
    payload = bytes(over_len)  # zeroed
    header = bytes([SOF_HOST_TO_FPGA, OPCODE_WRITE, REG_OBS0, over_len])
    body = header + payload
    frame = body + bytes([xor_bytes(body)])
    await send_bytes(dut, frame)

    resp = parse_response_frame(await collect_response(dut, captured))
    assert resp.status == ST_BAD_LEN, f"got status 0x{resp.status:02X}"


@cocotb.test()
async def test_bad_len_read(dut):
    """READ with LEN > MAX_PAYLOAD returns ST_BAD_LEN.

    READ frames carry no payload, so the FSM goes straight from RX_GET_LEN
    to RX_GET_CSUM, but the bound check on LEN must still fire to keep the
    response from overflowing tx_frame.
    """
    captured = await setup_test(dut)

    await send_bytes(dut, build_read_frame(REG_OBS0, MAX_PAYLOAD + 1))
    resp = parse_response_frame(await collect_response(dut, captured))
    assert resp.status == ST_BAD_LEN, f"got status 0x{resp.status:02X}"
