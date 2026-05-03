"""
End-to-end cocotb test for top_uart_accel_wrapper. Drives bit-serial UART on
uart_rx_i and captures the wrapper's response on uart_tx_o, exercising the
full host_if stack: uart_rx -> uart_accel_ctrl -> neural_network -> uart_tx.

To keep simulation tractable, BAUD is overridden to 10 Mbaud (10 cycles per
bit at 100 MHz) — same protocol, faster bit clock.

Assumes the Makefile sets MODEL_TYPE=0 (standard LIF) and points the weight
files at common/sv/cocotb/tests/weights/lif-64-16/.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.queue import Queue
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge, Timer
from cocotb.utils import get_sim_time

# Protocol constants (must match accel_uart_pkg.sv)
SOF_HOST_TO_FPGA = 0xA5
SOF_FPGA_TO_HOST = 0x5A

OPCODE_WRITE = 0x01
OPCODE_READ = 0x02
OPCODE_EXEC = 0x03
OPCODE_PING = 0x7F

ST_OK = 0x00

REG_CONTROL = 0x00
REG_STATUS = 0x04
REG_ACTION = 0x08
REG_OBS0 = 0x10

# Bit timing for the simulator-friendly baud override
CLOCK_HZ = 100_000_000
SIM_BAUD = 10_000_000
CLKS_PER_BIT = CLOCK_HZ // SIM_BAUD  # 10


def xor_bytes(data) -> int:
    acc = 0
    for b in data:
        acc ^= b & 0xFF
    return acc & 0xFF


def build_host_frame(opcode: int, addr: int, payload: bytes = b"") -> bytes:
    header = bytes([SOF_HOST_TO_FPGA, opcode & 0xFF, addr & 0xFF, len(payload) & 0xFF])
    body = header + payload
    return body + bytes([xor_bytes(body)])


def build_read_frame(addr: int, read_len: int) -> bytes:
    """READ has no payload on the wire — see common/sv/host_if/README.md."""
    header = bytes([SOF_HOST_TO_FPGA, OPCODE_READ, addr & 0xFF, read_len & 0xFF])
    return header + bytes([xor_bytes(header)])


def parse_response_frame(frame: bytes) -> tuple[int, bytes, bool]:
    assert len(frame) >= 4 and frame[0] == SOF_FPGA_TO_HOST, f"bad SOF: {frame!r}"
    status = frame[1]
    length = frame[2]
    assert len(frame) == 4 + length, f"length mismatch: {frame!r}"
    payload = frame[3 : 3 + length]
    csum_ok = frame[-1] == xor_bytes(frame[:-1])
    return status, bytes(payload), csum_ok


# --- UART bit-serial drivers -----------------------------------------------


async def uart_send_byte(dut, byte: int):
    """Drive one 8N1 byte on dut.uart_rx_i."""
    # Start bit
    dut.uart_rx_i.value = 0
    await ClockCycles(dut.clk, CLKS_PER_BIT)
    # Data bits, LSB first
    for i in range(8):
        dut.uart_rx_i.value = (byte >> i) & 1
        await ClockCycles(dut.clk, CLKS_PER_BIT)
    # Stop bit
    dut.uart_rx_i.value = 1
    await ClockCycles(dut.clk, CLKS_PER_BIT)


async def uart_send_frame(dut, frame: bytes):
    for b in frame:
        await uart_send_byte(dut, b)


async def uart_rx_monitor(dut, captured: Queue):
    """Background coroutine: capture every byte transmitted on dut.uart_tx_o.

    Waits on FallingEdge(uart_tx_o) for each start bit (cheap; cocotb only
    wakes on the actual transition rather than polling every clock).
    """
    while True:
        # Wait for start bit (line goes 1 -> 0).
        await FallingEdge(dut.uart_tx_o)
        # Move from the falling edge to mid-bit of the start bit, then
        # advance a full bit time to land in the middle of LSB of data.
        await ClockCycles(dut.clk, CLKS_PER_BIT // 2)
        await ClockCycles(dut.clk, CLKS_PER_BIT)
        byte = 0
        for i in range(8):
            byte |= (int(dut.uart_tx_o.value) & 1) << i
            await ClockCycles(dut.clk, CLKS_PER_BIT)
        # We're now mid-stop-bit. Skip the rest of the stop bit before
        # going back to the FallingEdge wait so a back-to-back transmission
        # is picked up cleanly.
        await ClockCycles(dut.clk, CLKS_PER_BIT // 2 + 1)
        captured.put_nowait(byte)


async def collect_response(captured: Queue, max_bytes: int = 80) -> bytes:
    """Pop SOF, STATUS, LEN, payload bytes, CSUM from the queue."""
    out: list[int] = []
    out.append(await captured.get())  # SOF
    out.append(await captured.get())  # STATUS
    out.append(await captured.get())  # LEN
    length = out[2]
    assert length <= max_bytes, f"response payload too long: LEN={length}"
    for _ in range(length):
        out.append(await captured.get())
    out.append(await captured.get())  # CSUM
    return bytes(out)


# --- Test harness ----------------------------------------------------------


async def reset_dut(dut):
    dut.reset.value = 1
    dut.uart_rx_i.value = 1  # idle high
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 5)


async def watchdog(dut, timeout_ns: int = 50_000):
    """Hang detector. Silent during a normal-paced test; if the simulation
    runs longer than `timeout_ns` without the test completing, dumps the
    controller's internal state and fails the test so we don't hang.
    """
    await Timer(timeout_ns, unit="ns")
    try:
        rx_state = int(dut.u_ctrl.rx_state.value)
        tx_active = int(dut.u_ctrl.tx_active.value)
        resp_pending = int(dut.u_ctrl.resp_pending.value)
        tx_o = int(dut.uart_tx_o.value)
        tx_busy = int(dut.tx_busy.value)
        dut._log.error(
            f"watchdog timeout @{get_sim_time('ns'):.0f}ns: "
            f"rx_state={rx_state} tx_active={tx_active} "
            f"resp_pending={resp_pending} tx_o={tx_o} tx_busy={tx_busy}"
        )
    except Exception as e:  # noqa: BLE001
        dut._log.error(f"watchdog timeout; signal probe failed: {e}")
    raise AssertionError(
        f"simulation exceeded {timeout_ns}ns without test completion"
    )


async def setup_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    captured: Queue = Queue()
    cocotb.start_soon(uart_rx_monitor(dut, captured))
    cocotb.start_soon(watchdog(dut))
    return captured


# --- Tests ------------------------------------------------------------------


@cocotb.test()
async def test_wrapper_ping(dut):
    """PING through the full UART path returns ST_OK + 'P'."""
    captured = await setup_test(dut)

    await uart_send_frame(dut, build_host_frame(OPCODE_PING, 0x00))
    frame = await collect_response(captured)
    status, payload, csum_ok = parse_response_frame(frame)

    assert csum_ok, "response CSUM bad"
    assert status == ST_OK, f"PING status: 0x{status:02X}"
    assert payload == b"P", f"PING payload: {payload!r}"


@cocotb.test()
async def test_wrapper_full_inference(dut):
    """Full inference round-trip:

    1. WRITE 8 bytes at REG_OBS0 (all four observations in one frame).
    2. EXEC.
    3. Poll STATUS until done bit is set.
    4. READ ACTION; assert it's a valid action index (0 or 1).
    """
    captured = await setup_test(dut)

    # 1. WRITE observations. Using a deterministic CartPole-shaped vector;
    # actual quantization scale doesn't matter here — we're testing the
    # pipeline plumbing, not numerical correctness.
    obs_int16 = [0x0100, -0x0050, 0x00C0, 0x0010]
    obs_payload = b""
    for v in obs_int16:
        obs_payload += int(v & 0xFFFF).to_bytes(2, "little", signed=False)
    await uart_send_frame(dut, build_host_frame(OPCODE_WRITE, REG_OBS0, obs_payload))
    status, _, csum_ok = parse_response_frame(await collect_response(captured))
    assert csum_ok and status == ST_OK, f"WRITE OBS status: 0x{status:02X}"

    # 2. EXEC.
    await uart_send_frame(dut, build_host_frame(OPCODE_EXEC, 0x00))
    status, _, csum_ok = parse_response_frame(await collect_response(captured))
    assert csum_ok and status == ST_OK, f"EXEC status: 0x{status:02X}"

    # 3. Poll STATUS. With NUM_TIMESTEPS=10 and a small network the inference
    # finishes in well under a millisecond, so a handful of polls is plenty.
    done = False
    for poll in range(20):
        await uart_send_frame(dut, build_read_frame(REG_STATUS, 1))
        status, payload, csum_ok = parse_response_frame(await collect_response(captured))
        assert csum_ok and status == ST_OK
        if payload[0] & 0x01:  # done bit
            done = True
            dut._log.info(f"done bit set after {poll + 1} STATUS poll(s)")
            break
    assert done, "STATUS done bit never set after 20 polls"

    # 4. READ ACTION.
    await uart_send_frame(dut, build_read_frame(REG_ACTION, 1))
    status, payload, csum_ok = parse_response_frame(await collect_response(captured))
    assert csum_ok and status == ST_OK
    action = payload[0] & 0x01  # 1-bit action for NUM_ACTIONS=2
    dut._log.info(f"selected action: {action}")
    assert action in (0, 1)
