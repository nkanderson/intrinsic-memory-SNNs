"""
Cocotb test for the UART RX/TX loopback. Wires uart_tx directly into uart_rx
through tb_uart_loopback and confirms each byte sent is received unchanged.

Configured for 100 MHz clock and 115_200 baud, matching the wrapper defaults.
"""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.queue import Queue
from cocotb.triggers import ClockCycles, RisingEdge

CLOCK_HZ = 100_000_000
BAUD = 115_200
CLKS_PER_BIT = CLOCK_HZ // BAUD
# Ten UART bit times (start + 8 data + stop) plus margin for FSM transitions.
BYTE_CYCLES = CLKS_PER_BIT * 11


async def reset_dut(dut):
    dut.reset.value = 1
    dut.tx_start.value = 0
    dut.tx_data_in.value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


async def rx_monitor(dut, captured: Queue):
    """Background coroutine: pushes every byte the RX accepts into `captured`.

    `rx_data_valid` is a single-cycle pulse, so we sample on every rising
    edge and forward immediately.
    """
    while True:
        await RisingEdge(dut.clk)
        if int(dut.rx_data_valid.value) == 1:
            captured.put_nowait(int(dut.rx_data_out.value) & 0xFF)


async def send_byte(dut, value: int):
    """Pulse tx_start with the given byte and wait for tx_done."""
    dut.tx_data_in.value = value & 0xFF
    dut.tx_start.value = 1
    await RisingEdge(dut.clk)
    dut.tx_start.value = 0

    for _ in range(BYTE_CYCLES + 64):
        await RisingEdge(dut.clk)
        if int(dut.tx_done.value) == 1:
            return
    raise AssertionError(f"tx_done never fired for byte 0x{value:02X}")


async def expect_byte(dut, captured: Queue, expected: int):
    """Pop the next byte the monitor captured and assert equality.

    RX samples mid-bit, so rx_data_valid fires about CLKS_PER_BIT/2 cycles
    after tx_done. Wait up to a full bit time before declaring a miss.
    """
    for _ in range(CLKS_PER_BIT * 2):
        if not captured.empty():
            break
        await RisingEdge(dut.clk)
    if captured.empty():
        raise AssertionError(
            f"no RX byte captured within {CLKS_PER_BIT*2} cycles of tx_done "
            f"(expected 0x{expected:02X})"
        )
    received = await captured.get()
    assert received == expected, (
        f"loopback mismatch: sent 0x{expected:02X}, received 0x{received:02X}"
    )


@cocotb.test()
async def test_uart_loopback_single_byte(dut):
    """A single byte sent through tx must come back through rx unchanged."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    captured: Queue = Queue()
    cocotb.start_soon(rx_monitor(dut, captured))

    test_byte = 0xA5
    await send_byte(dut, test_byte)
    await expect_byte(dut, captured, test_byte)
    dut._log.info(f"Loopback single byte 0x{test_byte:02X} OK")


@cocotb.test()
async def test_uart_loopback_all_bit_patterns(dut):
    """Exercise corner-case bit patterns: 0x00, 0xFF, alternating, walking ones."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    captured: Queue = Queue()
    cocotb.start_soon(rx_monitor(dut, captured))

    patterns = [0x00, 0xFF, 0x55, 0xAA, 0x01, 0x80, 0x10, 0x08]
    for value in patterns:
        await send_byte(dut, value)
        await expect_byte(dut, captured, value)
    dut._log.info(f"Loopback {len(patterns)} bit-pattern bytes OK")


@cocotb.test()
async def test_uart_loopback_burst(dut):
    """Send a burst of pseudo-random bytes back-to-back and check ordering."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    captured: Queue = Queue()
    cocotb.start_soon(rx_monitor(dut, captured))

    rng = random.Random(0xDEAD_BEEF)
    burst = [rng.randrange(0, 256) for _ in range(16)]
    for value in burst:
        await send_byte(dut, value)
        await expect_byte(dut, captured, value)
    dut._log.info(f"Loopback burst of {len(burst)} bytes OK")
