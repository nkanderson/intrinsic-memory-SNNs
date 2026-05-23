"""Golden-vector test for bitshift-64-32-4 (multi-seed bitshift custom_slow_decay)."""
import cocotb

from golden_vectors_common import run_golden_vectors


@cocotb.test()
async def test_golden_vectors_bitshift_64_32_4(dut):
    await run_golden_vectors(dut, "bitshift-64-32-4")
