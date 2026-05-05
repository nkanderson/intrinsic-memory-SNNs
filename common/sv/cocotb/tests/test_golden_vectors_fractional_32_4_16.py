"""Golden-vector test for fractional-32-4-16."""
import cocotb

from golden_vectors_common import run_golden_vectors


@cocotb.test()
async def test_golden_vectors_fractional_32_4_16(dut):
    await run_golden_vectors(dut, "fractional-32-4-16")
