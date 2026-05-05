"""Golden-vector test for fractional-16-4-32."""
import cocotb

from golden_vectors_common import run_golden_vectors


@cocotb.test()
async def test_golden_vectors_fractional_16_4_32(dut):
    await run_golden_vectors(dut, "fractional-16-4-32")
