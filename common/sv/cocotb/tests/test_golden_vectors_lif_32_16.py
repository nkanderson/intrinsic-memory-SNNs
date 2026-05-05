"""Golden-vector test for lif-32-16."""
import cocotb

from golden_vectors_common import run_golden_vectors


@cocotb.test()
async def test_golden_vectors_lif_32_16(dut):
    await run_golden_vectors(dut, "lif-32-16")
