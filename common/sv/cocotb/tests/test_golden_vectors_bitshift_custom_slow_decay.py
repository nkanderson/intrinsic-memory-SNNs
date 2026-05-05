"""Golden-vector test for bitshift-custom_slow_decay."""
import cocotb

from golden_vectors_common import run_golden_vectors


@cocotb.test()
async def test_golden_vectors_bitshift_custom_slow_decay(dut):
    await run_golden_vectors(dut, "bitshift-custom_slow_decay")
