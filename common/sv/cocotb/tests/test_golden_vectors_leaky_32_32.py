"""Golden-vector test for leaky-32-32 (v3 Optuna LIF model, QS2.13 weights)."""
import cocotb

from golden_vectors_common import run_golden_vectors


@cocotb.test()
async def test_golden_vectors_leaky_32_32(dut):
    await run_golden_vectors(dut, "leaky-32-32")
