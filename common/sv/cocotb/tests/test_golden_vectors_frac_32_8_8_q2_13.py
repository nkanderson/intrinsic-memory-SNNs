"""Golden-vector test for frac-32-8-8-q2_13 (v3 Optuna model, QS2.13 weights)."""
import cocotb

from golden_vectors_common import run_golden_vectors


@cocotb.test()
async def test_golden_vectors_frac_32_8_8_q2_13(dut):
    await run_golden_vectors(dut, "frac-32-8-8-q2_13")
