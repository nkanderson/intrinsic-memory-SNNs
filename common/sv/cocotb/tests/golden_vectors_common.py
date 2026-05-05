"""Shared helpers for golden vector cocotb tests."""
from __future__ import annotations

import json
import os
from pathlib import Path

from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

NUM_INPUTS = 4
DATA_WIDTH = 16
DEFAULT_MAX_VECTORS = 500
DEFAULT_TIMEOUT_CYCLES = 15000


def to_unsigned(value: int, bits: int) -> int:
    if value < 0:
        return value + (1 << bits)
    return value


def _read_int_param(dut, name: str, default: int) -> int:
    try:
        return int(getattr(dut, name).value)
    except Exception:
        return default


def estimate_timeout(dut) -> int:
    num_timesteps = _read_int_param(dut, "NUM_TIMESTEPS", 10)
    hl1_size = _read_int_param(dut, "HL1_SIZE", 64)
    hl2_size = _read_int_param(dut, "HL2_SIZE", 16)
    history_length = _read_int_param(dut, "HISTORY_LENGTH", 1)
    scaled = num_timesteps * max(1, hl1_size + hl2_size) * max(4, history_length) * 4
    return max(DEFAULT_TIMEOUT_CYCLES, scaled)


def resolve_vectors_path(model_name: str) -> Path:
    env_path = os.getenv("GOLDEN_VECTORS")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parent / "golden_vectors" / f"{model_name}.json"


async def reset_dut(dut):
    dut.reset.value = 1
    dut.start.value = 0
    for i in range(NUM_INPUTS):
        dut.observations[i].value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


async def run_inference(dut, obs_qs213, timeout_cycles: int):
    for i in range(NUM_INPUTS):
        dut.observations[i].value = to_unsigned(int(obs_qs213[i]), DATA_WIDTH)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            return int(dut.selected_action.value)

    raise TimeoutError(f"Inference did not complete within {timeout_cycles} cycles")


async def run_golden_vectors(dut, model_name: str):
    clock = Clock(dut.clk, 10, unit="ns")
    dut._log.info(f"Golden vectors model: {model_name}")
    dut._log.info(f"Golden vectors path: {resolve_vectors_path(model_name)}")
    from cocotb import start_soon

    start_soon(clock.start())

    vectors_path = resolve_vectors_path(model_name)
    if not vectors_path.exists():
        raise FileNotFoundError(
            f"Golden vectors not found at {vectors_path}. "
            "Set GOLDEN_VECTORS or run golden_vectors.py"
        )

    data = json.loads(vectors_path.read_text())
    vectors = data.get("vectors", [])
    if not vectors:
        raise ValueError("Golden vectors file has no entries")

    max_vectors = int(os.getenv("GOLDEN_MAX", str(DEFAULT_MAX_VECTORS)))
    timeout_cycles = estimate_timeout(dut)

    await reset_dut(dut)

    for idx, entry in enumerate(vectors[:max_vectors]):
        obs_qs213 = entry["obs_qs213"]
        expected_action = int(entry["expected_action"])
        action = await run_inference(dut, obs_qs213, timeout_cycles)
        assert action == expected_action, (
            f"Vector {idx}: action {action} != expected {expected_action}"
        )

    dut._log.info(f"Golden vectors passed: {min(len(vectors), max_vectors)} cases")
