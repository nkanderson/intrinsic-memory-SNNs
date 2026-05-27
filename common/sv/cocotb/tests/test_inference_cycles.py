"""Per-FSM-stage inference cycle measurement.

Reuses the same DUT parameter blocks as test_cartpole_integration.py via the
existing cp_integration_<config> Makefile targets. Run as:

    make TEST=cp_integration_<config> MODULE=test_inference_cycles \\
         CONFIG_NAME=<config> CYCLES_CSV=<path>

The DUT (neural_network / neural_network_bitshift / neural_network_fractional)
shares the same 5-state top FSM enum:

    IDLE=0, LOAD_HL1=1, RUN_TIMESTEPS=2, FINISH_Q=3, DONE_STATE=4

This test samples `dut.state` on every rising edge between `start` and `done`,
counts cycles per state, and writes one CSV row.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


STATE_NAMES = {
    0: "idle",
    1: "load_hl1",
    2: "run_timesteps",
    3: "finish_q",
    4: "done_state",
}

RUN_TIMESTEPS_STATE = 2

# ts_state (timestep-level FSM) enum mappings per NN variant. The base SV
# type is `timestep_state_t` defined in each neural_network*.sv; values
# differ across variants because bitshift/fractional add TS_HL2_WAIT
# between TS_HL2_STEP and TS_NEXT.
TS_SUBSTATE_KEYS = ("hl1_step", "fc2_start", "fc2_wait", "hl2_step", "hl2_wait", "next")
TS_STATE_BY_NEURON = {
    "lif": {0: "hl1_step", 1: "fc2_start", 2: "fc2_wait", 3: "hl2_step", 4: "next"},
    "bitshift": {0: "hl1_step", 1: "fc2_start", 2: "fc2_wait", 3: "hl2_step", 4: "hl2_wait", 5: "next"},
    "fractional": {0: "hl1_step", 1: "fc2_start", 2: "fc2_wait", 3: "hl2_step", 4: "hl2_wait", 5: "next"},
}


def _read_int(dut, name: str, default: int = 0) -> int:
    try:
        return int(getattr(dut, name).value)
    except Exception:
        return default


def _read_observation_width(dut) -> int:
    return _read_int(dut, "NUM_INPUTS", 4)


async def _reset(dut):
    dut.reset.value = 1
    dut.start.value = 0
    n_obs = _read_observation_width(dut)
    try:
        for i in range(n_obs):
            dut.observations[i].value = 0
    except Exception:
        pass
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_inference_cycles(dut):
    """Run one inference, count cycles per top-level FSM state, emit CSV row."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await _reset(dut)

    # Set a non-zero observation set so the network actually has work to do.
    # The exact values do not affect cycle counts — the FSM is data-independent
    # in cycle structure.
    n_obs = _read_observation_width(dut)
    try:
        for i in range(n_obs):
            dut.observations[i].value = 0x0100  # arbitrary small positive
    except Exception:
        pass

    cycles_in_state: dict[int, int] = {k: 0 for k in STATE_NAMES}
    # ts_cycles_in_state accumulates only while top-level state == RUN_TIMESTEPS,
    # so the per-substate totals sum exactly to cycles_run_timesteps.
    ts_cycles_in_state: dict[int, int] = {}
    total = 0
    timeout = 200_000  # generous; larger than any current config's expected runtime

    def _sample_ts_state(top_state: int) -> None:
        if top_state != RUN_TIMESTEPS_STATE:
            return
        try:
            ts = int(dut.ts_state.value)
        except Exception:
            return
        ts_cycles_in_state[ts] = ts_cycles_in_state.get(ts, 0) + 1

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    total += 1
    try:
        s = int(dut.state.value)
        cycles_in_state[s] = cycles_in_state.get(s, 0) + 1
        _sample_ts_state(s)
    except Exception as exc:
        raise RuntimeError("Could not read dut.state — module must expose top-level FSM state") from exc

    while True:
        if int(dut.done.value) == 1:
            break
        await RisingEdge(dut.clk)
        total += 1
        s = int(dut.state.value)
        cycles_in_state[s] = cycles_in_state.get(s, 0) + 1
        _sample_ts_state(s)
        if total > timeout:
            raise TimeoutError(f"Inference did not complete in {timeout} cycles")

    num_timesteps = _read_int(dut, "NUM_TIMESTEPS", 0)
    hl1_size = _read_int(dut, "HL1_SIZE", 0)
    hl2_size = _read_int(dut, "HL2_SIZE", 0)
    history_length = _read_int(dut, "HISTORY_LENGTH", 1)

    cycles_load_hl1 = cycles_in_state.get(1, 0)
    cycles_run_ts = cycles_in_state.get(2, 0)
    cycles_finish_q = cycles_in_state.get(3, 0)
    cycles_per_ts = cycles_run_ts / num_timesteps if num_timesteps > 0 else 0.0

    # Resolve raw ts_state ints -> symbolic substate names per NN variant.
    neuron_type = os.environ.get("NEURON_TYPE", "lif").lower()
    if neuron_type not in TS_STATE_BY_NEURON:
        dut._log.warning(
            f"Unknown NEURON_TYPE={neuron_type!r}; defaulting ts substate map to 'lif'."
        )
        neuron_type = "lif"
    ts_int_to_name = TS_STATE_BY_NEURON[neuron_type]
    ts_cycles_by_name: dict[str, int] = {k: 0 for k in TS_SUBSTATE_KEYS}
    for raw, count in ts_cycles_in_state.items():
        name = ts_int_to_name.get(raw)
        if name is not None:
            ts_cycles_by_name[name] += count

    dut._log.info(
        f"Cycles: total={total} idle={cycles_in_state.get(0, 0)} "
        f"load_hl1={cycles_load_hl1} run_ts={cycles_run_ts} "
        f"finish_q={cycles_finish_q} done={cycles_in_state.get(4, 0)} "
        f"per_ts={cycles_per_ts:.2f}"
    )
    dut._log.info(
        "  ts substates: " + " ".join(
            f"{k}={ts_cycles_by_name[k]}" for k in TS_SUBSTATE_KEYS
        )
    )

    config_name = os.environ.get("CONFIG_NAME", os.environ.get("TEST", "unknown"))
    csv_path_env = os.environ.get("CYCLES_CSV")
    if csv_path_env:
        csv_path = Path(csv_path_env)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "config": config_name,
            "neuron_type": neuron_type,
            "total_cycles": total,
            "cycles_idle": cycles_in_state.get(0, 0),
            "cycles_load_hl1": cycles_load_hl1,
            "cycles_run_timesteps": cycles_run_ts,
            "cycles_finish_q": cycles_finish_q,
            "cycles_done_state": cycles_in_state.get(4, 0),
            **{f"cycles_ts_{k}": ts_cycles_by_name[k] for k in TS_SUBSTATE_KEYS},
            "num_timesteps": num_timesteps,
            "cycles_per_timestep": f"{cycles_per_ts:.4f}",
            "hl1_size": hl1_size,
            "hl2_size": hl2_size,
            "history_length": history_length,
        }
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        dut._log.info(f"Wrote {csv_path}")

    assert total > 0
    assert cycles_run_ts > 0, "Expected cycles in RUN_TIMESTEPS state"
