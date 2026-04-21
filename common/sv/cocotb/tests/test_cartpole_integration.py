"""
Cocotb integration test for CartPole using the SystemVerilog neural_network module.

This test runs the actual CartPole-v1 gymnasium environment and uses the
hardware neural_network module to compute actions, verifying that the
trained SNN model works correctly in hardware.

Test configuration uses full model parameters:
- NUM_INPUTS = 4 (CartPole observation space)
- HL1_SIZE = 64
- HL2_SIZE = 16
- NUM_ACTIONS = 2 (left/right)
- NUM_TIMESTEPS = 30
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

import csv
import os
import gymnasium as gym
import numpy as np

# Fixed-point format constants (default QS2.13; can be overridden by DUT FRAC_BITS)
TOTAL_BITS = 16
FRAC_BITS = 13
SCALE_FACTOR = 2**FRAC_BITS  # 8192
MAX_SIGNED = 2 ** (TOTAL_BITS - 1) - 1
MIN_SIGNED = -(2 ** (TOTAL_BITS - 1))
UNSIGNED_RANGE = 2**TOTAL_BITS

# Test configuration (must match Verilog parameters)
NUM_INPUTS = 4
NUM_ACTIONS = 2
NUM_TIMESTEPS = 30
DEFAULT_CARTPOLE_MAX_STEPS = 500
FULL_DEBUG = os.getenv("FULL_DEBUG", os.getenv("SNAPSHOT_FULL_DEBUG", "0")) == "1"


def get_cartpole_max_steps() -> int:
    """Return configured CartPole max steps, defaulting to 500."""
    raw = os.getenv("CARTPOLE_MAX_STEPS", str(DEFAULT_CARTPOLE_MAX_STEPS))
    try:
        steps = int(raw)
    except ValueError:
        return DEFAULT_CARTPOLE_MAX_STEPS
    return steps if steps > 0 else DEFAULT_CARTPOLE_MAX_STEPS


def _read_int_param(dut, name: str, default: int) -> int:
    """Best-effort read of integer DUT parameter/constant handle."""
    try:
        return int(getattr(dut, name).value)
    except Exception:
        return default


def estimate_inference_timeout_cycles(dut) -> int:
    """Estimate a safe inference timeout from DUT sizing parameters.

    This scales timeout for multi-cycle neuron variants (e.g. larger HISTORY_LENGTH)
    while preserving current behavior for single-cycle LIF.
    """
    num_timesteps = _read_int_param(dut, "NUM_TIMESTEPS", NUM_TIMESTEPS)
    hl1_size = _read_int_param(dut, "HL1_SIZE", get_hl1_size(dut) or 64)
    hl2_size = _read_int_param(dut, "HL2_SIZE", get_hl2_size(dut) or 16)
    history_length = _read_int_param(dut, "HISTORY_LENGTH", 1)

    # Heuristic: work scales with timesteps, layer widths, and neuron history latency.
    # Keep a floor so standard LIF remains unchanged from prior behavior.
    scaled = num_timesteps * max(1, hl1_size + hl2_size) * max(4, history_length) * 4
    return max(50_000, scaled)


def get_frac_bits(dut=None) -> int:
    """Read FRAC_BITS from DUT when available; otherwise use default."""
    if dut is not None:
        try:
            return int(dut.FRAC_BITS.value)
        except Exception:
            pass
    return FRAC_BITS


def float_to_fixed(value: float, frac_bits: int = FRAC_BITS) -> int:
    """Convert float to fixed-point (unsigned representation for cocotb)."""
    scale_factor = 2**frac_bits
    scaled = int(round(value * scale_factor))
    if scaled > MAX_SIGNED:
        scaled = MAX_SIGNED
    elif scaled < MIN_SIGNED:
        scaled = MIN_SIGNED
    if scaled < 0:
        scaled = scaled + UNSIGNED_RANGE
    return scaled


def to_signed(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    value &= mask
    if value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


def get_hl1_size(dut) -> int:
    """Best-effort: derive HL1 size from DUT handles."""
    try:
        return len(dut.hl1_spikes)
    except Exception:
        try:
            return len(dut.hl1_currents)
        except Exception:
            return 0


def get_hl2_size(dut) -> int:
    """Best-effort: derive HL2 size from DUT handles."""
    try:
        return len(dut.hl2_membranes)
    except Exception:
        try:
            return len(dut.hl2_spikes)
        except Exception:
            return 0


async def reset_dut(dut):
    """Apply reset sequence to the DUT."""
    dut.reset.value = 1
    dut.start.value = 0
    for i in range(NUM_INPUTS):
        dut.observations[i].value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


async def run_inference(
    dut, observations: np.ndarray, timeout_cycles: int | None = None
) -> tuple:
    """
    Run a single inference through the neural network.

    Args:
        dut: Device under test
        observations: NumPy array of 4 float observations from CartPole
        timeout_cycles: Maximum cycles to wait for completion

    Returns:
        int: selected action (0 or 1)
    """
    # Set observations (convert to fixed-point)
    frac_bits = get_frac_bits(dut)
    for i in range(NUM_INPUTS):
        dut.observations[i].value = float_to_fixed(float(observations[i]), frac_bits)

    # Start inference
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    if timeout_cycles is None:
        timeout_cycles = estimate_inference_timeout_cycles(dut)

    for cycle in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            break
    else:
        raise TimeoutError(f"Inference did not complete within {timeout_cycles} cycles")

    # Read action selected by hardware from full-precision Q-values
    action = int(dut.selected_action.value)

    return action


async def run_inference_with_trace(
    dut, observations: np.ndarray, timeout_cycles: int | None = None
) -> tuple[int, dict]:
    """Run one inference and return action plus traceable internal summaries."""
    action = await run_inference(dut, observations, timeout_cycles=timeout_cycles)

    hl2_vals = []
    for i in range(get_hl2_size(dut)):
        raw = int(dut.hl2_membranes[i].value)
        hl2_vals.append(to_signed(raw, 24))

    q0 = ""
    q1 = ""
    try:
        q0 = int(dut.q_accum.q_divided["0"].value)
        q1 = int(dut.q_accum.q_divided["1"].value)
    except Exception:
        pass

    trace = {
        "q0": q0,
        "q1": q1,
        "hl2_mem_mean": float(np.mean(hl2_vals)) if hl2_vals else "",
        "hl2_mem_max": int(np.max(hl2_vals)) if hl2_vals else "",
        "hl2_mem_min": int(np.min(hl2_vals)) if hl2_vals else "",
    }

    return action, trace


def _trace_signature(action: int, trace: dict) -> tuple[int, int, int]:
    """Return stable integer signature for repeatability checks."""
    q0 = trace.get("q0", "")
    q1 = trace.get("q1", "")
    return (
        action,
        int(q0) if q0 != "" else 0,
        int(q1) if q1 != "" else 0,
    )


async def run_episode_trace(dut, env, seed: int, max_steps: int = 500):
    """Run one episode and return list of per-step action records."""
    await reset_dut(dut)
    observation, _ = env.reset(seed=seed)
    records = []

    for step in range(max_steps):
        action, trace = await run_inference_with_trace(dut, observation)
        records.append(
            {
                "seed": seed,
                "step": step,
                "action": action,
                "q0": trace["q0"],
                "q1": trace["q1"],
                "hl2_mem_mean": trace["hl2_mem_mean"],
                "hl2_mem_max": trace["hl2_mem_max"],
                "hl2_mem_min": trace["hl2_mem_min"],
                "obs0": float(observation[0]),
                "obs1": float(observation[1]),
                "obs2": float(observation[2]),
                "obs3": float(observation[3]),
            }
        )

        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    return records


def _safe_int(value_obj, bits: int | None = None):
    """Best-effort integer conversion for internal probe signals."""
    try:
        value = int(value_obj.value)
        if bits is not None:
            value = to_signed(value, bits)
        return value
    except Exception:
        return ""


def _safe_array_int(parent_obj, array_name: str, idx: int, bits: int | None = None):
    """Best-effort read for internal array elements across simulator handle styles."""
    try:
        arr = getattr(parent_obj, array_name)
    except Exception:
        return ""

    for key in (idx, str(idx), f"[{idx}]"):
        try:
            value = int(arr[key].value)
            if bits is not None:
                value = to_signed(value, bits)
            return value
        except Exception:
            pass

    for alt_name in (f"{array_name}[{idx}]", f"{array_name}_{idx}"):
        try:
            value = int(getattr(parent_obj, alt_name).value)
            if bits is not None:
                value = to_signed(value, bits)
            return value
        except Exception:
            pass

    return ""


def _safe_hl1_spike_count(dut):
    """Best-effort count of HL1 spikes at current cycle."""
    try:
        count = 0
        num_hl1 = len(dut.hl1_spikes)
        for i in range(num_hl1):
            count += int(dut.hl1_spikes[i].value)
        return count
    except Exception:
        return ""


def _safe_hl1_sample(dut, sample_size: int = 8) -> tuple[str, str]:
    """Capture small HL1 current/spike samples for timestep-0 sensitivity checks."""
    try:
        num_curr = len(dut.hl1_currents)
        num_spk = len(dut.hl1_spikes)
        sample_n = min(sample_size, num_curr, num_spk)

        currents = []
        spikes = []
        for i in range(sample_n):
            curr = _safe_array_int(dut, "hl1_currents", i, bits=TOTAL_BITS)
            spk = _safe_array_int(dut, "hl1_spikes", i)
            currents.append("" if curr == "" else str(curr))
            spikes.append("" if spk == "" else str(spk))

        return ";".join(currents), ";".join(spikes)
    except Exception:
        return "", ""


async def run_inference_with_timestep_snapshots(
    dut,
    observations: np.ndarray,
    inference_idx: int,
    full_debug: bool = False,
    timeout_cycles: int | None = None,
) -> tuple[int, list[dict]]:
    """Run one inference and collect fc2/HL2/Q snapshots during execution."""
    frac_bits = get_frac_bits(dut)
    scale_factor = 2**frac_bits
    obs_floats = [float(observations[i]) for i in range(NUM_INPUTS)]
    obs_fixed = [float_to_fixed(v, frac_bits) for v in obs_floats]

    for i in range(NUM_INPUTS):
        dut.observations[i].value = obs_fixed[i]

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    records = []
    cycle = 0
    last_q_timestep = None
    hl1_t0_curr_sample = ""
    hl1_t0_spike_sample = ""
    fc2_sat_pos_count = 0
    fc2_sat_neg_count = 0

    if timeout_cycles is None:
        timeout_cycles = estimate_inference_timeout_cycles(dut)

    for cycle in range(timeout_cycles):
        await RisingEdge(dut.clk)

        if int(dut.fc2_output_valid.value) == 1:
            ts = int(dut.current_timestep.value)
            fc2_idx = int(dut.fc2_output_idx.value)
            fc2_raw = int(dut.fc2_output_current.value)
            fc2_bits = len(dut.fc2_output_current)
            fc2_signed = to_signed(fc2_raw, fc2_bits)
            fc2_sat_pos = _safe_int(dut.fc2.sat_pos)
            fc2_sat_neg = _safe_int(dut.fc2.sat_neg)
            if fc2_sat_pos != "":
                fc2_sat_pos_count += int(fc2_sat_pos)
            if fc2_sat_neg != "":
                fc2_sat_neg_count += int(fc2_sat_neg)

            if full_debug and ts == 0 and fc2_idx == 0 and hl1_t0_curr_sample == "":
                hl1_t0_curr_sample, hl1_t0_spike_sample = _safe_hl1_sample(dut)

            records.append(
                {
                    "inference": inference_idx,
                    "cycle": cycle,
                    "stage": "fc2_stream",
                    "timestep": ts,
                    "fc2_idx": fc2_idx,
                    "fc2_raw": fc2_raw,
                    "fc2_signed": fc2_signed,
                    "fc2_float": fc2_signed / scale_factor,
                    "fc2_sat_pos": fc2_sat_pos,
                    "fc2_sat_neg": fc2_sat_neg,
                    "fc2_sat_pos_count": fc2_sat_pos_count,
                    "fc2_sat_neg_count": fc2_sat_neg_count,
                    "obs0": obs_floats[0],
                    "obs1": obs_floats[1],
                    "obs2": obs_floats[2],
                    "obs3": obs_floats[3],
                    "obs0_fixed": obs_fixed[0],
                    "obs1_fixed": obs_fixed[1],
                    "obs2_fixed": obs_fixed[2],
                    "obs3_fixed": obs_fixed[3],
                    "hl1_spike_count": _safe_hl1_spike_count(dut) if full_debug else "",
                    "hl1_t0_curr_sample": hl1_t0_curr_sample if full_debug else "",
                    "hl1_t0_spike_sample": hl1_t0_spike_sample if full_debug else "",
                    "hl2_mem_mean": "",
                    "hl2_mem_max": "",
                    "hl2_mem_min": "",
                    "q_read_timestep": "",
                    "q_state": "",
                    "q_accum0": "",
                    "q_accum1": "",
                    "q_div0": "",
                    "q_div1": "",
                    "selected_action": "",
                }
            )

        if int(dut.fc2_done.value) == 1:
            ts = int(dut.current_timestep.value)
            hl2_mem_vals = [
                to_signed(int(dut.hl2_membranes[i].value), 24)
                for i in range(get_hl2_size(dut))
            ]
            records.append(
                {
                    "inference": inference_idx,
                    "cycle": cycle,
                    "stage": "timestep_summary",
                    "timestep": ts,
                    "fc2_idx": "",
                    "fc2_raw": "",
                    "fc2_signed": "",
                    "fc2_float": "",
                    "fc2_sat_pos": "",
                    "fc2_sat_neg": "",
                    "fc2_sat_pos_count": fc2_sat_pos_count,
                    "fc2_sat_neg_count": fc2_sat_neg_count,
                    "obs0": obs_floats[0],
                    "obs1": obs_floats[1],
                    "obs2": obs_floats[2],
                    "obs3": obs_floats[3],
                    "obs0_fixed": obs_fixed[0],
                    "obs1_fixed": obs_fixed[1],
                    "obs2_fixed": obs_fixed[2],
                    "obs3_fixed": obs_fixed[3],
                    "hl1_spike_count": _safe_hl1_spike_count(dut) if full_debug else "",
                    "hl1_t0_curr_sample": hl1_t0_curr_sample if full_debug else "",
                    "hl1_t0_spike_sample": hl1_t0_spike_sample if full_debug else "",
                    "hl2_mem_mean": (
                        float(np.mean(hl2_mem_vals)) if hl2_mem_vals else ""
                    ),
                    "hl2_mem_max": int(np.max(hl2_mem_vals)) if hl2_mem_vals else "",
                    "hl2_mem_min": int(np.min(hl2_mem_vals)) if hl2_mem_vals else "",
                    "q_read_timestep": _safe_int(dut.q_read_timestep),
                    "q_state": _safe_int(dut.q_accum.state),
                    "q_accum0": _safe_array_int(dut.q_accum, "q_accum", 0),
                    "q_accum1": _safe_array_int(dut.q_accum, "q_accum", 1),
                    "q_div0": _safe_array_int(dut.q_accum, "q_divided", 0),
                    "q_div1": _safe_array_int(dut.q_accum, "q_divided", 1),
                    "selected_action": "",
                }
            )

        q_state = _safe_int(dut.q_accum.state)
        q_ts = _safe_int(dut.q_accum.timestep_counter)
        if q_state != "" and q_state != 0 and q_ts != "" and q_ts != last_q_timestep:
            records.append(
                {
                    "inference": inference_idx,
                    "cycle": cycle,
                    "stage": "q_progress",
                    "timestep": "",
                    "fc2_idx": "",
                    "fc2_raw": "",
                    "fc2_signed": "",
                    "fc2_float": "",
                    "fc2_sat_pos": "",
                    "fc2_sat_neg": "",
                    "fc2_sat_pos_count": fc2_sat_pos_count,
                    "fc2_sat_neg_count": fc2_sat_neg_count,
                    "obs0": obs_floats[0],
                    "obs1": obs_floats[1],
                    "obs2": obs_floats[2],
                    "obs3": obs_floats[3],
                    "obs0_fixed": obs_fixed[0],
                    "obs1_fixed": obs_fixed[1],
                    "obs2_fixed": obs_fixed[2],
                    "obs3_fixed": obs_fixed[3],
                    "hl1_spike_count": _safe_hl1_spike_count(dut) if full_debug else "",
                    "hl1_t0_curr_sample": hl1_t0_curr_sample if full_debug else "",
                    "hl1_t0_spike_sample": hl1_t0_spike_sample if full_debug else "",
                    "hl2_mem_mean": "",
                    "hl2_mem_max": "",
                    "hl2_mem_min": "",
                    "q_read_timestep": _safe_int(dut.q_read_timestep),
                    "q_state": q_state,
                    "q_accum0": (
                        _safe_array_int(dut.q_accum, "q_accum", 0) if full_debug else ""
                    ),
                    "q_accum1": (
                        _safe_array_int(dut.q_accum, "q_accum", 1) if full_debug else ""
                    ),
                    "q_div0": (
                        _safe_array_int(dut.q_accum, "q_divided", 0)
                        if full_debug
                        else ""
                    ),
                    "q_div1": (
                        _safe_array_int(dut.q_accum, "q_divided", 1)
                        if full_debug
                        else ""
                    ),
                    "selected_action": "",
                }
            )
            last_q_timestep = q_ts

        if dut.done.value == 1:
            break
    else:
        raise TimeoutError(f"Inference did not complete within {timeout_cycles} cycles")

    action = int(dut.selected_action.value)
    records.append(
        {
            "inference": inference_idx,
            "cycle": cycle,
            "stage": "final",
            "timestep": "",
            "fc2_idx": "",
            "fc2_raw": "",
            "fc2_signed": "",
            "fc2_float": "",
            "fc2_sat_pos": "",
            "fc2_sat_neg": "",
            "fc2_sat_pos_count": fc2_sat_pos_count,
            "fc2_sat_neg_count": fc2_sat_neg_count,
            "obs0": obs_floats[0],
            "obs1": obs_floats[1],
            "obs2": obs_floats[2],
            "obs3": obs_floats[3],
            "obs0_fixed": obs_fixed[0],
            "obs1_fixed": obs_fixed[1],
            "obs2_fixed": obs_fixed[2],
            "obs3_fixed": obs_fixed[3],
            "hl1_spike_count": _safe_hl1_spike_count(dut) if full_debug else "",
            "hl1_t0_curr_sample": hl1_t0_curr_sample if full_debug else "",
            "hl1_t0_spike_sample": hl1_t0_spike_sample if full_debug else "",
            "hl2_mem_mean": "",
            "hl2_mem_max": "",
            "hl2_mem_min": "",
            "q_read_timestep": _safe_int(dut.q_read_timestep),
            "q_state": _safe_int(dut.q_accum.state),
            "q_accum0": (
                _safe_array_int(dut.q_accum, "q_accum", 0) if full_debug else ""
            ),
            "q_accum1": (
                _safe_array_int(dut.q_accum, "q_accum", 1) if full_debug else ""
            ),
            "q_div0": (
                _safe_array_int(dut.q_accum, "q_divided", 0) if full_debug else ""
            ),
            "q_div1": (
                _safe_array_int(dut.q_accum, "q_divided", 1) if full_debug else ""
            ),
            "selected_action": action,
        }
    )

    return action, records


@cocotb.test()
async def test_cartpole_single_episode(dut):
    """
    Run a single CartPole episode using the hardware neural network.

    This test validates that the trained model can balance the pole.
    A successful episode should last close to the configured max steps.
    """
    clock = Clock(dut.clk, 10, unit="ns")  # 100 MHz
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Create CartPole environment
    env = gym.make("CartPole-v1")

    # Reset environment
    observation, info = env.reset(seed=42)

    total_reward = 0
    step_count = 0
    max_steps = get_cartpole_max_steps()
    min_passing_steps = int(np.ceil(0.95 * max_steps))

    dut._log.info("Starting CartPole episode...")

    while step_count < max_steps:
        # Run inference on hardware
        action = await run_inference(dut, observation)

        # Log every 50 steps
        if step_count % 50 == 0:
            dut._log.info(f"Step {step_count}: obs={observation}, action={action}")

        # Take action in environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        if terminated or truncated:
            break

    env.close()

    dut._log.info(f"Episode finished: {step_count} steps, total reward: {total_reward}")

    assert step_count >= min_passing_steps, (
        f"Episode too short: {step_count} steps "
        f"(expected >= {min_passing_steps}, 95% of max_steps={max_steps})"
    )

    dut._log.info(f"SUCCESS: CartPole balanced for {step_count} steps")


@cocotb.test()
async def test_cartpole_multiple_episodes(dut):
    """
    Run multiple CartPole episodes and compute average performance.

    This provides a more robust test of the model's performance by
    averaging over multiple random initial conditions.
    """
    clock = Clock(dut.clk, 10, unit="ns")  # 100 MHz
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Create CartPole environment
    env = gym.make("CartPole-v1")

    max_steps = get_cartpole_max_steps()

    async def run_episode(seed: int) -> tuple[float, int]:
        await reset_dut(dut)

        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        step_count = 0
        while step_count < max_steps:
            action = await run_inference(dut, observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        return total_reward, step_count

    # Repeatability check: same seed should produce nearly identical performance
    # if there is no cross-episode state leakage in the DUT.
    rep_seed = 42
    rep_rewards = []
    for idx in range(2):
        reward, steps = await run_episode(rep_seed)
        rep_rewards.append(reward)
        dut._log.info(
            f"Repeat seed check {idx + 1}: seed={rep_seed}, reward={reward} ({steps} steps)"
        )

    assert abs(rep_rewards[0] - rep_rewards[1]) <= 5.0, (
        f"Same-seed repeatability failed (possible state carryover): "
        f"{rep_rewards[0]} vs {rep_rewards[1]}"
    )

    num_episodes = 10
    episode_rewards = []

    for episode in range(num_episodes):
        total_reward, step_count = await run_episode(seed=episode * 7 + 42)

        episode_rewards.append(total_reward)
        dut._log.info(
            f"Episode {episode + 1}: {total_reward} reward ({step_count} steps)"
        )

    env.close()

    avg_reward = np.mean(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)

    dut._log.info(f"Performance over {num_episodes} episodes:")
    dut._log.info(f"  Average reward: {avg_reward:.1f}")
    dut._log.info(f"  Min reward: {min_reward:.1f}")
    dut._log.info(f"  Max reward: {max_reward:.1f}")

    expected_avg = 0.90 * max_steps
    assert (
        avg_reward >= expected_avg
    ), (
        f"Average reward too low: {avg_reward:.1f} "
        f"(expected >= {expected_avg:.1f}, 90% of max_steps={max_steps})"
    )

    dut._log.info(f"SUCCESS: Average reward {avg_reward:.1f} meets threshold")


@cocotb.test()
async def test_inference_timing(dut):
    """
    Measure and report inference timing.

    This test measures how many clock cycles a single inference takes,
    which is useful for performance characterization.
    """
    clock = Clock(dut.clk, 10, unit="ns")  # 100 MHz
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Create a sample observation
    test_obs = np.array([0.01, -0.02, 0.03, -0.01])
    frac_bits = get_frac_bits(dut)

    # Measure timing
    for i in range(NUM_INPUTS):
        dut.observations[i].value = float_to_fixed(float(test_obs[i]), frac_bits)

    # Start inference and count cycles
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    cycle_count = 0
    max_cycles = 100000

    while cycle_count < max_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1
        if dut.done.value == 1:
            break

    # Calculate timing at 100 MHz
    time_us = cycle_count * 0.01  # 10ns per cycle

    dut._log.info(f"Inference timing:")
    dut._log.info(f"  Cycles: {cycle_count}")
    dut._log.info(f"  Time @ 100MHz: {time_us:.2f} µs")
    dut._log.info(f"  Throughput: {1_000_000 / time_us:.0f} inferences/sec")

    # Sanity check - inference should complete
    assert cycle_count < max_cycles, "Inference did not complete"

    # Verify inference produced a valid action
    action = int(dut.selected_action.value)
    dut._log.info(f"  selected_action: {action}")

    dut._log.info("SUCCESS: Inference completed and timing measured")


@cocotb.test()
async def test_fc2_no_saturation_observation_ranges(dut):
    """
    Verify FC2 does not saturate across representative observation ranges.

    This is a stronger overflow-oriented check than action validity alone,
    because selected_action being 0/1 does not guarantee intermediate math
    stayed in range.
    """
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    test_cases = [
        ("Small values", np.array([0.01, 0.02, 0.03, 0.01])),
        ("Moderate values", np.array([0.5, -0.5, 0.2, -0.3])),
        ("Near limits", np.array([2.0, 1.5, 0.3, 1.0])),
        ("Edge of representable", np.array([3.0, -2.0, 0.4, -1.5])),
    ]

    for case_idx, (name, obs) in enumerate(test_cases):
        action, records = await run_inference_with_timestep_snapshots(
            dut,
            obs,
            inference_idx=case_idx,
            full_debug=False,
        )

        assert action in [0, 1], f"Invalid action: {action}"
        assert records, f"No snapshot records captured for case: {name}"

        final_record = next(
            (r for r in reversed(records) if r["stage"] == "final"), None
        )
        assert (
            final_record is not None
        ), f"Missing final snapshot record for case: {name}"

        sat_pos = int(final_record["fc2_sat_pos_count"])
        sat_neg = int(final_record["fc2_sat_neg_count"])

        assert sat_pos == 0 and sat_neg == 0, (
            f"FC2 saturation detected for {name}: "
            f"sat_pos_count={sat_pos}, sat_neg_count={sat_neg}, obs={obs}"
        )

        dut._log.info(
            f"{name}: obs={obs}, action={action}, "
            f"fc2_sat_pos_count={sat_pos}, fc2_sat_neg_count={sat_neg}"
        )

    dut._log.info("SUCCESS: No FC2 saturation across tested observation ranges")


@cocotb.test()
async def test_inference_state_leakage(dut):
    """
    Detect cross-inference state carryover using repeated identical observations.

    This is a focused dynamics test: repeated inferences with unchanged input
    should produce identical action/Q signatures if per-inference state is
    correctly reset/contained.
    """
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    obs = np.array([0.02, -0.03, 0.01, 0.04], dtype=np.float32)

    signatures = []
    repeats = 6
    for idx in range(repeats):
        action, trace = await run_inference_with_trace(dut, obs)
        sig = _trace_signature(action, trace)
        signatures.append(sig)
        dut._log.info(
            f"repeat={idx}: action={sig[0]}, q0={sig[1]}, q1={sig[2]}, "
            f"hl2_mean={trace['hl2_mem_mean']}"
        )

    baseline = signatures[0]
    mismatches = [
        (i, sig) for i, sig in enumerate(signatures[1:], start=1) if sig != baseline
    ]

    assert not mismatches, (
        "Cross-inference drift detected for fixed observation; "
        f"baseline={baseline}, mismatches={mismatches[:3]}"
    )

    await reset_dut(dut)
    post_reset_action, post_reset_trace = await run_inference_with_trace(dut, obs)
    post_reset_sig = _trace_signature(post_reset_action, post_reset_trace)

    assert post_reset_sig == baseline, (
        "Post-reset signature differs from initial baseline; "
        f"baseline={baseline}, post_reset={post_reset_sig}"
    )

    dut._log.info("SUCCESS: No cross-inference state leakage for fixed observation")


@cocotb.test()
async def test_cartpole_action_trace(dut):
    """Export deterministic hardware action traces for seeds 42 and 49."""
    if not FULL_DEBUG:
        cocotb.log.info("Skipping test_cartpole_action_trace (set FULL_DEBUG=1)")
        return

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    env = gym.make("CartPole-v1")
    seeds = [42, 49]
    all_records = []

    for seed in seeds:
        records = await run_episode_trace(dut, env, seed=seed, max_steps=500)
        all_records.extend(records)
        dut._log.info(f"Trace seed={seed}: {len(records)} steps")

    env.close()

    results_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results")
    )
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "cartpole_action_trace_hw.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "step",
                "action",
                "q0",
                "q1",
                "hl2_mem_mean",
                "hl2_mem_max",
                "hl2_mem_min",
                "obs0",
                "obs1",
                "obs2",
                "obs3",
            ],
        )
        writer.writeheader()
        writer.writerows(all_records)

    dut._log.info(f"Wrote hardware action trace: {out_csv}")


@cocotb.test()
async def test_cartpole_timestep_snapshots(dut):
    """Capture per-timestep fc2/HL2/Q snapshots for early seed-42 inferences."""
    if not FULL_DEBUG:
        cocotb.log.info("Skipping test_cartpole_timestep_snapshots (set FULL_DEBUG=1)")
        return

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    env = gym.make("CartPole-v1")
    observation, _ = env.reset(seed=42)

    all_records = []
    max_inferences = 5

    for inference_idx in range(max_inferences):
        action, records = await run_inference_with_timestep_snapshots(
            dut,
            observation,
            inference_idx=inference_idx,
            full_debug=FULL_DEBUG,
        )
        all_records.extend(records)

        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()

    results_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results")
    )
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "cartpole_timestep_snapshots_hw.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "inference",
                "cycle",
                "stage",
                "timestep",
                "fc2_idx",
                "fc2_raw",
                "fc2_signed",
                "fc2_float",
                "fc2_sat_pos",
                "fc2_sat_neg",
                "fc2_sat_pos_count",
                "fc2_sat_neg_count",
                "obs0",
                "obs1",
                "obs2",
                "obs3",
                "obs0_fixed",
                "obs1_fixed",
                "obs2_fixed",
                "obs3_fixed",
                "hl1_spike_count",
                "hl1_t0_curr_sample",
                "hl1_t0_spike_sample",
                "hl2_mem_mean",
                "hl2_mem_max",
                "hl2_mem_min",
                "q_read_timestep",
                "q_state",
                "q_accum0",
                "q_accum1",
                "q_div0",
                "q_div1",
                "selected_action",
            ],
        )
        writer.writeheader()
        writer.writerows(all_records)

    assert all_records, "No snapshot records captured"
    dut._log.info(f"Snapshot full debug: {FULL_DEBUG}")
    dut._log.info(f"Wrote timestep snapshots: {out_csv} ({len(all_records)} rows)")
