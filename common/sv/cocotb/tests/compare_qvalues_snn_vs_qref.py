"""Compare Q-values: snnTorch float reference vs fixed-point reference.

For each of N observations drawn from a golden-vectors JSON, runs three
pipelines and prints per-action Q-values plus pairwise deltas:

  A) snn_policy.SNNPolicy.forward
       Float snnTorch model loaded from the trained .pth checkpoint.
       No HL1 -> fc2 delay (snnTorch uses same-timestep spikes).

  B) qs213_reference.run_inference with use_delayed_hl1=True
       QS<frac_bits> fixed-point reference, with the HL1 -> fc2 one-timestep
       delay that matches the multi-cycle hardware (fractional/bitshift).
       This is the reference the SV golden_vectors are generated against,
       so it stands in for the hardware Q-values without needing cocotb.

  C) qs213_reference.run_inference with use_delayed_hl1=False
       Same fixed-point reference, but without the HL1 -> fc2 delay.
       Lets us isolate the off-by-one's effect inside the quantized domain.

Pairwise deltas:
  A vs C : pure quantization error (no delay on either side)
  B vs C : pure off-by-one effect (both quantized; only delay differs)
  A vs B : total deviation of hardware behaviour from training behaviour

Defaults target frac-32-8-8-q2_13 (fractional-32-8-8 v3 Optuna seed 44, QS2.13).
Pass --model / --checkpoint / --vectors to retarget another model.

Run with the project venv:
  ./venv/bin/python common/sv/cocotb/tests/compare_qvalues_snn_vs_qref.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
TRAIN_DIR = REPO_ROOT / "2_training_and_simulation" / "train"
SCRIPTS_DIR = TRAIN_DIR / "scripts"

sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from snn_policy import SNNPolicy  # noqa: E402
from qs213_reference import (  # noqa: E402
    BETA,
    DATA_WIDTH,
    MEMBRANE_WIDTH,
    THRESHOLD,
    ModelConfig,
    bitshift_step,
    fractional_step,
    lif_step,
    linear_layer,
    load_mem_file,
    load_model_config,
    q_accumulate,
    reshape,
)

DEFAULT_VECTORS = THIS_DIR / "golden_vectors" / "frac-32-8-8-q2_13.json"
DEFAULT_CHECKPOINT = (
    TRAIN_DIR / "models" / "fractional-32-8-8" / "fractional-32-8-8-seed44-final.pth"
)
DEFAULT_MODEL = "frac-32-8-8-q2_13"
DEFAULT_N = 10


def fixed_to_float(value: int, frac_bits: int, total_bits: int = DATA_WIDTH) -> float:
    if value >= (1 << (total_bits - 1)):
        value -= 1 << total_bits
    return value / (1 << frac_bits)


def run_qref_inference(
    obs_fixed: List[int], cfg: ModelConfig, use_delayed_hl1: bool
) -> List[int]:
    """Vendored qs213_reference.run_inference with use_delayed_hl1 as a parameter.

    Identical to qs213_reference.run_inference at qs213_reference.py:457 except
    the delay flag is explicit rather than implied by cfg.neuron_type. Keeping
    a local copy avoids monkey-patching the shared reference module.
    """
    if len(obs_fixed) != cfg.num_inputs:
        raise ValueError(
            f"Expected {cfg.num_inputs} observations, got {len(obs_fixed)}"
        )

    fc1_weights = reshape(
        load_mem_file(cfg.weights_dir / "fc1_weights.mem", DATA_WIDTH),
        cfg.hl1_size,
        cfg.num_inputs,
    )
    fc1_biases = load_mem_file(cfg.weights_dir / "fc1_bias.mem", DATA_WIDTH)
    fc2_weights = reshape(
        load_mem_file(cfg.weights_dir / "fc2_weights.mem", DATA_WIDTH),
        cfg.hl2_size,
        cfg.hl1_size,
    )
    fc2_biases = load_mem_file(cfg.weights_dir / "fc2_bias.mem", DATA_WIDTH)
    fc_out_weights = reshape(
        load_mem_file(cfg.weights_dir / "fc_out_weights.mem", DATA_WIDTH),
        cfg.num_actions,
        cfg.hl2_size,
    )
    fc_out_biases = load_mem_file(cfg.weights_dir / "fc_out_bias.mem", DATA_WIDTH)

    hl1_currents = linear_layer(
        obs_fixed, fc1_weights, fc1_biases, DATA_WIDTH, cfg.frac_bits
    )

    hl1_mem = [0 for _ in range(cfg.hl1_size)]
    hl1_spike_prev = [0 for _ in range(cfg.hl1_size)]
    hl2_mem = [0 for _ in range(cfg.hl2_size)]
    hl2_spike_prev = [0 for _ in range(cfg.hl2_size)]

    hl1_history = [
        [0 for _ in range(cfg.history_length)] for _ in range(cfg.hl1_size)
    ]
    hl1_history_ptr = [0 for _ in range(cfg.hl1_size)]
    hl2_history = [
        [0 for _ in range(cfg.history_length)] for _ in range(cfg.hl2_size)
    ]
    hl2_history_ptr = [0 for _ in range(cfg.hl2_size)]

    membranes_by_timestep: List[List[int]] = []
    hl1_spikes_for_fc2 = [0 for _ in range(cfg.hl1_size)]

    for _t in range(cfg.num_timesteps):
        hl1_spikes: List[int] = []
        for i in range(cfg.hl1_size):
            if cfg.neuron_type == "lif":
                hl1_mem[i], hl1_spike_prev[i] = lif_step(
                    hl1_mem[i],
                    hl1_spike_prev[i],
                    hl1_currents[i],
                    data_width=DATA_WIDTH,
                    membrane_width=MEMBRANE_WIDTH,
                    threshold=THRESHOLD,
                    beta=BETA,
                )
            elif cfg.neuron_type == "fractional":
                (
                    hl1_mem[i],
                    hl1_spike_prev[i],
                    hl1_history[i],
                    hl1_history_ptr[i],
                ) = fractional_step(
                    hl1_mem[i],
                    hl1_spike_prev[i],
                    hl1_currents[i],
                    hl1_history[i],
                    hl1_history_ptr[i],
                    cfg,
                )
            else:
                (
                    hl1_mem[i],
                    hl1_spike_prev[i],
                    hl1_history[i],
                    hl1_history_ptr[i],
                ) = bitshift_step(
                    hl1_mem[i],
                    hl1_spike_prev[i],
                    hl1_currents[i],
                    hl1_history[i],
                    hl1_history_ptr[i],
                    cfg,
                )
            hl1_spikes.append(hl1_spike_prev[i])

        if use_delayed_hl1:
            fc2_inputs = [cfg.threshold if s else 0 for s in hl1_spikes_for_fc2]
            hl1_spikes_for_fc2 = hl1_spikes
        else:
            fc2_inputs = [cfg.threshold if s else 0 for s in hl1_spikes]

        hl2_currents = linear_layer(
            fc2_inputs,
            fc2_weights,
            fc2_biases,
            cfg.fc2_output_width,
            cfg.frac_bits,
        )

        mem_t: List[int] = []
        for i in range(cfg.hl2_size):
            if cfg.neuron_type == "lif":
                hl2_mem[i], hl2_spike_prev[i] = lif_step(
                    hl2_mem[i],
                    hl2_spike_prev[i],
                    hl2_currents[i],
                    data_width=cfg.fc2_output_width,
                    membrane_width=MEMBRANE_WIDTH,
                    threshold=THRESHOLD,
                    beta=BETA,
                )
            elif cfg.neuron_type == "fractional":
                (
                    hl2_mem[i],
                    hl2_spike_prev[i],
                    hl2_history[i],
                    hl2_history_ptr[i],
                ) = fractional_step(
                    hl2_mem[i],
                    hl2_spike_prev[i],
                    hl2_currents[i],
                    hl2_history[i],
                    hl2_history_ptr[i],
                    cfg,
                )
            else:
                (
                    hl2_mem[i],
                    hl2_spike_prev[i],
                    hl2_history[i],
                    hl2_history_ptr[i],
                ) = bitshift_step(
                    hl2_mem[i],
                    hl2_spike_prev[i],
                    hl2_currents[i],
                    hl2_history[i],
                    hl2_history_ptr[i],
                    cfg,
                )
            mem_t.append(hl2_mem[i])
        membranes_by_timestep.append(mem_t)

    return q_accumulate(
        membranes_by_timestep, fc_out_weights, fc_out_biases, cfg.frac_bits
    )


def build_snn_policy(checkpoint_path: Path) -> SNNPolicy:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    policy = SNNPolicy(
        n_observations=cfg["n_observations"],
        n_actions=cfg["n_actions"],
        num_steps=cfg["num_steps"],
        beta=cfg["beta"],
        neuron_type=cfg["neuron_type"],
        hidden1_size=cfg["hidden1_size"],
        hidden2_size=cfg["hidden2_size"],
        alpha=cfg.get("alpha", 0.5),
        lam=cfg.get("lam", 0.111),
        history_length=cfg.get("history_length", 256),
        dt=cfg.get("dt", 1.0),
    )
    policy.load_state_dict(ckpt["policy_net_state_dict"])
    policy.eval()
    return policy


def run_snn_policy(policy: SNNPolicy, obs_float: List[float]) -> List[float]:
    with torch.no_grad():
        obs = torch.tensor([obs_float], dtype=torch.float32)
        q = policy(obs)
    return q[0].tolist()


def fixed_q_to_float(q_accum: List[int], cfg: ModelConfig) -> List[float]:
    """Normalize accumulated fixed-point q_accum to per-timestep float.

    qs213_reference accumulates q across num_timesteps with no division
    (matches the HW q_accumulator). snn_policy averages over num_steps and
    returns float. Dividing q_accum by (num_timesteps * 2^frac_bits) puts
    both in the same float per-timestep units.
    """
    scale = cfg.num_timesteps * (1 << cfg.frac_bits)
    return [q / scale for q in q_accum]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of vectors to process")
    parser.add_argument("--vectors", default=str(DEFAULT_VECTORS), help="Path to golden_vectors JSON")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Path to PyTorch .pth checkpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="qs213_reference model name")
    args = parser.parse_args()

    cfg = load_model_config(args.model)
    vectors = json.loads(Path(args.vectors).read_text())["vectors"][: args.n]
    policy = build_snn_policy(Path(args.checkpoint))

    print(f"# Model:      {args.model}")
    print(f"# Vectors:    {args.vectors}")
    print(f"# Checkpoint: {args.checkpoint}")
    print(f"# N vectors:  {len(vectors)}")
    print(f"# A = snn_policy (float snnTorch, NO HL1 delay)")
    print(f"# B = qs213_reference (Q{cfg.frac_bits} fixed, WITH HL1 delay  = HW behaviour)")
    print(f"# C = qs213_reference (Q{cfg.frac_bits} fixed, NO HL1 delay)")
    print()

    header = (
        f"{'idx':>3}  "
        f"{'A_q0':>9} {'A_q1':>9}   "
        f"{'B_q0':>9} {'B_q1':>9}   "
        f"{'C_q0':>9} {'C_q1':>9}   "
        f"{'A-B q0':>9} {'A-B q1':>9}   "
        f"{'B-C q0':>9} {'B-C q1':>9}   "
        f"{'aA':>2} {'aB':>2} {'aC':>2}"
    )
    print(header)
    print("-" * len(header))

    diff_act_AB = 0
    diff_act_AC = 0
    diff_act_BC = 0
    sum_abs_AB = [0.0, 0.0]
    sum_abs_BC = [0.0, 0.0]
    sum_abs_AC = [0.0, 0.0]

    for idx, vec in enumerate(vectors):
        obs_fixed = vec["obs_qs213"]
        obs_float = [fixed_to_float(o, cfg.frac_bits) for o in obs_fixed]

        q_A = run_snn_policy(policy, obs_float)
        q_B_fixed = run_qref_inference(obs_fixed, cfg, use_delayed_hl1=True)
        q_C_fixed = run_qref_inference(obs_fixed, cfg, use_delayed_hl1=False)
        q_B = fixed_q_to_float(q_B_fixed, cfg)
        q_C = fixed_q_to_float(q_C_fixed, cfg)

        act_A = 0 if q_A[0] >= q_A[1] else 1
        act_B = 0 if q_B[0] >= q_B[1] else 1
        act_C = 0 if q_C[0] >= q_C[1] else 1
        if act_A != act_B:
            diff_act_AB += 1
        if act_A != act_C:
            diff_act_AC += 1
        if act_B != act_C:
            diff_act_BC += 1

        for a in range(2):
            sum_abs_AB[a] += abs(q_A[a] - q_B[a])
            sum_abs_BC[a] += abs(q_B[a] - q_C[a])
            sum_abs_AC[a] += abs(q_A[a] - q_C[a])

        print(
            f"{idx:>3}  "
            f"{q_A[0]:>9.5f} {q_A[1]:>9.5f}   "
            f"{q_B[0]:>9.5f} {q_B[1]:>9.5f}   "
            f"{q_C[0]:>9.5f} {q_C[1]:>9.5f}   "
            f"{q_A[0] - q_B[0]:>+9.5f} {q_A[1] - q_B[1]:>+9.5f}   "
            f"{q_B[0] - q_C[0]:>+9.5f} {q_B[1] - q_C[1]:>+9.5f}   "
            f"{act_A:>2} {act_B:>2} {act_C:>2}"
        )

    n = len(vectors)
    print()
    print(f"Mean |delta| over {n} vectors:")
    print(f"  A vs B (snn_policy vs HW-equiv):  q0={sum_abs_AB[0]/n:.5f}  q1={sum_abs_AB[1]/n:.5f}")
    print(f"  A vs C (quantization only):        q0={sum_abs_AC[0]/n:.5f}  q1={sum_abs_AC[1]/n:.5f}")
    print(f"  B vs C (off-by-one only):          q0={sum_abs_BC[0]/n:.5f}  q1={sum_abs_BC[1]/n:.5f}")
    print()
    print(f"Action disagreements (out of {n}):")
    print(f"  A vs B: {diff_act_AB}")
    print(f"  A vs C: {diff_act_AC}")
    print(f"  B vs C: {diff_act_BC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
