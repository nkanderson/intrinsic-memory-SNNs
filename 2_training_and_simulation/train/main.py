"""
CLI entry point for training or evaluating an SNN-based DQN agent on CartPole.

Training delegates entirely to train_fn.train(). Evaluation loads a checkpoint
and runs agent.evaluate().

For workflows not supported here (resume training from checkpoint, live
rendering during training, per-episode generalization eval), use legacy_main.py.
"""

import argparse
import re
import sys
from pathlib import Path

import gymnasium as gym
import torch
import torch.optim as optim
import yaml
from snntorch import surrogate

from dqn_agent import DQNAgent, ReplayMemory
from snn_policy import SNNPolicy
from train_fn import SHIFT_FUNC_MAP, train

N_ACTIONS = 2
N_OBSERVATIONS = 4


def resolve_shift_func(shift_func_name):
    if not shift_func_name:
        return None
    func = SHIFT_FUNC_MAP.get(shift_func_name)
    if func is None:
        raise ValueError(
            f"Unknown shift_func: {shift_func_name}. "
            f"Valid options: {list(SHIFT_FUNC_MAP.keys())}"
        )
    return func


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_config_name(config_path=None, pretrained_file=None):
    if config_path is not None:
        return Path(config_path).stem
    if pretrained_file is not None:
        stem = Path(pretrained_file).stem
        for suffix in ["-best", "-final", "-quantized"]:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        return re.sub(r"-Q[A-Z]?\d+_\d+$", "", stem)
    raise ValueError("get_config_name() requires config_path or pretrained_file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or evaluate an SNN-based DQN agent on CartPole"
    )
    parser.add_argument("--config", "-c", type=str, default=None,
        help="Path to YAML config file")
    parser.add_argument("--load", type=str, default=None, metavar="FILE",
        help="Load a pre-trained model (evaluate-only mode only; "
             "use legacy_main.py to resume training)")
    parser.add_argument("--evaluate-only", action="store_true",
        help="Evaluate a loaded model without training (requires --load)")
    parser.add_argument("--no-hw-acceleration", dest="hw_acceleration",
        action="store_false", help="Disable hardware acceleration (CUDA/MPS)")
    parser.set_defaults(hw_acceleration=True)
    parser.add_argument("--max-episode-steps", type=int, default=500,
        help="Maximum steps per CartPole episode (default: 500)")
    parser.add_argument("--metrics-csv", type=str, default=None,
        help="Path for training metrics CSV output")
    parser.add_argument("--save-best", action="store_true",
        help="Also save best-model checkpoint during training")
    parser.add_argument("--no-save-final", action="store_true",
        help="Do not save the final model checkpoint")
    parser.add_argument("--seed", type=int, default=None,
        help="RNG seed for reproducibility")
    parser.add_argument("--human-render", action="store_true",
        help="Show CartPole rendering during evaluation")
    args = parser.parse_args()

    if args.evaluate_only and not args.load:
        parser.error("--evaluate-only requires --load")
    if args.load and not args.evaluate_only:
        parser.error(
            "--load without --evaluate-only (resume training) is not supported here. "
            "Use legacy_main.py for that workflow."
        )

    if args.config is None:
        if args.load:
            print("No config specified; will use config saved in the model checkpoint.")
        else:
            args.config = "configs/baseline.yaml"
            print("No config specified, using default: configs/baseline.yaml")

    config = load_config(args.config) if args.config else None
    config_name = get_config_name(config_path=args.config, pretrained_file=args.load)

    if args.hw_acceleration and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.hw_acceleration and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Using config: {config_name}")

    # ── Evaluate-only path ─────────────────────────────────────────────────────
    if args.evaluate_only:
        if config is not None:
            snn_cfg = config["snn"]
            training_cfg = config["training"]
            hidden1_size = snn_cfg["hidden1_size"]
            hidden2_size = snn_cfg["hidden2_size"]
            alpha = snn_cfg.get("alpha", 0.5)
            lam = snn_cfg.get("lam", 0.111)
            history_length = snn_cfg.get("history_length", 256)
            dt = snn_cfg.get("dt", 1.0)
            num_steps = snn_cfg["num_steps"]
            beta = snn_cfg["beta"]
            neuron_type = snn_cfg["neuron_type"]
            shift_func_name = snn_cfg.get("shift_func", None)
            surrogate_gradient_slope = snn_cfg.get("surrogate_gradient_slope", 25)
            lr = training_cfg["lr"]
            eval_seed_base = training_cfg.get("eval_seed_base", 42)
            eval_seed_stride = training_cfg.get("eval_seed_stride", 7)
        else:
            hidden1_size = 64
            hidden2_size = 16
            alpha = 0.5
            lam = 0.111
            history_length = 64
            dt = 1.0
            num_steps = 30
            beta = 0.9
            neuron_type = "leaky"
            shift_func_name = None
            surrogate_gradient_slope = 25
            lr = 0.0003
            eval_seed_base = 42
            eval_seed_stride = 7

        print(f"Loading pre-trained model from {args.load}")
        checkpoint = torch.load(args.load, map_location=device, weights_only=False)
        checkpoint_config = checkpoint.get("config", {})

        # Architecture: checkpoint > config file > defaults
        net_hidden1_size = checkpoint_config.get("hidden1_size", hidden1_size)
        net_hidden2_size = checkpoint_config.get("hidden2_size", hidden2_size)
        net_alpha = checkpoint_config.get("alpha", alpha)
        net_lam = checkpoint_config.get("lam", lam)
        net_history_length = checkpoint_config.get("history_length", history_length)
        net_dt = checkpoint_config.get("dt", dt)
        net_num_steps = checkpoint_config.get("num_steps", num_steps)
        net_beta = checkpoint_config.get("beta", beta)
        net_neuron_type = checkpoint_config.get("neuron_type", neuron_type)
        net_shift_func_name = checkpoint_config.get("shift_func") or shift_func_name
        net_shift_func = resolve_shift_func(net_shift_func_name)
        spike_grad = surrogate.fast_sigmoid(slope=surrogate_gradient_slope)

        policy_net = SNNPolicy(
            N_OBSERVATIONS, N_ACTIONS,
            num_steps=net_num_steps, beta=net_beta,
            spike_grad=spike_grad, neuron_type=net_neuron_type,
            hidden1_size=net_hidden1_size, hidden2_size=net_hidden2_size,
            alpha=net_alpha, lam=net_lam,
            history_length=net_history_length, dt=net_dt,
            shift_func=net_shift_func,
        ).to(device)
        target_net = SNNPolicy(
            N_OBSERVATIONS, N_ACTIONS,
            num_steps=net_num_steps, beta=net_beta,
            spike_grad=spike_grad, neuron_type=net_neuron_type,
            hidden1_size=net_hidden1_size, hidden2_size=net_hidden2_size,
            alpha=net_alpha, lam=net_lam,
            history_length=net_history_length, dt=net_dt,
            shift_func=net_shift_func,
        ).to(device)
        optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True, weight_decay=0)

        agent = DQNAgent.load(
            args.load,
            policy_net, target_net, optimizer, ReplayMemory(1),
            device,
            config_overrides={
                "hidden1_size": net_hidden1_size,
                "hidden2_size": net_hidden2_size,
                "alpha": net_alpha,
                "lam": net_lam,
                "history_length": net_history_length,
                "dt": net_dt,
                "num_steps": net_num_steps,
                "beta": net_beta,
                "neuron_type": net_neuron_type,
                "shift_func": net_shift_func_name,
            },
        )

        print(
            f"Loaded model: neuron_type={agent.neuron_type}, "
            f"hidden_sizes=({agent.hidden1_size}, {agent.hidden2_size})"
        )
        if agent.neuron_type in ("fractional", "bitshift"):
            print(
                f"  lam={agent.lam}, history_length={agent.history_length}, dt={agent.dt}"
            )

        env = gym.make(
            "CartPole-v1",
            render_mode="human" if args.human_render else None,
            max_episode_steps=args.max_episode_steps,
        )
        eval_seeds = [eval_seed_base + i * eval_seed_stride for i in range(10)]
        print(f"Running 10 evaluation episodes with fixed seeds: {eval_seeds}")
        agent.evaluate(env, num_episodes=10, render=True, seeds=eval_seeds)
        env.close()
        print("Evaluation complete.")
        sys.exit(0)

    # ── Training path ──────────────────────────────────────────────────────────
    flat_config = {**config["training"], **config["snn"]}

    Path("metrics").mkdir(exist_ok=True)
    metrics_csv_path = (
        Path(args.metrics_csv)
        if args.metrics_csv
        else Path("metrics") / f"{config_name}-training-metrics.csv"
    )

    save_final = not args.no_save_final
    result = train(
        config=flat_config,
        device=str(device),
        verbose=True,
        save_models=save_final or args.save_best,
        save_final_model=save_final,
        save_best_model=args.save_best,
        model_prefix=config_name,
        seed=args.seed,
        metrics_csv_path=metrics_csv_path,
        max_episode_steps=args.max_episode_steps,
    )

    print("Training complete.")
    print(f"  final_avg_reward   : {result['final_avg_reward']:.2f}")
    print(f"  best_avg_reward    : {result['best_avg_reward']:.2f}")
    if result.get("convergence_episode") is not None:
        print(f"  convergence_episode: {result['convergence_episode']}")
    print(f"  metrics CSV        : {metrics_csv_path}")
