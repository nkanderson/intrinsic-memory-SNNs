import argparse
import csv
import math
import os
import yaml
from pathlib import Path
import gymnasium as gym
import torch
import torch.optim as optim
from snntorch import surrogate
from snn_policy import SNNPolicy
from dqn_agent import DQNAgent, ReplayMemory
from scripts.history_coefficients import (
    simple_bitshift,
    slow_decay_bitshift,
    custom_bitshift,
    custom_slow_decay_bitshift,
)
import matplotlib

# Use non-interactive backend if no display is available (e.g., headless/tmux)
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import count
import random

# TODO: Get this from the env.action_space.n, and len(state) instead of hardcoding
# Network dimensions
n_actions = 2  # CartPole has 2 actions: left (0) and right (1)
n_observations = (
    4  # CartPole has 4 observations: [position, velocity, angle, angular_velocity]
)

SHIFT_FUNC_MAP = {
    "simple": simple_bitshift,
    "slow_decay": slow_decay_bitshift,
    "custom": custom_bitshift,
    "custom_slow_decay": custom_slow_decay_bitshift,
}


def resolve_shift_func(shift_func_name):
    if not shift_func_name:
        return None
    shift_func = SHIFT_FUNC_MAP.get(shift_func_name)
    if shift_func is None:
        raise ValueError(
            f"Unknown shift_func: {shift_func_name}. "
            f"Valid options: {list(SHIFT_FUNC_MAP.keys())}"
        )
    return shift_func


def load_config(config_path):
    """Load hyperparameters from YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_config_name(config_path=None, pretrained_file=None):
    """
    Determine config name for model filenames.

    Args:
        config_path: Path to config file (if provided)
        pretrained_file: Path to pretrained model file (if loading)

    Returns:
        String to use as config name for model filenames

    Raises:
        ValueError: If both config_path and pretrained_file are None
    """
    if config_path is not None:
        # Use config file basename
        return Path(config_path).stem
    elif pretrained_file is not None:
        # Derive from loaded model filename
        import re

        loaded_name = Path(pretrained_file).stem
        # Remove common suffixes like -best, -final, -quantized, etc
        for suffix in ["-best", "-final", "-quantized"]:
            if loaded_name.endswith(suffix):
                loaded_name = loaded_name[: -len(suffix)]
        # Also remove quantization format suffixes like -QS2_5
        loaded_name = re.sub(r"-Q[A-Z]?\d+_\d+$", "", loaded_name)
        return loaded_name
    else:
        # Both parameters are None - this is a programming error
        raise ValueError(
            "get_config_name() requires either config_path or pretrained_file to be provided"
        )


# TODO: Consider moving this to a method of the DQNAgent class
def select_action(state, steps_done, policy_net, device, eps_start, eps_end, eps_decay):
    """
    Select an action using epsilon-greedy policy.

    Args:
        state: tensor shape [1, n_observations] on device
        steps_done: number of steps taken so far
        policy_net: policy network for action selection
        device: torch device
        eps_start: starting epsilon value
        eps_end: minimum epsilon value
        eps_decay: epsilon decay rate

    Returns:
        action tensor shape [1,1]
    """
    sample = random.random()
    # Exponential decay of epsilon
    # math.exp(-1.0 * steps_done / eps_decay) is the decay factor, which starts
    # at 1.0 when steps_done is 0, and approaches 0 as steps_done -> infinity.
    # This value should decay from eps_start to eps_end following an exponential curve.
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(
        -1.0 * steps_done / eps_decay
    )
    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)  # forwards SNN, returns [1, n_actions]
            action = q_values.max(1).indices.view(1, 1)
            return action
    else:
        return torch.tensor(
            [[random.randint(0, n_actions - 1)]], device=device, dtype=torch.long
        )


# TODO: Consider creating a plotting util module
def plot_durations(episode_durations, show_result=False):
    """
    Plot episode durations and running average.

    Args:
        episode_durations: List of episode durations (number of steps per episode)
        show_result: If True, displays final result. If False, shows live training progress.
    """
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("SNN Training Results")
    else:
        plt.clf()
        plt.title("In-progress SNN Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())

    # Plot 100-episode moving average
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # For live updates: brief pause to update the plot
    # For final results: this ensures the plot is rendered before plt.show()
    plt.pause(0.001)


# -----------------------
# Optionally load a model, then train
# -----------------------
# TODO: Consider quantization-aware training to gain greater accuracy with
# lower bit-width weights. Currently, 8 bits does not appear to be sufficient
# for parity in performance between full-precision and quantized models.
if __name__ == "__main__":
    #
    # Section 0: Parse command line arguments and load configuration
    #
    parser = argparse.ArgumentParser(
        description="Train or evaluate an SNN-based DQN agent on CartPole"
    )

    # Configuration
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: configs/baseline.yaml if not loading a model, otherwise use model's saved config)",
    )

    # Training/loading options
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        metavar="FILE",
        help="Load pre-trained model from FILE",
    )

    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate the loaded model, do not train",
    )

    # Hardware and rendering
    parser.add_argument(
        "--no-hw-acceleration",
        dest="hw_acceleration",
        action="store_false",
        help="Disable hardware acceleration (CUDA/MPS)",
    )
    parser.set_defaults(hw_acceleration=True)

    parser.add_argument(
        "--human-render",
        action="store_true",
        help="Show environment rendering and live training plot",
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=500,
        help="Maximum steps per CartPole episode (default: 500)",
    )

    parser.add_argument(
        "--save-plot-image",
        action="store_true",
        help="Save final training plot image (default: disabled; metrics CSV is always saved)",
    )

    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="Optional path for training metrics CSV output",
    )

    parser.add_argument(
        "--save-final",
        action="store_true",
        help="Save final model checkpoint at end of training (default: disabled)",
    )

    args = parser.parse_args()

    # Determine config file to use
    # If loading a model and no config specified, we'll use the model's saved config
    # If not loading a model and no config specified, use default baseline config
    if args.config is None:
        if args.load is None:
            # Training from scratch, need a config file
            args.config = "configs/baseline.yaml"
            print("No config specified, using default: configs/baseline.yaml")
        else:
            # Loading a model, will use its saved config (no config file needed)
            print("No config specified, will use config from loaded model")

    # Load configuration file (if we have one)
    config = None
    if args.config is not None:
        config = load_config(args.config)

    # Determine config name for model filenames (works with or without config file)
    config_name = get_config_name(config_path=args.config, pretrained_file=args.load)

    # Extract hyperparameters from config (with defaults for when loading from model)
    if config is not None:
        batch_size = config["training"]["batch_size"]
        gamma = config["training"]["gamma"]
        eps_start = config["training"]["eps_start"]
        eps_end = config["training"]["eps_end"]
        eps_decay = config["training"]["eps_decay"]
        tau = config["training"]["tau"]
        lr = config["training"]["lr"]
        num_episodes = config["training"]["num_episodes"]
        eval_every_episodes = config["training"].get("eval_every_episodes", 0)
        eval_num_episodes = config["training"].get("eval_num_episodes", 10)
        eval_seed_base = config["training"].get("eval_seed_base", 42)
        eval_seed_stride = config["training"].get("eval_seed_stride", 7)

        num_steps = config["snn"]["num_steps"]
        beta = config["snn"]["beta"]
        surrogate_gradient_slope = config["snn"]["surrogate_gradient_slope"]
        neuron_type = config["snn"]["neuron_type"]
        hidden1_size = config["snn"]["hidden1_size"]
        hidden2_size = config["snn"]["hidden2_size"]

        # Fractional-order LIF parameters (optional, used only if neuron_type == "fractional")
        alpha = config["snn"].get("alpha", 0.5)
        lam = config["snn"].get("lam", 0.111)
        history_length = config["snn"].get("history_length", 256)
        dt = config["snn"].get("dt", 1.0)

        # BitshiftLIF parameter (optional, used only if neuron_type == "bitshift")
        shift_func_name = config["snn"].get("shift_func", None)
        shift_func = resolve_shift_func(shift_func_name)
    else:
        # No config file - will load everything from model checkpoint
        # Set placeholder defaults (will be overridden by checkpoint values)
        batch_size = 128
        gamma = 0.99
        eps_start = 0.9
        eps_end = 0.01
        eps_decay = 2500
        tau = 0.005
        lr = 0.0003
        num_episodes = 600
        eval_every_episodes = 0
        eval_num_episodes = 10
        eval_seed_base = 42
        eval_seed_stride = 7

        num_steps = 30
        beta = 0.9
        surrogate_gradient_slope = 25
        neuron_type = "leaky"
        hidden1_size = 64
        hidden2_size = 16
        alpha = 0.5
        lam = 0.111
        history_length = 64
        dt = 1.0
        shift_func_name = None
        shift_func = None  # No shift_func in placeholder defaults

    # Apply other parsed arguments
    pretrained_file = args.load
    evaluate_only = args.evaluate_only
    hw_acceleration = args.hw_acceleration
    human_render = args.human_render
    max_episode_steps = args.max_episode_steps
    save_plot_image = args.save_plot_image
    metrics_csv_arg = args.metrics_csv
    save_final = args.save_final

    # Create surrogate gradient function
    spike_grad = surrogate.fast_sigmoid(slope=surrogate_gradient_slope)

    #
    # Section 1: Initialize plotting, device, default params, and basic training variables
    # like steps_done and episode_durations
    #
    if human_render:
        plt.ion()  # Enable interactive plotting for live updates

    # Set device based on hw_acceleration flag and availability
    if hw_acceleration and torch.cuda.is_available():
        device = torch.device("cuda")
    elif hw_acceleration and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Using config: {config_name}")

    episode_durations = []
    steps_done = 0

    # Training parameters
    best_avg_reward = 0
    best_eval_avg_reward = 0

    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Generate model filenames based on config name
    best_model_filename = str(models_dir / f"{config_name}-best.pth")
    final_model_filename = str(models_dir / f"{config_name}-final.pth")

    # Metrics logging path (CSV)
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)
    metrics_csv_path = (
        Path(metrics_csv_arg)
        if metrics_csv_arg is not None
        else metrics_dir / f"{config_name}-training-metrics.csv"
    )

    #
    # Section 2: Initialize replay memory and create the CartPole environment
    #
    memory = ReplayMemory(10000)
    env = gym.make(
        "CartPole-v1",
        render_mode="human" if human_render else None,
        max_episode_steps=max_episode_steps,
    )

    #
    # Section 3: Load networks and optimizer
    # If loading a pre-trained model, load it and optionally evaluate then exit.
    # Otherwise, configure target_net from policy_net.
    #
    start_episode = 0

    if pretrained_file:
        # Load pre-trained model using DQNAgent.load()
        print(f"Loading pre-trained model from {pretrained_file}")

        # Load checkpoint to inspect saved config
        checkpoint = torch.load(
            pretrained_file, map_location=device, weights_only=False
        )
        checkpoint_config = checkpoint.get("config", {})

        # Determine values for network creation
        # Precedence: checkpoint config > config file > placeholder defaults
        # The following gets the values from the checkpoint if they exist,
        # otherwise fall back to the defaults defined earlier. If config is not None,
        # the defaults will be from the config file.
        net_hidden1_size = checkpoint_config.get("hidden1_size", hidden1_size)
        net_hidden2_size = checkpoint_config.get("hidden2_size", hidden2_size)
        net_alpha = checkpoint_config.get("alpha", alpha)
        net_lam = checkpoint_config.get("lam", lam)
        net_history_length = checkpoint_config.get("history_length", history_length)
        net_dt = checkpoint_config.get("dt", dt)
        net_num_steps = checkpoint_config.get("num_steps", num_steps)
        net_beta = checkpoint_config.get("beta", beta)
        net_neuron_type = checkpoint_config.get("neuron_type", neuron_type)
        net_shift_func_name = checkpoint_config.get("shift_func", shift_func_name)
        net_shift_func = resolve_shift_func(net_shift_func_name)

        # Set config_overrides for loading the agent. This is only necessary for
        # older models which did not have all parameters saved in the checkpoint.
        # These have the precedence: checkpoint > config file > placeholder
        config_overrides = {
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
        }

        # Create policy and target nets with precedence: checkpoint > config file > placeholder
        policy_net = SNNPolicy(
            n_observations,
            n_actions,
            num_steps=net_num_steps,
            beta=net_beta,
            spike_grad=spike_grad,
            neuron_type=net_neuron_type,
            hidden1_size=net_hidden1_size,
            hidden2_size=net_hidden2_size,
            alpha=net_alpha,
            lam=net_lam,
            history_length=net_history_length,
            dt=net_dt,
            shift_func=net_shift_func,
        ).to(device)
        target_net = SNNPolicy(
            n_observations,
            n_actions,
            num_steps=net_num_steps,
            beta=net_beta,
            spike_grad=spike_grad,
            neuron_type=net_neuron_type,
            hidden1_size=net_hidden1_size,
            hidden2_size=net_hidden2_size,
            alpha=net_alpha,
            lam=net_lam,
            history_length=net_history_length,
            dt=net_dt,
            shift_func=net_shift_func,
        ).to(device)
        optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)

        # Load agent with optional config overrides (hybrid approach)
        agent = DQNAgent.load(
            pretrained_file,
            policy_net,
            target_net,
            optimizer,
            memory,
            device,
            config_overrides=config_overrides if config_overrides else None,
        )

        # Extract components from agent
        policy_net = agent.policy_net
        target_net = agent.target_net
        optimizer = agent.optimizer
        start_episode = agent.episode
        prev_avg = agent.avg_reward

        # If evaluate-only mode, run evaluation and exit
        if evaluate_only:
            print(
                f"Loaded mode for evaluation: neuron_type={agent.neuron_type}, "
                f"hidden_sizes=({agent.hidden1_size}, {agent.hidden2_size})"
            )
            if agent.neuron_type == "fractional":
                print(
                    f"  Fractional parameters: alpha={agent.alpha}, lam={agent.lam}, "
                    f"history_length={agent.history_length}, dt={agent.dt}"
                )
            print(f"Resuming from episode {start_episode} (prev avg: {prev_avg:.2f})")
            eval_seeds = [eval_seed_base + i * eval_seed_stride for i in range(10)]
            print(f"Running evaluation for 10 episodes with fixed seeds: {eval_seeds}")
            episode_rewards, avg_reward = agent.evaluate(
                env,
                num_episodes=10,
                render=True,
                seeds=eval_seeds,
            )
            print("Evaluation complete. Exiting.")
            env.close()
            exit(0)
    else:
        # Create network instances
        policy_net = SNNPolicy(
            n_observations,
            n_actions,
            num_steps=num_steps,
            beta=beta,
            spike_grad=spike_grad,
            neuron_type=neuron_type,
            hidden1_size=hidden1_size,
            hidden2_size=hidden2_size,
            alpha=alpha,
            lam=lam,
            history_length=history_length,
            dt=dt,
            shift_func=shift_func,
        ).to(device)
        target_net = SNNPolicy(
            n_observations,
            n_actions,
            num_steps=num_steps,
            beta=beta,
            spike_grad=spike_grad,
            neuron_type=neuron_type,
            hidden1_size=hidden1_size,
            hidden2_size=hidden2_size,
            alpha=alpha,
            lam=lam,
            history_length=history_length,
            dt=dt,
            shift_func=shift_func,
        ).to(device)

        # Create optimizer
        optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
        # Initialize target network from policy network for fresh start
        target_net.load_state_dict(policy_net.state_dict())

        # Create fresh agent instance
        agent = DQNAgent(
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            memory=memory,
            n_observations=n_observations,
            n_actions=n_actions,
            num_steps=num_steps,
            beta=beta,
            neuron_type=neuron_type,
            device=device,
            episode=0,
            avg_reward=0.0,
            # SNN architecture parameters
            hidden1_size=hidden1_size,
            hidden2_size=hidden2_size,
            # Fractional-order LIF parameters
            alpha=alpha,
            lam=lam,
            history_length=history_length,
            dt=dt,
            shift_func_name=shift_func_name,
        )

    # Print configuration info after agent is created
    print(f"Using neuron type: {agent.neuron_type}")
    if agent.neuron_type == "fractional":
        print(
            f"  Fractional parameters: alpha={agent.alpha}, lam={agent.lam}, "
            f"history_length={agent.history_length}, dt={agent.dt}"
        )
    print(f"Training for {num_episodes} episodes")

    # Initialize per-episode metrics CSV (overwrite for a fresh run)
    with open(metrics_csv_path, "w", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "episode",
                "episode_steps",
                "episode_reward",
                "epsilon",
                "running_avg_100",
                "generalization_avg",
                "generalization_seeds",
                "generalization_rewards",
                "best_running_avg_100",
                "best_generalization_avg",
                "saved_best_running_model",
                "saved_best_generalization_model",
            ],
        )
        writer.writeheader()

    last_generalization_avg = None

    #
    # Section 4: Main training loop
    #
    for i_episode in range(start_episode, start_episode + num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(
            0
        )  # [1, obs]

        for t in count():
            action = select_action(
                state, steps_done, policy_net, device, eps_start, eps_end, eps_decay
            )
            steps_done += 1
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # store transition (state tensors already on device)
            memory.push(state, action, next_state, reward)

            # move to next state
            state = next_state

            # optimization step (on the policy network)
            agent.optimize(batch_size=batch_size, gamma=gamma)

            # Soft update target network: θ' <- τ θ + (1 − τ) θ'
            with torch.no_grad():
                for target_param, policy_param in zip(
                    target_net.parameters(), policy_net.parameters()
                ):
                    target_param.mul_(1.0 - tau).add_(policy_param, alpha=tau)

                # Soft-update buffers (e.g. snnTorch membrane potentials, synaptic currents)
                # The following may be necessary if any neuron implementations used in the SNNPolicy
                # have buffers that track state (e.g. membrane potentials) across time steps.
                # Right now, the custom neurons use `persistent=False` so those buffers are excluded
                # from `state_dict()`. If new implementations use `persistent=True` for any buffers,
                # then this manual buffer update may be necessary to ensure the target network's internal
                # state stays in sync with the policy network.
                # snnTorch stock neurons such as snn.Leaky register their state (e.g. membrane potential)
                # via `init_hidden()` as plain tensor attributes, not as `register_buffer()` calls, so they
                # won't be included in either `.parameters()` or `buffers()`. Their state was not included
                # in the prior implementation either using a full copy-in with intermediate tensors.
                # for (target_buf_name, target_buf), (_, policy_buf) in zip(
                #     target_net.named_buffers(), policy_net.named_buffers()
                # ):
                #     if target_buf.dtype.is_floating_point:
                #         target_buf.mul_(1.0 - tau).add_(policy_buf, alpha=tau)
                #     else:
                #         target_buf.copy_(policy_buf)

            if done:
                episode_steps = t + 1
                episode_reward = float(episode_steps)
                eps_threshold = eps_end + (eps_start - eps_end) * math.exp(
                    -1.0 * steps_done / eps_decay
                )
                episode_durations.append(t + 1)
                if human_render:
                    plot_durations(
                        episode_durations
                    )  # Only update plot live if human_render is enabled

                # Check if this is the best model so far (save only when we beat the record)
                recent_avg = None
                saved_best_running_model = False
                if len(episode_durations) >= 100:
                    recent_avg = sum(episode_durations[-100:]) / 100
                    if recent_avg > best_avg_reward:
                        best_avg_reward = recent_avg
                        # Update agent's episode and avg_reward before saving
                        agent.episode = i_episode
                        agent.avg_reward = recent_avg
                        agent.save(best_model_filename)
                        print(f"New best model saved! Avg reward: {recent_avg:.2f}")
                        saved_best_running_model = True

                # Optional fixed-seed evaluation for generalization-based model selection
                saved_best_generalization_model = False
                eval_seeds = []
                eval_rewards = []
                current_episode = i_episode - start_episode + 1
                if eval_every_episodes > 0 and (
                    current_episode % eval_every_episodes == 0
                ):
                    eval_seeds = [
                        eval_seed_base + i * eval_seed_stride
                        for i in range(eval_num_episodes)
                    ]
                    eval_rewards, eval_avg = agent.evaluate(
                        env,
                        num_episodes=eval_num_episodes,
                        render=False,
                        seeds=eval_seeds,
                    )
                    last_generalization_avg = eval_avg
                    print(
                        f"Generalization eval @ episode {i_episode}: avg={eval_avg:.2f} on seeds={eval_seeds}"
                    )

                    if eval_avg > best_eval_avg_reward:
                        best_eval_avg_reward = eval_avg
                        agent.episode = i_episode
                        agent.avg_reward = eval_avg
                        generalization_best_model = (
                            models_dir / f"{config_name}-best-generalization.pth"
                        )
                        agent.save(generalization_best_model)
                        print(
                            f"New best generalization model saved! Avg reward: {eval_avg:.2f}"
                        )
                        saved_best_generalization_model = True

                # Append one row of metrics per training episode
                with open(metrics_csv_path, "a", newline="") as metrics_file:
                    writer = csv.DictWriter(
                        metrics_file,
                        fieldnames=[
                            "episode",
                            "episode_steps",
                            "episode_reward",
                            "epsilon",
                            "running_avg_100",
                            "generalization_avg",
                            "generalization_seeds",
                            "generalization_rewards",
                            "best_running_avg_100",
                            "best_generalization_avg",
                            "saved_best_running_model",
                            "saved_best_generalization_model",
                        ],
                    )
                    writer.writerow(
                        {
                            "episode": i_episode,
                            "episode_steps": episode_steps,
                            "episode_reward": episode_reward,
                            "epsilon": eps_threshold,
                            "running_avg_100": recent_avg,
                            "generalization_avg": last_generalization_avg,
                            "generalization_seeds": (
                                "|".join(map(str, eval_seeds)) if eval_seeds else ""
                            ),
                            "generalization_rewards": (
                                "|".join(map(str, eval_rewards)) if eval_rewards else ""
                            ),
                            "best_running_avg_100": best_avg_reward,
                            "best_generalization_avg": best_eval_avg_reward,
                            "saved_best_running_model": int(saved_best_running_model),
                            "saved_best_generalization_model": int(
                                saved_best_generalization_model
                            ),
                        }
                    )

                break

    #
    # Section 5: Save final model
    #
    final_avg = sum(episode_durations[-min(len(episode_durations), 100) :]) / min(
        len(episode_durations), 100
    )

    print("Complete")
    if save_final:
        # Update agent's episode and avg_reward before saving
        agent.episode = start_episode + num_episodes - 1
        agent.avg_reward = final_avg
        final_model_file = agent.save(final_model_filename)
        print(f"Final model saved to: {final_model_file}")

    if best_avg_reward > 0:
        print(
            f"Best model saved to: {best_model_filename} (avg reward: {best_avg_reward:.2f})"
        )
    if best_eval_avg_reward > 0:
        print(f"Best generalization model avg reward: {best_eval_avg_reward:.2f}")

    print(f"Training metrics CSV saved to: {metrics_csv_path}")

    if save_plot_image:
        if human_render:
            plt.ioff()  # Turn off interactive mode
        plot_durations(episode_durations, show_result=True)

        plot_filename = f"images/{config_name}.png"
        Path("images").mkdir(exist_ok=True)
        plt.savefig(plot_filename)
        print(f"Training plot saved to: {plot_filename}")

        if os.environ.get("DISPLAY") or human_render:
            plt.show()
