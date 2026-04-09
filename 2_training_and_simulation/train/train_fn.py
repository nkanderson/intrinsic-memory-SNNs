"""
Reusable training function for SNN-DQN on CartPole.

This module extracts the core training loop from main.py into a callable
function, allowing both main.py (for manual training) and optimize.py
(for Optuna hyperparameter search) to share the same training logic.

Usage:
    from train_fn import train

    result = train(config_dict, device="cpu", verbose=False)
    print(result["best_avg_reward"])
"""

import math
import random
from itertools import count
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

# CartPole constants
N_ACTIONS = 2
N_OBSERVATIONS = 4

# Map shift function names to actual functions
SHIFT_FUNC_MAP = {
    "simple": simple_bitshift,
    "slow_decay": slow_decay_bitshift,
    "custom": custom_bitshift,
    "custom_slow_decay": custom_slow_decay_bitshift,
}


def _select_action(
    state, steps_done, policy_net, device, eps_start, eps_end, eps_decay
):
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
            return q_values.max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[random.randint(0, N_ACTIONS - 1)]], device=device, dtype=torch.long
        )


def train(
    config: dict,
    device: str = "cpu",
    verbose: bool = True,
    save_models: bool = False,
    model_prefix: str = "optuna",
    optuna_trial=None,
) -> dict:
    """
    Run a full SNN-DQN training session on CartPole and return metrics.

    Args:
        config: Flat dictionary of all hyperparameters. Expected keys:
            Training: batch_size, gamma, eps_start, eps_end, eps_decay, tau, lr, num_episodes
            SNN: num_steps, beta, surrogate_gradient_slope, neuron_type,
                 hidden1_size, hidden2_size
            Fractional (optional): alpha, lam, history_length, dt
            Bitshift (optional): shift_func (string name)
        device: Torch device string ("cpu", "cuda", "mps").
        verbose: If True, print progress during training.
        save_models: If True, save best/final model checkpoints.
        model_prefix: Prefix for saved model filenames.
        optuna_trial: If provided, an optuna.Trial object for pruning support.
            Reports intermediate values (100-episode avg reward) for Optuna's
            pruner to decide whether to stop unpromising trials early.

    Returns:
        Dictionary with training metrics:
            - best_avg_reward: Best 100-episode rolling average reward
            - final_avg_reward: Final 100-episode average reward
            - episode_durations: List of all episode durations
            - total_episodes: Number of episodes completed
    """
    device = torch.device(device)

    # ── Extract hyperparameters from flat config ──
    batch_size = config["batch_size"]
    gamma = config["gamma"]
    eps_start = config["eps_start"]
    eps_end = config["eps_end"]
    eps_decay = config["eps_decay"]
    tau = config["tau"]
    lr = config["lr"]
    num_episodes = config["num_episodes"]

    num_steps = config["num_steps"]
    beta = config["beta"]
    surrogate_gradient_slope = config.get("surrogate_gradient_slope", 25)
    neuron_type = config["neuron_type"]
    hidden1_size = config["hidden1_size"]
    hidden2_size = config["hidden2_size"]

    # Fractional / bitshift params (with defaults)
    alpha = config.get("alpha", 0.5)
    lam = config.get("lam", 0.111)
    history_length = config.get("history_length", 64)
    dt = config.get("dt", 1.0)

    # Bitshift shift function
    shift_func_name = config.get("shift_func", None)
    shift_func = SHIFT_FUNC_MAP.get(shift_func_name) if shift_func_name else None
    if neuron_type == "bitshift" and shift_func is None:
        raise ValueError(
            "shift_func must be provided for bitshift neuron type. "
            f"Valid options: {list(SHIFT_FUNC_MAP.keys())}"
        )

    spike_grad = surrogate.fast_sigmoid(slope=surrogate_gradient_slope)

    # ── Create environment, networks, optimizer, memory ──
    env = gym.make("CartPole-v1")
    memory = ReplayMemory(10000)

    policy_net = SNNPolicy(
        N_OBSERVATIONS,
        N_ACTIONS,
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
        N_OBSERVATIONS,
        N_ACTIONS,
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

    # Initialize target network from policy network for fresh start
    target_net.load_state_dict(policy_net.state_dict())
    # Create optimizer
    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)

    # Create fresh agent instance
    agent = DQNAgent(
        policy_net=policy_net,
        target_net=target_net,
        optimizer=optimizer,
        memory=memory,
        n_observations=N_OBSERVATIONS,
        n_actions=N_ACTIONS,
        num_steps=num_steps,
        beta=beta,
        neuron_type=neuron_type,
        device=device,
        hidden1_size=hidden1_size,
        hidden2_size=hidden2_size,
        alpha=alpha,
        lam=lam,
        history_length=history_length,
        dt=dt,
    )

    # ── Training loop ──
    episode_durations = []
    steps_done = 0
    best_avg_reward = 0.0
    best_model_file = None

    if save_models:
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

    for i_episode in range(num_episodes):
        state, _info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = _select_action(
                state, steps_done, policy_net, device, eps_start, eps_end, eps_decay
            )
            steps_done += 1
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward_t = torch.tensor([reward], device=device, dtype=torch.float32)
            done = terminated or truncated

            next_state = (
                None
                if terminated
                else torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
            )

            # store transition (state tensors already on device)
            memory.push(state, action, next_state, reward_t)
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
                episode_durations.append(t + 1)
                break

        # ── Track best rolling average ──
        if len(episode_durations) >= 100:
            recent_avg = sum(episode_durations[-100:]) / 100
            if recent_avg > best_avg_reward:
                best_avg_reward = recent_avg
                if save_models:
                    agent.episode = i_episode
                    agent.avg_reward = recent_avg
                    fname = str(Path("models") / f"dqn_{model_prefix}-best.pth")
                    agent.save(fname)
                    best_model_file = fname

        # ── Optuna pruning support ──
        # Report the 100-episode rolling average (or raw duration for early episodes)
        # so the pruner can stop unpromising trials.
        if optuna_trial is not None:
            if len(episode_durations) >= 100:
                report_value = sum(episode_durations[-100:]) / 100
            else:
                report_value = sum(episode_durations) / len(episode_durations)
            optuna_trial.report(report_value, i_episode)
            if optuna_trial.should_prune():
                env.close()
                import optuna

                raise optuna.TrialPruned()

        # ── Periodic logging ──
        if verbose and (i_episode + 1) % 50 == 0:
            avg = sum(episode_durations[-min(len(episode_durations), 100) :]) / min(
                len(episode_durations), 100
            )
            print(f"  Episode {i_episode + 1}/{num_episodes}  avg(100): {avg:.1f}")

    env.close()

    # ── Compute final metrics ──
    n = min(len(episode_durations), 100)
    final_avg = sum(episode_durations[-n:]) / n if n > 0 else 0.0

    if save_models:
        agent.episode = num_episodes - 1
        agent.avg_reward = final_avg
        fname = str(Path("models") / f"dqn_{model_prefix}-final.pth")
        agent.save(fname)

    return {
        "best_avg_reward": best_avg_reward,
        "final_avg_reward": final_avg,
        "episode_durations": episode_durations,
        "total_episodes": num_episodes,
        "best_model_file": best_model_file,
    }
