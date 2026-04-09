from datetime import datetime
import torch
import torch.nn as nn
from itertools import count
from typing import Optional, Tuple, List
from collections import namedtuple, deque
import random

# Transition and ReplayMemory are used for training
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """
    DQN agent using Spiking Neural Networks.

    This class encapsulates the training and evaluation logic for an SNN-based
    DQN agent. The neural networks, optimizer, and replay memory are provided
    as pre-initialized instances.
    """

    def __init__(
        self,
        policy_net: nn.Module,
        target_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        memory: ReplayMemory,
        n_observations: int,
        n_actions: int,
        num_steps: int,
        beta: float,
        neuron_type: str,
        device: torch.device,
        episode: int = 0,
        avg_reward: float = 0.0,
        # Optional SNN architecture parameters, with defaults
        hidden1_size: int = 64,
        hidden2_size: int = 64,
        # Optional fractional-order LIF parameters, with defaults
        alpha: float = 0.5,
        lam: float = 0.111,
        history_length: int = 64,
        dt: float = 1.0,
    ):
        """
        Initialize the agent.

        Args:
            policy_net: The policy network (already initialized)
            target_net: The target network (already initialized)
            optimizer: The optimizer (already initialized)
            memory: Replay memory buffer (already initialized)
            n_observations: Number of observation features
            n_actions: Number of possible actions
            num_steps: Number of timesteps for SNN simulation
            beta: LIF neuron decay parameter
            neuron_type: Type of spiking neuron ('leaky', 'leakysv', 'fractional')
            device: Torch device (CPU/CUDA/MPS)
            episode: Current episode number (default: 0)
            avg_reward: Running average reward (default: 0.0)
            hidden1_size: Size of first hidden layer (default: 64)
            hidden2_size: Size of second hidden layer (default: 64)
            alpha: Fractional order for FLIF neurons (default: 0.5)
            lam: Lambda parameter for FLIF neurons (default: 0.111)
            history_length: History buffer length for FLIF neurons (default: 64)
            dt: Time step for FLIF neurons (default: 1.0)
        """
        # Training hyperparameters
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.device = device

        # SNN and network architecture parameters
        self.num_steps = num_steps
        self.beta = beta
        self.neuron_type = neuron_type
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size

        # Fractional-order LIF parameters
        self.alpha = alpha
        self.lam = lam
        self.history_length = history_length
        self.dt = dt

        # Model components (pre-initialized instances)
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.memory = memory

        # Training state
        self.episode = episode
        self.avg_reward = avg_reward

    @classmethod
    def from_config(
        cls,
        config: dict,
        policy_net: nn.Module,
        target_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        memory: ReplayMemory,
    ):
        """
        Create an DQNAgent instance from a configuration dictionary.

        Args:
            config: Dictionary containing agent configuration
            policy_net: Pre-initialized policy network
            target_net: Pre-initialized target network
            optimizer: Pre-initialized optimizer
            memory: Pre-initialized replay memory

        Returns:
            DQNAgent instance
        """
        return cls(
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            memory=memory,
            n_observations=config["n_observations"],
            n_actions=config["n_actions"],
            num_steps=config["num_steps"],
            beta=config["beta"],
            neuron_type=config["neuron_type"],
            device=config["device"],
            episode=config.get("episode", 0),
            avg_reward=config.get("avg_reward", 0.0),
        )

    def save(self, filename: Optional[str] = None) -> str:
        """
        Save model checkpoint including networks, optimizer state, and training info.

        Args:
            filename: Path to save checkpoint. If None, generates timestamped filename.

        Returns:
            The filename where the checkpoint was saved
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snn_dqn_cartpole_{timestamp}.pth"

        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode": self.episode,
            "avg_reward": self.avg_reward,
            "config": {
                "n_observations": self.n_observations,
                "n_actions": self.n_actions,
                # SNN architecture parameters
                "num_steps": self.num_steps,
                "beta": self.beta,
                "neuron_type": self.neuron_type,
                "hidden1_size": self.hidden1_size,
                "hidden2_size": self.hidden2_size,
                # Fractional-order LIF parameters
                "alpha": self.alpha,
                "lam": self.lam,
                "history_length": self.history_length,
                "dt": self.dt,
            },
        }

        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")
        return filename

    @classmethod
    def load(
        cls,
        filename: str,
        policy_net: nn.Module,
        target_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        memory: ReplayMemory,
        device: torch.device,
        # Optional config overrides (for hybrid config loading approach)
        config_overrides: Optional[dict] = None,
    ):
        """
        Load a model checkpoint and return a DQNAgent instance.

        The provided policy_net, target_net, optimizer, and memory instances
        will have their states updated from the checkpoint.

        Hybrid approach: Checkpoint parameters are used by default, but can be
        overridden by config_overrides (with warnings printed).

        Args:
            filename: Path to checkpoint file
            policy_net: Policy network instance (state will be loaded)
            target_net: Target network instance (state will be loaded)
            optimizer: Optimizer instance (state will be loaded)
            memory: Replay memory instance (not loaded from checkpoint)
            device: Torch device
            config_overrides: Optional dict of parameters to override from checkpoint
                            (useful when loading with a different config file)

        Returns:
            DQNAgent instance with loaded state
        """
        checkpoint = torch.load(filename, map_location=device, weights_only=False)

        # Load network states
        policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        target_net.load_state_dict(checkpoint["target_net_state_dict"])

        # Load optimizer state if available (not present in quantized exports)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Start with checkpoint config (source of truth)
        config = checkpoint["config"].copy()

        # Apply overrides if provided (hybrid approach)
        if config_overrides:
            for key, new_value in config_overrides.items():
                if key in config and config[key] != new_value:
                    print(
                        f"WARNING: Overriding checkpoint parameter '{key}': {config[key]} -> {new_value}"
                    )
                config[key] = new_value

        # Create agent with merged config
        agent = cls(
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            memory=memory,
            n_observations=config["n_observations"],
            n_actions=config["n_actions"],
            num_steps=config["num_steps"],
            beta=config["beta"],
            neuron_type=config["neuron_type"],
            device=device,
            episode=checkpoint["episode"],
            avg_reward=checkpoint["avg_reward"],
            # SNN architecture and fractional params (with defaults for old checkpoints)
            hidden1_size=config.get("hidden1_size", 64),
            hidden2_size=config.get("hidden2_size", 64),
            alpha=config.get("alpha", 0.5),
            lam=config.get("lam", 0.111),
            history_length=config.get("history_length", 64),
            dt=config.get("dt", 1.0),
        )

        return agent

    def optimize(self, batch_size: int = 128, gamma: float = 0.99) -> Optional[float]:
        """
        Perform one optimization step using a batch from replay memory.

        Samples a batch from memory, computes Q-learning loss, and updates
        the policy network via gradient descent.

        Args:
            batch_size: Number of transitions to sample from memory
            gamma: Discount factor for future rewards

        Returns:
            Loss value if optimization occurred, None if not enough samples in memory
        """
        if len(self.memory) < batch_size:
            return None

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Mask for non-final next states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )

        # Concatenate next states where present
        non_final_next_states = (
            torch.cat([s for s in batch.next_state if s is not None], dim=0)
            if any(non_final_mask)
            else torch.empty((0, self.n_observations), device=self.device)
        )

        state_batch = torch.cat(batch.state, dim=0)
        action_batch = torch.cat(batch.action, dim=0)
        reward_batch = torch.cat(batch.reward, dim=0)

        # Compute Q(s_t, a) using the policy network (SNN forward does time simulation)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states using target_net
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            if non_final_next_states.numel() != 0:
                next_q = self.target_net(
                    non_final_next_states
                )  # returns [num_nonfinal, n_actions]
                next_state_values[non_final_mask] = next_q.max(1).values

        # Compute expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute loss (Huber/SmoothL1)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def evaluate(
        self,
        env,
        num_episodes: int = 10,
        render: bool = False,
        seeds: Optional[List[int]] = None,
    ) -> Tuple[List[float], float]:
        """
        Evaluate the agent's current policy without training.

        Runs the agent greedily (no exploration) for a number of episodes
        and returns performance metrics.

        Args:
            env: Gymnasium environment
            num_episodes: Number of episodes to evaluate
            render: Whether to print per-episode results
            seeds: Optional list of per-episode seeds for deterministic evaluation.
                If provided, its length should be >= num_episodes.

        Returns:
            Tuple of (episode_rewards, average_reward)
            - episode_rewards: List of total rewards for each episode
            - average_reward: Mean reward across all episodes
        """
        self.policy_net.eval()  # Set to evaluation mode
        total_rewards = []

        with torch.no_grad():
            for episode in range(num_episodes):
                if seeds is not None:
                    state, info = env.reset(seed=seeds[episode])
                else:
                    state, info = env.reset()
                state = torch.tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                total_reward = 0

                for t in count():
                    # Select action greedily (no exploration)
                    action = self.policy_net(state).max(1).indices.view(1, 1)
                    observation, reward, terminated, truncated, _ = env.step(
                        action.item()
                    )
                    total_reward += reward
                    done = terminated or truncated

                    if not done:
                        next_state = torch.tensor(
                            observation, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        state = next_state
                    else:
                        break

                total_rewards.append(total_reward)
                if render:
                    if seeds is not None:
                        print(
                            f"Episode {episode + 1} (seed={seeds[episode]}): {total_reward} steps"
                        )
                    else:
                        print(f"Episode {episode + 1}: {total_reward} steps")

        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")

        # Return to training mode
        self.policy_net.train()

        return total_rewards, avg_reward

    # -----------------------
    # Future methods (stubs)
    # This functionality is currently handled in the main loop in main.py. We may want to move
    # that logic here in the future for better encapsulation.
    # -----------------------

    def update_target_network(self):
        """
        Update target network to match policy network.

        TODO: Implement target network synchronization.
        Typically called every N episodes to stabilize training.
        """
        pass

    def train_episode(
        self, env, epsilon: float, batch_size: int = 128, gamma: float = 0.99
    ):
        """
        Run a single training episode.

        TODO: Implement full episode training loop including:
        - Action selection with epsilon-greedy
        - Environment interaction
        - Memory storage
        - Optimization steps

        Args:
            env: Gymnasium environment
            epsilon: Exploration rate for epsilon-greedy policy
            batch_size: Batch size for optimization
            gamma: Discount factor

        Returns:
            Episode statistics (reward, steps, etc.)
        """
        pass

    def select_action(self, state: torch.Tensor, epsilon: Optional[float] = None):
        """
        Select an action using epsilon-greedy policy.

        TODO: Implement action selection with exploration.
        If epsilon is None, select greedily (for evaluation).

        Args:
            state: Current state tensor
            epsilon: Exploration rate. If None, act greedily.

        Returns:
            Selected action tensor
        """
        pass
