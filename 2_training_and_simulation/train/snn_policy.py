import torch
import torch.nn as nn
import snntorch as snn
from leakysv import LeakySV
from fractional_lif import FractionalLIF
from bitshift_lif import BitshiftLIF


class SNNPolicy(nn.Module):
    """
    Spiking Neural Network policy for Deep Q-Network (DQN) reinforcement learning.

    This network uses spiking neurons to process observations and output Q-values
    for each possible action. The SNN simulates neural dynamics over multiple
    timesteps, with spikes accumulated to produce final Q-value estimates.

    Architecture:
        - Input layer: Linear transformation of observations to hidden1_size features
        - Hidden layer 1: hidden1_size spiking neurons (LIF)
        - Hidden layer 2: hidden2_size spiking neurons (LIF)
        - Output layer: Linear transformation to Q-values (one per action)

    The network processes the same input observation for `num_steps` timesteps,
    accumulating Q-values across time and averaging them to produce the final
    action-value estimates.
    """

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        num_steps=30,
        beta=0.9,
        spike_grad=None,
        neuron_type="leaky",
        hidden1_size=128,
        hidden2_size=128,
        # Fractional-order LIF specific parameters
        alpha=0.5,
        lam=0.111,
        history_length=256,
        dt=1.0,
        # BitshiftLIF specific parameter
        shift_func=None,
    ):
        """
        - num_steps: number of timesteps to simulate the SNN per environment step
        - beta: membrane decay for LIF (leaky neurons only)
        - spike_grad: surrogate gradient function from snntorch.surrogate (optional)
        - neuron_type: "leakysv", "leaky", "fractional", or "bitshift" - type of spiking neuron to use
        - hidden1_size: number of neurons in first hidden layer (default: 128)
        - hidden2_size: number of neurons in second hidden layer (default: 128)
        - alpha: fractional order for derivative (fractional neurons only, default: 0.5)
        - lam: leakage parameter in fractional equation (fractional/bitshift neurons, default: 0.111)
        - history_length: number of past values for GL/bitshift approximation
                         (fractional/bitshift neurons, default: 256)
        - dt: discrete timestep for approximation (fractional/bitshift neurons, default: 1.0)
        - shift_func: shift amount function for bitshift neurons. Pass the actual function object
                     (e.g., simple_bitshift) not a string name
                     (bitshift neurons only, required if neuron_type='bitshift')
        """
        super().__init__()
        self.num_steps = num_steps
        self.n_actions = n_actions
        self.neuron_type = neuron_type
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size

        # feedforward linear layers
        self.fc1 = nn.Linear(n_observations, hidden1_size)

        # Create neurons based on type
        if neuron_type == "leaky":
            self.lif1 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
            self.lif2 = snn.Leaky(
                beta=beta, init_hidden=True, spike_grad=spike_grad, output=True
            )
        elif neuron_type == "fractional":
            self.lif1 = FractionalLIF(
                beta=beta,  # For compatibility, not used
                alpha=alpha,
                lam=lam,
                history_length=history_length,
                dt=dt,
                spike_grad=spike_grad,
                init_hidden=True,
            )
            self.lif2 = FractionalLIF(
                beta=beta,  # For compatibility, not used
                alpha=alpha,
                lam=lam,
                history_length=history_length,
                dt=dt,
                spike_grad=spike_grad,
                init_hidden=True,
                output=True,
            )
        elif neuron_type == "bitshift":
            if shift_func is None:
                raise ValueError("shift_func must be provided for bitshift neuron type")
            self.lif1 = BitshiftLIF(
                beta=beta,  # For compatibility, not used
                shift_func=shift_func,
                lam=lam,
                history_length=history_length,
                dt=dt,
                spike_grad=spike_grad,
                init_hidden=True,
            )
            self.lif2 = BitshiftLIF(
                beta=beta,  # For compatibility, not used
                shift_func=shift_func,
                lam=lam,
                history_length=history_length,
                dt=dt,
                spike_grad=spike_grad,
                init_hidden=True,
                output=True,
            )
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")

        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        # decode membrane potential to Q-values per timestep
        self.fc_out = nn.Linear(hidden2_size, n_actions)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the spiking neural network.

        Processes observations through the SNN for num_steps timesteps,
        accumulating Q-values across time to produce final estimates.

        Args:
            observations: [batch, n_observations] float tensor of environment observations

        Returns:
            q_values: [batch, n_actions] tensor of Q-values (averaged over time)
        """
        # Reset snnTorch hidden states for all LIF instances
        # This avoids leak between separate forward passes / episodes.
        if self.neuron_type == "leakysv":
            LeakySV.reset_hidden()  # reset for LeakySV neurons (includes refractory counter)
        elif self.neuron_type == "fractional":
            FractionalLIF.reset_hidden()  # reset for FractionalLIF neurons (includes history buffer)
        elif self.neuron_type == "bitshift":
            BitshiftLIF.reset_hidden()  # reset for BitshiftLIF neurons (includes history buffer)
        else:
            snn.Leaky.reset_hidden()  # reset for Leaky neurons

        batch_size = observations.size(0)
        out_accum = torch.zeros(batch_size, self.n_actions, device=observations.device)

        # Simulate for num_steps timesteps with the SAME input each step (rate coding)
        for _t in range(self.num_steps):
            h1 = self.fc1(observations)  # current input -> hidden current
            spk1 = self.lif1(h1)  # spike output from layer1
            h2 = self.fc2(spk1)  # pass spikes into next layer
            spk2, mem2 = self.lif2(h2)  # output spikes and membrane of final LIF
            q_t = self.fc_out(mem2)  # decode membrane -> Q-values at this step
            # The Accumulation Strategy
            # A key aspect here is that out_accum accumulates the Q-values
            # across all timesteps. Since the network processes the same input
            # repeatedly, neurons that respond more strongly to this input will
            # fire more frequently, building up higher accumulated values.
            # After the loop completes, the final Q-values are obtained by
            # averaging: q_pred = out_accum / float(self.num_steps).
            out_accum += q_t

        q_values = out_accum / float(self.num_steps)
        return q_values
