import snntorch as snn
import torch


class LeakySV(snn.Leaky):
    """
    Training-optimized Leaky neuron with refractory period.

    This model uses floating-point LIF dynamics for training, with an added
    refractory period feature that matches an intended SystemVerilog hardware implementation.

    Key features:
    - Uses standard snnTorch Leaky membrane dynamics (floating-point)
    - Adds refractory period (neurons can't spike for N steps after spiking)
    - Compatible with gradient-based learning
    - Trained weights can be converted to fixed-point for hardware deployment

    Parameters:
    - refractory_period: Number of time steps after spiking during which neuron cannot spike again
    - All other parameters inherited from snn.Leaky

    Hardware Translation Notes:
    - Training uses floating-point in natural neuron ranges
    - Weights should be scaled/quantized post-training for hardware
    """

    def __init__(
        self,
        beta,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="zero",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        reset_delay=True,
        refractory_period=5,
    ):
        super().__init__(
            beta,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )

        # Refractory period parameter
        self.refractory_period = refractory_period

        # Initialize refractory counter (resized during forward pass)
        self.refractory_counter = torch.zeros(1)

    def forward(self, input_, mem=None):
        """
        Forward pass with refractory period.

        Uses standard Leaky membrane dynamics, but prevents spiking
        during refractory period after each spike.

        Args:
            input_: Input current
            mem: Membrane potential (optional, not used if init_hidden=True)

        Returns:
            spk: Output spikes
            mem: Updated membrane potential (if output=True or init_hidden=False)
        """
        # Initialize or resize refractory counter if needed
        if (
            not hasattr(self, "refractory_counter")
            or self.refractory_counter.shape != input_.shape
        ):
            self.refractory_counter = torch.zeros_like(input_)

        # Call parent Leaky forward to get membrane dynamics and spikes
        # This handles all the standard Leaky behavior
        result = super().forward(input_, mem)

        # Extract spike and membrane based on return format
        if self.output or not self.init_hidden:
            spk, mem = result
        else:
            spk = result
            mem = self.mem if hasattr(self, "mem") else None

        # Apply refractory period logic
        if self.refractory_period > 0:
            spk, mem = self._apply_refractory(spk, mem)

        # Return in the same format as parent
        if self.output:
            return spk, mem
        elif self.init_hidden:
            return spk
        else:
            return spk, mem

    def _apply_refractory(self, spk, mem):
        """
        Apply refractory period: suppress spikes if neuron is in refractory state.

        Args:
            spk: Spike tensor from parent Leaky
            mem: Membrane potential tensor

        Returns:
            spk: Modified spikes with refractory suppression
            mem: Membrane potential (unchanged)
        """
        # Check which neurons are in refractory period
        in_refractory = self.refractory_counter > 0

        # Suppress spikes for neurons in refractory
        spk_suppressed = spk * (~in_refractory).float()

        # Update refractory counter
        with torch.no_grad():
            # Decrement counter for neurons in refractory
            self.refractory_counter = torch.clamp(self.refractory_counter - 1, min=0)

            # Set counter for neurons that just spiked (use original spk, not suppressed)
            # This ensures we track all spikes for refractory timing
            just_spiked = (spk > 0) & (~in_refractory)
            self.refractory_counter = torch.where(
                just_spiked,
                torch.full_like(self.refractory_counter, float(self.refractory_period)),
                self.refractory_counter,
            )

        return spk_suppressed, mem

    @classmethod
    def reset_hidden(cls):
        """
        Reset hidden states for all LeakySV instances.
        Calls parent Leaky.reset_hidden() and also resets refractory counters.
        """
        # Reset parent class hidden states (mem)
        super(LeakySV, cls).reset_hidden()

        # Reset refractory counters for all instances
        # This iterates through all instances that have init_hidden=True
        for instance in cls.instances:
            if hasattr(instance, "refractory_counter"):
                instance.refractory_counter = torch.zeros_like(
                    instance.refractory_counter
                )
