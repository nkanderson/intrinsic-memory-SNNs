"""
Bit-shift approximation Leaky Integrate-and-Fire (BitshiftLIF) Neuron

This module implements a LIF neuron that approximates fractional-order dynamics
(alpha=0.5) using hardware-efficient bit-shift operations instead of multiplication.
Uses right-shift operations which can be directly implemented in hardware.

Inherits from snnTorch Leaky neuron, overriding only the state function.
"""

import torch
import snntorch as snn


class BitshiftLIF(snn.Leaky):
    """
    Bit-shift approximation Leaky Integrate-and-Fire neuron.

    Similar to FractionalLIF but uses bit-shift operations instead of GL coefficients.
    Designed to approximate alpha=0.5 fractional dynamics using hardware-efficient
    right-shift operations instead of floating-point multiplication.

    The shift_func should accept history_length as an argument and return a list of
    integer shift amounts. These shifts are applied directly using right-shift (>>)
    operations in the forward pass.

    Args:
        beta: Membrane potential decay rate for compatibility. Not used in bitshift dynamics,
              but passed to parent. Use lam instead for fractional leak.
        shift_func: Function that takes history_length and returns list of integer shift amounts.
                   Should be callable as: shifts = shift_func(history_length)
                   Examples: simple_bitshift, slow_decay_bitshift, custom_slow_decay_bitshift
        lam: Leakage parameter in fractional equation (>= 0). Default: 0.111 (matches beta=0.9)
        history_length: Number of past values for approximation. Default: 256
        dt: Discrete timestep for approximation. Default: 1.0
        threshold: Spike threshold. Default: 1.0
        spike_grad: Optional spike gradient surrogate function.
        init_hidden: If True, instantiates state variables as instance variables. Default: False
        output: If True with init_hidden=True, states are returned. Default: False
        **kwargs: Additional arguments passed to snnTorch Leaky parent class

    Note:
        This neuron approximates fractional-order dynamics with alpha=0.5 (hardcoded).
        The alpha value cannot be changed as the bit-shift sequences are designed
        specifically for alpha=0.5 approximation.

        WARNING: When using simple bit-shift functions (like simple_bitshift), keep
        history_length <= 20 to avoid numerical instability. For longer histories, use
        slow_decay or custom shift functions that prevent coefficients from becoming
        too small (e.g., 2^-63 ≈ 1e-19).

    Forward API (inherited from snnTorch):
        forward(input, mem) -> (spike, mem) or spike (if init_hidden=True and output=False)

    Example:
        >>> from scripts.history_coefficients import simple_bitshift
        >>> neuron = BitshiftLIF(shift_func=simple_bitshift, history_length=64)
    """

    def __init__(
        self,
        beta: float = 0.9,  # For compatibility, not used in bitshift dynamics
        shift_func=None,
        lam: float = 0.111,  # Default matches beta=0.9: lam = (1-0.9)/0.9
        history_length: int = 256,
        dt: float = 1.0,
        threshold: float = 1.0,
        spike_grad=None,
        init_hidden: bool = False,
        output: bool = False,
        **kwargs,
    ):
        # Initialize parent snnTorch Leaky neuron
        super().__init__(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            init_hidden=init_hidden,
            output=output,
            **kwargs,
        )

        # Validate parameters
        assert shift_func is not None, "shift_func must be provided"
        assert callable(shift_func), "shift_func must be callable"
        assert lam >= 0, "Leakage parameter lam must be non-negative"
        assert history_length > 0, "History length must be positive"

        # Store parameters
        self.shift_func = shift_func
        self.lam = lam
        self.history_length = history_length
        self.dt = dt
        self.alpha = 0.5  # Hardcoded - bitshift sequences approximate alpha=0.5

        # Compute shift amounts using provided function
        shift_amounts = shift_func(history_length)
        # Store as tensor of integers for efficient indexing
        self._shifts = torch.tensor(shift_amounts, dtype=torch.int64)
        self._shifts_device = None  # Track which device shifts are on

        # Initialize history buffer if init_hidden=True
        if self.init_hidden:
            hist = torch.zeros(0)
            self.register_buffer("hist", hist, persistent=False)

    def _get_shifts(self, device: torch.device) -> torch.Tensor:
        """Get shift amounts on the correct device."""
        # Move shifts to correct device if needed
        if self._shifts_device != device:
            self._shifts = self._shifts.to(device=device)
            self._shifts_device = device
        return self._shifts

    def _base_state_function(self, input_):
        """
        Override parent's state function to use bit-shift approximation dynamics.

        Uses right-shift operations instead of multiplication with coefficients.
        This is hardware-efficient and can be directly mapped to digital circuits.

        mem = (input - C * Σ (hist[k] >> shift[k])) / (C + λ)

        This is called by parent's forward() after handling reset logic.
        """
        device = input_.device
        dtype = input_.dtype

        # Get shift amounts
        shifts = self._get_shifts(device)

        # Initialize or reshape history buffer if needed
        if not hasattr(self, "hist") or self.hist.shape[0] == 0:
            hist_shape = (self.history_length,) + input_.shape
            self.hist = torch.zeros(hist_shape, device=device, dtype=dtype)
        elif self.hist.shape[1:] != input_.shape:
            # Reshape if batch size changed
            hist_shape = (self.history_length,) + input_.shape
            self.hist = torch.zeros(hist_shape, device=device, dtype=dtype)

        # Compute fractional membrane update using bit-shifts
        # V[n] = (I[n] - C * Σ_{k=1}^{H-1} (V[n-k] >> shift[k])) / (C + λ)
        C = 1.0 / (self.dt**self.alpha)

        # Extract past shift amounts (skip shift[0]=0 which corresponds to g_0=1 multiplying V[n])
        shifts_past = shifts[1 : self.history_length]

        # History buffer for k=1..H-1
        history_valid = self.hist[: self.history_length - 1]  # (H-1, batch, features)

        # Apply bit-shifts to history values using vectorized operations
        # For each position k, we right-shift by shifts_past[k] bits
        # Using torch.pow and division is faster than a Python loop
        # In actual hardware, these would be right-shift (>>) operations

        # Compute all shift divisors at once: [2^shift[0], 2^shift[1], ...]
        # Shape: (H-1,) -> (H-1, 1, 1) for broadcasting
        shift_divisors = torch.pow(2.0, shifts_past.float()).view(-1, 1, 1)

        # Apply all shifts at once via broadcasting
        # history_valid: (H-1, batch, features)
        # shift_divisors: (H-1, 1, 1)
        # Result: (H-1, batch, features)
        history_shifted = history_valid / shift_divisors

        # Sum over time dimension
        history_sum = history_shifted.sum(dim=0)  # (batch, features)

        # Compute new membrane potential
        numerator = input_ - C * history_sum
        denominator = C + self.lam
        mem_new = numerator / denominator

        # Update history buffer: shift in-place
        # Roll the history buffer and insert current mem at position 0
        self.hist = torch.roll(self.hist, shifts=1, dims=0)
        self.hist[0] = self.mem

        return mem_new

    @classmethod
    def reset_hidden(cls):
        """
        Reset hidden states for all BitshiftLIF instances.

        Overrides parent's reset_hidden to also clear the history buffer.

        Called automatically by snnTorch's instance tracking system.
        """
        # First call parent's reset_hidden to clear mem (and any other parent state)
        super(BitshiftLIF, cls).reset_hidden()

        # Then clear history buffers for all BitshiftLIF instances
        for instance in cls.instances:
            if isinstance(instance, BitshiftLIF) and hasattr(instance, "hist"):
                instance.hist = torch.zeros_like(
                    instance.hist, device=instance.hist.device
                )

    def extra_repr(self) -> str:
        """String representation of module parameters."""
        parent_repr = super().extra_repr()
        bitshift_repr = (
            f"shift_func={self.shift_func.__name__}, lam={self.lam}, "
            f"history_length={self.history_length}, dt={self.dt}, alpha=0.5 (hardcoded)"
        )
        return f"{parent_repr}, {bitshift_repr}"
