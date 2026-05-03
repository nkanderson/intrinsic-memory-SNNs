"""Hardware-accelerated SNN policy wrapping the FPGA accelerator.

Implements the same nn.Module interface as SNNPolicy but dispatches each
forward pass to the FPGA over UART. The FPGA computes the argmax internally;
forward() returns a [1, n_actions] tensor with 1.0 at the selected action
so that .max(1).indices gives the correct result.

Callers own the FpgaInterface lifetime — open it as a context manager and
pass it in:

    with FpgaInterface(port) as fpga:
        policy = SNNPolicyHardware(4, 2, fpga)
        action = policy(state).max(1).indices.item()
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

FRAC_BITS = 13
_SCALE = 1 << FRAC_BITS  # 8192


def float_to_qs2_13(values: np.ndarray) -> list[int]:
    """Convert float observations to QS2.13 int16 values (clipped, not saturated)."""
    return (
        np.round(values * _SCALE)
        .clip(-32768, 32767)
        .astype(np.int16)
        .tolist()
    )


def qs2_13_to_float(values: list[int]) -> np.ndarray:
    return np.array(values, dtype=np.float32) / _SCALE


class SNNPolicyHardware(nn.Module):
    """FPGA-backed SNN policy. No weights stored locally — all compute is on hardware."""

    def __init__(self, n_observations: int, n_actions: int, fpga_interface) -> None:
        super().__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self._fpga = fpga_interface

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Run one inference on the FPGA.

        Args:
            observations: [1, n_observations] float tensor

        Returns:
            [1, n_actions] tensor — 1.0 at selected action index, 0.0 elsewhere.
            Calling .max(1).indices on this gives the selected action.
        """
        obs_np = observations.squeeze(0).detach().cpu().numpy()
        obs_q = float_to_qs2_13(obs_np)

        self._fpga.write_obs(obs_q)
        self._fpga.start_inference()
        self._fpga.wait_done()
        action = self._fpga.read_action()

        result = torch.zeros(1, self.n_actions)
        result[0, action] = 1.0
        return result
