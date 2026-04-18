"""
Utility functions for fractional-order neural network implementations.
"""

import torch
import unittest


def compute_gl_coefficients(alpha: float, history_length: int) -> torch.Tensor:
    """
    Compute Grunwald-Letnikov binomial coefficients for fractional derivative.

    The GL coefficients are defined as:
        g_k = (-1)^k * binom(alpha, k)
    where binom(alpha, k) = Gamma(alpha+1) / (Gamma(k+1) * Gamma(alpha - k + 1))

    This is the generalized binomial coefficient extending Pascal's triangle to
    fractional values of alpha.

    Implementation: Stable Recurrence Relation
    -------------------------------------------
    Instead of computing Gamma functions directly (which is numerically unstable
    and computationally expensive for large k), we use the recurrence:

        g_0 = 1
        g_k = g_{k-1} * (alpha - (k-1)) / k * (-1)

    Derivation:
        From the binomial coefficient property:
            binom(alpha, k) / binom(alpha, k-1) = (alpha - k + 1) / k

        Therefore:
            g_k / g_{k-1} = [(-1)^k * binom(alpha, k)] / [(-1)^(k-1) * binom(alpha, k-1)]
                          = (-1) * (alpha - k + 1) / k
                          = (alpha - (k-1)) / k * (-1)

    Benefits of recurrence over direct Gamma computation:
        1. Numerically stable (no huge Gamma values or log subtractions)
        2. Fast: O(n) simple arithmetic vs O(n) transcendental functions
        3. No scipy dependency - pure PyTorch
        4. Works directly on any device/dtype

    Special cases:
        - alpha = 1.0: g_0=1, g_1=-1, g_k=0 for k>1 (first-order difference)
        - alpha = 0.5: g_0=1, g_1=-0.5, g_2=-0.125, ... (fractional diffusion)

    Args:
        alpha: Fractional order (0 < alpha <= 1 typically)
        history_length: Number of coefficients to compute

    Returns:
        Tensor of shape (history_length,) containing g_0, g_1, ..., g_{H-1}
        Returned on CPU with dtype float64 for precision; caller should convert
        to target device/dtype as needed.
    """
    # Compute coefficients using recurrence
    coeffs = torch.zeros(history_length, dtype=torch.float64)
    coeffs[0] = 1.0

    for k in range(1, history_length):
        # g_k = g_{k-1} * (alpha - (k-1)) / k * (-1)
        coeffs[k] = coeffs[k - 1] * (alpha - (k - 1)) / k * (-1.0)

    return coeffs


# ============================================================================
# Tests
# ============================================================================


class TestGLCoefficients(unittest.TestCase):
    """Test GL binomial coefficient computation."""

    def test_alpha_1_recovers_first_order(self):
        """For alpha=1, should get first-order difference: g=[1, -1, 0, 0, ...]"""
        coeffs = compute_gl_coefficients(alpha=1.0, history_length=10)

        self.assertAlmostEqual(coeffs[0].item(), 1.0, places=10)
        self.assertAlmostEqual(coeffs[1].item(), -1.0, places=10)
        # All subsequent coefficients should be zero (or very close)
        for k in range(2, 10):
            self.assertLess(
                abs(coeffs[k].item()), 1e-10, f"g_{k} should be ~0 for alpha=1"
            )

    def test_alpha_05_known_values(self):
        """Validate first few coefficients for alpha=0.5 against known values."""
        coeffs = compute_gl_coefficients(alpha=0.5, history_length=5)

        # Known values for alpha=0.5:
        # g_0 = 1
        # g_1 = 0.5 * (-1) = -0.5
        # g_2 = -0.5 * (0.5 - 1) / 2 * (-1) = -0.5 * (-0.5) / 2 * (-1) = -0.125
        # g_3 = -0.125 * (0.5 - 2) / 3 * (-1) = -0.125 * (-1.5) / 3 * (-1) = -0.0625

        self.assertAlmostEqual(coeffs[0].item(), 1.0, places=10)
        self.assertAlmostEqual(coeffs[1].item(), -0.5, places=10)
        self.assertAlmostEqual(coeffs[2].item(), -0.125, places=10)
        self.assertAlmostEqual(coeffs[3].item(), -0.0625, places=10)

    def test_coefficients_using_scipy(self):
        """Validate recurrence against direct Gamma computation using scipy."""
        try:
            from scipy.special import gamma
        except ImportError:
            self.skipTest("scipy not available")

        alpha = 0.7
        history_length = 20

        # Compute using our recurrence
        coeffs_recurrence = compute_gl_coefficients(alpha, history_length)

        # Compute using direct Gamma formula
        coeffs_gamma = torch.zeros(history_length, dtype=torch.float64)
        for k in range(history_length):
            binom_val = gamma(alpha + 1) / (gamma(k + 1) * gamma(alpha - k + 1))
            coeffs_gamma[k] = ((-1) ** k) * binom_val

        # Should match to high precision
        torch.testing.assert_close(
            coeffs_recurrence, coeffs_gamma, rtol=1e-10, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
