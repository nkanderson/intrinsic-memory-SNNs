import math
import torch

from fractional_lif import FractionalLIF
from common.scripts.utils import compute_gl_coefficients


def _c_flif_gl_reference_update(state, input_val, *, dt, alpha, coeffs, tau_m, V_rest, bias):
    """
    Reference update that mirrors spires/src/neurons/flif_gl.c timing exactly.

    History stores the newly computed V at each micro-step.
    """
    history = state["history"]
    internal_step = state["internal_step"]
    mem_len = state["mem_len"]

    # V_prev corresponds to previous micro-step (initialized to V_rest)
    V_prev = history[-1]

    # History sum (k = 1..limit)
    limit = internal_step if internal_step < mem_len else mem_len - 1
    history_sum = 0.0
    for k in range(1, limit + 1):
        history_sum += coeffs[k] * history[-k]

    if math.isfinite(tau_m):
        rhs = (-(V_prev - V_rest) / tau_m) + input_val + bias
    else:
        rhs = input_val + bias

    V_new = (dt**alpha) * rhs - history_sum

    # Update history with newly computed V (C timing)
    history.append(V_new)
    if len(history) > mem_len:
        history.pop(0)

    state["internal_step"] = internal_step + 1
    return V_new


def _c_flif_gl_reference_update_with_reset(
    state,
    input_val,
    *,
    dt,
    alpha,
    coeffs,
    tau_m,
    V_rest,
    bias,
    V_th,
    V_reset,
):
    """
    C reference update including spike/reset behavior.
    """
    V_new = _c_flif_gl_reference_update(
        state,
        input_val,
        dt=dt,
        alpha=alpha,
        coeffs=coeffs,
        tau_m=tau_m,
        V_rest=V_rest,
        bias=bias,
    )

    if V_new >= V_th:
        V_new = V_reset
        state["history"][-1] = V_new

    return V_new


def test_fractional_lif_matches_c_gl_update_no_leak():
    """
    Basic membrane update comparison with leak disabled.

    We set tau_m = inf and lam = 0 so both updates reduce to:
        V_n = dt^alpha * input - sum_{k=1} g_k V_{n-k}
    """
    torch.set_default_dtype(torch.float64)

    # Shared parameters
    alpha = 0.5
    dt = 1.0
    history_length = 16
    coeffs = compute_gl_coefficients(alpha, history_length).tolist()

    # Inputs (simple deterministic sequence)
    inputs = [0.1, 0.2, -0.05, 0.0, 0.15, -0.1]

    # C reference state
    c_state = {
        "history": [0.0 for _ in range(history_length)],
        "internal_step": 0,
        "mem_len": history_length,
    }

    # FractionalLIF instance (no leak, no spikes)
    neuron = FractionalLIF(
        alpha=alpha,
        lam=0.0,
        history_length=history_length,
        dt=dt,
        threshold=1e9,
        init_hidden=False,
    )

    # Initialize membrane state
    neuron.mem = torch.zeros(1, 1, dtype=torch.float64)

    # Step through inputs and compare
    for input_val in inputs:
        input_tensor = torch.tensor([[input_val]], dtype=torch.float64)

        # Python FractionalLIF update (base state function)
        mem_new = neuron._base_state_function(input_tensor)
        neuron.mem = mem_new

        # C reference update
        c_mem_new = _c_flif_gl_reference_update(
            c_state,
            input_val,
            dt=dt,
            alpha=alpha,
            coeffs=coeffs,
            tau_m=math.inf,
            V_rest=0.0,
            bias=0.0,
        )

        torch.testing.assert_close(
            mem_new.squeeze(),
            torch.tensor(c_mem_new, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-12,
        )


def test_fractional_lif_differs_from_c_with_leak_and_reset():
    """
    Incorporate leak and reset under the assumptions:
      - V_rest = 0
      - lambda = 1 / tau_m

    This test expects a mismatch vs C behavior.
    """
    torch.set_default_dtype(torch.float64)

    alpha = 0.5
    dt = 1.0
    history_length = 16
    coeffs = compute_gl_coefficients(alpha, history_length).tolist()

    tau_m = 5.0
    lam = 1.0 / tau_m
    V_th = 0.25
    V_reset = 0.0

    inputs = [0.2, 0.25, 0.3, 0.1, 0.05]

    c_state = {
        "history": [0.0 for _ in range(history_length)],
        "internal_step": 0,
        "mem_len": history_length,
    }

    neuron = FractionalLIF(
        alpha=alpha,
        lam=lam,
        history_length=history_length,
        dt=dt,
        threshold=V_th,
        init_hidden=False,
    )
    neuron.mem = torch.zeros(1, 1, dtype=torch.float64)

    mismatched = False
    for input_val in inputs:
        input_tensor = torch.tensor([[input_val]], dtype=torch.float64)

        mem_new = neuron._base_state_function(input_tensor)
        neuron.mem = mem_new

        c_mem_new = _c_flif_gl_reference_update_with_reset(
            c_state,
            input_val,
            dt=dt,
            alpha=alpha,
            coeffs=coeffs,
            tau_m=tau_m,
            V_rest=0.0,
            bias=0.0,
            V_th=V_th,
            V_reset=V_reset,
        )

        if not torch.isclose(
            mem_new.squeeze(),
            torch.tensor(c_mem_new, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-12,
        ):
            mismatched = True
            break

    assert mismatched, "Expected a mismatch vs C with leak/reset, but all steps matched."


def _run_as_script():
    test_fractional_lif_matches_c_gl_update_no_leak()
    test_fractional_lif_differs_from_c_with_leak_and_reset()
    print("OK: test_fractional_lif_matches_c_gl_update_no_leak")
    print("OK: test_fractional_lif_differs_from_c_with_leak_and_reset")


if __name__ == "__main__":
    _run_as_script()
