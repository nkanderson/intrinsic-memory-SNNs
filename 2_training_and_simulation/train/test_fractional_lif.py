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

        # Python FractionalLIF update
        spk, mem_new = neuron(input_tensor, neuron.mem)
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

    This test expects a mismatch vs C behavior due to differences in the leak application
    and reset behavior.
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

        spk, mem_new = neuron(input_tensor, neuron.mem)
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


def test_history_dependence():
    """Verify that history length directly affects membrane updates for alpha < 1."""
    torch.set_default_dtype(torch.float64)
    # Create two neurons with identical parameters
    n1 = FractionalLIF(alpha=0.5, lam=0.1, history_length=16, dt=1.0, threshold=100.0, init_hidden=True)
    n2 = FractionalLIF(alpha=0.5, lam=0.1, history_length=16, dt=1.0, threshold=100.0, init_hidden=True)
    
    FractionalLIF.reset_hidden()
    
    # Drive n1 with a specific sequence
    n1_inputs = torch.tensor([[1.0], [2.0], [0.5]])
    for inp in n1_inputs:
        n1(inp.unsqueeze(0))
    
    # Drive n2 with a different sequence
    n2_inputs = torch.tensor([[-0.5], [0.1]])
    for inp in n2_inputs:
        n2(inp.unsqueeze(0))
        
    # Analytically find input X for n2 such that its next mem matches n1's current mem.
    device = n2.mem.device
    dtype = n2.mem.dtype
    C = 1.0 / (n2.dt ** n2.alpha)
    coeffs = n2._get_coeffs(device, dtype)[1:]
    history_valid = n2.hist[:n2.history_length - 1]
    history_sum = (coeffs.view(-1, 1, 1) * history_valid).sum(dim=0)

    target_mem = n1.mem
    next_n2_input = target_mem * (C + n2.lam) + C * history_sum
    
    n2(next_n2_input)
    
    # Ensure they reached the exact same membrane potential
    assert torch.isclose(n1.mem, n2.mem, atol=1e-10), "Failed to match membrane potentials"
    
    # Now feed them the exact same input
    shared_input = torch.tensor([[1.0]])
    n1(shared_input)
    n2(shared_input)
    
    # They should diverge because their histories are different
    assert not torch.isclose(n1.mem, n2.mem, atol=1e-10), "Neurons did not diverge despite different histories"


def test_alpha_1_classic_lif():
    """Verify that alpha=1.0 acts like classic Leaky without history dependence."""
    torch.set_default_dtype(torch.float64)
    beta = 0.9
    lam = (1 - beta) / beta
    
    # Create two identical alpha=1 neurons
    n1 = FractionalLIF(alpha=1.0, lam=lam, history_length=16, dt=1.0, threshold=100.0, init_hidden=True)
    n2 = FractionalLIF(alpha=1.0, lam=lam, history_length=16, dt=1.0, threshold=100.0, init_hidden=True)
    
    FractionalLIF.reset_hidden()
    
    # Drive n1
    for inp in torch.tensor([[1.0], [2.0], [0.5]]):
        n1(inp.unsqueeze(0))
    
    # Drive n2
    for inp in torch.tensor([[-0.5], [0.1]]):
        n2(inp.unsqueeze(0))
        
    # Match n2's mem to n1's
    C = 1.0 / (n2.dt ** n2.alpha)
    coeffs = n2._get_coeffs(n2.mem.device, n2.mem.dtype)[1:]
    history_valid = n2.hist[:n2.history_length - 1]
    history_sum = (coeffs.view(-1, 1, 1) * history_valid).sum(dim=0)

    next_n2_input = n1.mem * (C + n2.lam) + C * history_sum
    n2(next_n2_input)
    
    assert torch.isclose(n1.mem, n2.mem, atol=1e-10)
    
    # Feed same input
    shared_input = torch.tensor([[1.0]])
    n1(shared_input)
    n2(shared_input)
    
    # Because alpha=1, they only depend on the previous step (which is identical), so they MUST match
    assert torch.isclose(n1.mem, n2.mem, atol=1e-10), "alpha=1.0 neurons diverged, failing Markov property"


def test_spike_frequency_adaptation():
    """Verify that alpha < 1.0 produces decreasing ISIs (acceleration) under constant current."""
    torch.set_default_dtype(torch.float64)
    
    flif_adapt = FractionalLIF(alpha=0.5, lam=0.1, history_length=100, dt=1.0, threshold=1.0, init_hidden=True)
    flif_classic = FractionalLIF(alpha=1.0, lam=0.1, history_length=100, dt=1.0, threshold=1.0, init_hidden=True)
    
    FractionalLIF.reset_hidden()
    
    def get_isis(neuron, steps, current):
        spikes = []
        for t in range(steps):
            spk = neuron(torch.tensor([[current]]))
            if spk.item() > 0:
                spikes.append(t)
        
        isis = [spikes[i] - spikes[i-1] for i in range(1, len(spikes))]
        return isis
    
    isis_adapt = get_isis(flif_adapt, 100, 1.0)
    isis_classic = get_isis(flif_classic, 100, 1.0)
    
    assert len(isis_adapt) > 2, "Not enough spikes to measure adaptation"
    assert len(isis_classic) > 2, "Not enough spikes to measure adaptation"
    
    # Check that alpha < 1 adapts (ISI decreases / spike rate accelerates)
    early_mean = sum(isis_adapt[:10]) / 10
    late_mean = sum(isis_adapt[-10:]) / 10
    assert late_mean < early_mean, f"Spike frequency did not accelerate for alpha < 1 (early={early_mean}, late={late_mean})"
    
    # Check that alpha = 1 does NOT adapt (average ISI remains constant)
    early_mean_classic = sum(isis_classic[:20]) / 20
    late_mean_classic = sum(isis_classic[-20:]) / 20
    assert abs(late_mean_classic - early_mean_classic) < 1e-5, f"ISI drifted for alpha = 1: {early_mean_classic} vs {late_mean_classic}"


def _run_as_script():
    test_fractional_lif_matches_c_gl_update_no_leak()
    test_fractional_lif_differs_from_c_with_leak_and_reset()
    test_history_dependence()
    test_alpha_1_classic_lif()
    test_spike_frequency_adaptation()
    print("OK: test_fractional_lif_matches_c_gl_update_no_leak")
    print("OK: test_fractional_lif_differs_from_c_with_leak_and_reset")
    print("OK: test_history_dependence")
    print("OK: test_alpha_1_classic_lif")
    print("OK: test_spike_frequency_adaptation")


if __name__ == "__main__":
    _run_as_script()
