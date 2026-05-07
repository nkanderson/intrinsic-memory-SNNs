import torch

from bitshift_lif import BitshiftLIF
from scripts.history_coefficients import custom_slow_decay_bitshift

def test_history_dependence():
    """Verify that neurons with identical current states but different histories diverge."""
    torch.set_default_dtype(torch.float64)
    
    n1 = BitshiftLIF(shift_func=custom_slow_decay_bitshift, history_length=10, dt=1.0, threshold=100.0, init_hidden=True)
    n2 = BitshiftLIF(shift_func=custom_slow_decay_bitshift, history_length=10, dt=1.0, threshold=100.0, init_hidden=True)
    BitshiftLIF.reset_hidden()
    
    inp = torch.tensor([[1.0]])
    n1(inp)
    n2(inp)
    
    assert torch.isclose(n1.mem, n2.mem, atol=1e-10)
    
    # Manually modify history, but keep current state (hist[0] or mem) the same
    # hist is shape (10, 1, 1)
    n2.hist[1:] = 10.0
    
    n1(inp)
    n2(inp)
    
    assert not torch.isclose(n1.mem, n2.mem, atol=1e-10), "Neurons did not diverge despite different histories"


def test_shift_sequence_application():
    """Verify the vectorized bit-shift logic exactly matches mathematical division."""
    torch.set_default_dtype(torch.float64)
    
    history_length = 4
    # custom_slow_decay_bitshift(4) returns [0, 1, 3, 4]
    
    n = BitshiftLIF(shift_func=custom_slow_decay_bitshift, history_length=history_length, dt=1.0, threshold=100.0, init_hidden=True)
    BitshiftLIF.reset_hidden()
    
    # Manually populate history buffer to bypass rolling
    inp = torch.tensor([[0.0]])
    n(inp) # Initialize buffer
    
    # After one forward pass, history buffer is populated. Let's overwrite it manually.
    # Buffer length is 4. Index 0 is the current state.
    # We must populate it knowing that forward() will roll it by 1 BEFORE calculation.
    # So to get [10, 8, 16] at indices [1, 2, 3] AFTER rolling:
    # We must put them at indices [0, 1, 2] BEFORE rolling
    n.hist[0] = torch.tensor([[10.0]])
    n.hist[1] = torch.tensor([[8.0]])
    n.hist[2] = torch.tensor([[16.0]])
    n.hist[3] = torch.tensor([[0.0]])
    
    # We apply input=1.0
    inp = torch.tensor([[1.0]])
    n(inp)
    
    # Manually compute expected output:
    # shifts_past = [1, 3, 4]
    # history_shifted = [10.0 / 2, 8.0 / 8, 16.0 / 16] = [5.0, 1.0, 1.0]
    # history_sum = 7.0
    # C = 1.0 / (dt**alpha) = 1.0
    # numerator = 1.0 + 1.0 * 7.0 = 8.0
    # denominator = 1.0 + 0.111 = 1.111
    # expected = 8.0 / 1.111
    
    expected_mem = 8.0 / (1.0 + n.lam)
    
    assert torch.isclose(n.mem, torch.tensor([[expected_mem]]), atol=1e-10), f"Expected {expected_mem}, got {n.mem.item()}"


def test_spike_frequency_adaptation():
    """Verify that BitshiftLIF produces decreasing ISIs (acceleration) under constant current."""
    torch.set_default_dtype(torch.float64)
    
    # Low threshold and low lam to encourage multiple spikes
    n = BitshiftLIF(shift_func=custom_slow_decay_bitshift, history_length=100, dt=1.0, lam=0.1, threshold=1.0, init_hidden=True)
    BitshiftLIF.reset_hidden()
    
    inp = torch.tensor([[0.5]])
    
    spike_times = []
    for t in range(500):
        spk = n(inp)
        if spk.item() > 0:
            spike_times.append(t)
            
    # Need at least 3 spikes to measure 2 ISIs
    assert len(spike_times) >= 3, "Not enough spikes to measure adaptation"
    
    isis = [spike_times[i] - spike_times[i-1] for i in range(1, len(spike_times))]
    
    # Verify ISIs are strictly decreasing
    for i in range(1, len(isis)):
        assert isis[i] <= isis[i-1], f"ISI increased from {isis[i-1]} to {isis[i]} at spike {i+1}"
    
    # And strictly check that the last ISI is smaller than the first
    assert isis[-1] < isis[0], f"No overall spike frequency acceleration observed. First ISI: {isis[0]}, Last ISI: {isis[-1]}"


if __name__ == "__main__":
    print("Running BitshiftLIF tests...")
    
    test_history_dependence()
    print("OK: test_history_dependence")
    
    test_shift_sequence_application()
    print("OK: test_shift_sequence_application")
    
    test_spike_frequency_adaptation()
    print("OK: test_spike_frequency_adaptation")
    
    print("All tests passed!")
