# Common Neuron RTL Contract (`lif` / `fractional_lif` / `bitshift_lif`)

This contract is intended to let all three neuron variants plug into the same network scheduling logic.

## 1) Port-level contract

Required inputs:
- `clk`
- `reset`
- `clear`
- `enable` (request one timestep update)
- `current`

Required outputs:
- `spike_out`
- `membrane_out`
- `busy`
- `output_valid`

### Semantics
- `enable` is a request pulse for one update.
- Request acceptance condition:
  - Accepted when `enable && !busy`.
  - If `enable && busy`, request is ignored (no queueing).
- `busy`:
  - `1` while neuron is processing an accepted request.
  - `0` when ready to accept another request.
- `output_valid`:
  - Single-cycle pulse when `spike_out` and `membrane_out` are updated for the accepted request.
  - Must be exactly one pulse per accepted request.

## 2) Timing model

### Single-cycle neuron (`lif` default)
- Accept request at cycle `N` (`enable && !busy`).
- Update outputs and pulse `output_valid` at cycle `N+1`.
- `busy` may stay `0` (or assert briefly internally), but externally next request must still obey acceptance semantics.

### Multi-cycle neuron (`fractional_lif` MAC,`bitshift_lif` may also become multi-cycle)
- Accept request at cycle `N`.
- Assert `busy` during compute (`N+1 ... N+K-1`).
- Update outputs + pulse `output_valid` at completion cycle (`N+K`).
- Deassert `busy` once complete.

## 3) State update rules

On accepted request:
- Use current `current` input sample for this timestep only.
- Apply reset-delay semantics consistently with existing model (`spike_prev` subtraction style).
- Write history/membrane state exactly once for this timestep.

## 4) Network integration rules

Top-level scheduler should:
1. Pulse `enable` only when target neuron `busy==0`.
2. Use `output_valid` (not fixed cycle delays) to:
   - capture `membrane_out`
   - capture `spike_out`
   - advance per-neuron/timestep bookkeeping
3. Maintain one accepted neuron update per logical timestep per neuron.

## 5) Backward compatibility guidance

For existing code paths that expect 1-cycle behavior:
- Keep old behavior by implementing:
  - `busy = 0`
  - `output_valid` pulse aligned with existing output register update.
- This preserves functionality while standardizing interfaces.

## 6) Why this contract

- Makes latency explicit.
- Allows mixed neuron internals (single-cycle vs multi-cycle) under one scheduler.
- Enables safe synthesis optimizations (MAC FSM, BRAM-based history) without rewriting network control each time.
