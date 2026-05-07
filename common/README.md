# common/

Shared assets used across the design-space, training, and FPGA-benchmarking work:

- `sv/` — SystemVerilog neuron and network sources, plus the cocotb test bench
- `scripts/` — Python utilities; primarily plotting and waveform analysis
- `metrics/` — checked-in waveforms (`*.vcd`) and spike-cycle CSVs that back the figures
- `images/` — generated SVG figures referenced by the paper / write-ups

## Plotting scripts

Two scripts produce the spike-frequency-adaptation figures. Both consume artifacts written by the cocotb memory tests in `sv/cocotb/tests/`:

1. **`plot_membrane_potential.py`** — renders the per-step waveform: membrane, current, spike raster, and inter-spike interval (ISI). Reads a `.vcd` waveform produced by the cocotb capture testcases. Optionally takes a phase CSV (and its sidecar JSON) to shade the dropout window and switch to multi-panel layouts.
2. **`plot_spike_cycles.py`** — renders the cycle-by-cycle ISI / firing-frequency trend that summarizes how the neuron adapts. Reads only the CSV exported by the capture tests; no waveform parsing.

### Input artifacts and how they get to `metrics/`

The cocotb capture testcases write to `sv/cocotb/results/`:

- `sim_build/<toplevel>.fst` — Icarus waveform (binary FST format)
- `<variant>_spike_cycles.csv` — per-cycle membrane traces with `phase` labels
- `<variant>_phases.json` — sidecar with the actual current-on/current-off boundary times in nanoseconds (only written by the dropout/recovery captures)

`plot_membrane_potential.py` parses VCD, not FST, so convert first (inside the cocotb container that has GTKWave / `fst2vcd` installed):

```bash
fst2vcd sv/cocotb/results/sim_build/fractional_lif.fst \
    -o sv/cocotb/results/fractional_lif.vcd
```

The `metrics/` directory holds the canonical, regeneratable waveforms / CSVs used to produce the checked-in figures. Promote a fresh capture into it with:

```bash
cp sv/cocotb/results/fractional_lif_constant_current.vcd metrics/
cp sv/cocotb/results/fractional_lif_memory_spike_cycles.csv metrics/fractional_lif_constant_current_spike_cycles.csv
cp sv/cocotb/results/fractional_lif_dropout_current.vcd metrics/
cp sv/cocotb/results/fractional_lif_dropout_recovery_spike_cycles.csv   metrics/
cp sv/cocotb/results/fractional_lif_dropout_recovery_phases.json metrics/
```

The plot scripts auto-detect the phase sidecar by replacing `_spike_cycles.csv` with `_phases.json` next to the CSV — keep them paired.

### Figure catalog

Run from `common/`. Each command lists the SVG it writes.

#### `plot_membrane_potential.py`

**Constant-current capture (4-subplot view, auto-zoom to first 20 spikes):**
```bash
python scripts/plot_membrane_potential.py \
    metrics/fractional_lif_constant_current.vcd \
    --output images/fractional_lif_memory_constant.svg
```
→ `images/fractional_lif_memory_constant.svg`

**Dropout/recovery, full-span single-panel view (default when `--phase-csv` is given):**
```bash
python scripts/plot_membrane_potential.py \
    metrics/fractional_lif_dropout_current.vcd \
    --phase-csv metrics/fractional_lif_dropout_recovery_spike_cycles.csv \
    --output images/fractional_lif_memory_dropout.svg
```
→ `images/fractional_lif_memory_dropout.svg`

Phase shading for `startup` (gray), `dropout` (vermillion), `recovery` (green) is drawn from the sidecar `*_phases.json`. The cross-dropout ISI is excluded from the ISI subplot so within-phase variations stay legible.

**Dropout/recovery, single-panel with end trimmed to first N=20 recovery spikes:**
```bash
python scripts/plot_membrane_potential.py \
    metrics/fractional_lif_dropout_current.vcd \
    --phase-csv metrics/fractional_lif_dropout_recovery_spike_cycles.csv \
    --phase-zoom-spikes 20 \
    --output images/fractional_lif_memory_dropout-zoom-20.svg
```
→ `images/fractional_lif_memory_dropout-zoom-20.svg`

Trims the long convergent tail of recovery while keeping the full startup + dropout window — useful when the late recovery flattens out.

**Dropout/recovery, 3-panel per-phase view (first N=20 spikes per active phase):**
```bash
python scripts/plot_membrane_potential.py \
    metrics/fractional_lif_dropout_current.vcd \
    --phase-csv metrics/fractional_lif_dropout_recovery_spike_cycles.csv \
    --zoom-early-spikes-per-phase 20 \
    --output images/fractional_lif_memory_dropout-zoom-early.svg
```
→ `images/fractional_lif_memory_dropout-zoom-early.svg`

Side-by-side membrane panels (one per phase, `dropout` shows the empty current-off window). Panel widths roughly match each other, making the startup-vs-recovery spike-density comparison honest at a glance.

**Same per-phase view, more spikes (N=30) for fuller adaptation arc:**
```bash
python scripts/plot_membrane_potential.py \
    metrics/fractional_lif_dropout_current.vcd \
    --phase-csv metrics/fractional_lif_dropout_recovery_spike_cycles.csv \
    --zoom-early-spikes-per-phase 30 \
    --output images/fractional_lif_memory_dropout-zoom-early-30.svg
```
→ `images/fractional_lif_memory_dropout-zoom-early-30.svg`

#### `plot_spike_cycles.py`

By default the figure has no suptitle — summary statistics (early/late mean, gain, per-phase
early means) are printed to stdout for inclusion in the figure caption. Pass `--title "..."`
to embed the summary in the figure itself.

The fractional LIF fires in doublets at the chosen operating point: the cycle stream alternates
between a within-doublet length-2 ISI and a between-doublet length-13ish ISI. The within-doublet
cycles compress the y-axis and obscure the actual adaptation, so the recommended workflow uses
`--min-cycle-length` to drop them before plotting.

**Constant-current ISI adaptation (raw, no filtering):**
```bash
python scripts/plot_spike_cycles.py \
    metrics/fractional_lif_constant_current_spike_cycles.csv \
    --output images/fractional_lif_memory_spike_cycles.svg
```
→ `images/fractional_lif_memory_spike_cycles.svg`

Top subplot: cycle length vs cycle index. Bottom: instantaneous spike frequency. Shows the
full doublet pattern — useful as a "this is what the raw data looks like" reference, but
hard to read trend-wise.

**Raw, zoomed to first 20 cycles (no filtering):**
```bash
python scripts/plot_spike_cycles.py \
    metrics/fractional_lif_constant_current_spike_cycles.csv \
    --max-cycles 20 \
    --output images/fractional_lif_memory_spike_cycles_zoom-20.svg
```
→ `images/fractional_lif_memory_spike_cycles_zoom-20.svg`

Same as the raw view above, x-axis clipped to the first 20 cycles. The doublet alternation
is still visible but the early adaptation in the long-ISI cycles is more apparent because
the long stable tail no longer compresses the y-axis. Stats from this run: early=11.20,
late=7.90, gain=1.42.

**Constant-current, doublet-filtered (recommended):**
```bash
python scripts/plot_spike_cycles.py \
    metrics/fractional_lif_constant_current_spike_cycles.csv \
    --min-cycle-length 5 \
    --output images/fractional_lif_memory_spike_cycles_zoom-mincyc-5.svg
```
→ `images/fractional_lif_memory_spike_cycles_zoom-mincyc-5.svg`

Drops the within-doublet length-2 cycles (197 of 394), leaving only the between-doublet ISIs
that carry the adaptation signal. Y-axis now spans the meaningful range (~13–18 steps) and
the trend is visible at a glance. Stats: early=17.10, late=13.00, gain=1.32.

**Constant-current, doublet-filtered + zoomed to first 30 cycles, smoothed overlay:**
```bash
python scripts/plot_spike_cycles.py \
    metrics/fractional_lif_constant_current_spike_cycles.csv \
    --min-cycle-length 5 \
    --max-cycles 30 \
    --smooth-window 5 \
    --output images/fractional_lif_memory_spike_cycles_zoom.svg
```
→ `images/fractional_lif_memory_spike_cycles_zoom.svg`

Black moving-average line on top of the raw long-ISI scatter, x-axis clipped to the early
adaptation portion before convergence flattens the trend.

**Doublet-filtered + zoomed to first 20 raw cycles:**
```bash
python scripts/plot_spike_cycles.py \
    metrics/fractional_lif_constant_current_spike_cycles.csv \
    --min-cycle-length 5 \
    --max-cycles 20 \
    --output images/fractional_lif_memory_spike_cycles_zoom-mincyc-5-zoom-20.svg
```
→ `images/fractional_lif_memory_spike_cycles_zoom-mincyc-5-zoom-20.svg`

Tight view of the very earliest adaptation — `--max-cycles 20` filters by the original
cycle index, so combined with `--min-cycle-length 5` only ~10 long-ISI cycles survive.
Note: with only 10 cycles and the default `--early-count 10 --late-count 10`, the early
and late windows fully overlap, so the printed gain is 1.00 — an artifact of the window,
not the data. Use a wider zoom or shrink `--early-count`/`--late-count` for a meaningful
ratio.

**Doublet-filtered + zoomed to first 40 raw cycles:**
```bash
python scripts/plot_spike_cycles.py \
    metrics/fractional_lif_constant_current_spike_cycles.csv \
    --min-cycle-length 5 \
    --max-cycles 40 \
    --output images/fractional_lif_memory_spike_cycles_zoom-mincyc-5-zoom-40.svg
```
→ `images/fractional_lif_memory_spike_cycles_zoom-mincyc-5-zoom-40.svg`

Wider window (~20 long-ISI cycles after filtering) — the early/late windows no longer
overlap, so the gain (1.31) is meaningful and the visible trend matches the unzoomed
filtered view's 1.32.

**Dropout/recovery (multi-phase coloring auto-enabled by the CSV's `phase` column):**
```bash
python scripts/plot_spike_cycles.py \
    metrics/fractional_lif_dropout_recovery_spike_cycles.csv \
    --min-cycle-length 5 \
    --output images/fractional_lif_recovery_spike_cycles.svg
```
→ `images/fractional_lif_recovery_spike_cycles.svg`

Startup vs recovery points are colored separately; transitions are marked with dashed
verticals. Per-phase early-means and the `recovery_gain` ratio (startup_early /
recovery_early) are printed to stdout.

**Constant-current, custom early/late summary windows:**
```bash
python scripts/plot_spike_cycles.py \
    metrics/fractional_lif_constant_current_spike_cycles.csv \
    --min-cycle-length 5 \
    --early-count 6 --late-count 20 \
    --output images/fractional_lif_memory_spike_cycles-early6-late20.svg
```
→ `images/fractional_lif_memory_spike_cycles-early6-late20.svg`

Tightens the early window and widens the late window — sharpens the printed convergence
ratio when adaptation is fast.

## LIF Unit Tests

4430.01ns INFO     cocotb.regression                  ****************************************************************************************************
** TEST                                        STATUS  SIM TIME (ns)  REAL TIME (s)  RATIO (ns/s) **
****************************************************************************************************
** test_lif.test_lif_reset                      PASS          40.00           0.03       1159.97  **
** test_lif.test_lif_clear                      PASS         190.00           0.00      76989.45  **
** test_lif.test_lif_no_spike_below_threshold   PASS         670.00           0.01      57549.17  **
** test_lif.test_lif_spike_above_threshold      PASS          90.00           0.00      65547.38  **
** test_lif.test_lif_membrane_accumulation      PASS         670.00           0.01      80148.98  **
** test_lif.test_lif_consecutive_spiking        PASS         130.00           0.00      29941.22  **
** test_lif.test_lif_reset_by_subtraction       PASS         130.00           0.00      32026.99  **
** test_lif.test_lif_multiple_inferences        PASS         710.00           0.02      39464.03  **
** test_lif.test_lif_negative_input             PASS         670.00           0.01      44917.66  **
** test_lif.test_lif_beta_decay                 PASS         670.00           0.01      65017.44  **
** test_lif.test_lif_enable_hold                PASS         190.00           0.00      43310.75  **
** test_lif.test_lif_varying_current            PASS         270.00           0.01      29898.41  **
****************************************************************************************************
** TESTS=12 PASS=12 FAIL=0 SKIP=0                           4430.01           0.16      28100.29  **
****************************************************************************************************


8230.01ns INFO     cocotb.regression                  *************************************************************************************************************************************
** TEST                                                                         STATUS  SIM TIME (ns)  REAL TIME (s)  RATIO (ns/s) **
*************************************************************************************************************************************
** test_fractional_lif.test_fractional_lif_reset                                 PASS          60.00           0.04       1336.93  **
** test_fractional_lif.test_fractional_lif_clear                                 PASS         610.00           0.01      50945.33  **
** test_fractional_lif.test_fractional_lif_no_spike_small_input                  PASS        2670.00           0.06      42202.10  **
** test_fractional_lif.test_fractional_lif_spike_large_input                     PASS         200.00           0.00     117553.36  **
** test_fractional_lif.test_fractional_vs_standard_lif_pulse_response            PASS        3060.00           0.04      85861.45  **
** test_fractional_lif.test_fractional_lif_matches_fixed_point_golden_baseline   PASS        1630.00           0.03      65047.77  **
** test_fractional_lif.test_fractional_lif_matches_fixed_point_golden_hist64     PASS           0.00           0.00          0.00  **
*************************************************************************************************************************************
** TESTS=7 PASS=7 FAIL=0 SKIP=0                                                              8230.01           0.21      38455.28  **
*************************************************************************************************************************************



13440.00ns INFO     cocotb.regression                  *****************************************************************************************************************************
** TEST                                                                 STATUS  SIM TIME (ns)  REAL TIME (s)  RATIO (ns/s) **
*****************************************************************************************************************************
** test_bitshift_lif.test_bitshift_lif_reset                             PASS          60.00           0.03       1899.45  **
** test_bitshift_lif.test_bitshift_lif_clear                             PASS        2850.00           0.16      17728.56  **
** test_bitshift_lif.test_bitshift_lif_enable_hold_behavior              PASS         800.00           0.04      17783.50  **
** test_bitshift_lif.test_bitshift_lif_matches_fixed_point_golden        PASS        9730.00           0.48      20083.71  **
** test_bitshift_lif.test_bitshift_lif_mode0_simple_profile              PASS           0.00           0.00          0.00  **
** test_bitshift_lif.test_bitshift_lif_mode3_custom_slow_decay_profile   PASS           0.00           0.00          0.00  **
*****************************************************************************************************************************
** TESTS=6 PASS=6 FAIL=0 SKIP=0                                                     13440.00           0.75      17804.85  **
*****************************************************************************************************************************

## LIF Data Formats

### lif.sv

`THRESHOLD` = 8192, 1.0 in QS2.13
`BETA` = 115, ~0.9 in Q1.7

input `current` in QS2.13 when using default DATA_WIDTH = 16
- Unit tests use values in decimal such as 0.5, ~0.1, 1.5, ~0.3, ~1.2, ~0.55, ~0.3, etc.

output `membrane_out` is MEMBRANE_WIDTH bits (default 24), QS10.13 — same fractional scale as the input, with extra integer headroom (10 integer bits vs. 2 in QS2.13).

internal `membrane_potential` has 13 fractional bits, based on the input current and threshold constant format.
- Functionally identical to `membrane_out` (both flopped from `next_membrane` on the same edge); the duplicate names mark "internal recurrence state" vs. "public output," not separate values. Same relationship between `spike_prev` and `spike_out`. Synthesis may merge each pair via register equivalence, but it is not guaranteed.

`decay_temp` is QS11.20 (currently hardcoded to 32 bits wide, signed)
- This is due to the formats of `membrane_potential` (QS10.13) and BETA (Q1.7) which is sign-extended to QS1.7. The result is QS11.20, as the multiplication produces 2 sign bits, so 1 is redundant. The natural product width is 33 bits; the 32-bit declaration drops exactly that redundant sign bit, so no information is lost.

`decay_potential` is QS10.13
- It's `decay_temp` arithmetically right-shifted by 7, then cast to MEMBRANE_WIDTH (24 bits).
- Represents beta * mem in a format with 13 fractional bits.
- The shift relocates the binary point (32-bit QS11.20 → 32-bit QS18.13, same value, 7 fractional bits become sign-extension fill). The 32→24-bit cast then drops 8 MSBs, taking QS18.13 → QS10.13. The cast is lossless of *value* (those 8 MSBs are always sign-extension) only because BETA < 1 contracts the magnitude — pushing BETA toward 1.0 would invalidate this.

`next_membrane`, `current_extended`, and `decay_potential` are all MEMBRANE_WIDTH bits wide
- Max value for input current is about 4 (in format QS2.13).
- The 24-bit three-operand adder is **not** worst-case-overflow-safe — three 24-bit signed operands would need 26 bits to never overflow. It works because of bounded inputs: at sustained max-current with continuous spiking, steady-state |membrane| is ~30, max |current| ≈ 4, |reset| = 1, sum magnitude ≤ ~35 — easily inside QS10.13's ±1024 range. Anyone widening DATA_WIDTH or pushing BETA toward 1.0 should re-check this bound.

### fractional_lif.sv
**TODO:** Update with values post-C_SCALED removal.

`THRESHOLD` = 8192 (default), 1.0 in QS2.13

GL Coefficients:
- 15 fractional bits, Q1.15 by default, HOWEVER, the core fractional-order models (16-4-32 and 32-4-16) use Q0.16 as the coefficients do not need any integer bits since they are all less than one. This likely does not make a huge difference, but it is worth noting.

`C_SCALED` = 256 (default), 1.0 in Q8.8
- C = 1 / dt^alpha = 1.000000
- Since dt is 1 for all of our models, C is going to be 1
- Only present in v1 fractional LIF. Subsequent models assumed a fixed dt, meaning C_SCALED would always be 1 and the expanded bit widths resulting from its inclusion were not beneficial and made synthesis more difficult.

`INV_DENOM` = 58982 (default),  ~0.9 in Q0.16
- INV_DENOM = 1 / (C + lambda)

input `current` in QS2.13 when using default DATA_WIDTH = 16
- Unit tests use values in decimal such as ~0.6, ~0.03, ~0.7, ~0.35, ~0.8, ~0.2, ~1.2, 0.5, -0.1

All intermediate widths derive from `MEMBRANE_WIDTH`, `COEFF_WIDTH`, `COEFF_FRAC_BITS`, `C_SCALED_FRAC_BITS`, `INV_DENOM_FRAC_BITS`, `ACCUM_GUARD_BITS`, and `NUMERATOR_GUARD_BITS`. Changing the membrane or frac-bit constants ripples through every signal; the two GUARD_BITS are local knobs.

#### MAC signals
`mac_hist_val`
- 24-bit signed, QS10.13 (same as `membrane_potential`)

`mac_coeff_mag`
- 16-bit unsigned, QU1.15 (by default)

`mac_product`
- 41 bits wide, QS12.28
- Result of `mac_hist_val` * `mac_coeff_mag` (converted to signed), so add fractional bits (13 + 15) and integer bits (10 + 1) + 1 for signed conversion

`history_sum_acc`
- 44 bits width, QS15.28
- `ACCUM_GUARD_BITS = 3` adds the headroom over PRODUCT_WIDTH. Theoretical worst case for summing 63 signed products would need `$clog2(64) = 6` guard bits, but for GL coefficients (mixed-sign partial sums, decreasing magnitude) 3 is empirically sufficient and was chosen as a timing/area sweet spot. Re-evaluate if porting to a different α or non-GL kernel.

#### Prep numerator
`prep_scaled_history_mult`
- 61 bits, QS24.36
- Result of C_SCALED (signed) * `history_sum_acc`, which is QS8.8 * QS15.28.

`prep_scaled_history`
- Drops 23 fractional bits to become QS47.13
- The shift discards 15 LSBs of fractional precision (28 → 13 frac bits). Sub-LSB relative to the final QS_.13 scale, so benign for output, but the MAC's extra precision is intentionally collapsed here. Pure-fractional drop — cannot cause overflow.

`prep_numerator`
- QS48.13

#### Reciprocal multiply
`mul_scaled_result`
- QS49.29
- Result of `numerator_reg` (registered result of `prep_numerator`, which is QS48.13) * `INV_DENOM` (signed) (Q0.16 + sign bit)

#### Shift to divide-by-scale
`div_membrane_pre_reset`
- QS65.13
- Result of `mul_scaled_result_reg` (registered result of `mul_scaled_result`, which is QS49.29) with a right arithmetic shift by the number of fractional bits in `INV_DENOM`
- Same character as the `>>> 23` above: drops 16 LSBs of fractional precision (29 → 13). Pure-fractional drop, sub-LSB at QS_.13 scale.

#### Finalize
`finalize_membrane`
- 24 bits, QS10.13. Three steps in [fractional_lif.sv:179-191](common/sv/neurons/fractional_lif.sv#L179-L191):
  1. **Reset subtraction in the wide domain.** `reset_subtract` is sign-extended from 24 bits to SCALED_RESULT_WIDTH (79 bits) so the subtraction happens at QS65.13 — the membrane format matches (both 13 frac bits), only the integer headroom is wider.
  2. **Saturation against MEMBRANE_MAX / MEMBRANE_MIN** (≈ ±1024 in real terms). This is the only place in the entire datapath that catches genuine out-of-range values; every prior stage uses wider containers and cannot saturate. Saturation is a safety net, not a normal operating mode — with C ≈ 1, INV_DENOM ≈ 0.9, and bounded inputs the membrane should never reach ±1024 in real operation. Saturation firing in a waveform is a diagnostic that something upstream is wrong (corrupted coefficients, untrained dynamics blowing up, parameter mismatch).
  3. **Truncation to MEMBRANE_WIDTH** in the in-range branch: keeps the bottom 24 bits. Lossless of value because the upper 55 bits are all sign-extension once the saturation guard has passed.

### bitshift_lif.sv
**TODO:** Update with values post-C_SCALED removal.

`THRESHOLD` = 8192 (default), 1.0 in QS2.13

`C_SCALED` = 256 (default), 1.0 in Q8.8
- C = 1 / dt^alpha = 1.000000
- Since dt is 1 for all of our models, C is going to be 1

`INV_DENOM` = 58982 (default),  ~0.9 in Q0.16
- INV_DENOM = 1 / (C + lambda)

input `current` in QS2.13 when using default DATA_WIDTH = 16
- Unit tests use values in decimal such as ~0.6, ~0.8, ~1.2, ~0.3, ~0.1, 0.75, ~0.2, -0.05, ~0.4, ~0.55, -0.1, ~0.95, ~0.15

All intermediate widths derive from `MEMBRANE_WIDTH`, `C_SCALED_FRAC_BITS`, `INV_DENOM_FRAC_BITS`, `ACCUM_GUARD_BITS`, and `NUMERATOR_GUARD_BITS`. Changing the membrane or frac-bit constants ripples through every signal; the two GUARD_BITS are local knobs.

#### Accumulator signals
`accum_hist_val`, `accum_shifted_hist`
- 24-bit signed, QS10.13 (same as `membrane_potential`)

`accum_shifted_hist_ext`
- 30-bit signed to account for possible overflow due to summing of all history terms
- QS16.13

`accum_next`
- 30-bit signed to account for possible overflow due to summing of all history terms
- QS16.13

`history_sum_acc`
- 30-bit signed to account for possible overflow due to summing of all history terms
- QS16.13
- `ACCUM_GUARD_BITS = $clog2(HISTORY_LENGTH)` adds the headroom over MEMBRANE_WIDTH. It would likely be safe to reduce this slightly if it would result in synthesis improvements, as the worst case with adding maximum values is unlikely or not possible.


#### Prep numerator
`prep_scaled_history_mult`
- 47 bits, QS24.21
- Result of C_SCALED (signed) * `history_sum_acc`, which is QS8.8 * QS16.13.

`prep_scaled_history`
- Drops 8 fractional bits to become QS33.13

`prep_numerator`
- QS34.13

#### Reciprocal multiply
`mul_scaled_result`
- QS35.29
- Result of `numerator_reg` (registered result of `prep_numerator`, which is QS34.13) * `INV_DENOM` (signed) (Q0.16 + sign bit)

`mul_membrane_pre_reset`
- QS51.13
- Shift to scale back down to 13 fractional bits
- Could probably consider dropping upper bits of this value

#### Finalize
`membrane_after_reset`
- 65 bits, QS51.13
- Result of `membrane_pre_reset_reg` (QS51.13) - sign-extended `reset_subtract`

`post_finalize_membrane`
- Lower `MEMBRANE_WIDTH` bits of `membrane_after_reset`
- QS10.13

### Membrane Potential Bit Width
This section provides conservative steady-state bounds on membrane magnitude using a
real-valued input bound and the documented default decay parameters. These bounds
assume no reset (spike_prev=0), which yields the largest steady-state magnitude. If
you want the steady-state bound under continuous spiking, replace I_max with
I_max - THRESHOLD.

Defaults used below:
- I_max = 4.8 (cart-pole max input; note this exceeds QS2.13 range)
- THRESHOLD = 1.0
- beta = 0.9 -> lambda = (1 - beta) / beta = 0.111111...
- C = 1 (dt = 1)
- FRAC_BITS = 13

General sizing rule:
- Integer bits needed: ceil(log2(|V_max| + 1))
- Total width = 1 (sign) + integer_bits + FRAC_BITS

#### lif.sv (beta LIF)
No-reset recurrence (maximal steady-state magnitude):
- V[n] = beta * V[n-1] + I_max
- V_max = I_max / (1 - beta) = 4.8 / 0.1 = 48.0

Continuous-spike steady-state (reset every cycle):
- V_max_spike = (I_max - THRESHOLD) / (1 - beta) = 3.8 / 0.1 = 38.0

Sizing example (conservative, no-reset):
- integer_bits = ceil(log2(48.0 + 1)) = 6
- total width = 1 + 6 + 13 = 20 bits

#### fractional_lif.sv (GL coefficients)
For 0 < alpha <= 1, g_k <= 0 for k >= 1, so use magnitudes |g_k|. With C = 1:
- V[n] = (I_max + sum_{k=1..H-1} |g_k| * V[n-k]) / (1 + lambda)
- V_max = I_max / (1 + lambda - sum |g_k|)

Using alpha = 0.5, HISTORY_LENGTH = 64:
- sum |g_k| = 0.9290596866
- denom = 1 + lambda - sum |g_k| = 0.1820514245
- V_max = 4.8 / 0.1820514245 = 26.37

Continuous-spike steady-state:
- V_max_spike = 3.8 / 0.1820514245 = 20.87

Sizing example (conservative, no-reset):
- integer_bits = ceil(log2(26.37 + 1)) = 5
- total width = 1 + 5 + 13 = 19 bits

#### bitshift_lif.sv (shifted-history kernel)
Let a_k = 2^(-shift[k]) from the compile-time shift profile. With C = 1:
- V[n] = (I_max + sum_{k=1..H-1} a_k * V[n-k]) / (1 + lambda)
- V_max = I_max / (1 + lambda - sum a_k)

Using SHIFT_MODE = custom_slow_decay (3), CUSTOM_DECAY_RATE = 3, HISTORY_LENGTH = 64:
- sum a_k = 0.935546875
- denom = 1 + lambda - sum a_k = 0.1755642361
- V_max = 4.8 / 0.1755642361 = 27.34

Continuous-spike steady-state:
- V_max_spike = 3.8 / 0.1755642361 = 21.64

Sizing example (conservative, no-reset):
- integer_bits = ceil(log2(27.34 + 1)) = 5
- total width = 1 + 5 + 13 = 19 bits

Notes:
- If you clamp inputs to QS2.13 (|I| <= ~4.0), substitute I_max = 4.0.
- Weight quantization does not constrain input current; use the true bound of
    your observation pipeline.
- If alpha, HISTORY_LENGTH, or shift profile changes, recompute the coefficient
    sum and V_max.
