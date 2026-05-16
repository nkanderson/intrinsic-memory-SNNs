# Training

## Models

After Optuna studies, five models were chosen for training:
- Smallest LIF: 32hl1, 16hl2 (trial 81 from leaky-1000ep study)
- A larger LIF: 64hl1, 16hl2 (trial 44 from leaky-1000ep study)
- Smallest fractional-order LIF: 16hl1, 4hl2, history length of 32
- Shortest history-length fractional-order LIF: 32hl1, 4hl2, history length of 16 (Original best model came from config with -run2 naming variant. The duplicative files have been cleaned, and the -run2 naming has been removed.)
- Custom slow decay bitshift LIF: 32hl1, 8hl2, history length of 8

In training, the best generalized model was chosen from each.
The bitshift model was quantized to QS3.12, the others were all quantized to QS2.13.

## Steps

1. Training:
```bash
python main.py --config configs/optimized-leaky-32hl1-16hl2.yaml
python main.py --config configs/optimized-leaky-64hl1-16hl2.yaml
python main.py --config configs/fractional-16hl1-4hl2-32hist.yaml
python main.py --config configs/fractional-32hl1-4hl2-16hist.yaml
python main.py --config configs/bitshift-custom_slow_decay-32hl1-8hl2-8hist.yaml
```

2. Generate coefficients .mem file and SV module constants (for fractional and bitshift only):
```bash
python scripts/generate_coefficients.py --lam 0.17601697332161148 --output-dir ../../common/sv/cocotb/tests/weights/fractional-16-4-32 --coeff-frac-bits 16 --history-length 32
python scripts/generate_coefficients.py --lam 0.05263950403873735 --output-dir ../../common/sv/cocotb/tests/weights/fractional-32-4-16 --coeff-frac-bits 16 --history-length 16
python scripts/generate_coefficients.py --lam 0.09550310156350894 --output-dir ../../common/sv/cocotb/tests/weights/bitshift-custom_slow_decay --history-length 8 --constants-only
```

3. Generate quantized model .pth file for validation in software simulation (PyTorch):
```bash
python scripts/manage_weights.py export pytorch models/dqn_64hl1-16hl2-best.pth --bits 16 --frac 13 --signed
```

4. Run software simulation to validate quantized model:
```bash
python main.py --load models/dqn_fractional-64hl1-16hl2-best-quantized-QS2_13.pth --evaluate-only --max-episode-steps 5000
python main.py --load models/dqn_64hl1-16hl2-best-quantized-QS2_13.pth --evaluate-only
python main.py --load models/dqn_bitshift-custom_slow_decay-32hl1-8hl2-8hist-best-generalization-quantized-QS3_12.pth --evaluate-only --config configs/bitshift-custom_slow_decay-32hl1-8hl2-8hist.yaml
```

5. Export quantized weights to .mem files use in hardware simulation (using cocotb):
```bash
python scripts/manage_weights.py export hardware models/dqn_optimized-leaky-32hl1-16hl2-best-generalization.pth --bits 16 --frac 13 --output ../../common/sv/cocotb/tests/weights/lif-32-16
python scripts/manage_weights.py export hardware models/dqn_optimized-leaky-64hl1-16hl2-best-generalization.pth --bits 16 --frac 13 --output ../../common/sv/cocotb/tests/weights/lif-64-16
python scripts/manage_weights.py export hardware models/dqn_fractional-16hl1-4hl2-32hist-best-generalization.pth --bits 16 --frac 13 --output ../../common/sv/cocotb/tests/weights/fractional-16-4-32
python scripts/manage_weights.py export hardware models/dqn_fractional-32hl1-4hl2-16hist-run2-best-generalization.pth --bits 16 --frac 13 --output ../../common/sv/cocotb/tests/weights/fractional-32-4-16
```

## V2 Models

1. Training:
```bash
python main.py --config configs/bitshift-custom_slow_decay-32hl1-8hl2-8hist.yaml
python main.py --config configs/fractional-16hl1-4hl2-32hist.yaml
python main.py --config configs/fractional-32hl1-4hl2-16hist.yaml
python main.py --config configs/leaky-32hl1-16hl2.yaml
python main.py --config configs/leaky-64hl1-16hl2.yaml
```

2. **NOTE:** Not re-doing this step because it's the same as what was done above, but including it here to show it is a part of the process: Generate coefficients .mem file and SV module constants (for fractional and bitshift only):
```bash
python scripts/generate_coefficients.py --lam 0.09550310156350894 --output-dir ../../common/sv/cocotb/tests/weights/bitshift-custom_slow_decay --history-length 8 --constants-only
python scripts/generate_coefficients.py --lam 0.17601697332161148 --output-dir ../../common/sv/cocotb/tests/weights/fractional-16-4-32 --coeff-frac-bits 16 --history-length 32
python scripts/generate_coefficients.py --lam 0.05263950403873735 --output-dir ../../common/sv/cocotb/tests/weights/fractional-32-4-16 --coeff-frac-bits 16 --history-length 16
```

3. Generate quantized model .pth file for validation in software simulation (PyTorch):
```bash
python scripts/manage_weights.py export pytorch models/leaky-32hl1-16hl2-gen.pth --bits 16 --frac 13 --signed
python scripts/manage_weights.py export pytorch models/leaky-64hl1-16hl2-gen.pth --bits 16 --frac 13 --signed
python scripts/manage_weights.py export pytorch models/fractional-16hl1-4hl2-32hist-gen.pth --bits 16 --frac 13 --signed
python scripts/manage_weights.py export pytorch models/fractional-32hl1-4hl2-16hist-gen.pth --bits 16 --frac 13 --signed
python scripts/manage_weights.py export pytorch models/bitshift-custom_slow_decay-32hl1-8hl2-8hist-gen.pth --bits 16 --frac 12 --signed
```

4. Run software simulation to validate quantized model:
```bash
python main.py --load models/leaky-32hl1-16hl2-gen-quantized-QS2_13.pth --evaluate-only --max-episode-steps 1000
python main.py --load models/leaky-64hl1-16hl2-gen-quantized-QS2_13.pth --evaluate-only --max-episode-steps 1000
python main.py --load models/fractional-32hl1-4hl2-16hist-gen-quantized-QS2_13.pth --evaluate-only --max-episode-steps 1000
python main.py --load models/fractional-16hl1-4hl2-32hist-gen-quantized-QS2_13.pth --evaluate-only --max-episode-steps 1000
python main.py --load models/bitshift-custom_slow_decay-32hl1-8hl2-8hist-gen-quantized-QS3_12.pth --evaluate-only --max-episode-steps 1000
```

5. Export quantized weights to .mem files use in hardware simulation (using cocotb):
```bash
python scripts/manage_weights.py export hardware models/leaky-64hl1-16hl2-gen.pth --bits 16 --frac 13 --output ../../common/sv/cocotb/tests/weights/lif-64-16
python scripts/manage_weights.py export hardware models/leaky-32hl1-16hl2-gen.pth --bits 16 --frac 13 --output ../../common/sv/cocotb/tests/weights/lif-32-16
python scripts/manage_weights.py export hardware models/fractional-16hl1-4hl2-32hist-gen.pth --bits 16 --frac 13 --output ../../common/sv/cocotb/tests/weights/fractional-16-4-32
python scripts/manage_weights.py export hardware models/fractional-32hl1-4hl2-16hist-gen.pth --bits 16 --frac 13 --output ../../common/sv/cocotb/tests/weights/fractional-32-4-16 
python scripts/manage_weights.py export hardware models/bitshift-custom_slow_decay-32hl1-8hl2-8hist-gen.pth --bits 16 --frac 12 --output ../../common/sv/cocotb/tests/weights/bitshift-custom_slow_decay
```

## Visualizing training metrics

[`scripts/visualize_training_metrics.py`](train/scripts/visualize_training_metrics.py)
renders figures from the per-episode CSV logs that `main.py` writes to
`train/metrics/`. Given a directory it produces one comparison figure plus
one detail figure per CSV; given a single CSV it produces only that CSV's
detail figure.

```bash
# From train/. All core-v2 figures, PNG (default).
python scripts/visualize_training_metrics.py metrics/core-v2/ \
    --output-dir images/core-v2/

# SVG output (scales cleanly for presentations/papers).
python scripts/visualize_training_metrics.py metrics/core-v2/ \
    --output-dir images/core-v2/ --format svg

# One model only (detail figure only).
python scripts/visualize_training_metrics.py \
    metrics/core-v2/leaky-64hl1-16hl2-training-metrics.csv \
    --output-dir images/core-v2/
```

Useful flags:
- `--compare {running,both,efficiency,summary,stack-running,stack-generalization}`
  — what the comparison output is (default `running`). See the
  descriptions below.
- `--comparison-only` / `--details-only` — restrict output to one or the other.
- `--no-raw` — drop the noisy per-episode scatter in detail figures and
  in `stack-running`.
- `--roll-window N` — width in episodes of the rolling-std band around the
  running average in detail figures and `stack-running` (default 100).

### Reading the plots

All plots use the Okabe–Ito colorblind-safe palette. Comparison figures
have no in-figure title (the surrounding caption supplies it).

**Comparison outputs** — six options via `--compare`:

- `--compare running` (default) → `training_comparison_running.{png,svg}`.
  One solid line per model showing the 100-episode rolling mean of episode
  reward (`running_avg_100`). Each model also gets a single filled marker
  in its assigned shape at *(episode, value)* of `max(running_avg_100)`,
  so the eye can immediately compare both **when** and **how high** each
  model peaked. Higher and earlier rise = faster learning.
- `--compare both` → `training_comparison.{png,svg}`. Same running-average
  lines plus a thin dashed trace of `generalization_avg` per model with
  sparse anchor markers (color + shape together identify each model).
- `--compare efficiency` → `training_comparison_efficiency.{png,svg}`. One
  point per model at *(first episode where `generalization_avg` reached
  500, `max(running_avg_100)`)*. Upper-left = best: reached perfect
  generalization early **and** held a high running average. Lower-right =
  late convergence with weak sustained performance. Each point is labeled
  in-line. Models that never hit `generalization_avg = 500` are omitted
  with a stderr warning.
- `--compare summary` → markdown table printed to stdout, plus
  `training_comparison_summary.csv` saved on disk. No figure is written.
  Columns: `best_gen_avg`, `first_ep_at_gen_500`, `best_running_avg`,
  `final_running_avg`. Best when precise numbers matter more than a
  visual pattern.
- `--compare stack-running` →
  `training_comparison_stack_running.{png,svg}`. Small-multiples: one row
  per model showing the same content as the detail figure's top panel
  (raw scatter, running-avg line, rolling-std band, dashed
  best-running-avg-to-date). Save-event rug ticks are intentionally
  omitted to keep the stacked view clean.
- `--compare stack-generalization` →
  `training_comparison_stack_generalization.{png,svg}`. Small-multiples:
  one row per model showing a simplified version of the detail figure's
  bottom panel (generalization mean line + dashed best-to-date line +
  triangles only at episodes where the saved best-gen value hit 500).
  Per-seed min-max and IQR bands are intentionally omitted to keep the
  stacked view clean.

**Per-model detail figure** (`{model}.{png,svg}`) — two stacked panels
sharing the x-axis (episode). Both panels' legends sit in the upper-left
because several models bottom out near zero late in training.

Top panel — training reward (rolling average):
- Faint gray dots: raw `episode_reward` — every individual training
  episode. Very noisy. Hide with `--no-raw`.
- Blue line: `running_avg_100` — the 100-episode rolling mean of the raw
  rewards.
- Light blue band: `running_avg_100 ± rolling std` over `--roll-window`
  episodes of `episode_reward`. Narrow band = stable policy; wide band =
  high variance across episodes.
- Orange dashed line: `best_running_avg_100` — the running maximum of
  `running_avg_100` seen so far in training.
- Orange rug ticks along the top edge: episodes where
  `saved_best_running_model == 1`, i.e. a new best-running-avg checkpoint
  was written to disk.

Bottom panel — generalization:
- Pale green band: seed min–max range at each episode, computed from the
  pipe-delimited per-seed rewards in `generalization_rewards`.
- Darker green band: seed interquartile range (25th–75th percentile) from
  the same per-seed rewards. A tight IQR with a wide min–max means
  occasional bad seeds; both bands narrowing together means consistent
  cross-seed behavior.
- Green line: `generalization_avg` — the mean across the 30 eval seeds.
- Orange dashed line: `best_generalization_avg` — running max of the mean.
- Orange triangles: episodes where `saved_best_generalization_model == 1`,
  i.e. a new best-generalization checkpoint was written.

### What is raw vs. derived

| Shown on plot | Source column(s) | Derivation |
|---|---|---|
| Running-avg line | `running_avg_100` | Raw column from `main.py`. |
| Running-avg band | `episode_reward` | Rolling std over `--roll-window`, centered on `running_avg_100`. |
| Generalization mean line | `generalization_avg` | Raw. |
| Generalization IQR band | `generalization_rewards` | 25th/75th percentile of pipe-split per-seed rewards, per row. |
| Generalization min–max band | `generalization_rewards` | min/max of pipe-split per-seed rewards, per row. |
| Best-avg line | `best_running_avg_100` | Raw. |
| Best-gen line | `best_generalization_avg` | Raw. |
| Save-event marks | `saved_best_*_model` | Raw (boolean flags). |

## Visualizing multi-seed retraining

After hyperparameter optimization, each candidate configuration is retrained
across multiple random seeds with
[`multi_seed_train.py`](train/multi_seed_train.py) (or
[`run_multi_seed_parallel.py`](train/run_multi_seed_parallel.py) for parallel
execution across seeds). Both scripts write per-seed CSVs and an aggregate
summary CSV to `metrics/multi-seed/<config_name>/` by default, plus per-seed
model checkpoints to `models/<config_name>/`.

[`scripts/plot_multi_seed_curves.py`](train/scripts/plot_multi_seed_curves.py)
consumes those CSVs and produces six figures intended for paper-quality
reporting against the standard CartPole-v1 success criterion (mean reward ≥
475 over the final 100 episodes). All plots use the Okabe–Ito colorblind-safe
palette and report per-config success counts (K/N seeds reaching the
threshold) prominently. The aggregate central line is the **interquartile
mean (IQM)** across seeds, following Agarwal et al. (2021); single-value
statistics (e.g., the bar heights in `final_bars`) use the IQM with **95%
bootstrap confidence intervals**; per-episode time-series bands use the
**IQR (25th–75th percentile)** across seeds.

```bash
# From train/. Single config, default png output.
python scripts/plot_multi_seed_curves.py \
    --config-name fractional-64-8-8 \
    --metrics-dir metrics/multi-seed \
    --output-dir images/multi-seed/

# Cross-config comparison (repeat --config-name; produces all 6 figures).
python scripts/plot_multi_seed_curves.py \
    --config-name fractional-64-8-8 \
    --config-name fractional-64-32-8 \
    --config-name leaky-32-16 \
    --metrics-dir metrics/multi-seed \
    --output-dir images/multi-seed/

# Same as above but produce the stacked variant of the cross-config
# learning curve (one row per config, shared x-axis and y-limits).
python scripts/plot_multi_seed_curves.py \
    --config-name fractional-64-8-8 \
    --config-name fractional-64-32-8 \
    --config-name leaky-32-16 \
    --metrics-dir metrics/multi-seed \
    --output-dir images/multi-seed/ \
    --stacked

# SVG output for paper figures.
python scripts/plot_multi_seed_curves.py \
    --config-name fractional-64-8-8 \
    --metrics-dir metrics/multi-seed \
    --output-dir images/multi-seed/ \
    --format svg
```

Useful flags:
- `--config-name NAME` — repeatable. Each one is loaded from
  `metrics-dir/NAME/` (the nested layout written by `multi_seed_train.py`)
  with a fallback to `metrics-dir/` (older flat layout).
- `--metrics-dir DIR` — root directory containing the per-config
  subdirectories. Default `metrics/multi-seed`.
- `--output-dir DIR` — where plot files are written. Default
  `images/multi-seed`.
- `--format {png,svg,pdf}` / `-f` — output format. PNG (default) is at 150
  dpi; SVG/PDF are vector and preferred for paper figures.
- `--stacked` — replaces the overlaid cross-config learning curve with a
  one-row-per-config stacked layout (shared x-axis, shared y-limits,
  per-config color, faint per-seed lines + IQM line + IQR band in each
  panel). Only affects the cross-config figure; other plots are unchanged.
  Mirrors the stacked variant in `visualize_training_metrics.py`.

### Reading the plots

Six figures are produced. Plots that depend on at least two configs
(`cross_config-learning_curves`, `convergence_strip`) are only written when
two or more `--config-name` values are supplied.

- `{config}-learning_curve.{ext}` — per-config learning curve.
  - Light gray lines: each individual seed's `running_avg_100` (the
    trailing 100-episode mean of episode rewards).
  - Bold blue line: per-episode **IQM** across seeds (mean of the middle
    50% after sorting at each episode).
  - Light blue band: per-episode **IQR** (25th–75th percentile across
    seeds).
  - Vermillion dashed line: 475 success threshold.
  - Annotation top-left: `solved: K/N seeds` (count of seeds whose
    `final_avg_reward` ≥ 475).

- `cross_config-learning_curves.{ext}` — overlay of per-config IQM lines
  with their IQR bands. Each config gets a distinct Okabe–Ito color and
  the legend includes its `[solved K/N]` count. Useful for direct
  side-by-side comparison of two or more configurations' learning
  trajectories and converged-phase stability.

- `cross_config-learning_curves_stacked.{ext}` *(only with `--stacked`)* —
  one row per config, shared x-axis and shared y-limits (so vertical
  comparison is honest even when one config never approaches 500). Each
  panel shows the same content as a single-config learning curve (faint
  per-seed lines, IQM line, IQR band, 475 reference line) in that config's
  Okabe–Ito color, with the config name and K/N annotated in the upper
  left. A single shared legend lives in the bottom panel. Preferred over
  the overlaid variant when bands across configs would mutually obscure.

- `convergence_strip.{ext}` — one column per config. Each converged seed
  contributes a dot at *(config, convergence_episode)* with horizontal
  jitter for visibility, where `convergence_episode` is the first
  episode at which the trailing 100-ep average reached 475 and never
  subsequently dropped below. A short horizontal bar marks the median
  convergence episode per config. The K/N count for each column is
  annotated near the bottom of the data area. Configs where no seeds
  converged appear as an empty column with the K/N annotation visible.

- `final_bars.{ext}` — bar chart per config:
  - Blue bars: `final_avg_reward` IQM across seeds with asymmetric
    error bars showing the 95% bootstrap CI.
  - Green bars: `best_avg_reward` IQM (peak 100-ep average reached
    during training) with the same CI treatment.
  - Vermillion dashed line: 475 threshold.
  - Text above each pair: `K/N solved` count for that config.
  - y-axis capped at 520 so bars are interpretable on the CartPole-v1
    scale.

- `performance_profile.{ext}` — empirical performance profile
  (Dolan–Moré curve) per config, following Agarwal et al. (2021). For
  each score threshold τ ∈ [0, 500] (x-axis), the y-axis shows the
  fraction of seeds with `final_avg_reward` ≥ τ. Curves that stay flat
  longer (further right) represent configurations whose seed
  distributions are uniformly better. The y-value at the vermillion
  dashed line (τ = 475) is the success rate K/N for each config; the
  K/N count also appears in the legend.

- `{config}-loss_curve.{ext}` — per-config DQN loss trajectory. Same
  IQM-line + IQR-band treatment as the learning curves, plotted on
  `avg_loss` per episode. Useful for diagnosing optimization
  instability separately from reward trajectory.

### What is raw vs. derived

| Shown on plot | Source column(s) | Derivation |
|---|---|---|
| Per-seed lines (learning curve) | `running_avg_100` | Raw per-episode column from `train_fn.train`. |
| IQM central line | `running_avg_100` (or `avg_loss`) | Mean of middle 50% across seeds per episode. |
| IQR band | `running_avg_100` (or `avg_loss`) | 25th–75th percentile across seeds per episode. |
| Bar height (final_bars) | `final_avg_reward`, `best_avg_reward` | IQM across seeds (single value per config). |
| Bar error bars (final_bars) | same | Percentile bootstrap 95% CI for IQM, 10,000 resamples. |
| K/N "solved" count | `final_avg_reward` | Count of seeds where final_avg ≥ 475. |
| Convergence dots | `convergence_episode` | Raw per-seed column from summary CSV. |
| Convergence median bar | `convergence_episode` | Median over the converged subset of seeds. |
| Performance profile y | `final_avg_reward` | Fraction of seeds with score ≥ τ, evaluated at each τ on [0, 500]. |

### Methodology notes

The reporting framing across these plots intentionally separates:

- The **success metric** for individual reporting, which follows the
  standard CartPole-v1 criterion `final_avg_reward ≥ 475` over the final
  100 episodes. K/N counts on each plot reflect this metric directly.
- The **aggregate central tendency**, reported as the IQM rather than
  the mean. Per Agarwal et al. (2021, NeurIPS), the IQM is more robust
  to outlier seeds than the mean and more informative than the median
  at small N (it uses 50% of the data rather than a single point).
- The **uncertainty quantification**, reported via 95% percentile
  bootstrap CIs on the IQM for single-value statistics, and per-episode
  IQR bands for time-series plots. With N = 10 seeds the bootstrap CIs
  are intentionally wide; Colas et al. (2018) note that N ≤ 5 is
  under-powered for typical deep-RL effect sizes, so wider intervals
  are an honest reflection of the sample size.

## Fractional Dynamics Verification

To ensure the correctness of the Python `FractionalLIF` model, we provide a suite of automated unit tests and visualization scripts in `train/scripts/`.

### Automated Tests (`test_fractional_lif.py`)
- **`test_history_dependence`**: Validates that neurons with identically matched instantaneous membrane potentials but different historical trajectories diverge over time, a core requirement of fractional order memory.
- **`test_alpha_1_classic_lif`**: Ensures the model cleanly reverts to standard, history-independent Markovian leaky integrate-and-fire behavior when $\alpha = 1.0$.
- **`test_spike_frequency_adaptation`**: Verifies that fractional order neurons ($\alpha < 1.0$) exhibit **Spike Frequency Acceleration** under constant input current. The Inter-Spike Interval (ISI) decreases monotonically over time because past spikes contribute positive historical momentum (negative GL coefficients) that makes the neuron easier to fire.

### Sub-threshold Dynamics Visualization
`plot_python_subthreshold.py` applies a square-wave input to neurons with varying $\alpha$ values, with a high threshold to prevent spiking. 

The resulting charge and discharge curves illustrate the fundamental tradeoff in fractional dynamics:
- **Higher $\alpha$** values ($\alpha \approx 1.0$) receive a massive, immediate feedback boost from their most recent state (since $|g_1| = \alpha$). They charge up to a much higher peak extremely quickly, but their memory drops off rapidly, causing them to discharge completely very quickly.
- **Lower $\alpha$** values ($\alpha \approx 0.1$) have a sluggish initial response, but their historical memory coefficients decay via a very slow power-law. As a result, they retain their charge over incredibly long time horizons, producing the characteristic "flattened" curves with extremely long tails.

#### The Mittag-Leffler Transition
By passing the `--log` flag, the script plots the discharge phase on a **log-log** scale. On a log-log plot, an exponential decay drops almost vertically, while a pure power-law decay forms a straight line. 
Because the `FractionalLIF` model incorporates a leak term ($\lambda = 0.1$), the governing mathematical solution is not a pure power law, but rather the **Mittag-Leffler function**. This function is characterized by an exponential-like drop at early time steps that smoothly transitions into a heavy power-law tail. The log-log plot beautifully captures this "knee" transition, with the curves eventually straightening out into parallel power-law tails at large timescales.

### Spike Frequency Adaptation Visualization
`plot_python_spike_adaptation.py` feeds a constant low current ($0.3$) into the neuron and plots the Inter-Spike Interval (ISI) over the sequence of emitted spikes. 
The script generates stacked subplots for different $\alpha$ values against a baseline `snnTorch.Leaky` neuron. The baseline and $\alpha=1.0$ maintain a perfectly constant firing rate, while the lower fractional orders clearly display inverse power-law acceleration, their ISIs plummeting over time as the long-term memory buffer accumulates positive historical charge.
