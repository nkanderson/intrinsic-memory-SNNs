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
python scripts/generate_coefficients.py --lam 0.05263950403873735 --output-dir ../../common/sv/cocotb/tests/weights/fractional-32-4-16 --coeff-frac-bits 16 --history-length 16
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
- `--compare {running,both,efficiency,summary}` — what the comparison
  output is (default `running`). See the descriptions below.
- `--comparison-only` / `--details-only` — restrict output to one or the other.
- `--no-raw` — drop the noisy per-episode scatter in detail figures.
- `--roll-window N` — width in episodes of the rolling-std band around the
  running average (default 100).

### Reading the plots

All plots use the Okabe–Ito colorblind-safe palette.

**Comparison outputs** — four options via `--compare`:

- `--compare running` (default) → `training_comparison_running.{png,svg}`.
  One solid line per model showing the 100-episode rolling mean of episode
  reward (`running_avg_100`). Higher and earlier rise = faster learning.
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
  `final_running_avg`. This is the right output for a small number of
  models where precise numbers matter more than a visual pattern.

**Per-model detail figure** (`{model}.{png,svg}`) — two stacked panels
sharing the x-axis (episode).

Top panel — training reward:
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
