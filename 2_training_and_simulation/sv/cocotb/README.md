# Cocotb Testing Environment for CartPole SNN

Docker-based cocotb testing environment for the LIF neuron and linear layer modules using Icarus Verilog.

## Features

- **Multi-stage Dockerfile** with layer caching to avoid reinstalling tools on minor changes
- **oss-cad-suite** bundle (includes Icarus Verilog, cocotb, and other EDA tools)
- **No virtual environment** - uses oss-cad-suite's bundled cocotb to avoid version conflicts
- **Volume mounting** for easy access to source files and test results
- **Live reload** - changes to source and test files are immediately visible in container
- **FST waveform tracing** - enabled by default for all tests

## Directory Structure

```
cocotb/
├── Dockerfile           # Multi-stage Docker build
├── docker-compose.yml   # Docker Compose configuration
├── README.md            # This file
├── tests/               # Test directory (mounted as read-write)
│   ├── test_lif.py          # Tests for LIF neuron module
│   ├── test_linear_layer.py # Tests for linear layer module
│   ├── Makefile             # Makefile for running tests
│   └── weights/             # Test weight files
│       ├── test_weights.mem # Identity matrix weights
│       └── test_bias.mem    # Zero bias values
└── results/             # Test results and waveforms (generated)
    └── sim_build/       # Simulation artifacts and waveforms
        ├── lif.fst          # LIF neuron waveform
        └── linear_layer.fst # Linear layer waveform
```

## Prerequisites

- Docker
- Docker Compose (optional but recommended)

## Quick Start

### 1. Build the Docker Image

```bash
cd cocotb
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose build
```

This builds a multi-stage image with:
- Ubuntu 22.04 base
- oss-cad-suite (includes Verilator, Icarus, and cocotb bundled together)
- Non-root user matching host user's UID/GID (defaults to 1000)

**Layer caching:** The oss-cad-suite installation is cached unless the version is updated.

**File ownership:** The container runs as a non-root user. By default, it uses UID/GID 1000. To match the host user:

```bash
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose build
```

### 2. Start the Container

```bash
docker compose up -d
```

### 3. Run Tests

```bash
# Enter the container
docker compose exec cocotb bash

# Run tests for LIF module, for example
make test_lif
```

To specify the simulator to be Verilator instead of the default of Icarus:
```bash
# Example clean and run tests with Verilator as the simulator
make clean && make test_linear_layer SIM=verilator
```
NOTE: Verilator is necessary for viewing unpacked arrays in waveforms.

Or run tests directly:

```bash
docker compose exec cocotb bash -c "make test_lif"
```

### 4. View Results

Test results and waveforms are written to `results/`:
- `results.xml` - JUnit-style test results
- `sim_build/<module name>.fst` - Waveform file location when using Icarus
- `dump.fst` - Waveform file location when using Verilator

### 5. Stop the Container

```bash
docker compose down
```

## Test Descriptions

### LIF Neuron Tests (`test_lif.py`)

| Test | Description |
|------|-------------|
| `test_lif_reset` | Verifies reset initializes neuron state |
| `test_lif_no_spike_below_threshold` | Small inputs don't cause spikes |
| `test_lif_spike_above_threshold` | Large inputs cause immediate spike |
| `test_lif_membrane_accumulation` | Membrane builds up over time and spikes |
| `test_lif_consecutive_spiking` | High sustained input causes consecutive spikes |
| `test_lif_reset_by_subtraction` | Verifies reset-by-subtraction prevents immediate re-spike |
| `test_lif_multiple_starts` | Module can be started multiple times |
| `test_lif_negative_input` | Negative inputs don't cause spikes |
| `test_lif_beta_decay` | Membrane decays when input removed |

### Linear Layer Tests (`test_linear_layer.py`)

| Test | Description |
|------|-------------|
| `test_linear_layer_reset` | Verifies reset initializes layer state |
| `test_linear_layer_identity_weights` | Identity weights pass inputs through unchanged |
| `test_linear_layer_multiple_runs` | Module can be started multiple times |
| `test_linear_layer_timing` | Verifies NUM_OUTPUTS valid outputs in correct timing |

### Fractional LIF Golden Tests (`test_fractional_lif.py`)

The file includes two bit-accurate golden checks with intentional profile gating:

- `test_fractional_lif_matches_fixed_point_golden_baseline`
    - Runs only for baseline H=8 profile (e.g., `make test_fractional_lif`)
    - Skips when `HISTORY_LENGTH != 8`
- `test_fractional_lif_matches_fixed_point_golden_hist64`
    - Runs only for H=64 variant profile (e.g., `make test_fractional_lif_hist64`)
    - Skips when `HISTORY_LENGTH != 64`

This avoids false failures from comparing H=8 assumptions against H=64 settings (and vice versa).

### CartPole Integration Tests (`test_cartpole_integration.py`)

| Test | Description |
|------|-------------|
| `test_cartpole_single_episode` | Runs one CartPole episode with hardware policy |
| `test_cartpole_multiple_episodes` | Runs 10 episodes + repeatability check |
| `test_inference_state_leakage` | Repeats fixed observation to detect state carryover |
| `test_cartpole_action_trace` | Writes per-step hardware action trace CSV (**FULL_DEBUG only**) |
| `test_cartpole_timestep_snapshots` | Writes per-cycle FC2/HL2/Q snapshot CSV (**FULL_DEBUG only**) |

For software-vs-hardware trace comparison workflow (`export_action_trace.py` / `compare_action_traces.py`), see `../../train/README.md`.

## Fractional CartPole Targets

The cocotb tests provide two fractional integration targets in `tests/Makefile`:

- `test_cartpole_integration_fractional` → history length 64 (`weights/fractional_order`)
- `test_cartpole_integration_fractional_hist8` → history length 8 (`weights/fractional_hist8`)

For the generalized history-8 model, use `test_cartpole_integration_fractional_hist8`.

### Determining `FC2_OUTPUT_WIDTH`

`FC2_OUTPUT_WIDTH` is the transport width used for the FC2 layer output before it is consumed by FC_OUT.
If this width is too small, FC2 values clip/saturate and policy behavior can collapse even when earlier layers look healthy.

#### Why this parameter matters

- Fractional models can produce larger FC2 dynamic range than expected from nominal observation bounds.
- Clipping in FC2 can silently distort action logits and produce stable but wrong action choices.
- The failure mode is often distribution-dependent, so simple smoke tests can miss it.

#### Practical sizing workflow

1. Start from an analytical estimate:
   - Bound FC2 accumulation from quantized FC1 activations and FC2 weights.
   - Add margin for bias and worst-case alignment of signs.
2. Choose the smallest candidate width that should cover that bound.
3. Validate in cocotb using observation sweeps and saturation counters.
4. Increase width until FC2 saturation counters remain zero in the sweep/regression set.

#### Current project setting

For both fractional integration profiles currently in `sv/cocotb/tests/Makefile`:

- `cartpole_integration_fractional` (history length 64)
- `cartpole_integration_fractional_hist8` (history length 8)

`FC2_OUTPUT_WIDTH=24` is used as the smallest width that remained stable in the saturation-focused integration checks.

#### Validation recommendation

When changing quantization, weights, or fractional constants, re-run the FC2 saturation check in cocotb and treat any non-zero saturation count as a width/regime mismatch requiring re-sizing.

## Recommended Commands (History-8)

Run from `cocotb/tests` inside container.

```bash
# 1) Multi-episode pass/fail performance check
make clean && make test_cartpole_integration_fractional_hist8 SIM=icarus TESTCASE=test_cartpole_multiple_episodes

# 2) Snapshot export for low-level diagnostics
FULL_DEBUG=1 make clean && make test_cartpole_integration_fractional_hist8 SIM=icarus TESTCASE=test_cartpole_timestep_snapshots

# 3) Analyze snapshots
python analyze_timestep_snapshots.py --csv ../results/cartpole_timestep_snapshots_hw.csv
```

## Snapshot Debug Modes

Debug CSV export tests (`test_cartpole_action_trace`, `test_cartpole_timestep_snapshots`) only run when `FULL_DEBUG=1`.

`test_cartpole_timestep_snapshots`:
    - Runs export and captures FC2 stream, HL2 summaries, FC2 saturation counters, plus HL1/Q probe fields.
    - May be followed by running the `analyze_timestep_snapshots.py` script

Backward compatibility: `SNAPSHOT_FULL_DEBUG` is still accepted as a fallback.

Example full-debug run:

```bash
FULL_DEBUG=1 make clean && make test_cartpole_integration_fractional_hist8 SIM=icarus TESTCASE=test_cartpole_timestep_snapshots
```

## What to Look For in Snapshot Analysis

- `fc2_saturation` line:
    - Healthy run should show near-zero saturation events for FC2.
    - Large `pos_events`/`neg_events` indicates FC2 clipping and likely policy collapse.
- `action=` across inferences:
    - Constant action with low rewards usually indicates lost dynamic range.
- `fc2_count_mismatches` / `fc2_index_coverage_issues`:
    - Should be zero; non-zero indicates sequencing/indexing bugs.
- `obs_fixed` vs early FC2/HL2 response:
    - Different observations should induce different downstream responses.

## Fixed-Point Format

The LIF neuron uses **QS2.13** fixed-point format:
- 16-bit signed
- 2 integer bits, 13 fractional bits
- Scale factor: 8192 (2^13)
- Threshold 1.0 = 8192
- Beta 0.9 ≈ 115 in Q1.7 format

## Viewing Waveforms

After running tests, view waveforms with GTKWave:

```bash
# On host (if GTKWave installed)
gtkwave results/sim_build/lif.fst
gtkwave results/sim_build/linear_layer.fst
```
NOTE: Unpacked arrays will not be visible in waveforms generated using Icarus. It is necessary to use Verilator to view these signals.

## Troubleshooting

### Permission Issues

If there are permission errors, rebuild with the host system user ID:

```bash
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose build --no-cache
```

### Simulation Errors

Check the Icarus Verilog output in `results/sim_build/` for compilation errors.

### Test Failures

Examine the cocotb log output and `results/results.xml` for details.
