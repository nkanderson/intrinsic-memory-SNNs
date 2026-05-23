# Benchmarking on FPGA

Synthesis, board integration, and benchmarking for running trained SNN models
on a Nexys A7-100T (Artix-7 100T). Five model configurations are supported;
each gets its own bitstream.

## Directory layout

```
constraints/   Board constraints (.xdc) for each config + master XDC
sv/            Per-config board tops (board_top_<config>.sv)
scripts/       Host-side Python scripts (see below)
results/       Per-config benchmark output CSVs and synthesis reports
docs/          Design notes
```

## Supported configurations

| Config name                  | Neuron type     | HL1 / HL2 | Timesteps |
|------------------------------|-----------------|-----------|-----------|
| `lif-64-16`                  | Standard LIF    | 64 / 16   | 10        |
| `lif-32-16`                  | Standard LIF    | 32 / 16   | 10        |
| `frac-32-4-16`               | Fractional LIF  | 32 / 4    | 10        |
| `frac-16-4-32`               | Fractional LIF  | 16 / 4    | 20        |
| `bitshift-custom_slow_decay` | Bitshift LIF    | 32 / 8    | 10        |

Each configuration requires a separate Vivado build and bitstream.
Weight `.mem` files are read from `common/sv/cocotb/tests/weights/<config>/`
at synthesis time — add that directory to the Vivado project's source search
path so `$readmemh` resolves the basenames.

## One-time host setup

### 1. Python dependencies

```bash
pip install pyserial gymnasium numpy torch
```

All scripts are run from `3_benchmarking_on_FPGA/scripts/`.

### 2. FTDI latency timer (important for performance)

The FT2232 USB-UART bridge on the Nexys A7 has a 16 ms USB packet latency
timer by default. With ~4 serial transactions per inference step this adds
~64 ms/step at 115200 baud instead of the expected ~5 ms/step. Reduce it to
1 ms permanently with a udev rule:

```bash
sudo tee /etc/udev/rules.d/99-ftdi-latency.rules <<'EOF'
SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Verify it took effect (replug the board if needed):

```bash
cat /sys/bus/usb-serial/devices/ttyUSB1/latency_timer   # should print 1
```

The scripts also attempt to set this automatically at startup, but the udev
rule is more reliable across replugs and reboots.

### 3. Serial port permissions

Add your user to the `dialout` group to avoid needing `sudo` for `/dev/ttyUSB*`:

```bash
sudo usermod -aG dialout $USER   # log out and back in to take effect
```

## Scripts

All scripts default to `--port /dev/ttyUSB1` and `--baud 115200`.

### `uart_smoke.py` — quick board bring-up check

Sends a PING, writes zero observations, runs one inference, and prints the
raw hex responses. Use this immediately after programming a new bitstream to
confirm the UART path is alive before running longer tests.

```bash
python uart_smoke.py --port /dev/ttyUSB1
```

### `fpga_interface.py` — UART transport library

Not run directly. Provides `FpgaInterface`, a context-manager class used by
the other scripts. Handles frame building, checksum, and response parsing
per the protocol in `common/sv/host_if/README.md`.

### `validate_fpga.py` — bit-exact validation against golden vectors

Feeds pre-generated golden vectors (from
`common/sv/cocotb/tests/golden_vectors/<config>.json`) to the FPGA one by
one and compares the returned action against the reference model's expected
action. Prints a per-mismatch line and a final match-rate summary.

```bash
python validate_fpga.py \
  --golden ../../common/sv/cocotb/tests/golden_vectors/lif-64-16.json \
  --port /dev/ttyUSB1
```

A 100% match rate is the exit criterion before running live CartPole episodes.

### `snn_policy_hardware.py` — `nn.Module` wrapper (library)

Not run directly. Wraps `FpgaInterface` in a PyTorch `nn.Module` interface
so the FPGA can be used as a drop-in policy in the eval loop. Handles
float → QS2.13 fixed-point conversion for observations.

### `eval_cartpole_hw.py` — hardware-in-the-loop CartPole evaluation

Runs CartPole-v1 episodes with actions selected by the FPGA over UART.
Prints per-episode reward/step/latency and writes a CSV to
`results/<config>/hw_eval_seed<N>_ep<M>.csv`.

```bash
python eval_cartpole_hw.py \
  --config lif-64-16 \
  --port /dev/ttyUSB1 \
  --episodes 100 \
  --seed 0
```

Expected throughput at 921600 baud with latency timer = 1 ms: ~0.5–1 ms/step
(~0.5 ms UART byte time + ~0.4 ms USB overhead per transaction).

## Typical workflow for a new config

1. Build the bitstream in Vivado using `sv/board_top_<config>.sv` and
   `constraints/nexys_a7_<config>.xdc`. Run `phys_opt_design -directive
   AggressiveExplore` after `route_design` before `write_bitstream` — the
   fractional LIF configs have a marginal timing path that the standard
   router doesn't always close. Or use the TCL batch wrapper:
   `python scripts/build_config.py <config>`.
2. Program the board:
   `python scripts/flash_config.py <config>` (uses `vivado -mode batch`
   and reads `results/<config>/bitstream.bit`).
3. `python uart_smoke.py` — confirm PING returns `5A 00 01 50 <csum>`.
4. `python validate_fpga.py --golden ../../common/sv/cocotb/tests/golden_vectors/<config>.json`
5. `python eval_cartpole_hw.py --config <config> --episodes 100`

## Collecting PPA + inference-cycle metrics

Per-config FPGA metrics — area, timing, power — and simulated per-stage
inference cycle counts share one data-flow. All scripts live in `scripts/`
and read the config registry in [scripts/configs.py](scripts/configs.py).

### 1. Verify per-config parameters are consistent

```bash
python 3_benchmarking_on_FPGA/scripts/check_param_drift.py
```

Compares each config's `cp_integration_<config>` cocotb Makefile block to
its `board_top_<config>.sv` and warns if numeric parameters (`HL1_SIZE`,
`INV_DENOM`, `SHIFT_MODE`, ...) or weight-file basenames disagree. Run
this before any synthesis or cycle measurement so you don't compare apples
to oranges. Note: `Q_BATCH_SIZE` intentionally differs (sim uses 4, board
uses 1 because the UART feeds one observation at a time) — current output
lists this as MISMATCH; treat as expected drift unless that ever changes.

### 2. Build bitstreams + extract Vivado reports

```bash
# Build one config
python 3_benchmarking_on_FPGA/scripts/build_config.py lif-64-16

# Or every config that has a board_top + XDC
python 3_benchmarking_on_FPGA/scripts/build_config.py --all
```

The wrapper invokes `vivado -mode batch -source build_config.tcl` with the
right paths and copies the synth/impl/power reports plus the bitstream to
`results/<config>/` with canonical filenames (see
[docs/vivado_bringup_lif_64_16.md](docs/vivado_bringup_lif_64_16.md) §10
for the table). GUI-driven runs still work; you just need to copy the
reports to the canonical names manually after each run.

Parse one config's reports without rebuilding (handy after a GUI run):

```bash
python 3_benchmarking_on_FPGA/scripts/synth_metrics.py --config lif-64-16
```

### 3. Measure inference cycles per FSM stage (cocotb)

The cycle test reuses each `cp_integration_<config>` parameter block so
there's zero duplication of `HL1_SIZE` / `INV_DENOM` / weights paths.
Per-config Make wrappers live alongside the existing cocotb targets:

```bash
# Inside the cocotb Docker container (common/sv/cocotb/docker-compose.yml)
cd /workspace/tests
make cycle_breakdown_lif-64-16
make cycle_breakdown_frac-32-4-16
make cycle_breakdown_bitshift-custom_slow_decay
# ...etc, one per config in configs.py
```

Each writes a one-row CSV to
`common/sv/cocotb/results/<config>/cycles.csv` with columns:
`total_cycles, cycles_idle, cycles_load_hl1, cycles_run_timesteps,
cycles_finish_q, cycles_done_state, num_timesteps, cycles_per_timestep,
hl1_size, hl2_size, history_length`.

### 4. Aggregate + visualize

```bash
# Join PPA + cycles into a single CSV; compute latency, throughput, energy/inference
python 3_benchmarking_on_FPGA/scripts/aggregate_ppa.py
# -> results/summary/ppa_cycles_combined.csv

# Plots use common/scripts/plot_styles.py (Okabe-Ito palette, LaTeX-sized figs)
python 3_benchmarking_on_FPGA/scripts/plot_ppa.py                # PNG (default)
python 3_benchmarking_on_FPGA/scripts/plot_ppa.py --format svg   # SVG
```

Plots emitted:
- `area.{png,svg}` — grouped bars: LUT/FF/DSP/BRAM per config (log scale)
- `performance_fmax.{png,svg}` — bars colored by neuron type with 100 MHz target line
- `power_stacked.{png,svg}` — static + dynamic, total annotated
- `cycles_per_stage.{png,svg}` — LOAD_HL1 / RUN_TIMESTEPS / FINISH_Q stacked
- `figures_of_merit.{png,svg}` — three small bars: latency_us, throughput_hz, energy_per_inference_uj

Configs without synth artifacts yet (e.g. `frac-32-8-8-q2_13`, which has a
cocotb target but no `board_top_*` / XDC) appear in the master CSV with
cycle data only; PPA columns stay blank until those files are authored.
