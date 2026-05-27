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
# Build one config (writes reports to results/<config>/baseline/)
python 3_benchmarking_on_FPGA/scripts/build_config.py lif-64-16

# Build the same config under a different profile (e.g. after switching the
# top-FSM encoding directive in the SV and re-synthesizing)
python 3_benchmarking_on_FPGA/scripts/build_config.py lif-64-16 --profile onehot_top_fsm

# Or every config that has a board_top + XDC
python 3_benchmarking_on_FPGA/scripts/build_config.py --all
python 3_benchmarking_on_FPGA/scripts/build_config.py --all --profile onehot_top_fsm
```

The wrapper invokes `vivado -mode batch -source build_config.tcl` with the
right paths and copies the synth/impl/power reports plus the bitstream to
`results/<config>/<profile>/` with canonical filenames (see
[docs/vivado_bringup_lif_64_16.md](docs/vivado_bringup_lif_64_16.md) §10
for the table). The default profile is `baseline`.

**Vivado project location.** The Vivado project itself lives at
`results/<config>/vivado_project/` — at the config level, not under any
profile. All profiles for a given config share that project: re-running
`build_config.py` with a different `--profile` deletes and re-creates the
project from the current SV / XDC / weights, then writes the reports into
the new profile subdir. This means you can iterate on FSM encoding
changes by editing the SV, running `build_config.py <cfg> --profile foo`,
and getting a fresh report set without touching the existing baseline
reports. Alternative profiles live in sibling subdirs and are compared
automatically by the aggregate + plot scripts — see §4. GUI-driven runs
still work; you just need to copy the reports to the canonical names
manually after each run, into the profile dir of your choice.

Parse one config's reports without rebuilding (handy after a GUI run):

```bash
python 3_benchmarking_on_FPGA/scripts/synth_metrics.py --config lif-64-16
python 3_benchmarking_on_FPGA/scripts/synth_metrics.py --config lif-64-16 --profile onehot_top_fsm
```

### 3. Measure inference cycles per FSM stage (cocotb)

The cycle test reuses each `cp_integration_<config>` parameter block so
there's zero duplication of `HL1_SIZE` / `INV_DENOM` / weights paths. In
addition to the top-level FSM (LOAD_HL1 / RUN_TIMESTEPS / FINISH_Q) it
samples the per-timestep substate FSM (`ts_state`) so the cycles spent in
each of HL1_STEP / FC2_START / FC2_WAIT / HL2_STEP / HL2_WAIT / NEXT can
be broken out per inference. The `NEURON_TYPE` env var (set by each
`cycle_breakdown_*` target) selects the right substate enum mapping for
the variant under test.

```bash
# Inside the cocotb Docker container (common/sv/cocotb/docker-compose.yml)
cd /workspace/tests

# Run all three reference-model breakdowns in sequence (recommended):
make cycle_breakdown_all

# Or one at a time:
make cycle_breakdown_leaky-32-32
make cycle_breakdown_bitshift-64-32-4
make cycle_breakdown_frac-32-8-8-q2_13
```

Each writes a one-row CSV to
`common/sv/cocotb/cycle_results/<config>/cycles.csv` (a sibling of the
generic `results/` dir that is **preserved across `make clean`**, so prior
measurements survive subsequent simulation runs). Columns:

```
config, neuron_type, total_cycles, cycles_idle, cycles_load_hl1,
cycles_run_timesteps, cycles_finish_q, cycles_done_state,
cycles_ts_hl1_step, cycles_ts_fc2_start, cycles_ts_fc2_wait,
cycles_ts_hl2_step, cycles_ts_hl2_wait, cycles_ts_next,
num_timesteps, cycles_per_timestep, hl1_size, hl2_size, history_length
```

The six `cycles_ts_*` columns sum to `cycles_run_timesteps` (the LIF
variant has no `TS_HL2_WAIT` state, so `cycles_ts_hl2_wait` is `0` there).

After the container run, copy each cycles.csv into the corresponding
profile directory on the host so it joins the per-(config × profile) row
emitted by `aggregate_ppa.py`:

```bash
for cfg in lif-32-32 bitshift-64-32-4 frac-32-8-8-q2_13; do
    mkdir -p 3_benchmarking_on_FPGA/results/$cfg/baseline
    cp common/sv/cocotb/cycle_results/$cfg/cycles.csv \
       3_benchmarking_on_FPGA/results/$cfg/baseline/cycles.csv
done
```

For a non-baseline profile (e.g. `onehot_top_fsm`), copy into that
profile's directory instead — the aggregate script only auto-discovers
cycle CSVs at the canonical `results/<config>/<profile>/cycles.csv` path
for non-baseline profiles.

### 4. Aggregate + visualize

```bash
# Join PPA + cycles into a single CSV; compute latency, throughput, energy/inference
python 3_benchmarking_on_FPGA/scripts/aggregate_ppa.py
# -> results/summary/ppa_cycles_combined.csv  (one row per (config, profile))

# Plots + CSV tables use common/scripts/plot_styles.py (Okabe-Ito palette)
python 3_benchmarking_on_FPGA/scripts/plot_ppa.py                # PNG plots (default) + CSV tables
python 3_benchmarking_on_FPGA/scripts/plot_ppa.py --format svg   # SVG plots + CSV tables
```

Output layout:

```
results/summary/
    ppa_cycles_combined.csv     # one row per (config × profile)
    plots/
        area.{png,svg}                              # LUT/FF/DSP/BRAM grouped bars
                                                    # — LUTs split into Logic (solid)
                                                    #   + Memory (hatched) sub-bars
        power_stacked.{png,svg}                     # static + dynamic, total annotated
        cycles_per_stage.{png,svg}                  # LOAD_HL1 / RUN_TIMESTEPS / FINISH_Q
        cycles_per_ts_substate.{png,svg}            # HL1_STEP / FC2_* / HL2_* / NEXT (total)
        cycles_per_ts_substate_per_timestep.{png,svg}  # same, divided by num_timesteps
        figures_of_merit.{png,svg}                  # latency / throughput / energy
    tables/
        performance_timing.csv      # fmax, clock period, WNS, TNS, WHS per (config, profile)
        ppa_summary.csv             # one-stop LUT/FF/DSP/BRAM/fmax/power/latency/throughput/energy
```

When more than one profile exists for a config, the plots switch to a
grouped-bar layout: bars within a config cluster are distinguished by
hatch pattern, with `baseline` always rendered solid. A profile→hatch
legend appears to the right of each affected plot.

Configs without synth artifacts yet (e.g. `lif-64-16`'s sibling models in
`configs.py` that have a cocotb target but no `board_top_*` / XDC) appear
in the master CSV with blank PPA columns until those files are authored.
