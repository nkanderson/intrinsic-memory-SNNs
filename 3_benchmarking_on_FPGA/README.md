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
   router doesn't always close.
2. Program the board.
3. `python uart_smoke.py` — confirm PING returns `5A 00 01 50 <csum>`.
4. `python validate_fpga.py --golden ../../common/sv/cocotb/tests/golden_vectors/<config>.json`
5. `python eval_cartpole_hw.py --config <config> --episodes 100`
