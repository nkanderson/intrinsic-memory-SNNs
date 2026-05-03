# Vivado bring-up: lif-64-16 on Nexys A7-100T

First-light synthesis flow for `board_top_lif_64_16`. GUI-driven for now;
[promote to a Tcl build script](../README.md) once the design is stable.

## 1. Create the project

- Target part: **xc7a100tcsg324-1** (Nexys A7-100T).
- Project type: RTL Project, no block design.
- Add sources at create time, or follow step 2 afterwards.

## 2. Add design sources

Add the following directories *recursively* as design sources:

- `common/sv/` — all of it. The standard-LIF, fractional, and bitshift
  network modules are all referenced by `top_uart_accel_wrapper.sv`'s
  generate block; even though only `MODEL_TYPE=0` is elaborated, Vivado
  needs every referenced module to be parseable.
- `3_benchmarking_on_FPGA/sv/` — picks up `board_top_lif_64_16.sv`.

Skip these subtrees (they're test-only and reference cocotb internals
or sim-only constructs):

- `common/sv/cocotb/`
- `common/sv/host_if/tb_uart_loopback.sv`

## 3. Add the weights

Add `common/sv/cocotb/tests/weights/lif-64-16/` as a *design source*
directory. The six `.mem` files in there (`fc1_weights.mem`,
`fc1_bias.mem`, `fc2_weights.mem`, `fc2_bias.mem`, `fc_out_weights.mem`,
`fc_out_bias.mem`) are referenced by basename in `board_top_lif_64_16.sv`,
so they need to be on Vivado's source search path. Adding the directory
to the project as a source root does this.

Don't copy the `.mem` files into `3_benchmarking_on_FPGA/`. Keep them
in the cocotb tree so simulation and synthesis read the same data.

## 4. Add constraints

Add `3_benchmarking_on_FPGA/constraints/nexys_a7_lif_64_16.xdc`. **Do
not** also add `Nexys-A7-100T-Master.xdc` — the master enables every
pin on the board, including switches and LEDs the design doesn't drive,
which produces unconstrained-pin warnings.

## 5. Set the top module

In *Hierarchy*, right-click `board_top_lif_64_16` → *Set as Top*.

## 6. Run synthesis

`Run Synthesis`. Open the Synthesis report when it finishes and check:

- **No critical warnings** about unresolved modules, missing weight files,
  or unconnected required ports.
- **Utilization**: expect well under 5% LUT/FF, ~10–15 DSPs of 240,
  ~4–6 BRAMs of 135. Anything substantially higher is a flag —
  recheck that `MODEL_TYPE=0` actually pruned the fractional/bitshift
  generate branches.

## 7. Run implementation + bitstream

`Run Implementation` then `Generate Bitstream`. The implementation
report should show **positive WNS** at the 100 MHz target. If WNS is
negative, capture the timing report under
`3_benchmarking_on_FPGA/results/lif-64-16/` and we'll iterate.

## 8. Program the board

Connect the Nexys A7 USB cable, open *Hardware Manager*, open target,
program with the generated `.bit`. Heartbeat LED (LED[15]) should
start blinking ~1.5 Hz immediately.

## 9. Smoke-test over UART

Open a serial terminal at **115 200 8N1** to the FT2232's USB UART
(typically `/dev/ttyUSB1` on Linux — `dmesg | tail` after plugging in
to confirm).

**PING test** — send 5 bytes:

```
A5 7F 00 00 DA
```

Expect 5 bytes back:

```
5A 00 01 50 0B
```

(`5A` SOF, `00` ST_OK, `01` LEN, `50` `'P'`, `0B` CSUM.)

**Inference test** — send three frames:

1. Write four observations (all 16-bit LE QS2.13, here all zero):
   ```
   A5 01 10 08  00 00 00 00 00 00 00 00  BC
   ```
   Expect: `5A 00 00 5A`.

2. EXEC:
   ```
   A5 03 00 00 A6
   ```
   Expect: `5A 00 00 5A`.

3. Read STATUS until bit 0 is set, then read ACTION:
   ```
   READ STATUS:  A5 02 04 01 A2  -> 5A 00 01 <stat> <csum>
   READ ACTION:  A5 02 08 01 AE  -> 5A 00 01 <act>  <csum>
   ```

LED[0] should pulse briefly during EXEC. LED[1] should latch to the
returned action bit afterwards.

## 10. Capture results

Save the implementation/timing/utilization reports under
`3_benchmarking_on_FPGA/results/lif-64-16/` along with the `.bit` file
and a short note of the Vivado version used. This is the baseline we'll
compare every subsequent bitstream against.
