# Fractional-Order LIF Neuron Implementation

This project implements fractional-order Leaky Integrate-and-Fire (LIF) neuron models in SystemVerilog for FPGA synthesis.

## Module Overview

- **`lif.sv`**: Basic LIF neuron with decay parameter
- **`lapicque.sv`**: LIF neuron with physical parameters (R, C, h)
- **`frac_order_lif.sv`**: Fractional-order LIF neuron using Grünwald-Letnikov approximation
- **`binomial_coeff_lut.sv`**: Lookup table for generalized binomial coefficients

## Fractional-Order Implementation Limits

### Alpha Sweep Analysis - Maximum History Length

Analysis of different fixed-point formats with varying precision was performed in order to determine the best choice for the fractional order LIF neuron, given the hardware constraints. The table below shows the **maximum practical history size** for all possible 4-bit alpha values (14 total values) before coefficients become too small to represent accurately.

All formats use unsigned magnitude representation for k≥1 coefficients to maximize precision, since it is known that all weights are negative for k>0.

**Format Details:**
- **UQ0.8**: k=0 uses UQ1.7, k≥1 uses UQ0.8 (8-bit total)
- **UQ0.12**: k=0 uses UQ1.11, k≥1 uses UQ0.12 (12-bit total)
- **UQ0.16**: k=0 uses UQ1.15, k≥1 uses UQ0.16 (16-bit total)

**Fixed-Point Thresholds (1 LSB for k≥1 coefficients):**
- **UQ0.8**: 0.00390625 (3.91×10⁻³)
- **UQ0.12**: 0.000244141 (2.44×10⁻⁴)
- **UQ0.16**: 0.0000152588 (1.53×10⁻⁵)

| α Value | UQ0.8 | UQ0.12 | UQ0.16 |
|---------|-------|--------|--------|
| 0.067   | 13    | 184    | 2486   |
| 0.133   | 20    | 239    | 2768   |
| 0.200   | 23    | 236    | 2378   |
| 0.267   | 23    | 210    | 1876   |
| 0.333   | 22    | 179    | 1431   |
| 0.400   | 20    | 148    | 1078   |
| 0.467   | 18    | 122    | 808    |
| 0.533   | 16    | 99     | 605    |
| 0.600   | 14    | 80     | 452    |
| 0.667   | 12    | 64     | 337    |
| 0.733   | 10    | 50     | 248    |
| 0.800   | 8     | 38     | 179    |
| 0.867   | 6     | 28     | 124    |
| 0.933   | 4     | 18     | 75     |

### Key Observations

1. **UQ0.8 format** (8-bit): Limited to **4-23 history samples** depending on α value
   - Suitable for resource-constrained implementations
   - Best performance with lower α values (0.2-0.4 range)

2. **UQ0.12 format** (12-bit): Supports **18-239 history samples**
   - Good balance between precision and resource usage
   - Recommended for most applications

3. **UQ0.16 format** (16-bit): Supports **75-2768 history samples**
   - Highest precision, suitable for demanding applications
   - Can support very long histories, especially for lower α values

4. **Alpha value impact**: History length has a complex relationship with α value
   - **Peak performance** around α ≈ 0.133-0.200 for most formats
   - **Very low α** (0.067) has shorter history due to rapid coefficient decay
   - **Higher α values** (> 0.6) significantly reduce maximum history size

### Implementation Guidelines

**For HISTORY_SIZE ≤ 8:**
- Use combinational adder tree (single cycle)
- Pre-computed coefficient arrays
- Target frequency: 100+ MHz
- Safe for all α values

**For HISTORY_SIZE 9-16:**
- Still feasible with combinational logic for most α values
- May need pipelined implementation for timing closure
- Consider reduced target frequency

**For HISTORY_SIZE > 16:**
- Only beneficial for low α values (< 0.4)
- Requires pipelined implementation
- Monitor coefficient significance carefully

## Signal Characteristics

### Key Signals Format Table

| Signal Name | Width (bits) | Signed | Format | Scale Factor | Description |
|-------------|--------------|--------|--------|--------------|-------------|
| `tau_alpha` | N/A | N/A | real | N/A | Synthesis-time calculation only |
| `tau_over_h_alpha` | 16 | No | Q8.8 | 2^8 = 256 | Default value is 2.0 |
| `tau_scaled` | 16 | No | Q16.0 | 1 | Time constant parameter |
| `current` | 8 | No | Q8.0 | 1 | Input current value |
| `coefficients[k]` | 8 | No | UQ0.8 | 2^8 = 256 | Binomial coefficient magnitude |
| `current_term` | 16 | No | Q16.0 | 1 | R × I[n] intermediate result |
| `fractional_sum` | 32 | No | Q24.8 | 2^8 = 256 | Sum of fractional products |
| `history_buffer[i]` | 8 | No | Q8.0 | 1 | Past membrane potential values |
| `history_term` | 32 | No | Q16.16 | 2^8 = 256 | τ^α/h^α × fractional_sum |
| `unnormalized_potential` | 32 | No | Q16.16 | 2^16 | Before normalization (current - history) |
| `updated_potential` | 16 | No | Q16.0 | 1 | Final result after normalization |
| `membrane_potential` | 8 | No | Q8.0 | 1 | Current membrane potential |
| `products[j]` | 16 | No | Q8.8 | 2^8 = 256 | Individual fractional products |

### Format Notation
- **Q**: Fixed-point format (integer.fractional bits)
- **UQ**: Unsigned fixed-point
- **SQ**: Signed fixed-point (includes sign bit)
- **Scale Factor**: Multiplier to convert from real value to fixed-point representation
- **real**: SystemVerilog real type - IEEE 754 double-precision floating-point (64-bit), used only for synthesis-time calculations and automatically eliminated during synthesis

### Implementation Notes
- All coefficients stored as unsigned magnitudes with negative sign applied in computation
- Products use 24-bit signed arithmetic to prevent overflow in multiplication
- Fractional sum accumulates in 32-bit signed for extended range
- Final membrane potential saturated to 8-bit range [0, 255]

## Legacy Fixed-Point Formats

- **Decay factors**: 0.8 format (8 fractional bits)
- **Binomial coefficients**: 8.8 format (8 fractional bits, sign bit)
- **Membrane potential**: 8-bit unsigned
- **Internal calculations**: 16-32 bit intermediate precision

## Testing

### Cocotb Tests
```bash
# Test binomial coefficient LUT
make test_binomial

# Test all modules
make test_all
```

### Verilator Simulation
```bash
# Test basic LIF neuron
make run_lif

# Clean up
make clean-all
```

## FPGA Synthesis

### For iCEBreaker (iCE40 UP5K)
```bash
# Full synthesis flow
make fpga

# Individual steps
make synth      # Yosys synthesis
make pnr        # Place and route
make bitstream  # Generate programming file
make program    # Program FPGA
```

### Lapicque on Icebreaker

In-progress test for a basic Lapicque neuron on the Icebreaker.

From within the `syn/scripts` directory:
```sh
yosys synth_lapicque_ice40.ys
nextpnr-ice40 --up5k --package sg48 --json top_lapicque_icebreaker.json --pcf ../lapicque_icebreaker.pcf --asc top_lapicque_icebreaker.asc --report ../../data/top_lapicque_timing.rpt --timing-allow-fail
icepack top_lapicque_icebreaker.asc top_lapicque_icebreaker.bin
iceprog top_lapicque_icebreaker.bin
```

### Fractional Order LIF on Icebreaker

Basic synthesis commands for the frac_order_lif neuron module on the Icebreaker.

From within the `syn/scripts` directory:
```sh
yosys synth_flif_ice40.ys
nextpnr-ice40 --up5k --package sg48 --json top_flif_icebreaker.json --pcf ../flif_icebreaker.pcf --asc top_flif_icebreaker.asc
icepack top_flif_icebreaker.asc top_flif_icebreaker.bin
iceprog top_flif_icebreaker.bin
```

### `yosys` Synthesis Analysis

Commands run from `yosys` shell unless otherwise noted. In the following example, `yosys` shell was initialized from within the `syn/scripts` directory.

#### `read_verilog`

Start by calling `read_verilog` on modules, and setting the top-level module with `hierarchy` command. The following will use the top module used for configuring the icebreaker to show spike interactivity.
```bash
read_verilog -sv ../../src/lapicque.sv
read_verilog -sv ../top_lapicque_icebreaker.sv
hierarchy -top top_lapicque_icebreaker
```
The following will instead just synthesize the `lapicque` module, which is probably more often what is wanted for analsys.
```bash
read_verilog -sv ../../src/lapicque.sv
hierarchy -top lapicque
synth_ice40 -top lapicque -json lapicque.json
stat
```

#### `stat`

Use `stat` to get gate counts and LUT usage. This shows basic stats for all modules in the hierarchy. The following includes just output for the `lapicque` module, though stats for the top module were also output.

```bash
stat

=== lapicque ===

  Number of wires:                 24
  Number of wire bits:            192
  Number of public wires:           8
  Number of public wire bits:      41
  Number of ports:                  4
  Number of port bits:             11
  Number of memories:               0
  Number of memory bits:            0
  Number of processes:              3
  Number of cells:                 12
    $add                            1
    $ge                             1
    $gt                             2
    $logic_and                      1
    $logic_not                      1
    $mul                            2
    $shr                            2
    $sub                            2
```
Includes number of cells (gates, FFs, etc.), number of LUTs, and combinational cells.

#### `synth_ice40`

Call `synth_ice40` to synthesize the design. This will print stats for the top module at the end of synthesis. The command `techmap` should map generic logic to iCE40-specific cells, so `stat` should be more accurate for the FPGA. However, when running with the top module in the hierarchy, it no longer prints stats for the `lapicque` module.
```bash
synth_ice40 -top top_lapicque_icebreaker
techmap -map +/ice40/cells_map.v
stat
```

#### `show` to generate schematic

Schematics in Graphviz DOT format can be generated. NOTE: The following commands can only be run when lapicque module has not been optimized away, which I think is happening with the `synth_ice40` command above.
```bash
prep -top lapicque
show -format svg -prefix lapicque_schematic
```
If you run `show` right after `read_verilog`, you’ll see mostly high-level RTL cells (adders, multipliers, $mux, $dff, $add, etc.).

If you run `show` after `synth_ice40` + `techmap`, you’ll see FPGA primitives like SB_LUT4, SB_DFF, etc. These are Graphviz boxes for the cell types, which are not gate symbols, but map 1:1 to the hardware.

Can also use `rsvg-convert` to convert to a PDF:
```bash
rsvg-convert -f pdf \
    -o images/lapicque_clean.pdf syn/scripts/top_lapicque_icebreaker_schematic.svg
```

Temporarily make the `lapicque` module the top module:
```bash
hierarchy -top lapicque
synth_ice40 -top lapicque -json lapicque.json
```
Then re-run any commands above as desired.

#### `nextpnr-ice40`

NextPNR is for placement and routing, and `nextpnr-*` commands should provide timing, power, and device-level utilization data. NOTE: This is *not* run within the `yosys` shell. In addition, this relies on commands having been run which output the json and asc files referenced.
```bash
nextpnr-ice40 \
    --up5k --package sg48 \
    --json top_lapicque_icebreaker.json \
    --pcf ../lapicque_icebreaker.pcf \
    --asc top_lapicque_icebreaker.asc \
    --timing-allow-fail
```
The option `--freq-report` is supposed to generate a frequency report for each clock domain, and maybe allows for inspecting critical paths for timing bottlenecks.

```bash
nextpnr-ice40 --up5k --package sg48 --json top_lapicque_icebreaker.json --pcf ../lapicque_icebreaker.pcf --asc top_lapicque_icebreaker.asc --timing-allow-fail

Info: Max frequency for clock 'clk$SB_IO_IN_$glb_clk': 33.88 MHz (PASS at 12.00 MHz)
```
The above also includes a long section labeled the "critical path report". Has multiple sections like the one included above, with a max frequency, but it's unclear how they differ. It was suggested to check the `.asc` summary, but it's unclear where that is.

Overview of `nextpnr-ice40` flags:
--json → input netlist from Yosys.
--pcf → pin constraint file.
--asc → output ASCII bitstream for icepack.
--gui → launches the interactive P&R GUI.
--timing-allow-fail → still generate an .asc even if timing not met.
--freq <MHz> → constrain target clock.
--freq-report → print frequency domain report.
--placer / --router → choose algorithm variant.

Note: `detailed-timing-report` and `report` both appear to be options as well, based on review of [nextpnr code](https://github.com/YosysHQ/nextpnr/blob/master/common/kernel/command.cc).

For icebreaker v1.1a, flags `--up5k --package sg48` should be correct for setting the device and package.

#### Full Example Analysis

```bash
read_verilog -sv ../../src/lapicque.sv
hierarchy -top lapicque
synth_ice40 -dsp -top lapicque -json lapicque.json
stat
show -format svg -prefix lapicque_schematic
```

#### Full Example FGPA Build

Maybe... but I think the full list of commands I have above, including `icepack` and `iceprog` is better to reference.
```bash
yosys synth_lapicque_ice40.ys
nextpnr-ice40 --up5k --package sg48 \
              --json top_lapicque_icebreaker.json \
              --pcf ../lapicque_icebreaker.pcf \
              --asc top_lapicque_icebreaker.asc \
              --gui
```

#### Analysis Output

Running the above steps from full example analysis for Lapicque:

Excerpts from `synth_ice40`:
```bash

3.13. Executing WREDUCE pass (reducing word size of cells).
Removed top 31 bits (of 32) from port B of cell lapicque.$gt$../../src/lapicque.sv:77$1 ($gt).
Removed top 9 bits (of 16) from port A of cell lapicque.$add$../../src/lapicque.sv:82$9 ($add).
Removed top 9 bits (of 16) from port B of cell lapicque.$add$../../src/lapicque.sv:82$9 ($add).
Removed top 8 bits (of 16) from port Y of cell lapicque.$add$../../src/lapicque.sv:82$9 ($add).
Removed top 31 bits (of 32) from port B of cell lapicque.$sub$../../src/lapicque.sv:91$11 ($sub).
Removed top 27 bits (of 32) from port Y of cell lapicque.$sub$../../src/lapicque.sv:91$11 ($sub).
Removed top 8 bits (of 16) from port A of cell lapicque.$gt$../../src/lapicque.sv:97$13 ($gt).
Removed top 8 bits (of 16) from port B of cell lapicque.$gt$../../src/lapicque.sv:97$13 ($gt).
Removed top 27 bits (of 32) from wire lapicque.$sub$../../src/lapicque.sv:91$11_Y.
Removed top 8 bits (of 16) from wire lapicque.updated_potential.

3.20. Executing ALUMACC pass (create $alu and $macc cells).
Extracting $alu and $macc cells in module lapicque:
  creating $macc model for $add$../../src/lapicque.sv:82$9 ($add).
  creating $macc model for $sub$../../src/lapicque.sv:91$11 ($sub).
  creating $macc model for $sub$../../src/lapicque.sv:93$12 ($sub).
  creating $alu model for $macc $sub$../../src/lapicque.sv:93$12.
  creating $alu model for $macc $sub$../../src/lapicque.sv:91$11.
  creating $alu model for $macc $add$../../src/lapicque.sv:82$9.
  creating $alu model for $ge$../../src/lapicque.sv:78$2 ($ge): merged with $sub$../../src/lapicque.sv:93$12.
  creating $alu model for $gt$../../src/lapicque.sv:77$1 ($gt): new $alu
  creating $alu model for $gt$../../src/lapicque.sv:97$13 ($gt): new $alu
  creating $alu cell for $gt$../../src/lapicque.sv:97$13: $auto$alumacc.cc:495:replace_alu$430
  creating $alu cell for $gt$../../src/lapicque.sv:77$1: $auto$alumacc.cc:495:replace_alu$441
  creating $alu cell for $add$../../src/lapicque.sv:82$9: $auto$alumacc.cc:495:replace_alu$452
  creating $alu cell for $sub$../../src/lapicque.sv:91$11: $auto$alumacc.cc:495:replace_alu$455
  creating $alu cell for $sub$../../src/lapicque.sv:93$12, $ge$../../src/lapicque.sv:78$2: $auto$alumacc.cc:495:replace_alu$458
  created 5 $alu and 0 $macc cells.

3.44. Executing OPT_LUT pass (optimize LUTs).
Discovering LUTs.
Number of LUTs:       27
  1-LUT                1
  2-LUT                7
  3-LUT               15
  4-LUT                4
  with \SB_CARRY    (#0)   11
  with \SB_CARRY    (#1)   10

Eliminating LUTs.
Number of LUTs:       27
  1-LUT                1
  2-LUT                7
  3-LUT               15
  4-LUT                4
  with \SB_CARRY    (#0)   11
  with \SB_CARRY    (#1)   10

Combining LUTs.
Number of LUTs:       24
  1-LUT                1
  2-LUT                3
  3-LUT               14
  4-LUT                6
  with \SB_CARRY    (#0)   11
  with \SB_CARRY    (#1)   10

Eliminated 0 LUTs.
Combined 3 LUTs.

3.48. Printing statistics.

=== lapicque ===

   Number of wires:                 41
   Number of wire bits:             76
   Number of public wires:          41
   Number of public wire bits:      76
   Number of ports:                  4
   Number of port bits:             11
   Number of memories:               0
   Number of memory bits:            0
   Number of processes:              0
   Number of cells:                 50
     SB_CARRY                       14
     SB_DFFESR                      12
     SB_LUT4                        24
```
Using `stat` produces the same stats as printed above in step 3.48, by `synth_ice40`.

Analysis for the fractional order LIF neuron:

Running stat immediately after `read_verilog` and setting `frac_order_lif` as the top. This shows the gate-level rather than the lower-level cells that are eventually synthesized on FPGA.
```bash
yosys> stat

3. Printing statistics.

=== frac_order_lif ===

   Number of wires:                274
   Number of wire bits:           2941
   Number of public wires:          79
   Number of public wire bits:    1060
   Number of ports:                  4
   Number of port bits:             11
   Number of memories:               1
   Number of memory bits:           64
   Number of processes:             20
   Number of cells:                 79
     $add                           19
     $eq                             3
     $ge                             9
     $gt                             4
     $logic_and                      2
     $logic_not                      1
     $meminit_v2                     1
     $memrd                          8
     $mul                            3
     $mux                            2
     $shl                            1
     $sub                           18
     SB_MAC16                        8
```

And `stat` after running `synth_ice40`:
```bash
yosys> stat

5. Printing statistics.

=== frac_order_lif ===

   Number of wires:                568
   Number of wire bits:           2387
   Number of public wires:         568
   Number of public wire bits:    2387
   Number of ports:                  4
   Number of port bits:             11
   Number of memories:               0
   Number of memory bits:            0
   Number of processes:              0
   Number of cells:               1085
     SB_CARRY                       94
     SB_DFFER                       80
     SB_LUT4                       903
     SB_MAC16                        8
```

## References

- [SNNTorch Library](https://github.com/jeshraghian/snntorch)
- [Grünwald-Letnikov Fractional Derivative](https://en.wikipedia.org/wiki/Gr%C3%BCnwald%E2%80%93Letnikov_derivative)
