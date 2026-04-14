# Design Space Analysis

Code in the root of this directory represents the second iteration of the design space analysis work on custom neurons with intrinsic memory, including fractional-order neurons as well as bitshift variants, as compared to a baseline LIF neuron. The first iteration developed during Summer 2025 is located in [v1-092025](./v1-092025/).

## Custom Neuron Synthesis

Use the commands below from the [1_design_space](1_design_space) directory.

### 1) Check expected spike counts in software

Run:
```bash
python scripts/spike_count.py
```

You should see a table with expected counts (decimal and hex) for `lif`, `fractional_lif`, and `bitshift_lif` for the configured input currents.

### 2) Build bitstreams for each neuron variant

Basic LIF:
```bash
vivado -mode batch -source scripts/build_top_lif_demo.tcl
```

Fractional LIF:
```bash
vivado -mode batch -source scripts/build_top_fractional_lif_demo.tcl
```

Bitshift LIF:
```bash
vivado -mode batch -source scripts/build_top_bitshift_lif_demo.tcl
```

Each run generates timing/utilization reports and a `.bit` file under:
- [1_design_space/results/lif](1_design_space/results/lif)
- [1_design_space/results/fractional_lif](1_design_space/results/fractional_lif)
- [1_design_space/results/bitshift_lif](1_design_space/results/bitshift_lif)

### 3) Program the Nexys A7 in Vivado Hardware Manager

1. Open Vivado and Hardware Manager.
2. Open Target / Auto Connect.
3. Select the detected `xc7a100t` device.
4. Program Device and select the desired `.bit` file.

If creating a project through the GUI for convenience, use an RTL project with default part `xc7a100tcsg324-1`.
