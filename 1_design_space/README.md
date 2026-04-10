# Design Space Analysis

Code in the root of this directory represents the second iteration of the design space analysis work on custom neurons with intrinsic memory, including fractional-order neurons as well as bitshift variants, as compared to a baseline LIF neuron. The first iteration developed during Summer 2025 is located in [v1-092025](./v1-092025/).

## Custom Neuron Synthesis

To run the script to synthesis, implement, and produce a .bit file for the basic LIF test, run the following:
```bash
$ vivado -mode batch -source scripts/build_top_lif_demo.tcl
```
