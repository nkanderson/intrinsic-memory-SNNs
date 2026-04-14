# Benchmarking on FPGA

This directory contains synthesis, board-integration, and benchmarking work for running trained SNN models on FPGA hardware.

## Directory Layout

- `constraints/` — board constraints (`.xdc`) for Nexys A7-100T
- `sv/` — FPGA-targeted SystemVerilog tops/wrappers and copied/adapted compute modules
- `sim/` — simple simulation collateral for FPGA tops
- `scripts/` — build/run helper scripts (Vivado Tcl/shell helpers)
- `docs/` — design notes, interface specs, and phase plans
- `results/` — synthesis/timing/utilization and benchmark artifacts



