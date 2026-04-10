# Project structure
SRC_DIR = src
TB_DIR = tb
SIM_DIR = sim
SYN_DIR = syn

# Top module name (without .sv extension)
TOP = top

# FPGA configuration
FPGA_FAMILY = ice40
FPGA_DEVICE = up5k
FPGA_PACKAGE = sg48

# verilator is a perl script, and make can't correctly execute it
# without specifying the shell and path
SHELL := /bin/bash
export PATH := /usr/local/bin:/usr/bin:/bin:$(PATH)

# C++ testbench files
TB_CPP = $(SIM_DIR)/sim_main.cpp
LIF_TB_CPP = $(SIM_DIR)/lif_tb.cpp

# Waveform output
VCD = wave.vcd

# Verilator flags
VERILATOR_FLAGS = -Wall --trace --sv -CFLAGS -std=c++14

# Build and run targets
all: build run

# Original top module simulation
build:
	verilator --cc $(TOP).sv $(VERILATOR_FLAGS) --exe $(TB_CPP)
	make -C obj_dir -f V$(TOP).mk V$(TOP)

run:
	./obj_dir/V$(TOP)

#######
#
# Testing
#
#######

# LIF neuron testbench
lif_tb: $(SRC_DIR)/lif.sv $(LIF_TB_CPP)
	verilator --cc $(SRC_DIR)/lif.sv $(VERILATOR_FLAGS) --exe $(LIF_TB_CPP) --top-module lif
	make -C obj_dir -f Vlif.mk Vlif

run_lif: lif_tb
	./obj_dir/Vlif

# Cocotb test targets
test_binomial: $(SRC_DIR)/binomial_coeff_lut.sv
	@echo "Running cocotb tests for binomial coefficient LUT..."
	@echo "Using Verilator at: $$(which verilator)"
	@echo "Verilator version: $$(verilator --version)"
	$(MAKE) -C $(TB_DIR) binomial

# Run all tests
test_all: test_binomial test_frac_lif test_lapicque

# Test Lapicque LIF module
test_lapicque: $(SRC_DIR)/lapicque.sv
	@echo "Running cocotb tests for Lapicque LIF neuron..."
	@echo "Using Verilator at: $$(which verilator)"
	@echo "Verilator version: $$(verilator --version)"
	$(MAKE) -C $(TB_DIR) lapicque

# Test fractional-order LIF module
test_frac_lif: $(SRC_DIR)/frac_order_lif.sv
	@echo "Running cocotb tests for fractional-order LIF neuron..."
	@echo "Using Verilator at: $$(which verilator)"
	@echo "Verilator version: $$(verilator --version)"
	$(MAKE) -C $(TB_DIR) frac_lif

# Test fractional-order LIF memory effects
test_frac_lif_memory: $(SRC_DIR)/frac_order_lif.sv
	@echo "Running cocotb tests for fractional-order LIF memory effects..."
	@echo "Using Verilator at: $$(which verilator)"
	@echo "Verilator version: $$(verilator --version)"
	$(MAKE) -C $(TB_DIR) frac_lif_memory

#######
#
# Synthesis and FPGA programming
#
#######

# Synthesis for iCE40 FPGA
synth: $(SRC_DIR)/lif.sv $(SRC_DIR)/lif_top.sv
	@echo "Synthesizing LIF neuron for iCE40 FPGA..."
	cd $(SYN_DIR)/scripts && yosys synth_ice40.ys
	@echo "Synthesis complete."

# Place and route for iCEBreaker
pnr: synth
	@echo "Running place and route for iCEBreaker..."
	nextpnr-ice40 --$(FPGA_DEVICE) --package $(FPGA_PACKAGE) \
		--json $(SYN_DIR)/results/lif_top.json \
		--pcf $(SYN_DIR)/constraints/icebreaker.pcf \
		--asc $(SYN_DIR)/results/lif_top.asc

# Generate bitstream
bitstream: pnr
	@echo "Generating bitstream..."
	icepack $(SYN_DIR)/results/lif_top.asc $(SYN_DIR)/results/lif_top.bin
	@echo "Bitstream generated: $(SYN_DIR)/results/lif_top.bin"

# Program the FPGA
program: bitstream
	@echo "Programming iCEBreaker..."
	iceprog $(SYN_DIR)/results/lif_top.bin
	@echo "Programming complete!"

# Full FPGA flow - runs synthesis, place & route, and bitstream generation
fpga: synth pnr bitstream
	@echo "Complete FPGA build finished!"
	@echo "Bitstream ready: $(SYN_DIR)/results/lif_top.bin"

# Timing analysis
timing: pnr
	icetime -d $(FPGA_DEVICE) -mtr $(SYN_DIR)/results/timing_report.txt $(SYN_DIR)/results/lif_top.asc

#######
#
# Cleanup and misc
#
#######

print-path:
	@echo $$PATH

view: $(VCD)
	gtkwave $(VCD)

# NOTE: Can provide a filename to specify formatting, e.g.:
# gtkwave tb/dump.fst frac_order_signal_format.gtkw
view_tb_wave: $(TB_DIR)/dump.fst
	gtkwave $(TB_DIR)/dump.fst

clean:
	rm -rf obj_dir $(VCD) *.vcd *.log lif_tb

clean-synth:
	rm -rf $(SYN_DIR)/results/*

clean-test:
	$(MAKE) -C $(TB_DIR) clean_all

clean-all: clean clean-synth clean-test

.PHONY: all build run view view_tb_wave clean lif_tb run_lif synth pnr bitstream program fpga timing clean-synth clean-all test_binomial test_binomial_waves test_frac_lif test_all clean-test
