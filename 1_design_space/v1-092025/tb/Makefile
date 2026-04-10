# Cocotb Makefile for running tests

# Using Verilator for now instead of Icarus
SIM ?= verilator

# Top level module
TOPLEVEL_LANG ?= verilog

# Source directory (relative to tb/)
SRC_DIR = ../src

# Explicitly set Verilator path
VERILATOR_ROOT = /usr/local/bin
export PATH := $(VERILATOR_ROOT):$(PATH)

# Test for binomial coefficient LUT
ifeq ($(TEST),binomial)
	VERILOG_SOURCES = $(SRC_DIR)/binomial_coeff_lut.sv
	TOPLEVEL = binomial_coeff_lut
	MODULE = test_binomial_coeff_lut
endif

# Test for Lapicque LIF neuron
ifeq ($(TEST),lapicque)
	VERILOG_SOURCES = $(SRC_DIR)/lapicque.sv
	TOPLEVEL = lapicque
	MODULE = test_lapicque
endif

# Test for fractional order LIF
ifeq ($(TEST),frac_lif)
	VERILOG_SOURCES = $(SRC_DIR)/frac_order_lif.sv
	TOPLEVEL = frac_order_lif
	MODULE = test_frac_order_lif
# 	VERILATOR_COMPILE_ARGS +=
endif

# Test for fractional order LIF memory effects
ifeq ($(TEST),frac_lif_memory)
	VERILOG_SOURCES = $(SRC_DIR)/frac_order_lif.sv
	TOPLEVEL = frac_order_lif
	MODULE = test_frac_order_lif_memory
endif

# Cocotb configuration
export COCOTB_RESOLVE_X ?= ZEROS

# Verilator specific settings
# Set the absolute location of the coefficients file for the frac_order_lif module
ifeq ($(SIM),verilator)
	EXTRA_ARGS += --trace --trace-fst --trace-structs
	EXTRA_ARGS += --trace-max-array 256
	EXTRA_ARGS += -DCOEFF_FILE="\"$(abspath $(SRC_DIR))/coefficients.mem\""
	COMPILE_ARGS += -CFLAGS -std=c++14
	# Force cocotb to use specific verilator
	export VERILATOR_EXECUTABLE = $(VERILATOR_ROOT)/verilator
endif

# Include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim

# Custom targets
.PHONY: binomial frac_lif frac_lif_memory clean_all check_verilator

check_verilator:
	@echo "Checking Verilator version..."
	@which verilator
	@verilator --version

binomial: check_verilator
	$(MAKE) TEST=binomial SIM=verilator

lapicque: check_verilator
	$(MAKE) TEST=lapicque SIM=verilator

frac_lif: check_verilator
	$(MAKE) TEST=frac_lif SIM=verilator

frac_lif_memory: check_verilator
	$(MAKE) TEST=frac_lif_memory SIM=verilator

clean_all:
	rm -rf __pycache__ .pytest_cache sim_build *.vcd dump.fst results.xml
