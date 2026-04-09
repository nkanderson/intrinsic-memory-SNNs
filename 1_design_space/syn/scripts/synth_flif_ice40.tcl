# tcl synthesis script for fractional order LIF neuron on iCE40
# This import statement is required in order to have access to functions like read_verilog
yosys -import

# Retrieve the coefficient file path from the environment variable
set coeff_file $::env(COEFF_PATH)

# Read SystemVerilog files
# NOTE: COEFF_FILE needs to be set as an env var when calling this script with yosys, e.g.
# COEFF_PATH="$(dirname "$(realpath synth_flif_ice40.tcl)")/../../src/coefficients.mem" yosys synth_flif_ice40.tcl
# Alternatively, this file could have a .ys extension, but we'd need to include the -c option in the yosys command
read_verilog -DICE40 -DCOEFF_FILE="$coeff_file" -sv ../../src/frac_order_lif.sv
read_verilog -DICE40 -sv ../top_flif_icebreaker.sv

# For checking that the .mem file was read as expected
# Confirmed this file had the expected contents in the coefficients array on Aug 18, 2025
# write_verilog -noattr top_flif_icebreaker_out.v

# Set top-level module
hierarchy -top top_flif_icebreaker

# Synthesis for iCE40
# The -dsp option will attempt to use iCE40 UltraPlus DSP cells for large arithmetic
# The icebreaker from 1BitSquared has 8 of these DSP cells. Unfortunately, it
# does not allow for using the max then automatically using LUTs for the rest,
# so we'll use ifdef and the ICE40 macro above. But on larger FPGAs, we should
# use the -dsp option.
synth_ice40 -top top_flif_icebreaker -json top_flif_icebreaker.json

# Print statistics
stat
