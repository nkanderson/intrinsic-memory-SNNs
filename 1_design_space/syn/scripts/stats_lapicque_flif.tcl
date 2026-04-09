# tcl synthesis script for comparing Lapicque and fractional-order LIF neuron stats on iCE40
# This import statement is required in order to have access to functions like read_verilog
# NOTE: COEFF_FILE needs to be set as an env var when calling this script with yosys, e.g.
# COEFF_PATH="$(dirname "$(realpath stats_lapicque_flif.tcl)")/../../src/coefficients.mem" yosys stats_lapicque_flif.tcl
yosys -import

# Retrieve the coefficient file path from the environment variable
set coeff_file $::env(COEFF_PATH)

puts "Starting synthesis statistics comparison for Lapicque and Fractional-Order LIF neurons"
puts "Coefficient file path: $coeff_file"

#==============================================================================
# Fractional-Order LIF Module Analysis
#==============================================================================

puts "\n=== Processing fractional-order LIF module ==="

# Read fractional-order LIF SystemVerilog file
read_verilog -DICE40 -DCOEFF_FILE="$coeff_file" -sv ../../src/frac_order_lif.sv

# Set top-level module
hierarchy -top frac_order_lif

# Dump post-read_verilog stats to file
puts "Generating post-read_verilog stats for frac_order_lif..."
tee -o ../../data/frac_order_lif_stats.txt stat -top frac_order_lif

# Add divider to stats file
set divider_file [open "../../data/frac_order_lif_stats.txt" "a"]
puts $divider_file ""
puts $divider_file "==============================================="
puts $divider_file "Post-synthesis (iCE40 cells) Statistics"
puts $divider_file "==============================================="
puts $divider_file ""
close $divider_file

# Synthesize for iCE40
synth_ice40 -top frac_order_lif

# Dump post-synthesis stats to file
puts "Generating post-synthesis stats for frac_order_lif..."
tee -a ../../data/frac_order_lif_stats.txt stat -top frac_order_lif

# Clear design for next module
design -reset

#==============================================================================
# Lapicque LIF Module Analysis
#==============================================================================

puts "\n=== Processing Lapicque LIF module ==="

# Read Lapicque LIF SystemVerilog file
read_verilog -DICE40 -sv ../../src/lapicque.sv

# Set top-level module
hierarchy -top lapicque

# Dump post-read_verilog stats to file
puts "Generating post-read_verilog stats for lapicque..."
tee -o ../../data/lapicque_stats.txt stat -top lapicque

# Add divider to stats file
set divider_file [open "../../data/lapicque_stats.txt" "a"]
puts $divider_file ""
puts $divider_file "==============================================="
puts $divider_file "Post-synthesis (iCE40 cells) Statistics"
puts $divider_file "==============================================="
puts $divider_file ""
close $divider_file

# Synthesize for iCE40 with DSP optimization
synth_ice40 -dsp -top lapicque

# Dump post-synthesis stats to file
puts "Generating post-synthesis stats for lapicque..."
tee -a ../../data/lapicque_stats.txt stat -top lapicque

puts "\n=== Synthesis statistics comparison complete ==="
puts "Results saved to:"
puts "  - ../../data/frac_order_lif_stats.txt"
puts "  - ../../data/lapicque_stats.txt"
