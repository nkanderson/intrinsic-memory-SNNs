# Vivado batch build script for top_bitshift_lif_demo

set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ..]]
set common_neuron_dir [file normalize [file join $root_dir .. common sv neurons]]

set proj_name top_bitshift_lif_demo
set proj_dir [file join $root_dir results bitshift_lif vivado_project]
set top_name top_bitshift_lif_demo
set results_dir [file join $root_dir results bitshift_lif]

# Optional guard-bit overrides (can be passed via env vars)
set bitshift_accum_guard_bits 5
set bitshift_numerator_guard_bits 1
set bitshift_accum_lanes 4
if {[info exists ::env(BITSHIFT_ACCUM_GUARD_BITS)]} {
    set bitshift_accum_guard_bits $::env(BITSHIFT_ACCUM_GUARD_BITS)
}
if {[info exists ::env(BITSHIFT_NUMERATOR_GUARD_BITS)]} {
    set bitshift_numerator_guard_bits $::env(BITSHIFT_NUMERATOR_GUARD_BITS)
}
if {[info exists ::env(BITSHIFT_ACCUM_LANES)]} {
    set bitshift_accum_lanes $::env(BITSHIFT_ACCUM_LANES)
}

file mkdir $results_dir
file mkdir $proj_dir

create_project $proj_name $proj_dir -part xc7a100tcsg324-1 -force
set_property target_language Verilog [current_project]

add_files [file join $common_neuron_dir bitshift_lif.sv]
add_files [file join $root_dir sv hex7seg.sv]
add_files [file join $root_dir sv top_bitshift_lif_demo.sv]
add_files -fileset constrs_1 [file join $root_dir sv top_lif_demo.xdc]

set_property top $top_name [current_fileset]
set_property generic [format "BITSHIFT_ACCUM_GUARD_BITS=%s BITSHIFT_NUMERATOR_GUARD_BITS=%s BITSHIFT_ACCUM_LANES=%s" \
    $bitshift_accum_guard_bits $bitshift_numerator_guard_bits $bitshift_accum_lanes] [current_fileset]
update_compile_order -fileset sources_1

puts "INFO: Config: BITSHIFT_ACCUM_GUARD_BITS=$bitshift_accum_guard_bits BITSHIFT_NUMERATOR_GUARD_BITS=$bitshift_numerator_guard_bits BITSHIFT_ACCUM_LANES=$bitshift_accum_lanes"

launch_runs synth_1 -jobs 8
wait_on_run synth_1
open_run synth_1
report_timing_summary -file [file join $results_dir synth_timing_summary.rpt]
report_utilization -file [file join $results_dir synth_utilization.rpt]

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
open_run impl_1
report_timing_summary -file [file join $results_dir impl_timing_summary.rpt]
report_utilization -file [file join $results_dir impl_utilization.rpt]

set impl_run_dir [get_property DIRECTORY [get_runs impl_1]]
set bitfile [file join $impl_run_dir ${top_name}.bit]
if {[file exists $bitfile]} {
    file copy -force $bitfile [file join $results_dir ${top_name}.bit]
} else {
    puts "WARNING: Expected bitstream not found at $bitfile"
}

close_project
