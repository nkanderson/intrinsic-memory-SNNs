# Vivado batch build script for top_bitshift_lif_demo

set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ..]]
set common_neuron_dir [file normalize [file join $root_dir .. common sv neurons]]

set proj_name top_bitshift_lif_demo
set proj_dir [file join $root_dir results bitshift_lif vivado_project]
set top_name top_bitshift_lif_demo
set results_dir [file join $root_dir results bitshift_lif]

file mkdir $results_dir
file mkdir $proj_dir

create_project $proj_name $proj_dir -part xc7a100tcsg324-1 -force
set_property target_language Verilog [current_project]

add_files [file join $common_neuron_dir bitshift_lif.sv]
add_files [file join $root_dir sv hex7seg.sv]
add_files [file join $root_dir sv top_bitshift_lif_demo.sv]
add_files -fileset constrs_1 [file join $root_dir sv top_lif_demo.xdc]

set_property top $top_name [current_fileset]
update_compile_order -fileset sources_1

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
