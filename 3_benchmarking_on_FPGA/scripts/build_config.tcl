# Vivado batch build for one stage-3 SNN configuration.
#
# Invoke via the build_config.py wrapper (which resolves paths from configs.py):
#     python 3_benchmarking_on_FPGA/scripts/build_config.py <config_name>
#
# Or directly:
#     vivado -mode batch -source build_config.tcl -tclargs \
#            <config_name> <board_top_module> <xdc_filename> <weights_subdir>
#
# All paths are interpreted relative to the repo root, which is the current
# working directory when invoked by the wrapper.
#
# Produces reports under 3_benchmarking_on_FPGA/results/<config_name>/ with
# canonical filenames:
#     synth_utilization.rpt    synth_timing_summary.rpt
#     impl_utilization.rpt     impl_timing_summary.rpt
#     impl_power.rpt           bitstream.bit

if {[llength $argv] < 4} {
    puts stderr "Usage: vivado -mode batch -source build_config.tcl -tclargs"
    puts stderr "       <config_name> <board_top_module> <xdc_filename> <weights_subdir>"
    exit 1
}

set config_name      [lindex $argv 0]
set board_top_module [lindex $argv 1]
set xdc_filename     [lindex $argv 2]
set weights_subdir   [lindex $argv 3]

set repo_root      [pwd]
set stage3_root    [file join $repo_root 3_benchmarking_on_FPGA]
set common_sv_root [file join $repo_root common sv]
set weights_root   [file join $common_sv_root cocotb tests weights]

set results_dir [file join $stage3_root results $config_name]
file mkdir $results_dir

set project_dir [file join $results_dir vivado_project]
set project_name "board_top_${config_name}"
if {[file exists $project_dir]} {
    puts "INFO: removing existing project dir $project_dir"
    file delete -force $project_dir
}

create_project $project_name $project_dir -part xc7a100tcsg324-1 -force

# --- Sources -----------------------------------------------------------------
# Recursively add common/sv/ but skip the cocotb tree and the UART loopback TB
# (per docs/vivado_bringup_lif_64_16.md).
set sv_files [glob -nocomplain -directory $common_sv_root -types f -- *.sv]
foreach subdir [glob -nocomplain -directory $common_sv_root -types d -- *] {
    set name [file tail $subdir]
    if {$name eq "cocotb"} { continue }
    foreach f [glob -nocomplain -directory $subdir -types f -- *.sv] {
        lappend sv_files $f
    }
    foreach f [glob -nocomplain -directory $subdir -types f -- */*.sv] {
        lappend sv_files $f
    }
}
# Walk one more level (common/sv/host_if/*.sv except loopback TB)
foreach subdir [glob -nocomplain -directory $common_sv_root -types d -- *] {
    if {[file tail $subdir] eq "cocotb"} { continue }
    foreach sub2 [glob -nocomplain -directory $subdir -types d -- *] {
        foreach f [glob -nocomplain -directory $sub2 -types f -- *.sv] {
            lappend sv_files $f
        }
    }
}
set sv_files [lsearch -all -inline -not $sv_files \
              "*common/sv/host_if/tb_uart_loopback.sv"]

# Stage-3 board top
lappend sv_files [file join $stage3_root sv "${board_top_module}.sv"]

add_files -norecurse $sv_files

# Weight .mem files (basename-referenced by $readmemh in board_top)
set weights_dir [file join $weights_root $weights_subdir]
if {![file isdirectory $weights_dir]} {
    puts stderr "ERROR: weights dir not found: $weights_dir"
    exit 1
}
set mem_files [glob -nocomplain -directory $weights_dir -- *.mem]
# If weights_subdir is nested (e.g. "fractional-32-8-8/q2_13"), also pick up
# shared .mem files in the parent directory — fractional configs put
# gl_coefficients.mem one level above the quantization-specific weights.
# Without this, $readmemh("gl_coefficients.mem", ...) synthesizes to a
# zero-initialized memory and the fractional history term silently collapses
# to zero. Symptom: FPGA returns a constant action regardless of input.
if {[string match "*/*" $weights_subdir]} {
    set parent_dir [file dirname $weights_dir]
    foreach f [glob -nocomplain -directory $parent_dir -- *.mem] {
        lappend mem_files $f
    }
}
if {[llength $mem_files] == 0} {
    puts stderr "ERROR: no .mem files under $weights_dir"
    exit 1
}
puts "INFO: adding [llength $mem_files] memory init file(s):"
foreach f $mem_files {
    puts "  $f"
}
add_files -norecurse $mem_files
set_property file_type {Memory Initialization Files} [get_files *.mem]

# Constraints
set xdc_path [file join $stage3_root constraints $xdc_filename]
if {![file exists $xdc_path]} {
    puts stderr "ERROR: XDC not found: $xdc_path"
    exit 1
}
add_files -fileset constrs_1 -norecurse $xdc_path

# Top
set_property top $board_top_module [current_fileset]
update_compile_order -fileset sources_1

# --- Synth ------------------------------------------------------------------
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts stderr "ERROR: synthesis failed"
    exit 1
}

open_run synth_1 -name synth_1
report_utilization -file [file join $results_dir synth_utilization.rpt]
report_timing_summary -file [file join $results_dir synth_timing_summary.rpt]
close_design

# --- Impl + bitstream (with phys_opt_design AggressiveExplore) --------------
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts stderr "ERROR: implementation/bitstream failed"
    exit 1
}

set impl_run_dir [get_property DIRECTORY [get_runs impl_1]]

# Locate the generated reports and copy to canonical names. Vivado names them
# <top>_utilization_placed.rpt / <top>_timing_summary_routed.rpt / *_power_routed.rpt.
proc copy_first_match {src_dir patterns dst} {
    foreach pat $patterns {
        foreach hit [glob -nocomplain -directory $src_dir -- $pat] {
            file copy -force $hit $dst
            return $hit
        }
    }
    puts stderr "WARN: none of $patterns found in $src_dir"
    return ""
}

copy_first_match $impl_run_dir \
    [list "*_utilization_placed.rpt" "*utilization*.rpt"] \
    [file join $results_dir impl_utilization.rpt]
copy_first_match $impl_run_dir \
    [list "*_timing_summary_routed.rpt" "*timing_summary*.rpt"] \
    [file join $results_dir impl_timing_summary.rpt]
copy_first_match $impl_run_dir \
    [list "*_power_routed.rpt" "*power*.rpt"] \
    [file join $results_dir impl_power.rpt]
copy_first_match $impl_run_dir \
    [list "${board_top_module}.bit" "*.bit"] \
    [file join $results_dir bitstream.bit]

puts "BUILD OK: $config_name"
puts "  Reports + bitstream in $results_dir"
exit 0
