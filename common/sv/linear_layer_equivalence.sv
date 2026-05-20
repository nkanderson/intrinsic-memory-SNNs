// linear_layer equivalence wrapper
//
// Instantiates the parallel `linear_layer` and the archived
// `linear_layer_serial` side-by-side with identical inputs, exposing both
// output interfaces for a cocotb testbench to compare element-wise.
//
// NOTE: This file deliberately expects the parallel interface on
// `linear_layer` (vector outputs + single done pulse). It will not compile
// until the canonical `linear_layer.sv` has been rewritten to the new
// interface. That is by design — the equivalence test is the spec for what
// the new module must produce, written before the rewrite to lock in the
// bit-exactness contract.
//
// Compile sources (cocotb Makefile target `linear_layer_equivalence`):
//   linear_layer_equivalence.sv     // this file
//   linear_layer.sv                  // new parallel implementation (post-step-3)
//   linear_layer_serial.sv           // archived legacy implementation

module linear_layer_equivalence #(
    parameter NUM_INPUTS = 4,
    parameter NUM_OUTPUTS = 4,
    parameter DATA_WIDTH = 16,
    parameter OUTPUT_WIDTH = DATA_WIDTH,
    parameter FRAC_BITS = 13,
    parameter WEIGHTS_FILE = "weights.mem",
    parameter BIAS_FILE = "bias.mem",
    // PARALLELISM only affects the parallel instance. Default = NUM_OUTPUTS
    // (full per-output parallel). Override at compile time to exercise the
    // batched (P < NUM_OUTPUTS) path.
    parameter PARALLELISM = NUM_OUTPUTS
) (
    input  wire clk,
    input  wire reset,
    input  wire start,
    input  wire signed [DATA_WIDTH-1:0] inputs [0:NUM_INPUTS-1],

    // Parallel interface
    output wire signed [OUTPUT_WIDTH-1:0] par_outputs [0:NUM_OUTPUTS-1],
    output wire                           par_done,
    output wire                           par_sat_pos,
    output wire                           par_sat_neg,

    // Serial (legacy) interface
    output wire signed [OUTPUT_WIDTH-1:0]      ser_output_current,
    output wire [$clog2(NUM_OUTPUTS)-1:0]      ser_output_idx,
    output wire                                ser_output_valid,
    output wire                                ser_done,
    output wire                                ser_sat_pos,
    output wire                                ser_sat_neg
);

    // ----- Parallel instance -----
    linear_layer #(
        .NUM_INPUTS(NUM_INPUTS),
        .NUM_OUTPUTS(NUM_OUTPUTS),
        .PARALLELISM(PARALLELISM),
        .DATA_WIDTH(DATA_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .WEIGHTS_FILE(WEIGHTS_FILE),
        .BIAS_FILE(BIAS_FILE)
    ) parallel_inst (
        .clk(clk),
        .reset(reset),
        .start(start),
        .inputs(inputs),
        .outputs(par_outputs),
        .done(par_done),
        .sat_pos(par_sat_pos),
        .sat_neg(par_sat_neg)
    );

    // ----- Serial (legacy) instance -----
    linear_layer_serial #(
        .NUM_INPUTS(NUM_INPUTS),
        .NUM_OUTPUTS(NUM_OUTPUTS),
        .DATA_WIDTH(DATA_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .WEIGHTS_FILE(WEIGHTS_FILE),
        .BIAS_FILE(BIAS_FILE)
    ) serial_inst (
        .clk(clk),
        .reset(reset),
        .start(start),
        .inputs(inputs),
        .output_current(ser_output_current),
        .output_idx(ser_output_idx),
        .output_valid(ser_output_valid),
        .sat_pos(ser_sat_pos),
        .sat_neg(ser_sat_neg),
        .done(ser_done)
    );

endmodule
