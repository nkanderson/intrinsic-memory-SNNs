// neural_network equivalence wrapper (baseline lif variant)
//
// Instantiates the parallel `neural_network` and the archived
// `neural_network_serial` side-by-side with identical observations,
// exposing both top-level outputs (`selected_action`, `done`) so a cocotb
// testbench can confirm bit-exact action selection across the rewrite.
//
// NOTE: This file deliberately expects the post-step-3 `neural_network` —
// same port list as today (start / observations / selected_action / done)
// but internally driven by the new parallel linear_layer. It will compile
// against today's serial `neural_network.sv` (the public port list is the
// same), but the test only becomes meaningful once the parallel rewrite
// has landed. The archived `neural_network_serial.sv` continues to point
// at `linear_layer_serial.sv` for fc1/fc2 and is locked-in.
//
// Compile sources (cocotb Makefile target `neural_network_equivalence`):
//   neural_network_equivalence.sv
//   neural_network.sv                 // new parallel implementation (post-step-3)
//   neural_network_serial.sv          // archived legacy implementation
//   linear_layer.sv                   // new parallel implementation
//   linear_layer_serial.sv            // archived legacy implementation
//   neurons/lif.sv
//   neuron_membrane_buffer.sv
//   q_accumulator.sv

module neural_network_equivalence #(
    parameter NUM_INPUTS = 4,
    parameter HL1_SIZE = 8,
    parameter HL2_SIZE = 4,
    parameter NUM_ACTIONS = 2,
    parameter NUM_TIMESTEPS = 4,
    parameter DATA_WIDTH = 16,
    parameter FC2_OUTPUT_WIDTH = DATA_WIDTH,
    parameter MEMBRANE_WIDTH = 24,
    parameter FRAC_BITS = 13,
    parameter THRESHOLD = 8192,
    parameter BETA = 115,
    parameter FC1_WEIGHTS_FILE = "fc1_weights.mem",
    parameter FC1_BIAS_FILE = "fc1_bias.mem",
    parameter FC2_WEIGHTS_FILE = "fc2_weights.mem",
    parameter FC2_BIAS_FILE = "fc2_bias.mem",
    parameter FC_OUT_WEIGHTS_FILE = "fc_out_weights.mem",
    parameter FC_OUT_BIAS_FILE = "fc_out_bias.mem",
    parameter Q_BATCH_SIZE = 2
) (
    input  wire clk,
    input  wire reset,
    input  wire start,
    input  wire signed [DATA_WIDTH-1:0] observations [0:NUM_INPUTS-1],

    // Parallel outputs
    output wire [$clog2(NUM_ACTIONS)-1:0] par_selected_action,
    output wire                            par_done,

    // Serial (legacy) outputs
    output wire [$clog2(NUM_ACTIONS)-1:0] ser_selected_action,
    output wire                            ser_done
);

    neural_network #(
        .NUM_INPUTS(NUM_INPUTS),
        .HL1_SIZE(HL1_SIZE),
        .HL2_SIZE(HL2_SIZE),
        .NUM_ACTIONS(NUM_ACTIONS),
        .NUM_TIMESTEPS(NUM_TIMESTEPS),
        .DATA_WIDTH(DATA_WIDTH),
        .FC2_OUTPUT_WIDTH(FC2_OUTPUT_WIDTH),
        .MEMBRANE_WIDTH(MEMBRANE_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .THRESHOLD(THRESHOLD),
        .BETA(BETA),
        .FC1_WEIGHTS_FILE(FC1_WEIGHTS_FILE),
        .FC1_BIAS_FILE(FC1_BIAS_FILE),
        .FC2_WEIGHTS_FILE(FC2_WEIGHTS_FILE),
        .FC2_BIAS_FILE(FC2_BIAS_FILE),
        .FC_OUT_WEIGHTS_FILE(FC_OUT_WEIGHTS_FILE),
        .FC_OUT_BIAS_FILE(FC_OUT_BIAS_FILE),
        .Q_BATCH_SIZE(Q_BATCH_SIZE)
    ) parallel_inst (
        .clk(clk),
        .reset(reset),
        .start(start),
        .observations(observations),
        .selected_action(par_selected_action),
        .done(par_done)
    );

    neural_network_serial #(
        .NUM_INPUTS(NUM_INPUTS),
        .HL1_SIZE(HL1_SIZE),
        .HL2_SIZE(HL2_SIZE),
        .NUM_ACTIONS(NUM_ACTIONS),
        .NUM_TIMESTEPS(NUM_TIMESTEPS),
        .DATA_WIDTH(DATA_WIDTH),
        .FC2_OUTPUT_WIDTH(FC2_OUTPUT_WIDTH),
        .MEMBRANE_WIDTH(MEMBRANE_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .THRESHOLD(THRESHOLD),
        .BETA(BETA),
        .FC1_WEIGHTS_FILE(FC1_WEIGHTS_FILE),
        .FC1_BIAS_FILE(FC1_BIAS_FILE),
        .FC2_WEIGHTS_FILE(FC2_WEIGHTS_FILE),
        .FC2_BIAS_FILE(FC2_BIAS_FILE),
        .FC_OUT_WEIGHTS_FILE(FC_OUT_WEIGHTS_FILE),
        .FC_OUT_BIAS_FILE(FC_OUT_BIAS_FILE),
        .Q_BATCH_SIZE(Q_BATCH_SIZE)
    ) serial_inst (
        .clk(clk),
        .reset(reset),
        .start(start),
        .observations(observations),
        .selected_action(ser_selected_action),
        .done(ser_done)
    );

endmodule
