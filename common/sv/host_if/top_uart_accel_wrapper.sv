module top_uart_accel_wrapper #(
    parameter integer MODEL_TYPE = 0, // 0=standard, 1=fractional, 2=bitshift
    parameter integer CLOCK_HZ = 100_000_000,
    parameter integer BAUD = 115_200,

    // Network architecture
    parameter integer NUM_INPUTS = 4,
    parameter integer HL1_SIZE = 64,
    parameter integer HL2_SIZE = 16,
    parameter integer NUM_ACTIONS = 2,
    parameter integer NUM_TIMESTEPS = 30,

    // Datapath
    parameter integer DATA_WIDTH = 16,
    parameter integer FC2_OUTPUT_WIDTH = DATA_WIDTH,
    parameter integer MEMBRANE_WIDTH = 24,
    parameter integer FRAC_BITS = 13,

    // LIF/Neuron params
    parameter integer THRESHOLD = 8192,
    parameter integer BETA = 115,

    // Fractional params
    parameter integer HISTORY_LENGTH = 64,
    parameter integer COEFF_WIDTH = 16,
    parameter integer COEFF_FRAC_BITS = 15,
    parameter [15:0] C_SCALED = 16'd256,
    parameter [15:0] INV_DENOM = 16'd58988,
    parameter GL_COEFF_FILE = "gl_coefficients.mem",

    // Bitshift params
    parameter integer SHIFT_WIDTH = 8,
    parameter [1:0] SHIFT_MODE = 2'd3,
    parameter integer CUSTOM_DECAY_RATE = 3,

    // Weight files
    parameter FC1_WEIGHTS_FILE = "fc1_weights.mem",
    parameter FC1_BIAS_FILE = "fc1_bias.mem",
    parameter FC2_WEIGHTS_FILE = "fc2_weights.mem",
    parameter FC2_BIAS_FILE = "fc2_bias.mem",
    parameter FC_OUT_WEIGHTS_FILE = "fc_out_weights.mem",
    parameter FC_OUT_BIAS_FILE = "fc_out_bias.mem",

    // q_accumulator tuning
    parameter integer Q_BATCH_SIZE = 4
) (
    input  wire clk,
    input  wire reset,
    input  wire uart_rx_i,
    output wire uart_tx_o
);

    localparam integer ACTION_WIDTH = (NUM_ACTIONS > 1) ? $clog2(NUM_ACTIONS) : 1;

    logic [7:0] rx_data;
    logic rx_valid;
    logic [7:0] tx_data;
    logic tx_start;
    logic tx_busy;

    logic start_pulse;
    logic signed [DATA_WIDTH-1:0] observations [0:NUM_INPUTS-1];
    logic accel_done;
    logic accel_busy;
    logic [ACTION_WIDTH-1:0] selected_action;

    uart_rx #(
        .CLOCK_HZ(CLOCK_HZ),
        .BAUD(BAUD)
    ) u_uart_rx (
        .clk(clk),
        .reset(reset),
        .rx(uart_rx_i),
        .data_out(rx_data),
        .data_valid(rx_valid)
    );

    uart_tx #(
        .CLOCK_HZ(CLOCK_HZ),
        .BAUD(BAUD)
    ) u_uart_tx (
        .clk(clk),
        .reset(reset),
        .start(tx_start),
        .data_in(tx_data),
        .tx(uart_tx_o),
        .busy(tx_busy),
        .done()
    );

    uart_accel_ctrl #(
        .NUM_INPUTS(NUM_INPUTS),
        .ACTION_WIDTH(ACTION_WIDTH)
    ) u_ctrl (
        .clk(clk),
        .reset(reset),
        .rx_data(rx_data),
        .rx_valid(rx_valid),
        .tx_data(tx_data),
        .tx_start(tx_start),
        .tx_busy(tx_busy),
        .start_pulse(start_pulse),
        .observations(observations),
        .accel_done(accel_done),
        .accel_busy(accel_busy),
        .accel_action(selected_action)
    );

    // Busy flag exported to control plane
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            accel_busy <= 1'b0;
        end else begin
            if (start_pulse) begin
                accel_busy <= 1'b1;
            end
            if (accel_done) begin
                accel_busy <= 1'b0;
            end
        end
    end

    generate
        if (MODEL_TYPE == 0) begin : gen_nn_standard
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
            ) u_accel (
                .clk(clk),
                .reset(reset),
                .start(start_pulse),
                .observations(observations),
                .selected_action(selected_action),
                .done(accel_done)
            );
        end else if (MODEL_TYPE == 1) begin : gen_nn_fractional
            neural_network_fractional #(
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
                .HISTORY_LENGTH(HISTORY_LENGTH),
                .COEFF_WIDTH(COEFF_WIDTH),
                .COEFF_FRAC_BITS(COEFF_FRAC_BITS),
                .C_SCALED(C_SCALED),
                .INV_DENOM(INV_DENOM),
                .GL_COEFF_FILE(GL_COEFF_FILE),
                .FC1_WEIGHTS_FILE(FC1_WEIGHTS_FILE),
                .FC1_BIAS_FILE(FC1_BIAS_FILE),
                .FC2_WEIGHTS_FILE(FC2_WEIGHTS_FILE),
                .FC2_BIAS_FILE(FC2_BIAS_FILE),
                .FC_OUT_WEIGHTS_FILE(FC_OUT_WEIGHTS_FILE),
                .FC_OUT_BIAS_FILE(FC_OUT_BIAS_FILE),
                .Q_BATCH_SIZE(Q_BATCH_SIZE)
            ) u_accel (
                .clk(clk),
                .reset(reset),
                .start(start_pulse),
                .observations(observations),
                .selected_action(selected_action),
                .done(accel_done)
            );
        end else begin : gen_nn_bitshift
            neural_network_bitshift #(
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
                .HISTORY_LENGTH(HISTORY_LENGTH),
                .SHIFT_WIDTH(SHIFT_WIDTH),
                .SHIFT_MODE(SHIFT_MODE),
                .CUSTOM_DECAY_RATE(CUSTOM_DECAY_RATE),
                .C_SCALED(C_SCALED),
                .INV_DENOM(INV_DENOM),
                .FC1_WEIGHTS_FILE(FC1_WEIGHTS_FILE),
                .FC1_BIAS_FILE(FC1_BIAS_FILE),
                .FC2_WEIGHTS_FILE(FC2_WEIGHTS_FILE),
                .FC2_BIAS_FILE(FC2_BIAS_FILE),
                .FC_OUT_WEIGHTS_FILE(FC_OUT_WEIGHTS_FILE),
                .FC_OUT_BIAS_FILE(FC_OUT_BIAS_FILE),
                .Q_BATCH_SIZE(Q_BATCH_SIZE)
            ) u_accel (
                .clk(clk),
                .reset(reset),
                .start(start_pulse),
                .observations(observations),
                .selected_action(selected_action),
                .done(accel_done)
            );
        end
    endgenerate

endmodule
