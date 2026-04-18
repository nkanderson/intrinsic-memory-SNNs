// Linear Layer Module (Sequential/Serial)
// Implements a fully connected layer: output = weights * input + bias
// Equivalent to nn.Linear in PyTorch
//
// Computes one output neuron per clock cycle (serial processing).
// Outputs are streamed one per cycle via output_current with corresponding output_idx.
//
// Fixed-point format: QS2.13 (16-bit signed, 2 integer bits, 13 fractional bits)
// Weights are stored in row-major order in a flattened array:
//   weights_flat[n*NUM_INPUTS + i] = weight for output n, input i
//
// Timing (registered outputs for clean timing closure):
//   - Cycle 0: Assert 'start' for one cycle with valid inputs
//   - Cycle 1: Inputs latched, state transitions to COMPUTING
//   - Cycle 2: First output valid (output_valid=1, output_idx=0)
//   - Cycle 2+N: Output N valid, done asserts on final output (N = NUM_OUTPUTS-1)
//   - Total latency: 1 + NUM_OUTPUTS cycles from start to done

module linear_layer #(
    parameter NUM_INPUTS = 4,
    parameter NUM_OUTPUTS = 16,
    parameter DATA_WIDTH = 16,
    parameter OUTPUT_WIDTH = DATA_WIDTH,
    parameter FRAC_BITS = 13,
    parameter WEIGHTS_FILE = "weights.mem",
    parameter BIAS_FILE = "bias.mem"
) (
    input wire clk,
    input wire reset,
    input wire start,
    input wire signed [DATA_WIDTH-1:0] inputs [0:NUM_INPUTS-1],
    output logic signed [OUTPUT_WIDTH-1:0] output_current, // One current per cycle
    output logic [$clog2(NUM_OUTPUTS)-1:0] output_idx,   // Which neuron (0 to NUM_OUTPUTS-1)
    output logic output_valid,                            // Current output is valid
    // TODO: Add debug flags so these signals are only added in debug mode
    output logic sat_pos,                                 // Current output saturated high
    output logic sat_neg,                                 // Current output saturated low
    output logic done
);

    // Index width for output counter
    localparam IDX_WIDTH = $clog2(NUM_OUTPUTS);

    // Flattened weights array: row-major order
    logic signed [DATA_WIDTH-1:0] weights_flat [0:NUM_OUTPUTS*NUM_INPUTS-1];
    logic signed [DATA_WIDTH-1:0] biases [0:NUM_OUTPUTS-1];

    // Load weights and biases from memory files
    initial begin
        $readmemh(WEIGHTS_FILE, weights_flat);
        $readmemh(BIAS_FILE, biases);
    end

    // Accumulator width: multiplication doubles bits, add extra for sum of NUM_INPUTS terms
    localparam ACCUM_WIDTH = 2 * DATA_WIDTH + $clog2(NUM_INPUTS + 1);

    // Saturation bounds
    localparam signed [OUTPUT_WIDTH-1:0] MAX_VAL = {1'b0, {(OUTPUT_WIDTH-1){1'b1}}};
    localparam signed [OUTPUT_WIDTH-1:0] MIN_VAL = {1'b1, {(OUTPUT_WIDTH-1){1'b0}}};

    // State machine
    typedef enum logic [1:0] {
        IDLE,
        COMPUTING
    } state_t;

    state_t state;
    logic [IDX_WIDTH-1:0] neuron_idx;  // Counter for current output neuron

    // Latched inputs (held stable during computation)
    logic signed [DATA_WIDTH-1:0] inputs_latched [0:NUM_INPUTS-1];

    // Combinational computation of current neuron's output
    logic signed [ACCUM_WIDTH-1:0] accum;
    logic signed [ACCUM_WIDTH-1:0] scaled;
    logic signed [OUTPUT_WIDTH-1:0] saturated;
    logic sat_pos_comb;
    logic sat_neg_comb;

    always_comb begin
        // Compute weighted sum for current neuron: Σ(input[i] * weight[neuron_idx][i])
        accum = 0;
        sat_pos_comb = 1'b0;
        sat_neg_comb = 1'b0;
        for (int i = 0; i < NUM_INPUTS; i++) begin
            accum = accum + inputs_latched[i] * weights_flat[neuron_idx * NUM_INPUTS + i];
        end

        // Scale back by FRAC_BITS (fixed-point multiply correction) and add bias
        scaled = (accum >>> FRAC_BITS) +
                 $signed({{(ACCUM_WIDTH-DATA_WIDTH){biases[neuron_idx][DATA_WIDTH-1]}},
                          biases[neuron_idx]});

        // Saturate to output range
        if (scaled > $signed({{(ACCUM_WIDTH-OUTPUT_WIDTH){1'b0}}, MAX_VAL})) begin
            saturated = MAX_VAL;
            sat_pos_comb = 1'b1;
        end else if (scaled < $signed({{(ACCUM_WIDTH-OUTPUT_WIDTH){1'b1}}, MIN_VAL})) begin
            saturated = MIN_VAL;
            sat_neg_comb = 1'b1;
        end else begin
            saturated = scaled[OUTPUT_WIDTH-1:0];
        end
    end

    // FSM
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            neuron_idx <= '0;
            done <= 1'b0;
            output_valid <= 1'b0;
            sat_pos <= 1'b0;
            sat_neg <= 1'b0;
            output_current <= '0;
            output_idx <= '0;
            for (int i = 0; i < NUM_INPUTS; i++) begin
                inputs_latched[i] <= '0;
            end
        end else begin
            unique case (state)
                IDLE: begin
                    done <= 1'b0;
                    output_valid <= 1'b0;
                    sat_pos <= 1'b0;
                    sat_neg <= 1'b0;
                    if (start) begin
                        // Latch inputs and start computation
                        for (int i = 0; i < NUM_INPUTS; i++) begin
                            inputs_latched[i] <= inputs[i];
                        end
                        neuron_idx <= '0;
                        state <= COMPUTING;
                    end
                end

                COMPUTING: begin
                    // Output current result
                    output_current <= saturated;
                    output_idx <= neuron_idx;
                    output_valid <= 1'b1;
                    sat_pos <= sat_pos_comb;
                    sat_neg <= sat_neg_comb;

                    // Check if this is the last neuron
                    if (neuron_idx == IDX_WIDTH'(NUM_OUTPUTS - 1)) begin
                        done <= 1'b1;
                        state <= IDLE;
                    end else begin
                        neuron_idx <= neuron_idx + 1'b1;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
