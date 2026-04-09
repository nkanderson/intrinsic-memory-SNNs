// Q-Value Accumulator Module
// Computes Q-values by accumulating weighted membrane potentials across all timesteps
//
// This module combines the functionality of:
// - Reading from per-neuron membrane buffers
// - Computing weighted sums (like linear_layer)
// - Accumulating Q-values across all timesteps
// - Averaging to produce final Q-values
//
// For each timestep t:
//   Q[0] += Σ(w[0][n] × membrane[n][t]) + bias[0]
//   Q[1] += Σ(w[1][n] × membrane[n][t]) + bias[1]
// Final: Q_avg[q] = Q_total[q] / NUM_TIMESTEPS
//
// Batched processing: BATCH_SIZE neurons processed per cycle
// Total multipliers: BATCH_SIZE × NUM_ACTIONS
//
// Fixed-point format: QS2.13 (16-bit signed, 2 integer bits, 13 fractional bits)
//
// Timing:
//   - Assert 'start' when all neuron membrane buffers are full
//   - Latency: NUM_TIMESTEPS × (NUM_NEURONS / BATCH_SIZE) + 2 cycles
//   - With defaults (30 × 4 + 2 = 122 cycles)
//   - Asserts 'done' when Q-values are ready

module q_accumulator #(
    parameter NUM_NEURONS = 16,          // Number of neurons in final hidden layer
    parameter NUM_TIMESTEPS = 30,        // Number of timesteps per inference
    parameter NUM_ACTIONS = 2,           // Number of Q-values (actions)
    parameter BATCH_SIZE = 4,            // Neurons processed per cycle (must divide NUM_NEURONS)
    parameter DATA_WIDTH = 16,           // Width of weights and outputs
    parameter MEMBRANE_WIDTH = 24,       // Width of membrane potentials
    parameter FRAC_BITS = 13,            // Fractional bits in fixed-point
    parameter WEIGHTS_FILE = "fc_out_weights.mem",
    parameter BIAS_FILE = "fc_out_bias.mem"
) (
    input wire clk,
    input wire reset,
    input wire start,                    // Begin Q-value computation

    // Interface to per-neuron membrane buffers
    output logic [$clog2(NUM_TIMESTEPS)-1:0] read_timestep,  // Shared timestep for all buffers
    // NOTE: We may need to split this into multiple accumulators or reduce this fan in
    // to BATCH_SIZE at a time if synthesis struggles with this size.
    input wire signed [MEMBRANE_WIDTH-1:0] membrane_in [0:NUM_NEURONS-1], // From all buffers

    // Action selection from full-precision Q-values, computed at internal accumulator width
    // Q-values are not output because they routinely exceed the DATA_WIDTH (QS2.13) range
    // and would saturate, losing the distinction between actions. The full-precision
    // comparison that produces selected_action is the authoritative result.
    output logic [$clog2(NUM_ACTIONS)-1:0] selected_action,
    output logic done
);

    // Derived parameters
    localparam NUM_BATCHES = NUM_NEURONS / BATCH_SIZE;
    localparam BATCH_IDX_WIDTH = $clog2(NUM_BATCHES) > 0 ? $clog2(NUM_BATCHES) : 1;

    // Counters
    logic [$clog2(NUM_TIMESTEPS)-1:0] timestep_counter;
    logic [BATCH_IDX_WIDTH-1:0] batch_counter;

    // Weights and biases
    // weights_flat[a * NUM_NEURONS + n] = weight for action a, neuron n
    logic signed [DATA_WIDTH-1:0] weights_flat [0:NUM_ACTIONS*NUM_NEURONS-1];
    logic signed [DATA_WIDTH-1:0] biases [0:NUM_ACTIONS-1];

    initial begin
        $readmemh(WEIGHTS_FILE, weights_flat);
        $readmemh(BIAS_FILE, biases);
    end

    // Accumulator width: needs headroom for sum across neurons and timesteps
    // membrane (24-bit) × weight (16-bit) = 40-bit product
    // Sum across NUM_NEURONS: +log2(NUM_NEURONS) bits
    // Sum across NUM_TIMESTEPS: +log2(NUM_TIMESTEPS) bits
    localparam ACCUM_WIDTH = MEMBRANE_WIDTH + DATA_WIDTH + $clog2(NUM_NEURONS) + $clog2(NUM_TIMESTEPS) + 2;

    // Q-value accumulators (one per action)
    logic signed [ACCUM_WIDTH-1:0] q_accum [0:NUM_ACTIONS-1];

    // State machine
    typedef enum logic [2:0] {
        IDLE,
        PROCESSING,     // Computing weighted sums for current batch
        NEXT_TIMESTEP,  // Move to next timestep
        DIVIDING,       // Divide by NUM_TIMESTEPS
        DONE_STATE
    } state_t;

    state_t state;

    // Batch products - one per (action, batch_neuron) pair
    // Total: NUM_ACTIONS × BATCH_SIZE multipliers
    logic signed [MEMBRANE_WIDTH+DATA_WIDTH-1:0] products [0:NUM_ACTIONS-1][0:BATCH_SIZE-1];

    // Batch sums per action
    logic signed [ACCUM_WIDTH-1:0] batch_sum [0:NUM_ACTIONS-1];

    // Division result
    logic signed [ACCUM_WIDTH-1:0] q_divided [0:NUM_ACTIONS-1];

    // Output read_timestep to buffers
    assign read_timestep = timestep_counter;

    // Combinational: compute all products and batch sums
    always_comb begin
        int neuron_idx;
        for (int a = 0; a < NUM_ACTIONS; a++) begin
            batch_sum[a] = '0;
            for (int b = 0; b < BATCH_SIZE; b++) begin
                // Neuron index = batch_counter * BATCH_SIZE + b
                neuron_idx = batch_counter * BATCH_SIZE + b;
                products[a][b] = membrane_in[neuron_idx] *
                                 weights_flat[a * NUM_NEURONS + neuron_idx];
                // Scale and add to batch sum
                batch_sum[a] = batch_sum[a] + (products[a][b] >>> FRAC_BITS);
            end
        end
    end

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            timestep_counter <= '0;
            batch_counter <= '0;
            done <= 1'b0;
            selected_action <= '0;
            for (int a = 0; a < NUM_ACTIONS; a++) begin
                q_accum[a] <= '0;
                q_divided[a] <= '0;
            end
        end else begin
            unique case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        // Initialize accumulators
                        for (int a = 0; a < NUM_ACTIONS; a++) begin
                            q_accum[a] <= '0;
                            q_divided[a] <= '0;
                        end
                        timestep_counter <= '0;
                        batch_counter <= '0;
                        state <= PROCESSING;
                    end else begin
                        state <= IDLE;
                    end
                end

                PROCESSING: begin
                    // Add batch sums to accumulators (computed combinationally)
                    for (int a = 0; a < NUM_ACTIONS; a++) begin
                        q_accum[a] <= q_accum[a] + batch_sum[a];
                    end

                    // Move to next batch or timestep
                    if (batch_counter == BATCH_IDX_WIDTH'(NUM_BATCHES - 1)) begin
                        // Finished all batches for this timestep, add biases
                        for (int a = 0; a < NUM_ACTIONS; a++) begin
                            q_accum[a] <= q_accum[a] + batch_sum[a] +
                                         $signed({{(ACCUM_WIDTH-DATA_WIDTH){biases[a][DATA_WIDTH-1]}}, biases[a]});
                        end
                        batch_counter <= '0;
                        state <= NEXT_TIMESTEP;
                    end else begin
                        batch_counter <= batch_counter + 1'b1;
                        state <= PROCESSING;
                    end
                end

                NEXT_TIMESTEP: begin
                    if (timestep_counter == $clog2(NUM_TIMESTEPS)'(NUM_TIMESTEPS - 1)) begin
                        // All timesteps done, divide by NUM_TIMESTEPS
                        state <= DIVIDING;
                    end else begin
                        timestep_counter <= timestep_counter + 1'b1;
                        state <= PROCESSING;
                    end
                end

                DIVIDING: begin
                    // Divide accumulated Q-values by NUM_TIMESTEPS
                    for (int a = 0; a < NUM_ACTIONS; a++) begin
                        q_divided[a] <= q_accum[a] / $signed(NUM_TIMESTEPS);
                    end
                    state <= DONE_STATE;
                end

                DONE_STATE: begin
                    // Select action from full-precision Q-values
                    // The comparison uses the full ACCUM_WIDTH bits, preserving
                    // distinctions that would be lost if we saturated to DATA_WIDTH
                    selected_action <= (q_divided[0] >= q_divided[1]) ? 1'b0 : 1'b1;
                    done <= 1'b1;

                    // Return to processing on next start, else idle
                    if (start) begin
                        for (int a = 0; a < NUM_ACTIONS; a++) begin
                            q_accum[a] <= '0;
                            q_divided[a] <= '0;
                        end
                        timestep_counter <= '0;
                        batch_counter <= '0;
                        done <= 1'b0;
                        state <= PROCESSING;
                    end else begin
                        state <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
