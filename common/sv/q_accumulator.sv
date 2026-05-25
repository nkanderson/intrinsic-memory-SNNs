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
// Final action: argmax of the accumulated Q-values.
//
// We deliberately do NOT divide by NUM_TIMESTEPS before the argmax: dividing
// each Q-value by the same positive constant preserves their order, and an
// earlier version's `q_divided[a] <= q_accum[a] / NUM_TIMESTEPS` synthesized
// to a ~37-deep CARRY4 chain (NUM_TIMESTEPS=10 isn't a power of two), which
// blew through 100 MHz timing. The output `selected_action` is identical
// whether the divide is performed or skipped.
//
// Batched processing: BATCH_SIZE neurons processed per cycle
// Total multipliers: BATCH_SIZE × NUM_ACTIONS
//
// Fixed-point format: QS2.13 (16-bit signed, 2 integer bits, 13 fractional bits)
//
// Truncation ordering: per-timestep products are summed at the full
// MEMBRANE_WIDTH+DATA_WIDTH product width into a per-timestep accumulator,
// then right-shifted by FRAC_BITS once at the end of the timestep before
// bias-add. This matches qs213_reference.q_accumulate exactly. An earlier
// version shifted every product individually inside batch_sum, which
// disagreed with the reference by up to ~NUM_NEURONS LSBs per timestep and
// could flip the chosen action when Q-margins were small.
//
// Timing:
//   - Assert 'start' when all neuron membrane buffers are full
//   - Latency: NUM_TIMESTEPS × (NUM_NEURONS / BATCH_SIZE + 4) + 2 cycles
//     Per timestep:
//       +1 READ_WAIT  (sync membrane_buffer read settles)
//       +1 mem_sel    (DSP A/B input register fills for first batch)
//       NUM_BATCHES   (issue mem_sel for each batch back-to-back)
//       +1 products_r (final batch's DSP P-register fills)
//       +1 NEXT_TIMESTEP transition
//   - Asserts 'done' when selected_action is valid
//
// Pipeline (per batch, in cycles after issue):
//   t=0: stage0 — register membrane_in[neuron_idx] → mem_sel_r,
//                 register weights_flat[...] → w_sel_r
//   t=1: stage1 — products_r <= mem_sel_r * w_sel_r  (DSP A_REG/B_REG → P_REG)
//   t=2: stage2 — batch_sum (comb) → timestep_accum or q_accum (registered)

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
    localparam NEURON_IDX_WIDTH = $clog2(NUM_NEURONS) > 0 ? $clog2(NUM_NEURONS) : 1;

    // Counters
    logic [$clog2(NUM_TIMESTEPS)-1:0] timestep_counter;
    logic [BATCH_IDX_WIDTH-1:0] batch_counter;
    logic [NEURON_IDX_WIDTH-1:0] base_idx;

    // Weights and biases
    // weights_flat[a * NUM_NEURONS + n] = weight for action a, neuron n
    logic signed [DATA_WIDTH-1:0] weights_flat [0:NUM_ACTIONS*NUM_NEURONS-1];
    logic signed [DATA_WIDTH-1:0] biases [0:NUM_ACTIONS-1];

    initial begin
        $readmemh(WEIGHTS_FILE, weights_flat);
        $readmemh(BIAS_FILE, biases);
    end

    // Per-timestep accumulator width: holds the un-shifted sum of full-width
    // (MEMBRANE_WIDTH+DATA_WIDTH) products across all neurons in one timestep,
    // before the FRAC_BITS shift. +2 bits of safety headroom.
    localparam TIMESTEP_ACCUM_WIDTH = MEMBRANE_WIDTH + DATA_WIDTH + $clog2(NUM_NEURONS) + 2;

    // Cross-timestep accumulator width: sum of shifted+biased per-timestep
    // results across NUM_TIMESTEPS. Sized conservatively at the legacy width.
    localparam ACCUM_WIDTH = MEMBRANE_WIDTH + DATA_WIDTH + $clog2(NUM_NEURONS) + $clog2(NUM_TIMESTEPS) + 2;

    // Q-value accumulators (one per action) and per-timestep product-sum
    // accumulators (cleared at each timestep boundary).
    logic signed [ACCUM_WIDTH-1:0]          q_accum        [0:NUM_ACTIONS-1];
    logic signed [TIMESTEP_ACCUM_WIDTH-1:0] timestep_accum [0:NUM_ACTIONS-1];

    // State machine
    typedef enum logic [2:0] {
        IDLE,
        READ_WAIT,      // One-cycle wait for sync membrane_buffer to register read_timestep
        PROCESSING,     // Computing weighted sums for current batch
        NEXT_TIMESTEP,  // Move to next timestep
        DONE_STATE      // Compare q_accum directly and emit selected_action
    } state_t;

    state_t state;

    // --- DSP input registers (stage 0) ---
    // Synthesis absorbs these into the DSP48E1 A_REG / B_REG, so the multiplier
    // sees registered inputs and the mux-mux-mult combinational chain is split
    // into mux → A/B_REG → multiply.
    logic signed [MEMBRANE_WIDTH-1:0] mem_sel_r [0:NUM_ACTIONS-1][0:BATCH_SIZE-1];
    logic signed [DATA_WIDTH-1:0]     w_sel_r   [0:NUM_ACTIONS-1][0:BATCH_SIZE-1];

    // --- DSP output register (stage 1) ---
    // products_r is the DSP P-register.
    (* use_dsp = "yes" *)
    logic signed [MEMBRANE_WIDTH+DATA_WIDTH-1:0] products_r [0:NUM_ACTIONS-1][0:BATCH_SIZE-1];

    // Batch sums per action: sum of full-width products within the current
    // batch (NOT shifted; shift happens once at end-of-timestep).
    logic signed [TIMESTEP_ACCUM_WIDTH-1:0] batch_sum [0:NUM_ACTIONS-1];

    // Combinational helpers for the end-of-timestep shift+bias path.
    logic signed [TIMESTEP_ACCUM_WIDTH-1:0] full_timestep_sum [0:NUM_ACTIONS-1];
    logic signed [TIMESTEP_ACCUM_WIDTH-1:0] timestep_shifted  [0:NUM_ACTIONS-1];

    // --- Pipeline valid/last signals ---
    //   sel_valid:  this cycle, mem_sel_r/w_sel_r hold a fresh batch (stage 0 → stage 1)
    //   sel_last:   the batch landing in mem_sel_r this cycle is the last of the timestep
    //   prod_valid: this cycle, products_r holds a fresh batch (stage 1 → stage 2)
    //   prod_last:  the batch landing in products_r this cycle is the last of the timestep
    //   sel_active: true while stage 0 should issue new batches (clears after last batch issued)
    logic sel_valid;
    logic sel_last;
    logic prod_valid;
    logic prod_last;
    logic sel_active;

    // Output read_timestep to buffers
    assign read_timestep = timestep_counter;
    assign base_idx = batch_counter * BATCH_SIZE;

    // Combinational: sum full-width products into batch_sum (no per-product
    // shift), then compute the running per-timestep sum and its FRAC_BITS-
    // shifted value for the end-of-timestep path.
    always_comb begin
        for (int a = 0; a < NUM_ACTIONS; a++) begin
            batch_sum[a] = '0;
            for (int b = 0; b < BATCH_SIZE; b++) begin
                // Sign-extend the MEMBRANE+DATA-wide product to
                // TIMESTEP_ACCUM_WIDTH and accumulate without any shift.
                batch_sum[a] = batch_sum[a] +
                    $signed({{(TIMESTEP_ACCUM_WIDTH-(MEMBRANE_WIDTH+DATA_WIDTH)){
                        products_r[a][b][MEMBRANE_WIDTH+DATA_WIDTH-1]}},
                        products_r[a][b]});
            end
            full_timestep_sum[a] = timestep_accum[a] + batch_sum[a];
            timestep_shifted[a]  = full_timestep_sum[a] >>> FRAC_BITS;
        end
    end

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            timestep_counter <= '0;
            batch_counter <= '0;
            done <= 1'b0;
            selected_action <= '0;
            sel_valid <= 1'b0;
            sel_last <= 1'b0;
            prod_valid <= 1'b0;
            prod_last <= 1'b0;
            sel_active <= 1'b0;
            for (int a = 0; a < NUM_ACTIONS; a++) begin
                q_accum[a] <= '0;
                timestep_accum[a] <= '0;
                for (int b = 0; b < BATCH_SIZE; b++) begin
                    mem_sel_r[a][b] <= '0;
                    w_sel_r[a][b]   <= '0;
                    products_r[a][b] <= '0;
                end
            end
        end else begin
            // --- Stage 1 → Stage 2 pipeline propagation (always live) ---
            // products_r is the DSP P-register; whether it's loaded with a
            // fresh product this cycle depends on sel_valid (the prior cycle).
            if (sel_valid) begin
                for (int a = 0; a < NUM_ACTIONS; a++) begin
                    for (int b = 0; b < BATCH_SIZE; b++) begin
                        products_r[a][b] <= mem_sel_r[a][b] * w_sel_r[a][b];
                    end
                end
            end
            prod_valid <= sel_valid;
            prod_last  <= sel_last;

            // --- Stage 2: accumulate when prod_valid is high ---
            if (prod_valid) begin
                for (int a = 0; a < NUM_ACTIONS; a++) begin
                    if (prod_last) begin
                        q_accum[a] <= q_accum[a] +
                            $signed({{(ACCUM_WIDTH-TIMESTEP_ACCUM_WIDTH){
                                timestep_shifted[a][TIMESTEP_ACCUM_WIDTH-1]}},
                                timestep_shifted[a]}) +
                            $signed({{(ACCUM_WIDTH-DATA_WIDTH){
                                biases[a][DATA_WIDTH-1]}},
                                biases[a]});
                        timestep_accum[a] <= '0;
                    end else begin
                        timestep_accum[a] <= full_timestep_sum[a];
                    end
                end
            end

            // Default: stage 0 idles unless explicitly issued below.
            // sel_valid/sel_last are cleared each cycle and re-asserted only
            // when stage 0 fires.
            sel_valid <= 1'b0;
            sel_last  <= 1'b0;

            unique case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        // Initialize accumulators for a fresh inference.
                        for (int a = 0; a < NUM_ACTIONS; a++) begin
                            q_accum[a] <= '0;
                            timestep_accum[a] <= '0;
                        end
                        timestep_counter <= '0;
                        batch_counter <= '0;
                        sel_active <= 1'b1;
                        // Spend one cycle in READ_WAIT so the sync membrane
                        // buffer can register the new read_timestep before
                        // stage 0 latches membrane_in.
                        state <= READ_WAIT;
                    end else begin
                        state <= IDLE;
                    end
                end

                READ_WAIT: begin
                    // membrane_in is valid one cycle after read_timestep
                    // changes (sync buffer). Move on to issuing computes.
                    state <= PROCESSING;
                end

                PROCESSING: begin
                    // Stage 0: while sel_active, issue a new batch each cycle
                    // by registering the selected operands into mem_sel_r /
                    // w_sel_r (DSP A_REG/B_REG inputs). Once the last batch's
                    // operands are latched, clear sel_active and let stage 1
                    // and stage 2 drain.
                    if (sel_active) begin
                        // neuron_idx declared at block scope (Icarus does not
                        // support per-variable lifetime overrides like
                        // `automatic int neuron_idx = ...` inside the loop).
                        int neuron_idx;
                        for (int a = 0; a < NUM_ACTIONS; a++) begin
                            for (int b = 0; b < BATCH_SIZE; b++) begin
                                neuron_idx = base_idx + b;
                                mem_sel_r[a][b] <= membrane_in[neuron_idx];
                                w_sel_r[a][b]   <= weights_flat[a * NUM_NEURONS + neuron_idx];
                            end
                        end
                        sel_valid <= 1'b1;
                        sel_last  <= (batch_counter == BATCH_IDX_WIDTH'(NUM_BATCHES - 1));

                        if (batch_counter == BATCH_IDX_WIDTH'(NUM_BATCHES - 1)) begin
                            batch_counter <= '0;
                            sel_active <= 1'b0;  // last batch issued; drain begins
                        end else begin
                            batch_counter <= batch_counter + 1'b1;
                        end
                    end

                    // Advance state once the last batch has flushed through
                    // stage 2 (q_accum has been updated for prod_last).
                    if (prod_valid && prod_last) begin
                        state <= NEXT_TIMESTEP;
                    end
                end

                NEXT_TIMESTEP: begin
                    if (timestep_counter == $clog2(NUM_TIMESTEPS)'(NUM_TIMESTEPS - 1)) begin
                        // All timesteps done, go straight to argmax — the
                        // divide-by-NUM_TIMESTEPS state is gone (see header).
                        state <= DONE_STATE;
                    end else begin
                        timestep_counter <= timestep_counter + 1'b1;
                        batch_counter <= '0;
                        sel_active <= 1'b1;
                        state <= READ_WAIT;
                    end
                end

                DONE_STATE: begin
                    // Select action by comparing the un-divided Q accumulators
                    // directly. argmax is invariant under positive scaling, so
                    // dropping the divide preserves correctness.
                    // The comparison uses the full ACCUM_WIDTH bits, preserving
                    // distinctions that would be lost if we saturated to DATA_WIDTH.
                    selected_action <= (q_accum[0] >= q_accum[1]) ? 1'b0 : 1'b1;
                    done <= 1'b1;

                    // Return to processing on next start, else idle
                    if (start) begin
                        for (int a = 0; a < NUM_ACTIONS; a++) begin
                            q_accum[a] <= '0;
                            timestep_accum[a] <= '0;
                        end
                        timestep_counter <= '0;
                        batch_counter <= '0;
                        sel_active <= 1'b1;
                        done <= 1'b0;
                        state <= READ_WAIT;
                    end else begin
                        state <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
