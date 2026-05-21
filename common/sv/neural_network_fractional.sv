// Neural Network Top-Level Module (fractional LIF variant, parallel fc1/fc2)
// Implements complete SNN inference for CartPole with fractional_lif neurons
// (Grünwald-Letnikov fractional-order leak).
//
// This module instantiates and coordinates all sub-modules to perform a complete
// forward pass matching snnTorch's behavior:
//   for t in range(num_steps):
//       h1 = fc1(observations)    # same input every timestep
//       spk1 = lif1(h1)           # HL1 processes static h1
//       h2 = fc2(spk1)            # new spike vector each timestep
//       spk2, mem2 = lif2(h2)     # HL2 processes varying h2
//       q_t = fc_out(mem2)        # Q-values from membrane
//       out_accum += q_t
//   q_values = out_accum / num_steps
//
// Architecture (parallel-MAC linear_layer + multi-cycle fractional_lif):
//   - fc1 (linear_layer, P=HL1_SIZE): observations -> HL1 currents in NUM_INPUTS+2 cycles
//   - HL1 fractional_lif (HL1_SIZE x): multi-cycle FSM (ST_MAC over HISTORY_LENGTH-1
//     terms, then ST_PREP_NUM / ST_MUL_RECIP / ST_SHIFT_DIV / ST_FINALIZE), all
//     fire synchronously each timestep
//   - fc2 (linear_layer, P=HL2_SIZE): HL1 spikes -> HL2 currents in HL1_SIZE+2 cycles
//   - HL2 fractional_lif (HL2_SIZE x): same multi-cycle FSM, all fire synchronously
//   - neuron_membrane_buffer (HL2_SIZE x): per-timestep membrane storage
//   - q_accumulator: computes Q-values and argmax selected_action
//
// Compared to the legacy serial version (previously: streaming linear_layer +
// per-neuron HL2 stagger), this variant follows the same refactor pattern as
// neural_network_bitshift.sv: vector + done fc1/fc2 interface, shared HL2
// enable, explicit TS_HL2_WAIT to drain the multi-cycle fractional FSM
// before advancing to the next timestep.

module neural_network_fractional #(
    // Network architecture
    parameter NUM_INPUTS = 4,
    parameter HL1_SIZE = 64,
    parameter HL2_SIZE = 16,
    parameter NUM_ACTIONS = 2,
    parameter NUM_TIMESTEPS = 30,
    // Fixed-point parameters
    parameter DATA_WIDTH = 16,
    parameter FC2_OUTPUT_WIDTH = DATA_WIDTH,
    parameter MEMBRANE_WIDTH = 24,
    parameter FRAC_BITS = 13,
    // LIF parameters
    parameter THRESHOLD = 8192,
    parameter BETA = 115,
    // Fractional LIF parameters
    parameter HISTORY_LENGTH = 64,
    parameter COEFF_WIDTH = 16,
    parameter COEFF_FRAC_BITS = 15,
    parameter [15:0] INV_DENOM = 16'd58988,
    parameter GL_COEFF_FILE = "gl_coefficients.mem",
    // Weight files
    parameter FC1_WEIGHTS_FILE = "fc1_weights.mem",
    parameter FC1_BIAS_FILE = "fc1_bias.mem",
    parameter FC2_WEIGHTS_FILE = "fc2_weights.mem",
    parameter FC2_BIAS_FILE = "fc2_bias.mem",
    parameter FC_OUT_WEIGHTS_FILE = "fc_out_weights.mem",
    parameter FC_OUT_BIAS_FILE = "fc_out_bias.mem",
    // q_accumulator tuning
    parameter Q_BATCH_SIZE = 4
) (
    input wire clk,
    input wire reset,
    input wire start,
    input wire signed [DATA_WIDTH-1:0] observations [0:NUM_INPUTS-1],
    output logic [$clog2(NUM_ACTIONS)-1:0] selected_action,
    output logic done
);

    // =========================================================================
    // Derived parameters
    // =========================================================================
    localparam TIMESTEP_WIDTH = $clog2(NUM_TIMESTEPS);

    // =========================================================================
    // State machine
    // =========================================================================
    typedef enum logic [2:0] {
        IDLE,
        LOAD_HL1,       // fc1 computing HL1 currents
        RUN_TIMESTEPS,  // Main timestep loop
        FINISH_Q,       // Wait for q_accumulator to finish
        DONE_STATE
    } state_t;

    state_t state;

    // Timestep counter
    logic [TIMESTEP_WIDTH-1:0] current_timestep;

    // Sub-state within RUN_TIMESTEPS
    // 6 states: extra TS_HL2_WAIT to drain multi-cycle fractional_lif before
    // the next TS_HL2_STEP (or q_accumulator) reads stale membranes.
    typedef enum logic [2:0] {
        TS_HL1_STEP,    // Pulse all HL1 LIFs (synchronous fire)
        TS_FC2_START,   // Pulse fc2 start
        TS_FC2_WAIT,    // Wait for fc2_done
        TS_HL2_STEP,    // Pulse all HL2 LIFs (synchronous fire)
        TS_HL2_WAIT,    // Wait for !hl2_any_busy (multi-cycle drain)
        TS_NEXT         // Advance timestep or finish
    } timestep_state_t;

    timestep_state_t ts_state;

    // =========================================================================
    // Saved observations
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] observations_registered [0:NUM_INPUTS-1];

    // =========================================================================
    // fc1 (linear_layer): observations -> HL1 currents (registered vector)
    // =========================================================================
    logic fc1_start;
    logic signed [DATA_WIDTH-1:0] fc1_outputs [0:HL1_SIZE-1];
    logic fc1_done;

    linear_layer #(
        .NUM_INPUTS(NUM_INPUTS),
        .NUM_OUTPUTS(HL1_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .WEIGHTS_FILE(FC1_WEIGHTS_FILE),
        .BIAS_FILE(FC1_BIAS_FILE)
        // PARALLELISM defaults to NUM_OUTPUTS = HL1_SIZE
    ) fc1 (
        .clk(clk),
        .reset(reset),
        .start(fc1_start),
        .inputs(observations_registered),
        .outputs(fc1_outputs),
        .sat_pos(/* unconnected - DCE removes the sticky flag register */),
        .sat_neg(/* unconnected - DCE removes the sticky flag register */),
        .done(fc1_done)
    );

    // =========================================================================
    // HL1 fractional_lif neurons (HL1_SIZE instances, all fire synchronously)
    // =========================================================================
    logic hl1_clear;
    logic hl1_enable;
    logic hl1_spikes [0:HL1_SIZE-1];
    logic signed [MEMBRANE_WIDTH-1:0] hl1_membranes [0:HL1_SIZE-1];
    logic hl1_busy [0:HL1_SIZE-1];
    logic hl1_output_valid [0:HL1_SIZE-1];

    // Pack spike outputs into vector (used for fc2 input conversion)
    logic [HL1_SIZE-1:0] hl1_spike_vector;
    always_comb begin
        for (int i = 0; i < HL1_SIZE; i++) begin
            hl1_spike_vector[i] = hl1_spikes[i];
        end
    end

    generate
        for (genvar i = 0; i < HL1_SIZE; i++) begin : gen_hl1_lif
            fractional_lif #(
                .THRESHOLD(THRESHOLD),
                .DATA_WIDTH(DATA_WIDTH),
                .MEMBRANE_WIDTH(MEMBRANE_WIDTH),
                .HISTORY_LENGTH(HISTORY_LENGTH),
                .COEFF_WIDTH(COEFF_WIDTH),
                .COEFF_FRAC_BITS(COEFF_FRAC_BITS),
                .INV_DENOM(INV_DENOM),
                .GL_COEFF_FILE(GL_COEFF_FILE)
            ) hl1_lif (
                .clk(clk),
                .reset(reset),
                .clear(hl1_clear),
                .enable(hl1_enable),
                // fc1_outputs[i] is registered inside linear_layer and stable
                // across the entire NUM_TIMESTEPS loop (fc1 runs once).
                .current(fc1_outputs[i]),
                .spike_out(hl1_spikes[i]),
                .membrane_out(hl1_membranes[i]),
                .busy(hl1_busy[i]),
                .output_valid(hl1_output_valid[i])
            );
        end
    endgenerate

    // =========================================================================
    // fc2 (linear_layer): HL1 spikes -> HL2 currents (registered vector)
    // =========================================================================
    logic fc2_start;
    logic signed [FC2_OUTPUT_WIDTH-1:0] fc2_outputs [0:HL2_SIZE-1];
    logic fc2_done;

    // Convert spike vector to signed input array for fc2.
    // Spikes are 0 or 1; represent as fixed-point (1.0 = THRESHOLD).
    logic signed [DATA_WIDTH-1:0] fc2_inputs [0:HL1_SIZE-1];
    always_comb begin
        for (int i = 0; i < HL1_SIZE; i++) begin
            fc2_inputs[i] = hl1_spike_vector[i] ? DATA_WIDTH'(THRESHOLD) : '0;
        end
    end

    linear_layer #(
        .NUM_INPUTS(HL1_SIZE),
        .NUM_OUTPUTS(HL2_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .OUTPUT_WIDTH(FC2_OUTPUT_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .WEIGHTS_FILE(FC2_WEIGHTS_FILE),
        .BIAS_FILE(FC2_BIAS_FILE)
        // PARALLELISM defaults to NUM_OUTPUTS = HL2_SIZE
    ) fc2 (
        .clk(clk),
        .reset(reset),
        .start(fc2_start),
        .inputs(fc2_inputs),
        .outputs(fc2_outputs),
        .sat_pos(/* unconnected - DCE removes the sticky flag register */),
        .sat_neg(/* unconnected - DCE removes the sticky flag register */),
        .done(fc2_done)
    );

    // =========================================================================
    // HL2 fractional_lif neurons (HL2_SIZE instances) + per-neuron membrane buffers
    // All HL2 LIFs share a single enable, fired synchronously after fc2_done.
    // =========================================================================
    logic hl2_clear;
    logic hl2_enable;  // Single shared enable (was per-neuron stagger)
    logic hl2_spikes [0:HL2_SIZE-1];
    logic signed [MEMBRANE_WIDTH-1:0] hl2_membranes [0:HL2_SIZE-1];
    logic hl2_busy [0:HL2_SIZE-1];
    logic hl2_output_valid [0:HL2_SIZE-1];

    // membrane_buffer interface signals
    logic [TIMESTEP_WIDTH-1:0] membrane_write_timestep;
    logic [TIMESTEP_WIDTH-1:0] q_read_timestep;
    logic signed [MEMBRANE_WIDTH-1:0] membrane_to_q [0:HL2_SIZE-1];

    // Delay output_valid by 1 cycle so membrane_buf captures the updated
    // membrane_out value (LIF registers membrane on the cycle output_valid
    // pulses, so 1-cycle delay aligns the buffer write).
    logic hl2_output_valid_delayed [0:HL2_SIZE-1];
    logic [TIMESTEP_WIDTH-1:0] membrane_write_timestep_delayed;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            for (int i = 0; i < HL2_SIZE; i++) begin
                hl2_output_valid_delayed[i] <= 1'b0;
            end
            membrane_write_timestep_delayed <= '0;
        end else begin
            for (int i = 0; i < HL2_SIZE; i++) begin
                hl2_output_valid_delayed[i] <= hl2_output_valid[i];
            end
            membrane_write_timestep_delayed <= membrane_write_timestep;
        end
    end

    // OR-reduce HL2 busy signals so the FSM can wait for all HL2 LIFs to
    // return to ST_IDLE before advancing.
    logic hl2_any_busy;
    always_comb begin
        hl2_any_busy = 1'b0;
        for (int i = 0; i < HL2_SIZE; i++) begin
            hl2_any_busy = hl2_any_busy || hl2_busy[i];
        end
    end

    // `hl2_started` latches on the first cycle of TS_HL2_WAIT where any HL2
    // LIF has transitioned out of ST_IDLE (busy goes high). Without this
    // gate, TS_HL2_WAIT samples hl2_any_busy on the same edge that the LIFs
    // sample enable — Verilog non-blocking semantics give the FSM the OLD
    // busy value (still 0 from ST_IDLE), so it would falsely advance to
    // TS_NEXT before HL2 even starts. The bug manifests as
    // membrane_write_timestep_delayed advancing past timestep T while the
    // HL2 LIFs for T are still computing, so their membranes get written
    // into timestep T+1's slot.
    logic hl2_started;

    generate
        for (genvar i = 0; i < HL2_SIZE; i++) begin : gen_hl2
            fractional_lif #(
                .THRESHOLD(THRESHOLD),
                .DATA_WIDTH(FC2_OUTPUT_WIDTH),
                .MEMBRANE_WIDTH(MEMBRANE_WIDTH),
                .HISTORY_LENGTH(HISTORY_LENGTH),
                .COEFF_WIDTH(COEFF_WIDTH),
                .COEFF_FRAC_BITS(COEFF_FRAC_BITS),
                .INV_DENOM(INV_DENOM),
                .GL_COEFF_FILE(GL_COEFF_FILE)
            ) hl2_lif (
                .clk(clk),
                .reset(reset),
                .clear(hl2_clear),
                .enable(hl2_enable),
                // fc2_outputs[i] is registered inside linear_layer and stable
                // from fc2_done through the next fc2 run.
                .current(fc2_outputs[i]),
                .spike_out(hl2_spikes[i]),
                .membrane_out(hl2_membranes[i]),
                .busy(hl2_busy[i]),
                .output_valid(hl2_output_valid[i])
            );

            neuron_membrane_buffer #(
                .NUM_TIMESTEPS(NUM_TIMESTEPS),
                .MEMBRANE_WIDTH(MEMBRANE_WIDTH)
            ) membrane_buf (
                .clk(clk),
                .reset(reset),
                .clear(hl2_clear),
                .write_en(hl2_output_valid_delayed[i]),
                .write_timestep(membrane_write_timestep_delayed),
                .membrane_in(hl2_membranes[i]),
                .read_timestep(q_read_timestep),
                .membrane_out(membrane_to_q[i]),
                .full()  // Not used - we track timesteps explicitly
            );
        end
    endgenerate

    // =========================================================================
    // q_accumulator
    // =========================================================================
    logic q_start;
    logic [$clog2(NUM_ACTIONS)-1:0] q_selected_action;
    logic q_done;

    q_accumulator #(
        .NUM_NEURONS(HL2_SIZE),
        .NUM_TIMESTEPS(NUM_TIMESTEPS),
        .NUM_ACTIONS(NUM_ACTIONS),
        .BATCH_SIZE(Q_BATCH_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .MEMBRANE_WIDTH(MEMBRANE_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .WEIGHTS_FILE(FC_OUT_WEIGHTS_FILE),
        .BIAS_FILE(FC_OUT_BIAS_FILE)
    ) q_accum (
        .clk(clk),
        .reset(reset),
        .start(q_start),
        .read_timestep(q_read_timestep),
        .membrane_in(membrane_to_q),
        .selected_action(q_selected_action),
        .done(q_done)
    );

    // =========================================================================
    // Output assignments
    // =========================================================================
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            selected_action <= '0;
        end else if (q_done) begin
            selected_action <= q_selected_action;
        end
    end

    // =========================================================================
    // Main state machine
    // =========================================================================
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            ts_state <= TS_HL1_STEP;
            current_timestep <= '0;
            done <= 1'b0;

            fc1_start <= 1'b0;
            fc2_start <= 1'b0;
            q_start <= 1'b0;
            hl1_clear <= 1'b0;
            hl1_enable <= 1'b0;
            hl2_clear <= 1'b0;
            hl2_enable <= 1'b0;
            hl2_started <= 1'b0;
            membrane_write_timestep <= '0;

            for (int i = 0; i < NUM_INPUTS; i++) begin
                observations_registered[i] <= '0;
            end

        end else begin
            // Default: deassert single-cycle signals
            fc1_start <= 1'b0;
            fc2_start <= 1'b0;
            q_start <= 1'b0;
            hl1_clear <= 1'b0;
            hl1_enable <= 1'b0;
            hl2_clear <= 1'b0;
            hl2_enable <= 1'b0;

            unique case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        for (int i = 0; i < NUM_INPUTS; i++) begin
                            observations_registered[i] <= observations[i];
                        end

                        hl1_clear <= 1'b1;
                        hl2_clear <= 1'b1;
                        fc1_start <= 1'b1;
                        state <= LOAD_HL1;
                    end
                end

                LOAD_HL1: begin
                    if (fc1_done) begin
                        current_timestep <= '0;
                        ts_state <= TS_HL1_STEP;
                        state <= RUN_TIMESTEPS;
                    end
                end

                RUN_TIMESTEPS: begin
                    unique case (ts_state)
                        TS_HL1_STEP: begin
                            // Pulse all HL1 fractional_lifs synchronously.
                            // They run their multi-cycle FSM in parallel with
                            // fc2 (started below); HL1 finishes well before
                            // fc2 (HL1 ~= ST_MAC over HISTORY_LENGTH-1 terms,
                            // fc2 ~= HL1_SIZE+2 cycles which is typically larger).
                            hl1_enable <= 1'b1;
                            ts_state <= TS_FC2_START;
                        end

                        TS_FC2_START: begin
                            fc2_start <= 1'b1;
                            membrane_write_timestep <= current_timestep;
                            ts_state <= TS_FC2_WAIT;
                        end

                        TS_FC2_WAIT: begin
                            // fc2_done is a 1-cycle pulse coincident with the
                            // EMIT cycle. fc2_outputs[] register on that cycle.
                            if (fc2_done) begin
                                ts_state <= TS_HL2_STEP;
                            end
                        end

                        TS_HL2_STEP: begin
                            // Pulse all HL2 fractional_lifs synchronously.
                            // They sample fc2_outputs[i] (registered, stable).
                            hl2_enable <= 1'b1;
                            hl2_started <= 1'b0;  // reset for this timestep
                            ts_state <= TS_HL2_WAIT;
                        end

                        TS_HL2_WAIT: begin
                            // Two-phase wait: first observe hl2_any_busy go
                            // high (LIFs left ST_IDLE for this timestep),
                            // then wait for it to fall (LIFs finished and
                            // returned to ST_IDLE). Without the started
                            // latch, the FSM would race the LIF enable-sample
                            // edge and advance immediately on the first
                            // cycle of this state.
                            if (hl2_any_busy) begin
                                hl2_started <= 1'b1;
                            end
                            if (hl2_started && !hl2_any_busy) begin
                                ts_state <= TS_NEXT;
                            end
                        end

                        TS_NEXT: begin
                            if (current_timestep == NUM_TIMESTEPS - 1) begin
                                q_start <= 1'b1;
                                state <= FINISH_Q;
                            end else begin
                                current_timestep <= current_timestep + 1'b1;
                                ts_state <= TS_HL1_STEP;
                            end
                        end

                        default: ts_state <= TS_HL1_STEP;
                    endcase
                end

                FINISH_Q: begin
                    if (q_done) begin
                        done <= 1'b1;
                        state <= DONE_STATE;
                    end
                end

                DONE_STATE: begin
                    if (start) begin
                        done <= 1'b0;

                        for (int i = 0; i < NUM_INPUTS; i++) begin
                            observations_registered[i] <= observations[i];
                        end

                        hl1_clear <= 1'b1;
                        hl2_clear <= 1'b1;
                        fc1_start <= 1'b1;
                        current_timestep <= '0;
                        state <= LOAD_HL1;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
