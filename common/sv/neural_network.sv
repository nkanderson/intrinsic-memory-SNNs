// Neural Network Top-Level Module (parallel fc1/fc2)
// Implements complete SNN inference for CartPole.
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
// Key insight: HL1 receives same current every timestep, HL2 receives different
// current. fc1 is computed once at the start and its outputs are held in
// linear_layer's registered output vector across the entire timestep loop.
//
// Architecture (parallel-MAC linear_layer):
//   - fc1 (linear_layer, P=HL1_SIZE): 4 -> HL1_SIZE in NUM_INPUTS+2 cycles
//   - HL1 LIFs (HL1_SIZE x): all fire synchronously each timestep
//   - fc2 (linear_layer, P=HL2_SIZE): HL1_SIZE -> HL2_SIZE in HL1_SIZE+2 cycles
//   - HL2 LIFs (HL2_SIZE x): all fire synchronously after fc2_done
//   - neuron_membrane_buffer (HL2_SIZE x): store membrane per timestep
//   - q_accumulator: computes Q-values from membrane buffers, argmax selected_action
//
// Compared to the legacy serial version (neural_network_serial.sv):
//   - HL1 LIFs consume fc1_outputs[i] directly (no streaming capture register)
//   - HL2 LIFs consume fc2_outputs[i] directly and share a single hl2_enable
//     (no per-neuron stagger keyed on fc2's output_idx)
//   - Per-timestep loop shrinks from ~(1 + HL2_SIZE*(HL1_SIZE+1)) cycles to
//     ~(HL1_SIZE + 4) cycles, dominated by fc2's serial-input MAC

module neural_network #(
    // Network architecture
    parameter NUM_INPUTS = 4,
    parameter HL1_SIZE = 64,
    parameter HL2_SIZE = 16,
    parameter NUM_ACTIONS = 2,
    parameter NUM_TIMESTEPS = 30,
    // Fixed-point parameters
    parameter DATA_WIDTH = 16,
    // FC2_OUTPUT_WIDTH is a transport/interface width (not fc2 internal accumulator width).
    // linear_layer computes in a wider ACCUM_WIDTH, then saturates to OUTPUT_WIDTH.
    // This width is carried on fc2_outputs[] -> HL2 LIF current input.
    // Choose the smallest width that avoids FC2 saturation for the target observation set.
    parameter FC2_OUTPUT_WIDTH = DATA_WIDTH,
    parameter MEMBRANE_WIDTH = 24,
    parameter FRAC_BITS = 13,
    // LIF parameters
    parameter THRESHOLD = 8192,
    parameter BETA = 115,
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
    // Action selected from full-precision Q-values inside q_accumulator
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
    typedef enum logic [2:0] {
        TS_HL1_STEP,    // Pulse all HL1 LIFs (synchronous fire)
        TS_FC2_START,   // Pulse fc2 start
        TS_FC2_WAIT,    // Wait for fc2_done
        TS_HL2_STEP,    // Pulse all HL2 LIFs (synchronous fire)
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
        // PARALLELISM defaults to NUM_OUTPUTS = HL1_SIZE (full per-output parallel)
    ) fc1 (
        .clk(clk),
        .reset(reset),
        .start(fc1_start),
        .inputs(observations_registered),
        .outputs(fc1_outputs),
        .sat_pos(/* unconnected — DCE removes the sticky flag register */),
        .sat_neg(/* unconnected — DCE removes the sticky flag register */),
        .done(fc1_done)
    );

    // =========================================================================
    // HL1 LIF neurons (HL1_SIZE instances, all fire synchronously)
    // =========================================================================
    logic hl1_clear;
    logic hl1_enable;
    logic hl1_spikes [0:HL1_SIZE-1];
    logic signed [MEMBRANE_WIDTH-1:0] hl1_membranes [0:HL1_SIZE-1];  // Not used externally
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
            lif #(
                .THRESHOLD(THRESHOLD),
                .BETA(BETA),
                .DATA_WIDTH(DATA_WIDTH),
                .MEMBRANE_WIDTH(MEMBRANE_WIDTH)
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
    // Spikes are 0 or 1; represent as fixed-point (1.0 = THRESHOLD = 8192 in QS2.13).
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
        .sat_pos(/* unconnected — DCE removes the sticky flag register */),
        .sat_neg(/* unconnected — DCE removes the sticky flag register */),
        .done(fc2_done)
    );

    // =========================================================================
    // HL2 LIF neurons (HL2_SIZE instances) + per-neuron membrane buffers
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
    // membrane_out value (lif registers membrane on the cycle output_valid
    // pulses, so we need a 1-cycle delay before sampling for buffer write).
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

    generate
        for (genvar i = 0; i < HL2_SIZE; i++) begin : gen_hl2
            // HL2 LIF neuron
            lif #(
                .THRESHOLD(THRESHOLD),
                .BETA(BETA),
                .DATA_WIDTH(FC2_OUTPUT_WIDTH),
                .MEMBRANE_WIDTH(MEMBRANE_WIDTH)
            ) hl2_lif (
                .clk(clk),
                .reset(reset),
                .clear(hl2_clear),
                .enable(hl2_enable),
                // fc2_outputs[i] is registered inside linear_layer; stable
                // from fc2_done through the rest of this timestep until the
                // next fc2 run.
                .current(fc2_outputs[i]),
                .spike_out(hl2_spikes[i]),
                .membrane_out(hl2_membranes[i]),
                .busy(hl2_busy[i]),
                .output_valid(hl2_output_valid[i])
            );

            // Per-neuron membrane buffer writes are aligned to delayed output_valid.
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

            // Control signals
            fc1_start <= 1'b0;
            fc2_start <= 1'b0;
            q_start <= 1'b0;
            hl1_clear <= 1'b0;
            hl1_enable <= 1'b0;
            hl2_clear <= 1'b0;
            hl2_enable <= 1'b0;
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
                    // Wait for fc1 to complete. fc1_outputs[] are registered
                    // inside linear_layer and remain valid until the next fc1
                    // start (which won't happen this inference).
                    if (fc1_done) begin
                        current_timestep <= '0;
                        ts_state <= TS_HL1_STEP;
                        state <= RUN_TIMESTEPS;
                    end
                end

                RUN_TIMESTEPS: begin
                    unique case (ts_state)
                        TS_HL1_STEP: begin
                            // Pulse all HL1 LIFs synchronously
                            hl1_enable <= 1'b1;
                            ts_state <= TS_FC2_START;
                        end

                        TS_FC2_START: begin
                            // HL1 spikes feed directly to fc2 via fc2_inputs[]
                            // (combinational from hl1_spike_vector). fc2 latches
                            // them when start pulses; we pulse start here.
                            fc2_start <= 1'b1;
                            membrane_write_timestep <= current_timestep;
                            ts_state <= TS_FC2_WAIT;
                        end

                        TS_FC2_WAIT: begin
                            // Wait for fc2 to register all HL2_SIZE outputs.
                            if (fc2_done) begin
                                ts_state <= TS_HL2_STEP;
                            end
                        end

                        TS_HL2_STEP: begin
                            // Pulse all HL2 LIFs synchronously. They sample
                            // fc2_outputs[i] (still valid, registered).
                            hl2_enable <= 1'b1;
                            ts_state <= TS_NEXT;
                        end

                        TS_NEXT: begin
                            // Membrane writes for this timestep land via
                            // hl2_output_valid_delayed during the next cycle.
                            // q_accumulator start remains deferred until all
                            // timesteps complete (matches serial behavior;
                            // pipelining across timesteps is future work).
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
                    // Hold done until next start.
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
