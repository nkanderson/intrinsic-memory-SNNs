// Neural Network Top-Level Module
// Implements complete SNN inference for CartPole
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
// Key insight: HL1 receives same current every timestep, HL2 receives different current.
//
// Architecture:
//   - fc1 (linear_layer): 4 → 64, computes HL1 currents once
//   - HL1 LIFs (64×): Process same current each timestep, output spikes
//   - fc2 (linear_layer): 64 → 16, computes HL2 currents from spikes directly connected from HL1
//   - HL2 LIFs (16×): Process varying currents, output membrane values
//   - neuron_membrane_buffer (16×): Store membrane values per timestep
//   - q_accumulator: Computes final Q-values from membrane buffers
//
// Pipelined execution: q_accumulator processes timestep T while HL2 works on T+1

module neural_network_fractional #(
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
    // This width is carried on fc2_output_current -> hl2_currents -> fractional_lif.DATA_WIDTH.
    // Choose the smallest width that avoids FC2 saturation for the target observation set.
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
    parameter [15:0] C_SCALED = 16'd256,
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
    // Action selected from full-precision Q-values inside q_accumulator
    // Q-values are not output because they routinely exceed QS2.13 range and saturate,
    // losing the distinction between actions. selected_action is the authoritative result.
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
    typedef enum logic [1:0] {
        TS_HL1_STEP,    // Process HL1 LIFs
        TS_FC2_START,   // Start fc2
        TS_FC2_HL2,     // fc2 + HL2 processing
        TS_NEXT         // Move to next timestep
    } timestep_state_t;

    timestep_state_t ts_state;

    // =========================================================================
    // Saved observations
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] observations_registered [0:NUM_INPUTS-1];

    // =========================================================================
    // fc1 (linear_layer): Input → HL1 currents
    // =========================================================================
    logic fc1_start;
    logic signed [DATA_WIDTH-1:0] fc1_output_current;
    logic [$clog2(HL1_SIZE)-1:0] fc1_output_idx;
    logic fc1_output_valid;
    logic fc1_done;

    linear_layer #(
        .NUM_INPUTS(NUM_INPUTS),
        .NUM_OUTPUTS(HL1_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .WEIGHTS_FILE(FC1_WEIGHTS_FILE),
        .BIAS_FILE(FC1_BIAS_FILE)
    ) fc1 (
        .clk(clk),
        .reset(reset),
        .start(fc1_start),
        .inputs(observations_registered),
        .output_current(fc1_output_current),
        .output_idx(fc1_output_idx),
        .output_valid(fc1_output_valid),
        .done(fc1_done)
    );

    // =========================================================================
    // HL1 current registers (saved for reuse each timestep)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] hl1_currents [0:HL1_SIZE-1];

    // Save fc1 outputs in registers as they stream out
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            for (int i = 0; i < HL1_SIZE; i++) begin
                hl1_currents[i] <= '0;
            end
        end else if (fc1_output_valid) begin
            hl1_currents[fc1_output_idx] <= fc1_output_current;
        end
    end

    // =========================================================================
    // HL1 LIF neurons (64 instances)
    // =========================================================================
    logic hl1_clear;
    logic hl1_enable;
    logic hl1_spikes [0:HL1_SIZE-1];
    logic signed [MEMBRANE_WIDTH-1:0] hl1_membranes [0:HL1_SIZE-1];  // Not used but kept for completeness

    // Pack spike outputs into vector for fc2
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
                .C_SCALED(C_SCALED),
                .INV_DENOM(INV_DENOM),
                .GL_COEFF_FILE(GL_COEFF_FILE)
            ) hl1_lif (
                .clk(clk),
                .reset(reset),
                .clear(hl1_clear),
                .enable(hl1_enable),
                .current(hl1_currents[i]),
                .spike_out(hl1_spikes[i]),
                .membrane_out(hl1_membranes[i])
            );
        end
    endgenerate

    // =========================================================================
    // fc2 (linear_layer): HL1 spikes → HL2 currents
    // =========================================================================
    logic fc2_start;
    logic signed [FC2_OUTPUT_WIDTH-1:0] fc2_output_current;
    logic [$clog2(HL2_SIZE)-1:0] fc2_output_idx;
    logic fc2_output_valid;
    logic fc2_done;

    // Convert spike vector to signed input array for fc2 direct from HL1 LIF outputs
    logic signed [DATA_WIDTH-1:0] fc2_inputs [0:HL1_SIZE-1];
    always_comb begin
        for (int i = 0; i < HL1_SIZE; i++) begin
            // Spikes are 0 or 1, represent as fixed-point (1.0 = THRESHOLD = 8192)
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
    ) fc2 (
        .clk(clk),
        .reset(reset),
        .start(fc2_start),
        .inputs(fc2_inputs),
        .output_current(fc2_output_current),
        .output_idx(fc2_output_idx),
        .output_valid(fc2_output_valid),
        .done(fc2_done)
    );

    // =========================================================================
    // HL2 LIF neurons (16 instances) + neuron_membrane_buffers
    // =========================================================================
    logic hl2_clear;
    logic hl2_enable [0:HL2_SIZE-1];  // Individual enable per neuron
    logic signed [FC2_OUTPUT_WIDTH-1:0] hl2_currents [0:HL2_SIZE-1];
    logic hl2_spikes [0:HL2_SIZE-1];  // Not used but output for completeness
    logic signed [MEMBRANE_WIDTH-1:0] hl2_membranes [0:HL2_SIZE-1];

    // Save fc2 outputs as they stream out
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            for (int i = 0; i < HL2_SIZE; i++) begin
                hl2_currents[i] <= '0;
            end
        end else if (fc2_output_valid) begin
            hl2_currents[fc2_output_idx] <= fc2_output_current;
        end
    end

    // Track which HL2 neuron should process next (based on fc2 streaming output)
    logic [$clog2(HL2_SIZE)-1:0] hl2_process_idx;
    logic hl2_process_valid;

    // Delay fc2 valid by one cycle to give LIF time to use registered current
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            hl2_process_idx <= '0;
            hl2_process_valid <= 1'b0;
        end else begin
            hl2_process_idx <= fc2_output_idx;
            hl2_process_valid <= fc2_output_valid;
        end
    end

    // membrane_buffer interface signals
    logic [TIMESTEP_WIDTH-1:0] membrane_write_timestep;
    logic [TIMESTEP_WIDTH-1:0] q_read_timestep;
    logic signed [MEMBRANE_WIDTH-1:0] membrane_to_q [0:HL2_SIZE-1];

    // Delayed write enable for membrane buffer (captures new membrane after LIF computes)
    logic hl2_enable_delayed [0:HL2_SIZE-1];
    logic [TIMESTEP_WIDTH-1:0] membrane_write_timestep_delayed;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            for (int i = 0; i < HL2_SIZE; i++) begin
                hl2_enable_delayed[i] <= 1'b0;
            end
            membrane_write_timestep_delayed <= '0;
        end else begin
            for (int i = 0; i < HL2_SIZE; i++) begin
                hl2_enable_delayed[i] <= hl2_enable[i];
            end
            membrane_write_timestep_delayed <= membrane_write_timestep;
        end
    end

    generate
        for (genvar i = 0; i < HL2_SIZE; i++) begin : gen_hl2
            // HL2 LIF neuron
            fractional_lif #(
                .THRESHOLD(THRESHOLD),
                .DATA_WIDTH(FC2_OUTPUT_WIDTH),
                .MEMBRANE_WIDTH(MEMBRANE_WIDTH),
                .HISTORY_LENGTH(HISTORY_LENGTH),
                .COEFF_WIDTH(COEFF_WIDTH),
                .COEFF_FRAC_BITS(COEFF_FRAC_BITS),
                .C_SCALED(C_SCALED),
                .INV_DENOM(INV_DENOM),
                .GL_COEFF_FILE(GL_COEFF_FILE)
            ) hl2_lif (
                .clk(clk),
                .reset(reset),
                .clear(hl2_clear),
                .enable(hl2_enable[i]),
                .current(hl2_currents[i]),
                .spike_out(hl2_spikes[i]),
                .membrane_out(hl2_membranes[i])
            );

            // Per-neuron membrane buffer with writes delayed by 1 cycle to capture new membrane value after LIF processing
            neuron_membrane_buffer #(
                .NUM_TIMESTEPS(NUM_TIMESTEPS),
                .MEMBRANE_WIDTH(MEMBRANE_WIDTH)
            ) membrane_buf (
                .clk(clk),
                .reset(reset),
                .clear(hl2_clear),
                .write_en(hl2_enable_delayed[i]),
                .write_timestep(membrane_write_timestep_delayed),
                .membrane_in(hl2_membranes[i]),
                .read_timestep(q_read_timestep),
                .membrane_out(membrane_to_q[i]),
                .full()  // Not used - we track timesteps explicitly
            );
        end
    endgenerate

    // HL2 enable logic: enable the neuron one cycle after fc2 outputs its current
    always_comb begin
        for (int i = 0; i < HL2_SIZE; i++) begin
            hl2_enable[i] = hl2_process_valid && (hl2_process_idx == i[$clog2(HL2_SIZE)-1:0]);
        end
    end

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
            membrane_write_timestep <= '0;

            // Reset observations
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

            unique case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        // Save observations
                        for (int i = 0; i < NUM_INPUTS; i++) begin
                            observations_registered[i] <= observations[i];
                        end

                        // Clear all state for new inference
                        hl1_clear <= 1'b1;
                        hl2_clear <= 1'b1;

                        // Start fc1
                        fc1_start <= 1'b1;
                        state <= LOAD_HL1;
                    end
                end

                LOAD_HL1: begin
                    // Wait for fc1 to complete
                    // fc1 outputs are saved in hl1_currents by the always_ff block above
                    if (fc1_done) begin
                        current_timestep <= '0;
                        ts_state <= TS_HL1_STEP;
                        state <= RUN_TIMESTEPS;
                    end
                end

                RUN_TIMESTEPS: begin
                    unique case (ts_state)
                        TS_HL1_STEP: begin
                            // Process all HL1 LIFs in parallel
                            hl1_enable <= 1'b1;
                            ts_state <= TS_FC2_START;
                        end

                        TS_FC2_START: begin
                            // HL1 spikes feed directly to fc2 via hl1_spike_vector
                            // Start fc2
                            fc2_start <= 1'b1;
                            membrane_write_timestep <= current_timestep;

                            ts_state <= TS_FC2_HL2;
                        end

                        TS_FC2_HL2: begin
                            // Wait for fc2 to complete
                            // HL2 LIFs are enabled individually as fc2 outputs their currents
                            // (handled by hl2_enable combinational logic)
                            if (fc2_done) begin
                                ts_state <= TS_NEXT;
                            end
                        end

                        TS_NEXT: begin
                            // Check if we need to start q_accumulator for previous timestep
                            // (pipelined: q_acc starts when HL2 data for timestep T is ready)
                            // For simplicity in this first version, we start q_acc after all timesteps
                            // TODO: Implement pipelined q_accumulator start

                            if (current_timestep == NUM_TIMESTEPS - 1) begin
                                // All timesteps complete, start q_accumulator
                                q_start <= 1'b1;
                                state <= FINISH_Q;
                            end else begin
                                // Move to next timestep
                                current_timestep <= current_timestep + 1'b1;
                                ts_state <= TS_HL1_STEP;
                            end
                        end

                        default: ts_state <= TS_HL1_STEP;
                    endcase
                end

                FINISH_Q: begin
                    // Wait for q_accumulator to complete
                    if (q_done) begin
                        done <= 1'b1;
                        state <= DONE_STATE;
                    end
                end

                DONE_STATE: begin
                    // Hold done and q_values until next start
                    if (start) begin
                        done <= 1'b0;

                        // Save new observations
                        for (int i = 0; i < NUM_INPUTS; i++) begin
                            observations_registered[i] <= observations[i];
                        end

                        // Clear all state for new inference
                        hl1_clear <= 1'b1;
                        hl2_clear <= 1'b1;

                        // Start fc1
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
