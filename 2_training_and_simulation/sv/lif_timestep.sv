// Leaky Integrate-and-Fire (LIF) Neuron Module
// Implements a spiking neuron with membrane potential integration, leak, and reset
// Matches snnTorch Leaky neuron with reset_mechanism="subtract" and reset_delay=True
//
// Membrane dynamics: mem[t] = beta * mem[t-1] + input[t] - reset[t-1] * threshold
// Spike generation: spike[t] = (mem[t] >= threshold)
// Reset delay: The threshold subtraction is applied one cycle after the spike
//
// Fixed-point format: QS2.13 (16-bit signed, 2 integer bits, 13 fractional bits)
// - Threshold 1.0 = 8192 (2^13)
// - Beta 0.9 in Q1.7 = 115
//
// Operation:
// - Assert 'start' to latch input current and begin timestep processing
// - Runs for NUM_TIMESTEPS timesteps automatically, outputting spike_out and membrane_out each cycle
// - Asserts 'done' when all timesteps complete
// - Holds state until next 'start'
//
// NOTE: This module manages its own timestep loop internally. For a single-step
// version where timesteps are managed externally, see lif.sv.

module lif_timestep #(
    parameter THRESHOLD = 8192,          // Spike threshold (1.0 in QS2.13)
    parameter BETA = 115,                // Decay factor in Q1.7 format (115 ≈ 0.9)
    parameter NUM_TIMESTEPS = 30         // Number of timesteps per inference
) (
    input wire clk,
    input wire reset,
    input wire start,                            // Latch current and begin timesteps
    input wire signed [15:0] current,            // Input current (QS2.13 format)
    output logic spike_out,                      // Current timestep spike
    output logic signed [23:0] membrane_out,     // Current membrane potential
    output logic [$clog2(NUM_TIMESTEPS)-1:0] timestep, // Current timestep (0 to NUM_TIMESTEPS-1)
    output logic done                            // All timesteps complete
);

    // Internal state
    logic signed [23:0] membrane_potential;  // Membrane potential (wider for accumulation headroom)
    logic signed [15:0] current_latched;     // Latched input current
    logic [$clog2(NUM_TIMESTEPS)-1:0] timestep_counter; // Timestep counter (0 to NUM_TIMESTEPS-1)
    logic running;                           // Processing timesteps flag
    logic spike_prev;                        // Previous spike for reset delay

    // Next state computation
    logic signed [23:0] next_membrane;
    logic signed [23:0] current_extended;
    logic signed [23:0] decay_potential;
    logic signed [31:0] decay_temp;
    logic signed [23:0] reset_subtract;

    // Combinational logic to compute next membrane potential
    // Implements: mem = beta * mem + current - prev_spike * threshold
    always_comb begin
        // Select current source: use input directly when starting, latched when running
        // This allows first timestep computation during the start cycle
        logic signed [15:0] current_to_use;
        current_to_use = (start && !running) ? current : current_latched;

        // Sign-extend current from 16-bit to 24-bit
        current_extended = {{8{current_to_use[15]}}, current_to_use};

        // Step 1: Apply decay to current membrane (multiply by beta)
        // BETA is in Q1.7 format (8 bits: 1 integer bit, 7 fractional bits)
        // membrane_potential is 24-bit signed (QS2.13 with extra headroom)
        // Result of multiplication is 32-bit, shift right by 7 to scale back
        decay_temp = membrane_potential * $signed({1'b0, BETA[7:0]});
        decay_potential = 24'(decay_temp >>> 7);  // Arithmetic right shift, truncate to 24 bits

        // Step 2: Calculate reset subtraction (threshold if prev spike, else 0)
        // `spike_prev` register holds the spike value from the previous timestep
        reset_subtract = spike_prev ? THRESHOLD : 24'sd0;

        // Step 3: Compute next membrane: beta*mem + input - reset*threshold
        next_membrane = decay_potential + current_extended - reset_subtract;
    end

    // Sequential logic - update state on clock edge
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            membrane_potential <= 24'sd0;
            current_latched <= 16'sd0;
            timestep_counter <= '0;
            running <= 1'b0;
            spike_out <= 1'b0;
            spike_prev <= 1'b0;
            membrane_out <= 24'sd0;
            timestep <= '0;
            done <= 1'b0;
        end else begin
            if (start && !running) begin
                // Latch current and begin timestep processing
                // First timestep (0) computation happens this cycle via combinational logic
                current_latched <= current;
                running <= 1'b1;
                done <= 1'b0;

                // Initialize state for timestep 0
                membrane_potential <= next_membrane;  // First timestep result
                spike_prev <= 1'b0;
                timestep_counter <= {{($clog2(NUM_TIMESTEPS)-1){1'b0}}, 1'b1};  // Next timestep will be 1

                // Output first timestep results
                spike_out <= (next_membrane >= 24'($signed(THRESHOLD)));
                membrane_out <= next_membrane;
                timestep <= '0;
            end else if (running) begin
                // Update membrane potential
                membrane_potential <= next_membrane;

                // Generate spike if membrane >= threshold
                spike_out <= (next_membrane >= 24'($signed(THRESHOLD)));

                // Update spike_prev with the NEW spike_out value for next cycle's reset
                spike_prev <= (next_membrane >= 24'($signed(THRESHOLD)));

                // Output current membrane and timestep
                membrane_out <= next_membrane;
                timestep <= timestep_counter;

                // Increment timestep counter and check for completion
                if (timestep_counter == NUM_TIMESTEPS - 1) begin
                    // Just output the last timestep, will complete next cycle
                    timestep_counter <= timestep_counter + {{($clog2(NUM_TIMESTEPS)-1){1'b0}}, 1'b1};
                end else if (timestep_counter == NUM_TIMESTEPS) begin
                    // All timesteps completed
                    running <= 1'b0;
                    done <= 1'b1;
                end else begin
                    timestep_counter <= timestep_counter + {{($clog2(NUM_TIMESTEPS)-1){1'b0}}, 1'b1};
                end
            end else begin
                // Hold done signal until next start
                done <= done;
            end
        end
    end


endmodule
