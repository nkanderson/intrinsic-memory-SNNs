// Leaky Integrate-and-Fire (LIF) Neuron Module (Single-Step Version)
// Implements a spiking neuron with membrane potential integration, leak, and reset
// Matches snnTorch Leaky neuron with reset_mechanism="subtract" and reset_delay=True
//
// This single-step version processes one timestep per 'enable' signal, allowing
// external control of timestep timing by the neural_network module.
//
// Membrane dynamics: mem[t] = beta * mem[t-1] + input[t] - reset[t-1] * threshold
// Spike generation: spike[t] = (mem[t] >= threshold)
// Reset delay: The threshold subtraction is applied one cycle after the spike
//
// Fixed-point format: QS2.13 (16-bit signed, 2 integer bits, 13 fractional bits)
// - Threshold 1.0 = 8192 (2^13)
// - Beta 0.9 in Q1.7 = 115
//
// For the legacy version with internal timestep loop, see lif_timestep.sv.

module lif #(
    parameter THRESHOLD = 8192,          // Spike threshold (1.0 in QS2.13)
    parameter BETA = 115,                // Decay factor in Q1.7 format (115 ≈ 0.9)
    parameter DATA_WIDTH = 16,
    parameter MEMBRANE_WIDTH = 24
) (
    input wire clk,
    input wire reset,
    input wire clear,                                    // Synchronous clear for new inference
    input wire enable,                                    // Process one timestep
    input wire signed [DATA_WIDTH-1:0] current,          // Input current (QS2.13 format)
    output logic spike_out,                               // Spike output this timestep
    output logic signed [MEMBRANE_WIDTH-1:0] membrane_out // Membrane potential after update
);

    // Internal state
    logic signed [MEMBRANE_WIDTH-1:0] membrane_potential;  // Membrane potential
    logic spike_prev;                                      // Previous spike for reset delay

    // Next state computation
    logic signed [MEMBRANE_WIDTH-1:0] next_membrane;
    logic signed [MEMBRANE_WIDTH-1:0] current_extended;
    logic signed [MEMBRANE_WIDTH-1:0] decay_potential;
    logic signed [31:0] decay_temp;
    logic signed [MEMBRANE_WIDTH-1:0] reset_subtract;
    logic next_spike;

    // Combinational logic to compute next membrane potential
    // Implements: mem = beta * mem + current - prev_spike * threshold
    always_comb begin
        // Default assignments to prevent latches
        current_extended = '0;
        decay_temp = '0;
        decay_potential = '0;
        reset_subtract = '0;
        next_membrane = '0;
        next_spike = 1'b0;

        // Sign-extend current from DATA_WIDTH to MEMBRANE_WIDTH
        current_extended = {{(MEMBRANE_WIDTH-DATA_WIDTH){current[DATA_WIDTH-1]}}, current};

        // Step 1: Apply decay to current membrane (multiply by beta)
        // BETA is in Q1.7 format (8 bits: 1 integer bit, 7 fractional bits)
        // membrane_potential is MEMBRANE_WIDTH-bit signed (QS2.13 with extra headroom)
        // Result of multiplication is 32-bit, shift right by 7 to scale back
        decay_temp = membrane_potential * $signed({1'b0, BETA[7:0]});
        decay_potential = MEMBRANE_WIDTH'(decay_temp >>> 7);  // Arithmetic right shift, truncate to MEMBRANE_WIDTH bits

        // Step 2: Calculate reset subtraction (threshold if prev spike, else 0)
        // `spike_prev` register holds the spike value from the previous timestep
        reset_subtract = spike_prev ? MEMBRANE_WIDTH'($signed(THRESHOLD)) : '0;

        // Step 3: Compute next membrane: beta*mem + input - reset*threshold
        next_membrane = decay_potential + current_extended - reset_subtract;

        // Step 4: Generate spike if membrane >= threshold
        next_spike = (next_membrane >= MEMBRANE_WIDTH'($signed(THRESHOLD)));
    end

    // Sequential logic - update state on clock edge
    always_ff @(posedge clk or posedge reset) begin
        // Default: hold state when not enabled
        membrane_potential <= membrane_potential;
        spike_prev <= spike_prev;
        spike_out <= spike_out;
        membrane_out <= membrane_out;
        if (reset) begin
            membrane_potential <= '0;
            spike_prev <= 1'b0;
            spike_out <= 1'b0;
            membrane_out <= '0;
        end else if (clear) begin
            // Synchronous clear for new inference (if needed)
            membrane_potential <= '0;
            spike_prev <= 1'b0;
            spike_out <= 1'b0;
            membrane_out <= '0;
        end else if (enable) begin
            // Process one timestep
            membrane_potential <= next_membrane;
            spike_prev <= next_spike;
            spike_out <= next_spike;
            membrane_out <= next_membrane;
        end
    end

endmodule
