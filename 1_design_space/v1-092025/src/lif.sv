//==============================================================================
// File: lif.sv
// Author: Niklas Anderson
// Date: July 2025
// Generated with assistance from GitHub Copilot
//
// Description:
//   Leaky Integrate-and-Fire (LIF) neuron model implementation in SystemVerilog.
//   Based on SNNTorch's Leaky Integrate-and-Fire neuron model: https://github.com/jeshraghian/snntorch/blob/master/snntorch/_neurons/leaky.py
//
//   This module implements a discrete-time LIF neuron with the following behavior:
//   - Membrane potential integrates input current with linear decay
//   - Current input affects membrane potential in the same clock cycle (instantaneous)
//   - Spike generation occurs when membrane potential exceeds threshold
//   - Refractory period prevents multiple spikes
//   - Saturation logic prevents overflow
//
// Timing Behavior:
//   - Input current changes are reflected immediately in the updated_potential
//     calculation via combinational logic
//   - Membrane potential updates on the rising clock edge using the current
//     cycle's input and previous cycle's membrane potential
//   - This creates proper discrete-time integration: V[n] = decay*V[n-1] + I[n]
//
// Parameters:
//   - THRESHOLD: Spike threshold (default: 200)
//   - DECAY: Fixed-point decay factor, 0.8 format (default: 128 = 128/256 = 0.5)
//   - REFRACTORY_PERIOD: Clock cycles of refractory period (default: 5)
//
//==============================================================================

module lif #(
    parameter [7:0] THRESHOLD = 8'd200,
    parameter [7:0] DECAY = 8'd128, // Fixed-point representation of decay (e.g., 128/256 = 0.5)
    parameter [4:0] REFRACTORY_PERIOD = 5'd5
) (
    input logic clk,
    input logic rst,
    input logic [7:0] current,
    output logic spike
);

    logic [7:0] membrane_potential;
    logic [4:0] refractory_counter;
    logic in_refractory;
    logic [15:0] updated_potential; // Temporary 16-bit value to handle overflow

    assign in_refractory = (refractory_counter > 0);
    assign spike = (membrane_potential >= THRESHOLD) && !in_refractory;

    // Fixed-point multiplication and clamping logic
    assign updated_potential = ((membrane_potential * DECAY) >> 8) + {8'b0, current};

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            membrane_potential <= 8'b0;
            refractory_counter <= 5'b0;
        end else begin
            if (in_refractory) begin
                refractory_counter <= refractory_counter - 1;
            end else if (spike) begin
                membrane_potential <= membrane_potential - THRESHOLD;
                refractory_counter <= REFRACTORY_PERIOD;
            end else begin
                // Saturation logic to prevent overflow
                if (updated_potential > 16'd255) begin
                    membrane_potential <= 8'd255; // Clamp to max value
                end else begin
                    membrane_potential <= updated_potential[7:0]; // Assign valid range
                end
            end
        end
    end

endmodule
