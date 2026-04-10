//==============================================================================
// File: lapicque.sv
// Author: Niklas Anderson
// Date: July 2025
// Generated with assistance from GitHub Copilot
//
// Description:
//   Lapicque Leaky Integrate-and-Fire (LIF) neuron model with physical parameters.
//   Based on SNNTorch's Lapicque model: https://github.com/jeshraghian/snntorch/blob/master/snntorch/_neurons/lapicque.py
//
//   This implementation uses physically meaningful parameters (R, C, h) instead
//   of abstract decay values. The discrete-time integration follows:
//
//   V[n] = decay * V[n-1] + K * I[n]
//
//   Where:
//   - decay = 1 - h/(RC)  : Membrane decay factor based on RC time constant
//   - K = h/C             : Current scaling factor based on capacitance
//   - h                   : Time step (simulation time resolution)
//   - R                   : Membrane resistance
//   - C                   : Membrane capacitance
//
// Physical Parameter Defaults (fixed-point representation):
//   - R = 100 (representing 100 MΩ in scaled units)
//   - C = 20 (representing 20 pF in scaled units)
//   - H = 1 (representing 1 ms time step in scaled units)
//   - RC = 100 * 20 = 2000 (scaled time constant)
//   - decay = 256 - 256*1/2000 = 256 - 0.128 ≈ 256 (≈1.0 for small h/RC)
//   - K = 256*1/20*10 = 128 (scaled current factor for 10pA input units)
//
// Fixed-Point Representation:
//   - All calculations use integer arithmetic
//   - DECAY: 0.8 format (0-255 maps to 0.0-0.996)
//   - K: 8-bit integer scaling factor
//
//==============================================================================

module lapicque #(
    // Physical parameters
    parameter real R_MOHM = 100.0,           // Membrane resistance - default 100 MΩ
    parameter real C_PF = 20.0,              // Membrane capacitance - default 20 pF
    parameter real H_MS = 1.0,               // Time step - default 1 ms

    // Intermediate calculations
    localparam real RC_MS = R_MOHM * C_PF * 1e-3,  // tau - result is in ms - default is 2 ms
    localparam real H_OVER_RC = H_MS / RC_MS,      // Dimensionless - default is 0.5
    localparam real K_REAL = H_MS / C_PF,          // (ms/pF = 10⁹ × s/F) - default is 0.05

    // Convert to fixed-point
    // Adding intermediate variables to satisfy verilator linting as well as yosys
    // TODO: Consider moving this to the body of the module and seeing if the bit
    // slicing works with yosys, so we can get rid of the intermediate variables
    localparam integer DECAY_TEMP = $rtoi((1.0 - H_OVER_RC) * 256.0),
    localparam integer K_TEMP = $rtoi(K_REAL * 256.0 * 10.0),
    localparam [7:0] DECAY = DECAY_TEMP[7:0],  // 0.5 × 256 = 128
    localparam [7:0] K = K_TEMP[7:0],         // 0.05 × 256 × 10 = 128 (10pA per input unit)

    // Standard LIF parameters
    parameter [7:0] THRESHOLD = 8'd200,      // Spike threshold
    parameter [4:0] REFRACTORY_PERIOD = 5'd5 // Refractory period in time steps
) (
    input logic clk,
    input logic rst,
    // Allows for a range of 10pA to 2.55nA. If larger values are needed,
    // the bit width here may be increased. Using smaller input granularity
    // (e.g., 1pA) causes issues with the K scaling value, where the input
    // current essentially disappears.
    input logic [7:0] current,               // Input current (represents 10pA per unit)
    output logic spike
);

    logic [7:0] membrane_potential = 8'd0;
    logic [4:0] refractory_counter = 5'd0;
    logic in_refractory;
    logic [15:0] updated_potential; // Temporary 16-bit value to handle overflow

    assign in_refractory = (refractory_counter > 0);
    assign spike = (membrane_potential >= THRESHOLD) && !in_refractory;

    // Lapicque model integration: V[n] = decay * V[n-1] + K * I[n]
    // Both DECAY and K are now in Q0.8 format, so both need >> 8 bit shifts
    assign updated_potential = (({8'b0, K} * current) >> 8) +
                              ((membrane_potential * DECAY) >> 8);

    always_ff @(posedge clk) begin
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

//==============================================================================
// Parameter Calculation Reference:
//
// Given physical parameters:
//   R = 100 MΩ, C = 20 pF, h = 1 ms
//
// RC time constant:
//   τ = RC = 100e6 * 20e-12 = 2e-3 = 2 ms
//
// Decay factor (continuous to discrete):
//   decay = exp(-h/τ) ≈ 1 - h/τ = 1 - 1ms/2ms = 0.5
//   In Q0.8 fixed-point: 0.5 * 256 = 128
//
// Current scaling factor:
//   K_actual = h/C = 1e-3 / 20e-12 = 5e10 (V·s/F)
//   With input current scaling (1 unit = 10 pA), K becomes 0.5 mV per input unit
//   K_scaled = 128 (accounts for 10 pA input scaling and Q0.8 format)
//
// This gives reasonable behavior with input currents in the range 0-255
// (representing 0-2.55 nA) producing membrane potentials that can reach the threshold of 200.
//==============================================================================
