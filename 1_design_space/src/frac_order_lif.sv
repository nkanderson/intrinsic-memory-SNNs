//==============================================================================
// File: frac_order_lif.sv
// Author: Niklas Anderson
// Date: July 2025
// Generated with assistance from GitHub Copilot
//
// Description:
//   Fractional-order Leaky Integrate-and-Fire (LIF) neuron model implementation.
//
//   This module implements a discrete-time fractional-order LIF neuron using
//   the Grünwald-Letnikov fractional derivative approximation:
//
//   V[n] = (1/(1+(tau/h^alpha))) * [R*I[n] - (tau/h^alpha) *
//          sum(k=1 to history_size)(-1^k * (alpha choose k) * V[n-k])]
//
//   Where:
//   - alpha: Fractional order parameter (0 < alpha < 1)
//   - tau: Time constant (related to RC in physical model)
//   - h: Time step
//   - R: Membrane resistance scaling factor
//   - (alpha choose k): Generalized binomial coefficients from LUT
//   - (-1)^k: Alternating sign factor for fractional derivative
//
// Key Features:
//   - Uses binomial coefficient LUT for fractional weights
//   - Maintains circular history buffer for past membrane potentials
//   - Implements proper fractional-order memory effects
//   - Fixed-point arithmetic throughout for FPGA efficiency
//
//==============================================================================

module frac_order_lif #(
    // Physical parameters
    parameter [7:0] R_SCALED = 8'd9,      // Resistance scaling factor (in MΩ) - reduced for better sensitivity

    // Current scaling to match lapicque.sv (10 pA per input unit)
    // parameter [7:0] CURRENT_SCALE = 8'd128,  // Matches lapicque K factor for 10pA scaling

    // The values for TAU_OVER_H_ALPHA and NORM_FACTOR come from preprocessing
    // in the syn/scripts/frac_order_lif_utils.py script. These values need to be calculated
    // ahead of time due to the requirements for floating point arithmetic.
    // TAU_OVER_H_ALPHA in Q8.8 format
    // Used to default to 2.0 * 256 = 512.0 (well below max 16-bit value),
    // currently fine-tuning the desired default
    localparam [15:0] TAU_OVER_H_ALPHA = 16'd6400,

    // NORM_FACTOR in Q0.8 format (0 integer bits, 8 fractional bits)
    localparam [7:0] NORM_FACTOR = 8'd9,

    parameter integer HISTORY_SIZE = 256, // Number of past samples to consider (max 8 for iCE40 when mapping to DSP slices for multiplication)

    // Standard LIF parameters
    parameter [7:0] THRESHOLD = 8'd40,      // Spike threshold
    parameter [4:0] REFRACTORY_PERIOD = 5'd5, // Refractory period in time steps

    // Coefficient configuration
    parameter integer COEFF_WIDTH = 16,       // Binomial coefficient bit width (16-bit for UQ0.16)
    // Number of DSPs to use on iCE40UP5K (max 8). Additional multipliers fall back to LUTs.
    parameter int NUM_DSP = 8
) (
    input logic clk,
    input logic rst,
    input logic [7:0] current,               // Input current (represents 10pA per unit, matches lapicque.sv)
    output logic spike
);

    // Address width for history indexing
    localparam integer ADDR_WIDTH = $clog2(HISTORY_SIZE);

    // Internal signals
    logic [7:0] membrane_potential;
    logic [4:0] refractory_counter;
    logic in_refractory;

    // History buffer (circular buffer for past membrane potentials)
    logic [7:0] history_buffer [0:HISTORY_SIZE-1];
    logic [ADDR_WIDTH-1:0] history_ptr;      // Points to next storage location (current V[n] will be stored here)

    // Parallel coefficient processing arrays
    logic [COEFF_WIDTH-1:0] coefficients [1:HISTORY_SIZE];  // Pre-computed coefficient magnitudes for k=1 to HISTORY_SIZE
    
    // Load pre-computed coefficients from memory file
    // The coefficients.mem file should contain HISTORY_SIZE lines with coefficient values
    // in Q0.COEFF_WIDTH format (one coefficient per line, in decimal or hex format)
    initial begin
        $readmemh(`COEFF_FILE, coefficients, 1, HISTORY_SIZE);
    end

    logic [7:0] history_values [0:HISTORY_SIZE-1];          // History values for parallel access
    logic [23:0] products [0:HISTORY_SIZE-1];               // Multiplication results: coeff_mag * history
    logic [31:0] fractional_sum;                            // Final accumulator

    // Final membrane potential calculation
    logic [31:0] current_term;               // R * I[n] term
    logic [31:0] history_term;               // (tau/h^alpha) * fractional_sum
    // Temporary 48-bit variable to capture the full result for the history term,
    // then saturate if needed, otherwise extract bits [31:0] for Q16.16 format
    logic [47:0] temp_history_mult;
    logic [31:0] unnormalized_potential;     // Before normalization (32-bit unsigned)
    logic [39:0] temp_result;                // Temporary result for normalization (32×8-bit mult)
    logic [15:0] updated_potential;          // Final result after normalization

    // Basic LIF logic
    assign in_refractory = (refractory_counter > 0);
    assign spike = (membrane_potential >= THRESHOLD) && !in_refractory;

    // Parallel fractional derivative calculation (combinational)
    // This block uses a generate-for loop to instantiate HISTORY_SIZE parallel always_comb blocks,
    // allowing synthesis tools to infer parallel combinational logic for each fractional term.
    // The generate/always_comb structure ensures each product is computed in parallel hardware,
    // improving performance and matching the intended architecture for FPGA synthesis.
    generate
        genvar j;
        for (j = 0; j < HISTORY_SIZE; j++) begin : gen_parallel_mult
            // j=0 corresponds to k=1, j=1 corresponds to k=2, etc.
            // Calculate history index for V[n-k]
            // history_ptr points to oldest value / next position to overwrite, so we go back (j+1) steps
            localparam integer K_VALUE = j + 1; // Mathematical k value

            always_comb begin
                // Get history value for V[n-k]
                // history_ptr points to next storage location, so most recent is at (history_ptr - 1)
                // For V[n-k], we go back K_VALUE steps from the most recent
                logic [ADDR_WIDTH-1:0] hist_idx;
                logic [ADDR_WIDTH-1:0] k_addr;

                k_addr = ADDR_WIDTH'(K_VALUE);  // Convert K_VALUE to address width
                if (history_ptr >= k_addr) begin
                    hist_idx = history_ptr - k_addr;
                end else begin
                    hist_idx = history_ptr + ADDR_WIDTH'(HISTORY_SIZE) - k_addr;
                end
                history_values[j] = history_buffer[hist_idx];
            end

            // Calculate weighted term: (-1)^k * (α choose k) * V[n-k]
            // For 0 < α < 1: All final weights are negative due to alternating patterns canceling
            // We account for this later by subtracting the sum.
            // Use DSP for first NUM_DSP terms when targeting iCE40, else LUTs
            `ifdef ICE40
            // TODO: Confirm products assignment is correct with updated bit widths.
            if (j < NUM_DSP) begin : use_dsp
              // Zero-extend to 16 bits for SB_MAC16 (unsigned math here)
              wire [15:0] a_in = {{(16-COEFF_WIDTH){1'b0}}, coefficients[K_VALUE]};
              wire [15:0] b_in = {8'b0, history_values[j]};
              wire [31:0] mac_y;

              SB_MAC16 #(
                  .A_SIGNED(1'b0),
                  .B_SIGNED(1'b0),
                  .MODE_8x8(1'b0), // full 16x16
                  // No internal regs/pipelines
                  .A_REG(1'b0), .B_REG(1'b0), .C_REG(1'b0), .D_REG(1'b0),
                  .TOP_8x8_MULT_REG(1'b0), .BOT_8x8_MULT_REG(1'b0),
                  .PIPELINE_16x16_MULT_REG1(1'b0), .PIPELINE_16x16_MULT_REG2(1'b0),
                  .TOPOUTPUT_SELECT(2'b00), .BOTOUTPUT_SELECT(2'b00)
              ) mac16 (
                  .A(a_in),
                  .B(b_in),
                  .C(16'd0),
                  .D(16'd0),
                  .O(mac_y),

                  // Tie-offs for unused control pins
                  .AHOLD(1'b0), .BHOLD(1'b0), .CHOLD(1'b0), .DHOLD(1'b0),
                  .ADDSUBTOP(1'b0), .ADDSUBBOT(1'b0),
                  .OLOADTOP(1'b0), .OLOADBOT(1'b0),
                  .OHOLDTOP(1'b0), .OHOLDBOT(1'b0),
                  .IRSTTOP(1'b0), .IRSTBOT(1'b0),
                  .ORSTTOP(1'b0), .ORSTBOT(1'b0),
                  .CLK(1'b0), .CE(1'b1), .CI(1'b0)
              );

              // Keep 16 LSBs to match the 8x8 -> 16-bit product format
              assign products[j] = mac_y[15:0];
          end else begin : use_lut
            // Use LUT for remaining terms
            assign products[j] = coefficients[K_VALUE] * history_values[j];
          end
          `else
            // Simulation / generic build: plain LUT multiplier for all
            assign products[j] = coefficients[K_VALUE] * history_values[j];
          `endif
        end
    endgenerate

    // Sum all products to get fractional derivative sum
    always_comb begin
        fractional_sum = 32'b0;  // Initialize to proper width
        for (int k = 0; k < HISTORY_SIZE; k++) begin
            // Max number of product terms before considering the possibility of overflow
            // is 2^24. Unlikely that even that would overflow given expected values,
            // but we should consider it as we increase the HISTORY_SIZE.
            fractional_sum += {8'b0, products[k]};  // Extend to 32 bits
        end
    end

    // Main membrane potential calculation
    always_comb begin
        // Optionally, consider adding a scaling factor CURRENT_SCALE for current
        // to match lapicque.sv K factor.
        // Current term: (R * CURRENT_SCALE * I[n]) >> 8
        // Apply current scaling similar to lapicque.sv K factor
        // current_term = (R_SCALED * CURRENT_SCALE * {24'b0, current}) >> 8;
        // Current term: R * I[n]
        current_term = R_SCALED * current;

        // History term: (tau/h^alpha) * fractional_sum
        // Get full result, then saturate if needed, otherwise extract bits [31:0] for Q12.20 format.
        // temp_history_mult is format Q24.24
        temp_history_mult = TAU_OVER_H_ALPHA * fractional_sum;
        if (temp_history_mult > 48'h0000000F_FFF_FFFFF) begin
            history_term = 32'hFFFF_FFFF;  // Saturate to max Q12.20
        end else begin
            // Select to get format of Q12.20, allowing for a max integer value
            // of 4095, leaving greater precision in fractional bits.
            history_term = temp_history_mult[35:4];
            // history_term = temp_history_mult[31:0];  // Normal case
        end

        // Combined: R*I[n] - (tau/h^alpha) * fractional_sum
        // We can add the history_term because we know the original equation would result
        // in the subtraction of a negative number, which is equivalent.
        // Use saturation arithmetic to handle potential overflow in 32-bit unsigned
        // TODO: Consider this saturation in light of increased bit widths. The current_term is
        // unlikely to reach this max value, so this is lower priority, but should be investigated.
        if (current_term == 32'hFFFF_FFFF && history_term > 32'h0000_FFFF) begin
            // Would cause overflow, saturate the addition
            unnormalized_potential = 32'hFFFF_FFFF;
        end else begin
            // Shift current_term to match Q12.20 format for addition with history_term
            unnormalized_potential = (current_term << 20) + history_term;
        end

        // Normalize: multiply by 1/(1+(tau/h^alpha))
        // NORM_FACTOR is in Q0.8 format, 32×8 = 40-bit result
        // temp_result is Q12.28
        temp_result = unnormalized_potential * NORM_FACTOR;
        // Take only the upper bits for Q12.4 format
        updated_potential = temp_result[39:24];  // Effectively a right shift by 24
    end

    // Sequential logic for membrane potential and history updates
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            membrane_potential <= 8'b0;
            refractory_counter <= 5'b0;
            history_ptr <= '0;

            // Clear history buffer
            for (int i = 0; i < HISTORY_SIZE; i++) begin
                history_buffer[i] <= 8'b0;
            end
        end else begin
            if (in_refractory) begin
                refractory_counter <= refractory_counter - 1;
                // NOTE: Now we have a doubling of the first value out of refractory
                // in history. That may be how this needs to function, given that there is no
                // other reasonable history value to insert.
                // Setup for exiting the refractory period:
                // During the last cycle of the refractory period, we want a zero value to
                // be present in the history, so the updated_potential can be correctly calculated.
                // As such, we need to push the zero-value membrane_potential into history when
                // the refractory_counter is 2, then in the last cycle (when refractory_counter is 1),
                // we can update the membrane potential with the new calculated value.
                // TODO: Decide whether to insert the reset membrane potential value at each clock cycle
                // in the refractory period, or only once when exiting.
                // if (refractory_counter >= 2) begin
                if (refractory_counter == 2) begin
                    // On last cycle of refractory, allow update with zero value in history
                    history_buffer[history_ptr] <= membrane_potential;
                    history_ptr <= (history_ptr == ADDR_WIDTH'(HISTORY_SIZE - 1)) ? '0 : history_ptr + 1;
                end
                if (refractory_counter == 1) begin
                    // Update membrane potential with saturation
                    // updated_potential is in Q12.4 format, so max value before saturation is 0x0FF0
                    if (updated_potential > 16'h0FF0) begin
                        membrane_potential <= 8'd255;
                    end else begin
                        membrane_potential <= updated_potential[11:4];
                    end
                end
            end else if (spike) begin
                // Reset logic on spike
                // NOTE: When we use the method of subracting the threshold, the remainder value
                // can have a significant impact on the next membrane potential calculated out of the
                // refractory period. This is due to the coefficient being relatively large for that
                // recent value.
                // We're using zero for now to produce more consistent behavior while testing. We may or
                // may not decide that a reset via subraction is desired later.
                // membrane_potential <= membrane_potential - THRESHOLD;
                membrane_potential <= '0;
                refractory_counter <= REFRACTORY_PERIOD;

                // Update history with current membrane potential before reset
                history_buffer[history_ptr] <= membrane_potential;
                history_ptr <= (history_ptr == ADDR_WIDTH'(HISTORY_SIZE - 1)) ? '0 : history_ptr + 1;
            end else begin
                // Normal update: store current membrane potential in history
                history_buffer[history_ptr] <= membrane_potential;
                history_ptr <= (history_ptr == ADDR_WIDTH'(HISTORY_SIZE - 1)) ? '0 : history_ptr + 1;

                // Update membrane potential with saturation
                // updated_potential is in Q12.4 format, so max value before saturation is 0x0FF0
                if (updated_potential > 16'h0FF0) begin
                    membrane_potential <= 8'd255;
                end else begin
                    membrane_potential <= updated_potential[11:4];
                end
            end
        end
    end

endmodule

//==============================================================================
// Implementation Notes:
//
// 1. Uses parallel coefficient generation in synthesis-time generate blocks
// 2. All binomial coefficients pre-computed using localparam for given ALPHA_4BIT
// 3. Parallel multipliers (HISTORY_SIZE count) for single-cycle computation
// 4. Coefficients stored as unsigned magnitude (UQ0.8 format for k≥1)
// 5. All final weights are negative: (-1)^k * (α choose k) always < 0 for k≥1
// 6. Simplified logic: store magnitude, apply negative sign directly
// 7. History buffer implemented as distributed memory for parallel access
// 8. Suitable for HISTORY_SIZE ≤ 8 on iCE40, larger on other FPGAs
//
// Mathematical Background:
// - Grünwald-Letnikov: sum(k=1 to N) [(-1)^k * (α choose k) * V[n-k]]
// - For 0 < α < 1: (α choose k) alternates: +α, -α(α-1)/2!, +α(α-1)(α-2)/3!, ...
// - Combined with (-1)^k: all final weights are negative (memory decay effect)
// - This eliminates complex alternating sign logic in hardware
//
// Resource Usage (approximate for HISTORY_SIZE=8):
// - DSP slices: 8 (one per k value)
// - Logic cells: ~150-200 (simplified from alternating sign removal)
// - Memory: 8 × 8-bit registers (history buffer)
//
// TODO for future enhancements:
// - Add scalable pipeline for HISTORY_SIZE > DSP slice count
// - Runtime alpha switching (multiple coefficient sets)
// - Power optimization with clock gating
// - Precision analysis and optimization
//
//==============================================================================
