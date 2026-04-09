//==============================================================================
// File: binomial_coeff_lut.sv
// Author: Niklas Anderson
// Date: July 2025
// Generated with assistance from GitHub Copilot
//
// Description:
//   Lookup table for generalized binomial coefficients (alpha choose k) where
//   alpha can be fractional. Uses fixed-point arithmetic for synthesis.
//
//   Generalized binomial coefficient formula:
//   (alpha choose k) = alpha * (alpha-1) * ... * (alpha-k+1) / k!
//                    = prod(i=0 to k-1) (alpha-i) / k!
//
//   For fractional alpha, this enables fractional-order differential equations
//   in discrete-time implementations.
//
// Parameters:
//   - ALPHA_4BIT: Alpha value as 4-bit integer (1-14, maps to 1/15 through 14/15)
//   - MAX_K: Maximum k value supported (determines LUT size, typically 4-23 for UQ0.8)
//   - COEFF_WIDTH: Bit width for coefficient storage (8-bit for UQ1.7/UQ0.8 formats)
//
// Implementation Notes:
//   - Coefficients are pre-computed at synthesis time for single alpha value
//   - k=0: Stored as unsigned magnitude in UQ1.7 format (1 integer bit + 7 fractional bits)
//   - k=0 coefficient is always 1.0, stored as 128
//   - k≥1: Stored as unsigned magnitude in UQ0.8 format (8 fractional bits)
//   - Parent module applies alternating signs (-1)^k for k≥1 coefficients
//   - Parent module reads k=0 directly as UQ1.7, k≥1 as UQ0.8 magnitude
//   - This approach maximizes precision for small negative coefficients
//   - LUT uses unpacked array for variable indexing
//   - Real arithmetic only used in generate loops (synthesis-time constants)
//
//==============================================================================

module binomial_coeff_lut #(
    parameter [3:0] ALPHA_4BIT = 4'd8,       // Alpha as 4-bit value (1-14, maps to 1/15-14/15)
    parameter integer MAX_K = 23,            // Maximum k value (supports history_size up to MAX_K)
    parameter integer COEFF_WIDTH = 8,       // Bit width for coefficients (signed, UQ0.8 format)
    parameter integer ADDR_WIDTH = $clog2(MAX_K+1) // Address width for k values
) (
    input  logic [ADDR_WIDTH-1:0] k,         // k value (0 to MAX_K)
    output logic [COEFF_WIDTH-1:0] coeff // Binomial coefficient (alpha choose k)
);

    // Pre-computed coefficient array - using unpacked array for synthesis compatibility
    // Most synthesis tools prefer unpacked arrays for variable indexing
    logic [COEFF_WIDTH-1:0] coeff_lut [0:MAX_K];

    // Function to calculate generalized binomial coefficient (performed at synthesis time only)
    function automatic real calc_binomial_coeff(real alpha, integer k);
        real result;
        integer i;

        if (k == 0) begin
            result = 1.0;
        end else begin
            result = 1.0;
            // Calculate: alpha * (alpha-1) * ... * (alpha-k+1) / k!
            for (i = 0; i < k; i++) begin
                result = result * (alpha - real'(i));
            end
            // Divide by k!
            for (i = 1; i <= k; i++) begin
                result = result / real'(i);
            end
        end

        return result;
    endfunction

    // Initialize packed LUT with pre-computed coefficients
    generate
        genvar i;
        // k=0 is always 1.0, stored as 128 in UQ1.7 format (bit 7 = 1, bits 6-0 = 0)
        initial coeff_lut[0] = 8'b10000000; // 1.0 in UQ1.7 format = 128 decimal
        for (i = 1; i <= MAX_K; i++) begin : gen_coeffs
            // Convert 4-bit alpha to real for coefficient calculation (synthesis-time only)
            localparam real ALPHA_REAL = real'(ALPHA_4BIT) / 15.0;
            localparam real COEFF_REAL = calc_binomial_coeff(ALPHA_REAL, i);

            // For k≥1: Use UQ0.8 format (0 integer bits + 8 fractional bits, unsigned magnitude)
            // Take absolute value since we store magnitudes only
            localparam real COEFF_MAG = (COEFF_REAL < 0.0) ? -COEFF_REAL : COEFF_REAL;
            localparam real SCALE_FACTOR = 256.0; // 2^8 for k≥1
            localparam real COEFF_SCALED = COEFF_MAG * SCALE_FACTOR; // Always positive scaling

            // Convert real to integer with proper saturation for each format
            localparam integer COEFF_INT =
                    // k≥1: UQ0.8 format (unsigned magnitude, 0 to 255)
                    (COEFF_SCALED > 255.0) ? 255 :
                    (COEFF_SCALED < 0.0) ? 0 :
                    int'(COEFF_SCALED);

            // Explicitly truncate to the correct width
            localparam [COEFF_WIDTH-1:0] COEFF_FIXED = COEFF_INT[COEFF_WIDTH-1:0];
            // Store coefficient in unpacked array at synthesis time
            initial coeff_lut[i] = COEFF_FIXED;
        end
    endgenerate

    // Extract coefficient from unpacked array based on k input
    // Extend k to 32 bits to match MAX_K width and avoid Verilator warnings
    assign coeff = (32'(k) <= MAX_K) ? coeff_lut[k] : '0;

    // Synthesis attributes for FPGA optimization
    (* rom_style = "distributed" *) // Use distributed ROM for small LUTs
    (* keep = "true" *) // Prevent optimization of LUT

endmodule

//==============================================================================
// Usage Example:
//
// binomial_coeff_lut #(
//     .ALPHA_4BIT(4'd8),       // alpha = 8/15 ≈ 0.533
//     .MAX_K(23),              // Support history_size up to 23 (UQ0.8 limit for α=0.2-0.267)
//     .COEFF_WIDTH(8)          // 8-bit coefficients (unsigned with offset for k=0, magnitude for k≥1)
// ) binom_lut (
//     .k(k_value),
//     .coeff(unsigned_coeff)   // k=0: subtract 128 to get signed value, k≥1: unsigned magnitude (apply (-1)^k in parent)
// );
//
// For fractional-order dynamics with 4-bit alpha representation:
// - Alpha values: 1/15, 2/15, ..., 14/15 (0.067 to 0.933)
// - k=0 coefficient: Stored as unsigned magnitude in UQ1.7 format (1 integer bit + 7 fractional bits)
// - k≥1 coefficients: Unsigned magnitude in UQ0.8 format
// - Parent module applies alternating signs: (-1)^k * coeff for k≥1
//
// This format maximizes precision for the small negative coefficients while
// maintaining hardware efficiency with simple sign application.
//
//==============================================================================
