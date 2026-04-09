// Fractional-Order Leaky Integrate-and-Fire (LIF) Neuron Module
// Implements a fractional-order spiking neuron using Grünwald-Letnikov approximation
// Drop-in replacement for lif.sv with matching interface
//
// Membrane dynamics (from fractional_lif.py):
//   V[n] = (I[n] - C * Σ_{k=1}^{H-1} g_k * V[n-k]) / (C + λ)
//   where C = 1 / dt^α
//
// Spike generation: spike[t] = (mem[t] >= threshold)
// Reset: subtract threshold from membrane on spike (reset_delay=True style)
//
// Relationship between standard LIF beta and fractional lambda:
//   λ = (1 - β) / β
//   For β = 0.9 (default): λ = 0.1 / 0.9 ≈ 0.111
//   This provides approximate equivalence in membrane decay behavior.
//
// Fixed-point format: QS2.13 (16-bit signed, 2 integer bits, 13 fractional bits)

module fractional_lif #(
    // Standard LIF parameters (match lif.sv interface)
    parameter THRESHOLD = 8192,              // Spike threshold (1.0 in QS2.13)
    parameter DATA_WIDTH = 16,
    parameter MEMBRANE_WIDTH = 24,
    
    // Fractional-order parameters
    parameter HISTORY_LENGTH = 64,           // Number of past values for GL approximation
    parameter COEFF_WIDTH = 16,              // GL coefficient magnitude width (QU1.15 unsigned)
    parameter COEFF_FRAC_BITS = 15,          // Fractional bits in coefficients
    
    // Precomputed constants (from generate_coefficients.py)
    // For α=0.5, dt=1.0, β=0.9 (→ λ=0.111): C=1.0, denom=1.111
    // Relationship: λ = (1 - β) / β
    // C_SCALED in Q8.8: 1.0 * 256 = 256
    parameter [15:0] C_SCALED = 16'd256,
    parameter integer C_SCALED_FRAC_BITS = 8,
    // INV_DENOM in Q0.16: 1/1.111 ≈ 0.9 * 65536 ≈ 58982
    parameter [15:0] INV_DENOM = 16'd58982,
    parameter integer INV_DENOM_FRAC_BITS = 16,
    
    // Coefficient file (GL coefficient magnitudes |g_1| to |g_{H-1}|)
    parameter GL_COEFF_FILE = "gl_coefficients.mem"
) (
    input wire clk,
    input wire reset,
    input wire clear,                                    // Synchronous clear for new inference
    input wire enable,                                   // Process one timestep
    input wire signed [DATA_WIDTH-1:0] current,          // Input current (QS2.13 format)
    output logic spike_out,                              // Spike output this timestep
    output logic signed [MEMBRANE_WIDTH-1:0] membrane_out // Membrane potential after update
);

    // Address width for history indexing
    localparam integer ADDR_WIDTH = $clog2(HISTORY_LENGTH);

    // Internal state
    logic signed [MEMBRANE_WIDTH-1:0] membrane_potential;  // Membrane potential
    logic spike_prev;                                      // Previous spike for reset delay

    // History buffer (circular buffer for past membrane potentials)
    logic signed [MEMBRANE_WIDTH-1:0] history_buffer [0:HISTORY_LENGTH-1];
    logic [ADDR_WIDTH-1:0] history_ptr;      // Points to oldest value (next write location)

    // GL coefficient magnitudes (unsigned, loaded from file)
    // |g_k| for k=1 to HISTORY_LENGTH-1 (g_0=1 is implicit and not stored)
    // Assumes 0 < alpha <= 1, where g_k (k>=1) are non-positive.
    logic [COEFF_WIDTH-1:0] gl_coeffs_mag [0:HISTORY_LENGTH-2];
    
    // Load pre-computed GL coefficients from memory file
    initial begin
        $readmemh(GL_COEFF_FILE, gl_coeffs_mag, 0, HISTORY_LENGTH-2);
    end

    // Intermediate computation signals
    logic signed [MEMBRANE_WIDTH-1:0] next_membrane;
    logic signed [MEMBRANE_WIDTH-1:0] current_extended;
    logic signed [MEMBRANE_WIDTH-1:0] reset_subtract;
    logic next_spike;
    
    // History magnitude-weighted sum computation (Σ |g_k| * V[n-k] for k=1 to H-1)
    // This needs to be pipelined for large HISTORY_LENGTH, but for now use combinational
    // Width derivation notes:
    // - product = signed(hist_val) * unsigned(coeff_mag) => MEMBRANE_WIDTH + COEFF_WIDTH + 1 bits
    // - summing up to (HISTORY_LENGTH-1) products needs ceil(log2(HISTORY_LENGTH-1)) guard bits
    // - C_SCALED / INV_DENOM are stored as 16-bit unsigned params, prepended with 1 sign bit before multiply
    localparam integer PRODUCT_WIDTH = MEMBRANE_WIDTH + COEFF_WIDTH + 1;
    localparam integer HISTORY_SUM_WIDTH = PRODUCT_WIDTH + $clog2(HISTORY_LENGTH);
    localparam integer C_SCALED_WIDTH = $bits(C_SCALED) + 1;
    localparam integer SCALED_HISTORY_WIDTH = HISTORY_SUM_WIDTH + C_SCALED_WIDTH;
    localparam integer NUMERATOR_INPUT_WIDTH = (SCALED_HISTORY_WIDTH > MEMBRANE_WIDTH) ? SCALED_HISTORY_WIDTH : MEMBRANE_WIDTH;
    localparam integer NUMERATOR_WIDTH = NUMERATOR_INPUT_WIDTH + 1;
    localparam integer INV_DENOM_WIDTH = $bits(INV_DENOM) + 1;
    localparam integer SCALED_RESULT_WIDTH = NUMERATOR_WIDTH + INV_DENOM_WIDTH;
    logic signed [HISTORY_SUM_WIDTH-1:0] history_sum;  // Accumulator for sum of products
    localparam signed [MEMBRANE_WIDTH-1:0] MEMBRANE_MAX = {1'b0, {(MEMBRANE_WIDTH-1){1'b1}}};
    localparam signed [MEMBRANE_WIDTH-1:0] MEMBRANE_MIN = {1'b1, {(MEMBRANE_WIDTH-1){1'b0}}};

    // Combinational logic to compute next membrane potential
    // Implements fractional LIF using |g_k| with 0<alpha<=1 sign property:
    // V[n] = (I[n] + C * Σ |g_k| * V[n-k]) / (C + λ)
    always_comb begin
        logic [ADDR_WIDTH-1:0] hist_idx;
        logic [ADDR_WIDTH-1:0] k_plus_1;
        logic signed [MEMBRANE_WIDTH-1:0] hist_val;
        logic [COEFF_WIDTH-1:0] coeff_mag;
        logic signed [MEMBRANE_WIDTH+COEFF_WIDTH:0] product;
        logic signed [SCALED_HISTORY_WIDTH-1:0] scaled_history;
        logic signed [NUMERATOR_WIDTH-1:0] numerator;
        logic signed [SCALED_RESULT_WIDTH-1:0] scaled_result;
        logic signed [SCALED_RESULT_WIDTH-1:0] membrane_pre_reset;
        logic signed [SCALED_RESULT_WIDTH-1:0] membrane_after_reset;
        logic signed [SCALED_RESULT_WIDTH-1:0] membrane_max_ext;
        logic signed [SCALED_RESULT_WIDTH-1:0] membrane_min_ext;

        // Default assignments to prevent latches
        current_extended = '0;
        reset_subtract = '0;
        next_membrane = '0;
        next_spike = 1'b0;
        history_sum = '0;
        scaled_history = '0;
        numerator = '0;
        scaled_result = '0;
        membrane_pre_reset = '0;
        membrane_after_reset = '0;
        membrane_max_ext = '0;
        membrane_min_ext = '0;

        // Sign-extend current from DATA_WIDTH to MEMBRANE_WIDTH
        current_extended = {{(MEMBRANE_WIDTH-DATA_WIDTH){current[DATA_WIDTH-1]}}, current};

        // Step 1: Compute history sum Σ_{k=1}^{H-1} |g_k| * V[n-k]
        // history_ptr points to oldest value, so most recent V[n-1] is at (history_ptr - 1)
        for (int k = 0; k < HISTORY_LENGTH - 1; k++) begin
            // k=0 in loop corresponds to |g_1| * V[n-1], k=1 to |g_2| * V[n-2], etc.
            k_plus_1 = ADDR_WIDTH'(k + 1);

            // Calculate circular buffer index for V[n-k-1]
            // history_ptr is where next write goes (oldest), so V[n-1] is at history_ptr-1
            if (history_ptr >= k_plus_1) begin
                hist_idx = history_ptr - k_plus_1;
            end else begin
                hist_idx = history_ptr + ADDR_WIDTH'(HISTORY_LENGTH) - k_plus_1;
            end

            hist_val = history_buffer[hist_idx];
            coeff_mag = gl_coeffs_mag[k];
            product = $signed({1'b0, coeff_mag}) * hist_val;

            // Accumulate (sign-extend product to accumulator width)
            history_sum = history_sum +
                          {{(HISTORY_SUM_WIDTH-PRODUCT_WIDTH){product[PRODUCT_WIDTH-1]}}, product};
        end

        // Step 2: Calculate reset subtraction (threshold if prev spike, else 0)
        // reset_delay=True: subtract threshold one cycle after spike
        reset_subtract = spike_prev ? MEMBRANE_WIDTH'($signed(THRESHOLD)) : '0;

        // Step 3: Compute V[n] = (I[n] + C * history_sum) / (C + λ)
        // Using precomputed C_SCALED and INV_DENOM with explicit frac-bit parameters
        // 
        // Numerator = current_extended + (C_SCALED * history_sum) >> C_SCALED_FRAC_BITS
        // Then multiply by INV_DENOM and shift to get final membrane
        begin
            // Scale history_sum by C and remove fractional bits from both fixed-point factors
            scaled_history = ($signed({1'b0, C_SCALED}) * history_sum) >>> (C_SCALED_FRAC_BITS + COEFF_FRAC_BITS);

            // Numerator = I[n] + C * Σ |g_k| * V[n-k]
            numerator = {{(NUMERATOR_WIDTH-MEMBRANE_WIDTH){current_extended[MEMBRANE_WIDTH-1]}}, current_extended} +
                       {{(NUMERATOR_WIDTH-SCALED_HISTORY_WIDTH){scaled_history[SCALED_HISTORY_WIDTH-1]}}, scaled_history};

            // Divide by (C + λ) via multiplication by 1/(C+λ) = INV_DENOM
            // Remove INV_DENOM fractional bits after multiply
            scaled_result = numerator * $signed({1'b0, INV_DENOM});

            // Apply reset subtraction at wide precision, then saturate to membrane width
            membrane_pre_reset = scaled_result >>> INV_DENOM_FRAC_BITS;
            membrane_after_reset = membrane_pre_reset -
                                   {{(SCALED_RESULT_WIDTH-MEMBRANE_WIDTH){reset_subtract[MEMBRANE_WIDTH-1]}}, reset_subtract};

            membrane_max_ext = {{(SCALED_RESULT_WIDTH-MEMBRANE_WIDTH){MEMBRANE_MAX[MEMBRANE_WIDTH-1]}}, MEMBRANE_MAX};
            membrane_min_ext = {{(SCALED_RESULT_WIDTH-MEMBRANE_WIDTH){MEMBRANE_MIN[MEMBRANE_WIDTH-1]}}, MEMBRANE_MIN};

            if (membrane_after_reset > membrane_max_ext) begin
                next_membrane = MEMBRANE_MAX;
            end else if (membrane_after_reset < membrane_min_ext) begin
                next_membrane = MEMBRANE_MIN;
            end else begin
                next_membrane = membrane_after_reset[MEMBRANE_WIDTH-1:0];
            end
        end

        // Step 4: Generate spike if membrane >= threshold
        next_spike = (next_membrane >= MEMBRANE_WIDTH'($signed(THRESHOLD)));
    end

    // Sequential logic - update state on clock edge
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            membrane_potential <= '0;
            spike_prev <= 1'b0;
            spike_out <= 1'b0;
            membrane_out <= '0;
            history_ptr <= '0;
            
            // Clear history buffer
            for (int i = 0; i < HISTORY_LENGTH; i++) begin
                history_buffer[i] <= '0;
            end
        end else if (clear) begin
            // Synchronous clear for new inference
            membrane_potential <= '0;
            spike_prev <= 1'b0;
            spike_out <= 1'b0;
            membrane_out <= '0;
            history_ptr <= '0;
            
            // Clear history buffer
            for (int i = 0; i < HISTORY_LENGTH; i++) begin
                history_buffer[i] <= '0;
            end
        end else if (enable) begin
            // Process one timestep
            
            // Store current membrane in history before updating
            history_buffer[history_ptr] <= membrane_potential;
            history_ptr <= (history_ptr == ADDR_WIDTH'(HISTORY_LENGTH - 1)) ? '0 : history_ptr + 1'b1;
            
            // Update membrane potential
            membrane_potential <= next_membrane;
            spike_prev <= next_spike;
            spike_out <= next_spike;
            membrane_out <= next_membrane;
        end
        // else: hold state when not enabled
    end

endmodule
