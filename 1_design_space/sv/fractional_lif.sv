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
    output logic signed [MEMBRANE_WIDTH-1:0] membrane_out, // Membrane potential after update
    output logic busy,                                   // High while multi-cycle update is in progress
    output logic output_valid                            // 1-cycle pulse when spike_out/membrane_out update
);

    // Address width for history indexing
    localparam integer ADDR_WIDTH = $clog2(HISTORY_LENGTH);

    // Width derivation notes:
    // - product = signed(hist_val) * unsigned(coeff_mag) => MEMBRANE_WIDTH + COEFF_WIDTH + 1 bits
    // - history accumulator adds up to (HISTORY_LENGTH-1) products
    localparam integer PRODUCT_WIDTH = MEMBRANE_WIDTH + COEFF_WIDTH + 1;
    localparam integer HISTORY_SUM_WIDTH = PRODUCT_WIDTH + $clog2(HISTORY_LENGTH);
    localparam integer C_SCALED_WIDTH = $bits(C_SCALED) + 1;
    localparam integer SCALED_HISTORY_WIDTH = HISTORY_SUM_WIDTH + C_SCALED_WIDTH;
    localparam integer NUMERATOR_INPUT_WIDTH = (SCALED_HISTORY_WIDTH > MEMBRANE_WIDTH) ? SCALED_HISTORY_WIDTH : MEMBRANE_WIDTH;
    localparam integer NUMERATOR_WIDTH = NUMERATOR_INPUT_WIDTH + 1;
    localparam integer INV_DENOM_WIDTH = $bits(INV_DENOM) + 1;
    localparam integer SCALED_RESULT_WIDTH = NUMERATOR_WIDTH + INV_DENOM_WIDTH;

    localparam signed [MEMBRANE_WIDTH-1:0] MEMBRANE_MAX = {1'b0, {(MEMBRANE_WIDTH-1){1'b1}}};
    localparam signed [MEMBRANE_WIDTH-1:0] MEMBRANE_MIN = {1'b1, {(MEMBRANE_WIDTH-1){1'b0}}};

    typedef enum logic [1:0] {
        ST_IDLE     = 2'b00,
        ST_MAC      = 2'b01,
        ST_FINALIZE = 2'b10
    } state_t;

    // Internal state
    state_t state;
    logic signed [MEMBRANE_WIDTH-1:0] membrane_potential;
    logic spike_prev;

    // Circular history buffer for past membrane potentials
    logic signed [MEMBRANE_WIDTH-1:0] history_buffer [0:HISTORY_LENGTH-1];
    logic [ADDR_WIDTH-1:0] history_ptr;

    // GL coefficient magnitudes (|g_1| to |g_{H-1}|)
    logic [COEFF_WIDTH-1:0] gl_coeffs_mag [0:HISTORY_LENGTH-2];

    // Multi-cycle MAC datapath registers
    logic signed [MEMBRANE_WIDTH-1:0] current_latched;
    logic [ADDR_WIDTH-1:0] mac_index;
    logic signed [HISTORY_SUM_WIDTH-1:0] history_sum_acc;

    // MAC combinational helper signals
    logic [ADDR_WIDTH-1:0] mac_k_plus_1;
    logic [ADDR_WIDTH-1:0] mac_hist_idx;
    logic signed [MEMBRANE_WIDTH-1:0] mac_hist_val;
    logic [COEFF_WIDTH-1:0] mac_coeff_mag;
    (* use_dsp = "yes" *) logic signed [PRODUCT_WIDTH-1:0] mac_product;
    logic signed [HISTORY_SUM_WIDTH-1:0] mac_product_ext;
    logic signed [HISTORY_SUM_WIDTH-1:0] mac_acc_next;

    // Finalize stage combinational helper signals
    logic signed [MEMBRANE_WIDTH-1:0] reset_subtract;
    (* use_dsp = "yes" *) logic signed [SCALED_HISTORY_WIDTH-1:0] scaled_history_mult;
    logic signed [SCALED_HISTORY_WIDTH-1:0] scaled_history;
    logic signed [NUMERATOR_WIDTH-1:0] numerator;
    (* use_dsp = "yes" *) logic signed [SCALED_RESULT_WIDTH-1:0] scaled_result;
    logic signed [SCALED_RESULT_WIDTH-1:0] membrane_pre_reset;
    logic signed [SCALED_RESULT_WIDTH-1:0] membrane_after_reset;
    logic signed [SCALED_RESULT_WIDTH-1:0] membrane_max_ext;
    logic signed [SCALED_RESULT_WIDTH-1:0] membrane_min_ext;
    logic signed [MEMBRANE_WIDTH-1:0] finalize_membrane;
    logic finalize_spike;

    // Load pre-computed GL coefficients from memory file
    initial begin
        $readmemh(GL_COEFF_FILE, gl_coeffs_mag, 0, HISTORY_LENGTH-2);
    end

    // One MAC term per cycle: |g_(k+1)| * V[n-(k+1)]
    always_comb begin
        mac_k_plus_1 = ADDR_WIDTH'(mac_index + 1'b1);
        if (history_ptr >= mac_k_plus_1) begin
            mac_hist_idx = history_ptr - mac_k_plus_1;
        end else begin
            mac_hist_idx = history_ptr + ADDR_WIDTH'(HISTORY_LENGTH) - mac_k_plus_1;
        end

        mac_hist_val = history_buffer[mac_hist_idx];
        mac_coeff_mag = gl_coeffs_mag[mac_index];
        mac_product = $signed({1'b0, mac_coeff_mag}) * mac_hist_val;
        mac_product_ext = {{(HISTORY_SUM_WIDTH-PRODUCT_WIDTH){mac_product[PRODUCT_WIDTH-1]}}, mac_product};
        mac_acc_next = history_sum_acc + mac_product_ext;
    end

    // Finalize stage: V[n] = (I[n] + C*sum) / (C+lambda), then delayed reset subtraction
    always_comb begin
        reset_subtract = spike_prev ? MEMBRANE_WIDTH'($signed(THRESHOLD)) : '0;

        scaled_history_mult = $signed({1'b0, C_SCALED}) * history_sum_acc;
        scaled_history = scaled_history_mult >>> (C_SCALED_FRAC_BITS + COEFF_FRAC_BITS);

        numerator = {{(NUMERATOR_WIDTH-MEMBRANE_WIDTH){current_latched[MEMBRANE_WIDTH-1]}}, current_latched} +
                    {{(NUMERATOR_WIDTH-SCALED_HISTORY_WIDTH){scaled_history[SCALED_HISTORY_WIDTH-1]}}, scaled_history};

        scaled_result = numerator * $signed({1'b0, INV_DENOM});
        membrane_pre_reset = scaled_result >>> INV_DENOM_FRAC_BITS;
        membrane_after_reset = membrane_pre_reset -
                               {{(SCALED_RESULT_WIDTH-MEMBRANE_WIDTH){reset_subtract[MEMBRANE_WIDTH-1]}}, reset_subtract};

        membrane_max_ext = {{(SCALED_RESULT_WIDTH-MEMBRANE_WIDTH){MEMBRANE_MAX[MEMBRANE_WIDTH-1]}}, MEMBRANE_MAX};
        membrane_min_ext = {{(SCALED_RESULT_WIDTH-MEMBRANE_WIDTH){MEMBRANE_MIN[MEMBRANE_WIDTH-1]}}, MEMBRANE_MIN};

        if (membrane_after_reset > membrane_max_ext) begin
            finalize_membrane = MEMBRANE_MAX;
        end else if (membrane_after_reset < membrane_min_ext) begin
            finalize_membrane = MEMBRANE_MIN;
        end else begin
            finalize_membrane = membrane_after_reset[MEMBRANE_WIDTH-1:0];
        end

        finalize_spike = (finalize_membrane >= MEMBRANE_WIDTH'($signed(THRESHOLD)));
    end

    // Sequential logic
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= ST_IDLE;
            membrane_potential <= '0;
            spike_prev <= 1'b0;
            spike_out <= 1'b0;
            membrane_out <= '0;
            output_valid <= 1'b0;
            history_ptr <= '0;
            current_latched <= '0;
            mac_index <= '0;
            history_sum_acc <= '0;

            for (int i = 0; i < HISTORY_LENGTH; i++) begin
                history_buffer[i] <= '0;
            end
        end else if (clear) begin
            state <= ST_IDLE;
            membrane_potential <= '0;
            spike_prev <= 1'b0;
            spike_out <= 1'b0;
            membrane_out <= '0;
            output_valid <= 1'b0;
            history_ptr <= '0;
            current_latched <= '0;
            mac_index <= '0;
            history_sum_acc <= '0;

            for (int i = 0; i < HISTORY_LENGTH; i++) begin
                history_buffer[i] <= '0;
            end
        end else begin
            output_valid <= 1'b0;
            unique case (state)
                ST_IDLE: begin
                    // Start a new timestep when enabled.
                    if (enable) begin
                        current_latched <= {{(MEMBRANE_WIDTH-DATA_WIDTH){current[DATA_WIDTH-1]}}, current};
                        history_sum_acc <= '0;
                        mac_index <= '0;

                        if (HISTORY_LENGTH <= 1) begin
                            state <= ST_FINALIZE;
                        end else begin
                            state <= ST_MAC;
                        end
                    end
                end

                ST_MAC: begin
                    history_sum_acc <= mac_acc_next;
                    if (mac_index == ADDR_WIDTH'(HISTORY_LENGTH - 2)) begin
                        state <= ST_FINALIZE;
                    end else begin
                        mac_index <= mac_index + 1'b1;
                    end
                end

                ST_FINALIZE: begin
                    // Store current membrane in history before updating.
                    history_buffer[history_ptr] <= membrane_potential;
                    history_ptr <= (history_ptr == ADDR_WIDTH'(HISTORY_LENGTH - 1)) ? '0 : history_ptr + 1'b1;

                    membrane_potential <= finalize_membrane;
                    spike_prev <= finalize_spike;
                    spike_out <= finalize_spike;
                    membrane_out <= finalize_membrane;
                    output_valid <= 1'b1;

                    state <= ST_IDLE;
                end

                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end

    assign busy = (state != ST_IDLE);

endmodule
