// Bit-shift Approximation Leaky Integrate-and-Fire (LIF) Neuron Module
// Drop-in replacement for lif.sv / fractional_lif.sv with matching interface.
//
// Dynamics mirror train/bitshift_lif.py with fixed-point arithmetic:
//   V[n] = (I[n] - C * Σ_{k=1}^{H-1} (V[n-k] >> shift[k])) / (C + λ)
//
// Spike generation/reset style matches lif.sv timing:
//   spike[t] = (V[t] >= threshold)
//   reset subtraction uses spike from previous timestep (reset_delay=True style)

module bitshift_lif #(
  // Standard LIF parameters (match lif.sv interface)
  parameter THRESHOLD = 8192,              // Spike threshold (1.0 in QS2.13)
  parameter DATA_WIDTH = 16,
  parameter MEMBRANE_WIDTH = 24,

  // Bit-shift approximation parameters
  parameter HISTORY_LENGTH = 64,
  parameter SHIFT_WIDTH = 8,
  // SHIFT_MODE selection (compile-time):
  //   0: simple            -> [0,1,2,3,4,...]
  //   1: slow_decay        -> [0,1,1,2,2,3,3,...]
  //   2: custom            -> [0,1,3,4,5xR,6xR,...], R=CUSTOM_DECAY_RATE
  //   3: custom_slow_decay -> [0,1,3,4,5x3,6x4,7x5,...]
  parameter [1:0] SHIFT_MODE = 2'd3,
  parameter integer CUSTOM_DECAY_RATE = 3,

  // Fixed-point constants for C=1/dt^alpha and 1/(C+lam)
  // Defaults correspond to dt=1.0, alpha=0.5, lam≈0.111:
  //   C = 1.0, INV_DENOM ≈ 1/1.111 ≈ 0.9
  parameter [15:0] C_SCALED = 16'd256,
  parameter integer C_SCALED_FRAC_BITS = 8,
  parameter [15:0] INV_DENOM = 16'd58982,
  parameter integer INV_DENOM_FRAC_BITS = 16
) (
  input wire clk,
  input wire reset,
  input wire clear,                                    // Synchronous clear for new inference
  input wire enable,                                   // Process one timestep
  input wire signed [DATA_WIDTH-1:0] current,          // Input current (QS2.13 format)
  output logic spike_out,                              // Spike output this timestep
  output logic signed [MEMBRANE_WIDTH-1:0] membrane_out // Membrane potential after update
);

  localparam integer ADDR_WIDTH = $clog2(HISTORY_LENGTH);
  localparam integer HISTORY_SUM_WIDTH = MEMBRANE_WIDTH + $clog2(HISTORY_LENGTH);
  localparam integer C_SCALED_WIDTH = $bits(C_SCALED) + 1;
  localparam integer SCALED_HISTORY_WIDTH = HISTORY_SUM_WIDTH + C_SCALED_WIDTH;
  localparam integer NUMERATOR_INPUT_WIDTH = (SCALED_HISTORY_WIDTH > MEMBRANE_WIDTH) ? SCALED_HISTORY_WIDTH : MEMBRANE_WIDTH;
  localparam integer NUMERATOR_WIDTH = NUMERATOR_INPUT_WIDTH + 1;
  localparam integer INV_DENOM_WIDTH = $bits(INV_DENOM) + 1;
  localparam integer SCALED_RESULT_WIDTH = NUMERATOR_WIDTH + INV_DENOM_WIDTH;

  localparam signed [MEMBRANE_WIDTH-1:0] MEMBRANE_MAX = {1'b0, {(MEMBRANE_WIDTH-1){1'b1}}};
  localparam signed [MEMBRANE_WIDTH-1:0] MEMBRANE_MIN = {1'b1, {(MEMBRANE_WIDTH-1){1'b0}}};

  // Internal state
  logic signed [MEMBRANE_WIDTH-1:0] membrane_potential;
  logic spike_prev;

  // History buffer (circular)
  logic signed [MEMBRANE_WIDTH-1:0] history_buffer [0:HISTORY_LENGTH-1];
  logic [ADDR_WIDTH-1:0] history_ptr;  // points to next write location (oldest sample)

  // Intermediate signals
  logic signed [MEMBRANE_WIDTH-1:0] next_membrane;
  logic signed [MEMBRANE_WIDTH-1:0] current_extended;
  logic signed [MEMBRANE_WIDTH-1:0] reset_subtract;
  logic next_spike;
  logic signed [HISTORY_SUM_WIDTH-1:0] history_sum;
  logic signed [MEMBRANE_WIDTH-1:0] history_shifted_terms [0:HISTORY_LENGTH-2];

  function automatic integer get_shift_amount_const(input integer idx);
    integer rem;
    integer shift;
    integer repeat_count;
    integer found;
    begin
      case (SHIFT_MODE)
        // simple_bitshift: [0,1,2,3,...]
        2'd0: begin
          get_shift_amount_const = idx;
        end

        // slow_decay_bitshift: [0,1,1,2,2,3,3,...]
        2'd1: begin
          if (idx == 0) begin
            get_shift_amount_const = 0;
          end else begin
            get_shift_amount_const = (idx + 1) / 2;
          end
        end

        // custom_bitshift: [0,1,3,4,5,5,5,6,6,6,...]
        2'd2: begin
          if (idx == 0) begin
            get_shift_amount_const = 0;
          end else if (idx == 1) begin
            get_shift_amount_const = 1;
          end else if (idx == 2) begin
            get_shift_amount_const = 3;
          end else if (idx == 3) begin
            get_shift_amount_const = 4;
          end else begin
            get_shift_amount_const = 5 + ((idx - 4) / CUSTOM_DECAY_RATE);
          end
        end

        // custom_slow_decay_bitshift:
        // [0,1,3,4,5x3,6x4,7x5,...] where shift N repeats (N-2) times
        default: begin
          if (idx == 0) begin
            get_shift_amount_const = 0;
          end else if (idx == 1) begin
            get_shift_amount_const = 1;
          end else if (idx == 2) begin
            get_shift_amount_const = 3;
          end else if (idx == 3) begin
            get_shift_amount_const = 4;
          end else begin
            rem = idx - 4;
            shift = 5;
            found = 0;
            for (int s = 5; s <= ((1 << SHIFT_WIDTH) - 1); s++) begin
              repeat_count = s - 2;
              if ((found == 0) && (rem < repeat_count)) begin
                shift = s;
                found = 1;
              end else if (found == 0) begin
                rem = rem - repeat_count;
              end
            end
            get_shift_amount_const = shift;
          end
        end
      endcase

      if (get_shift_amount_const < 0) begin
        get_shift_amount_const = 0;
      end
      if (get_shift_amount_const > ((1 << SHIFT_WIDTH) - 1)) begin
        get_shift_amount_const = (1 << SHIFT_WIDTH) - 1;
      end
    end
  endfunction

  // Build per-tap shifted history terms with compile-time constant shift amounts.
  // This avoids runtime shift-pattern selection logic in the datapath.
  generate
    for (genvar gi = 0; gi < HISTORY_LENGTH - 1; gi++) begin : gen_history_terms
      localparam integer SHIFT_AMT = get_shift_amount_const(gi + 1);
      always_comb begin
        logic [ADDR_WIDTH-1:0] k_plus_1;
        logic [ADDR_WIDTH-1:0] hist_idx_g;

        k_plus_1 = ADDR_WIDTH'(gi + 1);
        if (history_ptr >= k_plus_1) begin
          hist_idx_g = history_ptr - k_plus_1;
        end else begin
          hist_idx_g = history_ptr + ADDR_WIDTH'(HISTORY_LENGTH) - k_plus_1;
        end

        history_shifted_terms[gi] = history_buffer[hist_idx_g] >>> SHIFT_AMT;
      end
    end
  endgenerate

  // Combinational next-state computation
  always_comb begin
    logic signed [SCALED_HISTORY_WIDTH-1:0] scaled_history;
    logic signed [NUMERATOR_WIDTH-1:0] numerator;
    logic signed [SCALED_RESULT_WIDTH-1:0] scaled_result;
    logic signed [SCALED_RESULT_WIDTH-1:0] membrane_pre_reset;
    logic signed [SCALED_RESULT_WIDTH-1:0] membrane_after_reset;
    logic signed [SCALED_RESULT_WIDTH-1:0] membrane_max_ext;
    logic signed [SCALED_RESULT_WIDTH-1:0] membrane_min_ext;

    // Defaults
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

    // Sign-extend input current
    current_extended = {{(MEMBRANE_WIDTH-DATA_WIDTH){current[DATA_WIDTH-1]}}, current};

    // Step 1: Sum shifted history terms for k=1..H-1
    for (int k = 0; k < HISTORY_LENGTH - 1; k++) begin
      history_sum = history_sum +
              {{(HISTORY_SUM_WIDTH-MEMBRANE_WIDTH){history_shifted_terms[k][MEMBRANE_WIDTH-1]}}, history_shifted_terms[k]};
    end

    // Step 2: Delayed reset subtraction (lif.sv style)
    reset_subtract = spike_prev ? MEMBRANE_WIDTH'($signed(THRESHOLD)) : '0;

    // Step 3: V[n] = (I[n] - C * history_sum) / (C + λ)
    // C_SCALED and INV_DENOM are fixed-point constants
    scaled_history = ($signed({1'b0, C_SCALED}) * history_sum) >>> C_SCALED_FRAC_BITS;

    numerator = {{(NUMERATOR_WIDTH-MEMBRANE_WIDTH){current_extended[MEMBRANE_WIDTH-1]}}, current_extended} -
          {{(NUMERATOR_WIDTH-SCALED_HISTORY_WIDTH){scaled_history[SCALED_HISTORY_WIDTH-1]}}, scaled_history};

    scaled_result = numerator * $signed({1'b0, INV_DENOM});
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

    // Step 4: Spike generation
    next_spike = (next_membrane >= MEMBRANE_WIDTH'($signed(THRESHOLD)));
  end

  // Sequential state updates
  always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
      membrane_potential <= '0;
      spike_prev <= 1'b0;
      spike_out <= 1'b0;
      membrane_out <= '0;
      history_ptr <= '0;
      for (int i = 0; i < HISTORY_LENGTH; i++) begin
        history_buffer[i] <= '0;
      end
    end else if (clear) begin
      membrane_potential <= '0;
      spike_prev <= 1'b0;
      spike_out <= 1'b0;
      membrane_out <= '0;
      history_ptr <= '0;
      for (int i = 0; i < HISTORY_LENGTH; i++) begin
        history_buffer[i] <= '0;
      end
    end else if (enable) begin
      // Store current membrane in history before update
      history_buffer[history_ptr] <= membrane_potential;
      history_ptr <= (history_ptr == ADDR_WIDTH'(HISTORY_LENGTH - 1)) ? '0 : history_ptr + 1'b1;

      // Publish next state
      membrane_potential <= next_membrane;
      spike_prev <= next_spike;
      spike_out <= next_spike;
      membrane_out <= next_membrane;
    end
    // else: hold state
  end

endmodule
