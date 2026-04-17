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
  output logic signed [MEMBRANE_WIDTH-1:0] membrane_out, // Membrane potential after update
  output logic busy,                                   // High while update is in progress
  output logic output_valid                            // 1-cycle pulse when outputs update
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

  typedef enum logic [2:0] {
    ST_IDLE     = 3'b001,
    ST_ACCUM    = 3'b010,
    ST_FINALIZE = 3'b100
  } state_t;

  // Internal state
  (* fsm_encoding = "one_hot" *) state_t state;
  logic signed [MEMBRANE_WIDTH-1:0] membrane_potential;
  logic spike_prev;

  // History buffer (circular)
  logic signed [MEMBRANE_WIDTH-1:0] history_buffer [0:HISTORY_LENGTH-1];
  logic [ADDR_WIDTH-1:0] history_ptr;  // points to next write location (oldest sample)

  // Multi-cycle accumulation state
  logic signed [MEMBRANE_WIDTH-1:0] current_latched;
  logic [ADDR_WIDTH-1:0] accum_index;
  logic signed [HISTORY_SUM_WIDTH-1:0] history_sum_acc;

  // Intermediate helper signals
  logic [ADDR_WIDTH-1:0] accum_k_plus_1;
  logic [ADDR_WIDTH-1:0] accum_hist_idx;
  logic signed [MEMBRANE_WIDTH-1:0] accum_hist_val;
  logic [SHIFT_WIDTH-1:0] accum_shift_amt;
  logic signed [MEMBRANE_WIDTH-1:0] accum_shifted_hist;
  logic signed [HISTORY_SUM_WIDTH-1:0] accum_shifted_hist_ext;
  logic signed [HISTORY_SUM_WIDTH-1:0] accum_next;

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

  // One accumulation term per cycle: V[n-k] >> shift[k], k=1..H-1
  always_comb begin
    accum_k_plus_1 = ADDR_WIDTH'(accum_index + 1'b1);
    if (history_ptr >= accum_k_plus_1) begin
      accum_hist_idx = history_ptr - accum_k_plus_1;
    end else begin
      accum_hist_idx = history_ptr + ADDR_WIDTH'(HISTORY_LENGTH) - accum_k_plus_1;
    end

    accum_hist_val = history_buffer[accum_hist_idx];
    accum_shift_amt = SHIFT_WIDTH'(get_shift_amount_const(accum_index + 1));
    accum_shifted_hist = accum_hist_val >>> accum_shift_amt;
    accum_shifted_hist_ext =
      {{(HISTORY_SUM_WIDTH-MEMBRANE_WIDTH){accum_shifted_hist[MEMBRANE_WIDTH-1]}}, accum_shifted_hist};
    accum_next = history_sum_acc + accum_shifted_hist_ext;
  end

  // Finalize stage: V[n] = (I[n] - C*sum)/(C+lambda), then delayed reset subtraction
  always_comb begin
    reset_subtract = spike_prev ? MEMBRANE_WIDTH'($signed(THRESHOLD)) : '0;

    scaled_history_mult = $signed({1'b0, C_SCALED}) * history_sum_acc;
    scaled_history = scaled_history_mult >>> C_SCALED_FRAC_BITS;

    numerator = {{(NUMERATOR_WIDTH-MEMBRANE_WIDTH){current_latched[MEMBRANE_WIDTH-1]}}, current_latched} -
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

  // Sequential state updates
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
      accum_index <= '0;
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
      accum_index <= '0;
      history_sum_acc <= '0;
      for (int i = 0; i < HISTORY_LENGTH; i++) begin
        history_buffer[i] <= '0;
      end
    end else begin
      output_valid <= 1'b0;

      unique case (state)
        ST_IDLE: begin
          if (enable) begin
            current_latched <= {{(MEMBRANE_WIDTH-DATA_WIDTH){current[DATA_WIDTH-1]}}, current};
            history_sum_acc <= '0;
            accum_index <= '0;

            if (HISTORY_LENGTH > 1) begin
              state <= ST_ACCUM;
            end else begin
              state <= ST_FINALIZE;
            end
          end
        end

        ST_ACCUM: begin
          history_sum_acc <= accum_next;
          if (accum_index == ADDR_WIDTH'(HISTORY_LENGTH - 2)) begin
            state <= ST_FINALIZE;
          end else begin
            accum_index <= accum_index + 1'b1;
          end
        end

        ST_FINALIZE: begin
          // Store current membrane in history before updating
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
