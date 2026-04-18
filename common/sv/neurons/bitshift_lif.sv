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
  parameter integer ACCUM_LANES = 1,       // Parallel history terms per cycle (>=1)
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
  parameter integer INV_DENOM_FRAC_BITS = 16,

  // Internal precision controls
  // ACCUM_GUARD_BITS controls accumulation headroom in history_sum_acc.
  // NUMERATOR_GUARD_BITS controls extra headroom before reciprocal multiply.
  // Defaults preserve behavior validated across multiple configurations and models.
  parameter integer ACCUM_GUARD_BITS = $clog2(HISTORY_LENGTH),
  parameter integer NUMERATOR_GUARD_BITS = 1
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
  localparam integer TERM_COUNT = (HISTORY_LENGTH > 1) ? (HISTORY_LENGTH - 1) : 1;
  localparam integer ACCUM_LANES_NONZERO = (ACCUM_LANES < 1) ? 1 : ACCUM_LANES;
  localparam integer ACCUM_LANES_EFF = (ACCUM_LANES_NONZERO > TERM_COUNT) ? TERM_COUNT : ACCUM_LANES_NONZERO;
  localparam integer ACCUM_GUARD_BITS_EFF = (ACCUM_GUARD_BITS < 0) ? 0 : ACCUM_GUARD_BITS;
  localparam integer NUMERATOR_GUARD_BITS_EFF = (NUMERATOR_GUARD_BITS < 0) ? 0 : NUMERATOR_GUARD_BITS;

  localparam integer HISTORY_SUM_WIDTH = MEMBRANE_WIDTH + ACCUM_GUARD_BITS_EFF;
  localparam integer ACCUM_REDUCE_GUARD_BITS = $clog2(ACCUM_LANES_EFF);
  localparam integer ACCUM_BATCH_SUM_WIDTH = HISTORY_SUM_WIDTH + ACCUM_REDUCE_GUARD_BITS;
  localparam integer C_SCALED_WIDTH = $bits(C_SCALED) + 1;
  localparam integer SCALED_HISTORY_WIDTH = HISTORY_SUM_WIDTH + C_SCALED_WIDTH;
  localparam integer NUMERATOR_INPUT_WIDTH = (SCALED_HISTORY_WIDTH > MEMBRANE_WIDTH) ? SCALED_HISTORY_WIDTH : MEMBRANE_WIDTH;
  localparam integer NUMERATOR_WIDTH = NUMERATOR_INPUT_WIDTH + NUMERATOR_GUARD_BITS_EFF;
  localparam integer INV_DENOM_WIDTH = $bits(INV_DENOM) + 1;
  localparam integer SCALED_RESULT_WIDTH = NUMERATOR_WIDTH + INV_DENOM_WIDTH;

  localparam signed [MEMBRANE_WIDTH-1:0] MEMBRANE_MAX = {1'b0, {(MEMBRANE_WIDTH-1){1'b1}}};
  localparam signed [MEMBRANE_WIDTH-1:0] MEMBRANE_MIN = {1'b1, {(MEMBRANE_WIDTH-1){1'b0}}};

  typedef enum logic [5:0] {
    ST_IDLE      = 6'b000001,
    ST_ACCUM     = 6'b000010,
    ST_PREP_NUM  = 6'b000100,
    ST_MUL_DIV   = 6'b001000,
    ST_POST      = 6'b010000,
    ST_WRITEBACK = 6'b100000
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

  // Intermediate helper signals (parallel lanes)
  logic [ADDR_WIDTH-1:0] accum_term_idx_lane [0:ACCUM_LANES_EFF-1];
  logic [ADDR_WIDTH-1:0] accum_k_plus_1_lane [0:ACCUM_LANES_EFF-1];
  logic [ADDR_WIDTH-1:0] accum_hist_idx_lane [0:ACCUM_LANES_EFF-1];
  logic signed [MEMBRANE_WIDTH-1:0] accum_hist_val_lane [0:ACCUM_LANES_EFF-1];
  logic [SHIFT_WIDTH-1:0] accum_shift_amt_lane [0:ACCUM_LANES_EFF-1];
  logic signed [MEMBRANE_WIDTH-1:0] accum_shifted_hist_lane [0:ACCUM_LANES_EFF-1];
  logic accum_lane_active [0:ACCUM_LANES_EFF-1];
  logic signed [ACCUM_BATCH_SUM_WIDTH-1:0] accum_batch_sum;
  logic signed [HISTORY_SUM_WIDTH-1:0] accum_next;

  (* use_dsp = "yes" *) logic signed [SCALED_HISTORY_WIDTH-1:0] prep_scaled_history_mult;
  logic signed [SCALED_HISTORY_WIDTH-1:0] prep_scaled_history;
  logic signed [NUMERATOR_WIDTH-1:0] prep_numerator;
  logic signed [NUMERATOR_WIDTH-1:0] numerator_reg;

  (* use_dsp = "yes" *) logic signed [SCALED_RESULT_WIDTH-1:0] mul_scaled_result;
  logic signed [SCALED_RESULT_WIDTH-1:0] mul_membrane_pre_reset;
  logic signed [SCALED_RESULT_WIDTH-1:0] membrane_pre_reset_reg;

  logic signed [MEMBRANE_WIDTH-1:0] reset_subtract;
  logic signed [SCALED_RESULT_WIDTH-1:0] membrane_after_reset;
  logic signed [SCALED_RESULT_WIDTH-1:0] membrane_max_ext;
  logic signed [SCALED_RESULT_WIDTH-1:0] membrane_min_ext;
  logic signed [MEMBRANE_WIDTH-1:0] post_finalize_membrane;
  logic post_finalize_spike;
  logic signed [MEMBRANE_WIDTH-1:0] finalize_membrane_reg;
  logic finalize_spike_reg;

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

  // Parallel accumulation terms per cycle: sum_lanes (V[n-k] >> shift[k]), k=1..H-1
  always_comb begin
    accum_batch_sum = '0;

    for (int lane = 0; lane < ACCUM_LANES_EFF; lane++) begin
      accum_term_idx_lane[lane] = accum_index + ADDR_WIDTH'(lane);
      accum_lane_active[lane] = (accum_term_idx_lane[lane] < ADDR_WIDTH'(HISTORY_LENGTH - 1));

      if (accum_lane_active[lane]) begin
        accum_k_plus_1_lane[lane] = accum_term_idx_lane[lane] + ADDR_WIDTH'(1);
        if (history_ptr >= accum_k_plus_1_lane[lane]) begin
          accum_hist_idx_lane[lane] = history_ptr - accum_k_plus_1_lane[lane];
        end else begin
          accum_hist_idx_lane[lane] = history_ptr + ADDR_WIDTH'(HISTORY_LENGTH) - accum_k_plus_1_lane[lane];
        end

        accum_hist_val_lane[lane] = history_buffer[accum_hist_idx_lane[lane]];
        accum_shift_amt_lane[lane] = SHIFT_WIDTH'(get_shift_amount_const(accum_term_idx_lane[lane] + 1));
        accum_shifted_hist_lane[lane] = accum_hist_val_lane[lane] >>> accum_shift_amt_lane[lane];
        accum_batch_sum = accum_batch_sum +
          {{(ACCUM_BATCH_SUM_WIDTH-MEMBRANE_WIDTH){accum_shifted_hist_lane[lane][MEMBRANE_WIDTH-1]}}, accum_shifted_hist_lane[lane]};
      end else begin
        accum_k_plus_1_lane[lane] = '0;
        accum_hist_idx_lane[lane] = '0;
        accum_hist_val_lane[lane] = '0;
        accum_shift_amt_lane[lane] = '0;
        accum_shifted_hist_lane[lane] = '0;
      end
    end

    accum_next = history_sum_acc + accum_batch_sum[HISTORY_SUM_WIDTH-1:0];
  end

  // ST_PREP_NUM stage: numerator = I[n] - C*sum
  always_comb begin
    prep_scaled_history_mult = $signed({1'b0, C_SCALED}) * history_sum_acc;
    prep_scaled_history = prep_scaled_history_mult >>> C_SCALED_FRAC_BITS;

    prep_numerator = {{(NUMERATOR_WIDTH-MEMBRANE_WIDTH){current_latched[MEMBRANE_WIDTH-1]}}, current_latched} -
                     {{(NUMERATOR_WIDTH-SCALED_HISTORY_WIDTH){prep_scaled_history[SCALED_HISTORY_WIDTH-1]}}, prep_scaled_history};
  end

  // ST_MUL_DIV stage: divide by (C+lambda) using reciprocal multiply
  always_comb begin
    mul_scaled_result = numerator_reg * $signed({1'b0, INV_DENOM});
    mul_membrane_pre_reset = mul_scaled_result >>> INV_DENOM_FRAC_BITS;
  end

  // ST_POST stage: delayed reset subtraction, saturation, spike generation
  always_comb begin
    reset_subtract = spike_prev ? MEMBRANE_WIDTH'($signed(THRESHOLD)) : '0;

    membrane_after_reset = membrane_pre_reset_reg -
                           {{(SCALED_RESULT_WIDTH-MEMBRANE_WIDTH){reset_subtract[MEMBRANE_WIDTH-1]}}, reset_subtract};

    membrane_max_ext = {{(SCALED_RESULT_WIDTH-MEMBRANE_WIDTH){MEMBRANE_MAX[MEMBRANE_WIDTH-1]}}, MEMBRANE_MAX};
    membrane_min_ext = {{(SCALED_RESULT_WIDTH-MEMBRANE_WIDTH){MEMBRANE_MIN[MEMBRANE_WIDTH-1]}}, MEMBRANE_MIN};

    if (membrane_after_reset > membrane_max_ext) begin
      post_finalize_membrane = MEMBRANE_MAX;
    end else if (membrane_after_reset < membrane_min_ext) begin
      post_finalize_membrane = MEMBRANE_MIN;
    end else begin
      post_finalize_membrane = membrane_after_reset[MEMBRANE_WIDTH-1:0];
    end

    post_finalize_spike = (post_finalize_membrane >= MEMBRANE_WIDTH'($signed(THRESHOLD)));
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
      numerator_reg <= '0;
      membrane_pre_reset_reg <= '0;
      finalize_membrane_reg <= '0;
      finalize_spike_reg <= 1'b0;
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
      numerator_reg <= '0;
      membrane_pre_reset_reg <= '0;
      finalize_membrane_reg <= '0;
      finalize_spike_reg <= 1'b0;
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
              state <= ST_PREP_NUM;
            end
          end
        end

        ST_ACCUM: begin
          history_sum_acc <= accum_next;
          if (($unsigned(accum_index) + ACCUM_LANES_EFF) >= (HISTORY_LENGTH - 1)) begin
            state <= ST_PREP_NUM;
          end else begin
            accum_index <= accum_index + ADDR_WIDTH'(ACCUM_LANES_EFF);
          end
        end

        ST_PREP_NUM: begin
          numerator_reg <= prep_numerator;
          state <= ST_MUL_DIV;
        end

        ST_MUL_DIV: begin
          membrane_pre_reset_reg <= mul_membrane_pre_reset;
          state <= ST_POST;
        end

        ST_POST: begin
          finalize_membrane_reg <= post_finalize_membrane;
          finalize_spike_reg <= post_finalize_spike;
          state <= ST_WRITEBACK;
        end

        ST_WRITEBACK: begin
          // Store current membrane in history before updating
          history_buffer[history_ptr] <= membrane_potential;
          history_ptr <= (history_ptr == ADDR_WIDTH'(HISTORY_LENGTH - 1)) ? '0 : history_ptr + 1'b1;

          membrane_potential <= finalize_membrane_reg;
          spike_prev <= finalize_spike_reg;
          spike_out <= finalize_spike_reg;
          membrane_out <= finalize_membrane_reg;
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
