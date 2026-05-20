// Linear Layer Module (Per-Output Parallel MAC)
// Implements a fully connected layer: outputs = weights * inputs + bias
// Equivalent to nn.Linear in PyTorch.
//
// Architecture: PARALLELISM (P) MAC engines run in lockstep, one input per
// cycle multiplied against P different weight rows simultaneously. Each
// engine owns one accumulator. After NUM_INPUTS cycles a batch of P
// dot-products is complete; the registered, saturated, biased results are
// written into the corresponding slots of outputs[]. If P < NUM_OUTPUTS,
// the FSM iterates ceil(NUM_OUTPUTS/P) batches before asserting done. With
// the default P = NUM_OUTPUTS, the layer completes in NUM_INPUTS+2 cycles
// (one MAC per cycle, one EMIT cycle, one IDLE→COMPUTING transition).
//
// This replaces the earlier serial-MAC design (now linear_layer_serial.sv),
// which streamed one output per ~NUM_INPUTS cycles via output_valid +
// output_idx pulses. The new interface registers the full output vector
// and pulses a single 'done'. The bit-exact reference is preserved in
// linear_layer_serial.sv; linear_layer_equivalence.sv instantiates both
// and a cocotb test (test_linear_layer_equivalence.py) compares outputs.
//
// Bit-exactness: per-neuron accumulators are ACCUM_WIDTH wide (same as the
// serial version), and per-neuron inputs are accumulated in identical
// order (input_idx 0..NUM_INPUTS-1). With no narrowing mid-sum, fixed-point
// addition is associative — the parallel and serial implementations
// produce identical accumulator values, identical saturated outputs, and
// identical bias-added results.
//
// Weights are stored in row-major order in a flattened array:
//   weights_flat[n*NUM_INPUTS + i] = weight for output n, input i
//
// Fixed-point format: QS2.13 (16-bit signed, 2 integer bits, 13 fractional bits)
//
// Timing (handshake on done; no cycle count promises beyond start/done):
//   - Cycle 0:                            assert 'start' for one cycle with valid inputs
//   - Cycle 1:                            inputs latched, COMPUTING begins
//   - Cycle 1 + B*(NUM_INPUTS+1):         outputs[] registered for batch B, done pulses on last batch
//   - 'done' is a single-cycle pulse; outputs[] hold valid values starting that cycle
//   - Total latency: 1 + ceil(NUM_OUTPUTS/PARALLELISM) * (NUM_INPUTS + 1) cycles from start to done
//
// Resource budget: NUM_OUTPUTS = 32 with P = 32 maps to ~32 DSP48E1 on
// 7-series. With NUM_OUTPUTS = 64 and P = 32 (e.g. bitshift fc1), 32 DSPs
// processing two sequential batches of 32 outputs.
//
// =========================================================================
// Synthesis considerations (read before first Vivado run on Artix-7 100T)
// =========================================================================
// 1. weights_flat fan-out (HIGHEST PRIORITY)
//    Every COMPUTING cycle, P lanes simultaneously read weights_flat at P
//    different addresses (one per active output neuron). Vivado has no
//    BRAM that supports >2 read ports natively, so it will fall back to
//    one of:
//      (a) Mapping weights_flat to LUTRAM and replicating it P times.
//          At P=32, NUM_INPUTS=64, DATA_WIDTH=16, that is up to
//          ~32 KB × 32 ≈ 1 MB of LUTRAM — will not fit on XC7A100T.
//      (b) Synthesizing the entire table as combinational LUT logic with
//          a P-output mux tree. Costs LUTs but is feasible at typical
//          fc1/fc2 sizes (32×64×16 = 32 Kb).
//      (c) Heuristically partitioning into per-lane banks when the access
//          pattern is provably disjoint. Not guaranteed.
//
//    Diagnosis after first synth: open `report_utilization` (post-synth),
//    look at "LUT as Memory" count and any timing failures on paths
//    sourced from weights_flat. Also look at routing congestion around
//    the fc1/fc2 instance regions on the floorplan.
//
//    Mitigation if needed: replace weights_flat with explicit per-lane
//    ROMs using generate + (* rom_style = "distributed" *):
//      generate
//        for (genvar p = 0; p < PARALLELISM; p++) begin : gen_weight_bank
//          (* rom_style = "distributed" *)
//          logic signed [DATA_WIDTH-1:0] weights_lane [0:LANE_W-1];
//          initial $readmemh(per_lane_file(p), weights_lane);
//        end
//      endgenerate
//    For P = NUM_OUTPUTS this collapses cleanly (each lane owns one
//    neuron's weights, no muxing needed). For P < NUM_OUTPUTS each bank
//    holds NUM_OUTPUTS/P neurons' weights addressed by batch_idx.
//
// 2. inputs_latched broadcast (low concern)
//    inputs_latched[input_idx] is read by all P lanes per cycle. Fan-out
//    of P=32 on a 16-bit register is well below Vivado's default
//    MAX_FANOUT threshold; tool auto-replicates if placement demands it.
//    No action needed.
//
// 3. MAC critical-path pipelining (defensive)
//    Each lane's combinational chain is: address-gen → weight read →
//    16×16 multiply → sign-extend → 39-bit add → accum register. At -1
//    speed grade this should fit in 10 ns, but it is the timing-critical
//    path. If timing closure is tight after first synth (< 1 ns WNS),
//    register the multiplier output (mac_term[p]) to give Vivado room
//    to retime into the DSP48E1's internal M-register. Costs +1 cycle
//    of latency per layer (need to drain the pipeline before EMIT), but
//    the FSM is straightforward to extend. See the retiming notes near
//    the COMPUTING state below.

module linear_layer #(
    parameter NUM_INPUTS  = 4,
    parameter NUM_OUTPUTS = 16,
    parameter PARALLELISM = NUM_OUTPUTS,  // P: number of MAC engines (DSP count)
    parameter DATA_WIDTH  = 16,
    parameter OUTPUT_WIDTH = DATA_WIDTH,
    parameter FRAC_BITS   = 13,
    parameter WEIGHTS_FILE = "weights.mem",
    parameter BIAS_FILE    = "bias.mem"
) (
    input  wire clk,
    input  wire reset,
    input  wire start,
    input  wire signed [DATA_WIDTH-1:0]   inputs [0:NUM_INPUTS-1],
    output logic signed [OUTPUT_WIDTH-1:0] outputs [0:NUM_OUTPUTS-1],
    // Sticky saturation flags. Cleared on 'start'; OR'd across all
    // outputs in the run. Valid alongside 'done' (and persist until
    // the next start).
    output logic sat_pos,
    output logic sat_neg,
    output logic done
);

    // =========================================================================
    // Derived parameters
    // =========================================================================
    localparam integer NUM_BATCHES          = (NUM_OUTPUTS + PARALLELISM - 1) / PARALLELISM;
    localparam integer BATCH_IDX_WIDTH      = (NUM_BATCHES > 1) ? $clog2(NUM_BATCHES) : 1;
    localparam integer INPUT_IDX_WIDTH      = (NUM_INPUTS  > 1) ? $clog2(NUM_INPUTS)  : 1;
    localparam integer NEURON_IDX_WIDTH     = (NUM_OUTPUTS > 1) ? $clog2(NUM_OUTPUTS) : 1;
    // Wide-enough index to hold batch_idx*P + p before the < NUM_OUTPUTS check.
    localparam integer GLOBAL_IDX_WIDTH     = NEURON_IDX_WIDTH + 1;
    localparam integer ACCUM_WIDTH          = 2 * DATA_WIDTH + $clog2(NUM_INPUTS + 1);
    // Address of any weight in the flat array.
    localparam integer WEIGHT_ADDR_WIDTH    = $clog2(NUM_OUTPUTS * NUM_INPUTS + 1);

    // Saturation bounds in OUTPUT_WIDTH range
    localparam signed [OUTPUT_WIDTH-1:0] MAX_VAL = {1'b0, {(OUTPUT_WIDTH-1){1'b1}}};
    localparam signed [OUTPUT_WIDTH-1:0] MIN_VAL = {1'b1, {(OUTPUT_WIDTH-1){1'b0}}};

    // =========================================================================
    // Weight / bias storage
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] weights_flat [0:NUM_OUTPUTS*NUM_INPUTS-1];
    logic signed [DATA_WIDTH-1:0] biases       [0:NUM_OUTPUTS-1];

    initial begin
        $readmemh(WEIGHTS_FILE, weights_flat);
        $readmemh(BIAS_FILE, biases);
    end

    // =========================================================================
    // FSM and counters
    // =========================================================================
    typedef enum logic [1:0] {
        IDLE,
        COMPUTING,
        EMIT
    } state_t;

    state_t state;
    logic [BATCH_IDX_WIDTH-1:0] batch_idx;
    logic [INPUT_IDX_WIDTH-1:0] input_idx;

    // Latched inputs (held stable across the entire run)
    logic signed [DATA_WIDTH-1:0] inputs_latched [0:NUM_INPUTS-1];

    // =========================================================================
    // Per-lane MAC datapath
    // =========================================================================
    // P parallel accumulators, one per active output in the current batch.
    logic signed [ACCUM_WIDTH-1:0] accums [0:PARALLELISM-1];

    // Combinational per-lane index calculations: which global neuron each
    // lane is working on this batch, and whether that lane is active.
    logic [GLOBAL_IDX_WIDTH-1:0] lane_global_idx [0:PARALLELISM-1];
    logic                        lane_active     [0:PARALLELISM-1];
    logic [NEURON_IDX_WIDTH-1:0] lane_neuron_idx [0:PARALLELISM-1];
    logic [WEIGHT_ADDR_WIDTH-1:0] lane_weight_addr [0:PARALLELISM-1];

    always_comb begin
        for (int p = 0; p < PARALLELISM; p++) begin
            lane_global_idx[p]  = GLOBAL_IDX_WIDTH'({1'b0, batch_idx} * PARALLELISM + p);
            lane_active[p]      = (lane_global_idx[p] < NUM_OUTPUTS);
            lane_neuron_idx[p]  = lane_global_idx[p][NEURON_IDX_WIDTH-1:0];
            lane_weight_addr[p] = WEIGHT_ADDR_WIDTH'(
                lane_neuron_idx[p] * NUM_INPUTS + input_idx);
        end
    end

    // Per-lane MAC term and next-accumulator value (combinational).
    logic signed [2*DATA_WIDTH-1:0] mac_term      [0:PARALLELISM-1];
    logic signed [ACCUM_WIDTH-1:0]  mac_term_ext  [0:PARALLELISM-1];
    logic signed [ACCUM_WIDTH-1:0]  accum_next    [0:PARALLELISM-1];

    always_comb begin
        for (int p = 0; p < PARALLELISM; p++) begin
            if (lane_active[p]) begin
                mac_term[p] = inputs_latched[input_idx] * weights_flat[lane_weight_addr[p]];
            end else begin
                mac_term[p] = '0;
            end
            mac_term_ext[p] = {{(ACCUM_WIDTH - 2*DATA_WIDTH){mac_term[p][2*DATA_WIDTH-1]}},
                               mac_term[p]};
            accum_next[p]   = accums[p] + mac_term_ext[p];
        end
    end

    // =========================================================================
    // Per-lane saturate + bias-add (combinational; consumed in EMIT)
    // =========================================================================
    logic signed [ACCUM_WIDTH-1:0]  scaled       [0:PARALLELISM-1];
    logic signed [OUTPUT_WIDTH-1:0] saturated    [0:PARALLELISM-1];
    logic                            sat_pos_lane [0:PARALLELISM-1];
    logic                            sat_neg_lane [0:PARALLELISM-1];

    always_comb begin
        for (int p = 0; p < PARALLELISM; p++) begin
            // bias is indexed by the lane's current neuron (which is only
            // meaningful when lane is active; otherwise zero-default).
            scaled[p] = (accums[p] >>> FRAC_BITS) +
                        $signed({{(ACCUM_WIDTH-DATA_WIDTH){biases[lane_neuron_idx[p]][DATA_WIDTH-1]}},
                                  biases[lane_neuron_idx[p]]});
            sat_pos_lane[p] = 1'b0;
            sat_neg_lane[p] = 1'b0;

            if (!lane_active[p]) begin
                saturated[p] = '0;
            end else if (scaled[p] > $signed({{(ACCUM_WIDTH-OUTPUT_WIDTH){1'b0}}, MAX_VAL})) begin
                saturated[p]    = MAX_VAL;
                sat_pos_lane[p] = 1'b1;
            end else if (scaled[p] < $signed({{(ACCUM_WIDTH-OUTPUT_WIDTH){1'b1}}, MIN_VAL})) begin
                saturated[p]    = MIN_VAL;
                sat_neg_lane[p] = 1'b1;
            end else begin
                saturated[p] = scaled[p][OUTPUT_WIDTH-1:0];
            end
        end
    end

    // =========================================================================
    // FSM and registered datapath
    // =========================================================================
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            state     <= IDLE;
            batch_idx <= '0;
            input_idx <= '0;
            done      <= 1'b0;
            sat_pos   <= 1'b0;
            sat_neg   <= 1'b0;
            for (int p = 0; p < PARALLELISM; p++) accums[p] <= '0;
            for (int i = 0; i < NUM_INPUTS;  i++) inputs_latched[i] <= '0;
            for (int n = 0; n < NUM_OUTPUTS; n++) outputs[n] <= '0;
        end else begin
            unique case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        for (int i = 0; i < NUM_INPUTS; i++) begin
                            inputs_latched[i] <= inputs[i];
                        end
                        batch_idx <= '0;
                        input_idx <= '0;
                        sat_pos   <= 1'b0;  // clear sticky flags for new run
                        sat_neg   <= 1'b0;
                        for (int p = 0; p < PARALLELISM; p++) accums[p] <= '0;
                        state     <= COMPUTING;
                    end
                end

                COMPUTING: begin
                    // One MAC per cycle, per lane, accumulated into accums[p].
                    // After the final MAC we transition to EMIT for one cycle
                    // to saturate, bias-add, and write the batch into outputs[].
                    for (int p = 0; p < PARALLELISM; p++) begin
                        accums[p] <= accum_next[p];
                    end
                    if (input_idx == INPUT_IDX_WIDTH'(NUM_INPUTS - 1)) begin
                        state <= EMIT;
                    end else begin
                        input_idx <= input_idx + 1'b1;
                    end
                end

                EMIT: begin
                    // Register P saturated outputs into outputs[].
                    // Inactive lanes (last batch underfull) leave the
                    // corresponding outputs[] slots untouched.
                    for (int p = 0; p < PARALLELISM; p++) begin
                        if (lane_active[p]) begin
                            outputs[lane_neuron_idx[p]] <= saturated[p];
                            if (sat_pos_lane[p]) sat_pos <= 1'b1;
                            if (sat_neg_lane[p]) sat_neg <= 1'b1;
                        end
                    end

                    if (batch_idx == BATCH_IDX_WIDTH'(NUM_BATCHES - 1)) begin
                        done  <= 1'b1;
                        state <= IDLE;
                    end else begin
                        batch_idx <= batch_idx + 1'b1;
                        input_idx <= '0;
                        for (int p = 0; p < PARALLELISM; p++) accums[p] <= '0;
                        state     <= COMPUTING;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
