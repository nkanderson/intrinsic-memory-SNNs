// Linear Layer Module (Serial MAC, one multiply per cycle) — ARCHIVED REFERENCE
//
// This is the legacy serial implementation, preserved as the equivalence
// reference for the parallel rewrite of linear_layer.sv. It is not used in
// production builds; only the equivalence testbench and any explicit
// regression target should reference it. Module contents are an exact copy
// of the original linear_layer.sv with the module name suffixed to avoid
// elaboration collisions when both versions are instantiated together.
//
// Implements a fully connected layer: output = weights * input + bias
// Equivalent to nn.Linear in PyTorch.
//
// Architecture: one input is multiplied by its weight per clock cycle and
// accumulated into a register. After NUM_INPUTS cycles the per-neuron
// dot-product is complete and the saturated, biased result is emitted on
// the next cycle. This replaces an earlier design that summed all
// NUM_INPUTS products combinationally each cycle — fine for small layers
// (NUM_INPUTS<=8) but for HL1_SIZE=64 it inferred a 64-deep DSP cascade
// (~109 ns combinational delay), which made closing 100 MHz timing
// impossible. The serial form maps to a single DSP48E1 with internal
// pipeline registers, easily hits >250 MHz Fmax, and adds only NUM_INPUTS
// extra cycles per output neuron — well below UART round-trip time.
//
// Weights are stored in row-major order in a flattened array:
//   weights_flat[n*NUM_INPUTS + i] = weight for output n, input i
//
// Fixed-point format: QS2.13 (16-bit signed, 2 integer bits, 13 fractional bits)
//
// Timing (output_valid handshake; no cycle count promises beyond start/done):
//   - Cycle 0:                          assert 'start' for one cycle with valid inputs
//   - Cycle 1:                          inputs latched, COMPUTING for neuron 0 begins
//   - Cycle 1+NUM_INPUTS:               last MAC for neuron 0 completes
//   - Cycle 2+NUM_INPUTS:               output_valid=1, output_idx=0
//   - Cycle 2 + N*(NUM_INPUTS+1):       output_valid=1, output_idx=N
//   - 'done' asserts together with the final output_valid pulse
//   - Total latency: 1 + NUM_OUTPUTS * (NUM_INPUTS + 1) cycles from start to done
//
// Consumers should drive 'start' for one cycle and then sample on
// 'output_valid' rather than counting cycles.

module linear_layer_serial #(
    parameter NUM_INPUTS = 4,
    parameter NUM_OUTPUTS = 16,
    parameter DATA_WIDTH = 16,
    parameter OUTPUT_WIDTH = DATA_WIDTH,
    parameter FRAC_BITS = 13,
    parameter WEIGHTS_FILE = "weights.mem",
    parameter BIAS_FILE = "bias.mem"
) (
    input wire clk,
    input wire reset,
    input wire start,
    input wire signed [DATA_WIDTH-1:0] inputs [0:NUM_INPUTS-1],
    output logic signed [OUTPUT_WIDTH-1:0] output_current, // One current per cycle
    output logic [$clog2(NUM_OUTPUTS)-1:0] output_idx,   // Which neuron (0 to NUM_OUTPUTS-1)
    output logic output_valid,                            // Current output is valid
    // TODO: Add debug flags so these signals are only added in debug mode
    output logic sat_pos,                                 // Current output saturated high
    output logic sat_neg,                                 // Current output saturated low
    output logic done
);

    // Index widths
    localparam IDX_WIDTH       = (NUM_OUTPUTS > 1) ? $clog2(NUM_OUTPUTS) : 1;
    localparam INPUT_IDX_WIDTH = (NUM_INPUTS  > 1) ? $clog2(NUM_INPUTS)  : 1;

    // Flattened weights array: row-major order
    logic signed [DATA_WIDTH-1:0] weights_flat [0:NUM_OUTPUTS*NUM_INPUTS-1];
    logic signed [DATA_WIDTH-1:0] biases [0:NUM_OUTPUTS-1];

    // Load weights and biases from memory files
    initial begin
        $readmemh(WEIGHTS_FILE, weights_flat);
        $readmemh(BIAS_FILE, biases);
    end

    // Accumulator width: multiplication doubles bits, add extra for sum of NUM_INPUTS terms
    localparam ACCUM_WIDTH = 2 * DATA_WIDTH + $clog2(NUM_INPUTS + 1);

    // Saturation bounds
    localparam signed [OUTPUT_WIDTH-1:0] MAX_VAL = {1'b0, {(OUTPUT_WIDTH-1){1'b1}}};
    localparam signed [OUTPUT_WIDTH-1:0] MIN_VAL = {1'b1, {(OUTPUT_WIDTH-1){1'b0}}};

    // FSM
    typedef enum logic [1:0] {
        IDLE,
        COMPUTING,
        EMIT
    } state_t;

    state_t state;
    logic [IDX_WIDTH-1:0]       neuron_idx;  // Counter for current output neuron
    logic [INPUT_IDX_WIDTH-1:0] input_idx;   // Counter for current input within the neuron's dot product

    // Latched inputs (held stable during computation)
    logic signed [DATA_WIDTH-1:0] inputs_latched [0:NUM_INPUTS-1];

    // Per-cycle MAC term and running accumulator
    logic signed [2*DATA_WIDTH-1:0] mac_term;
    logic signed [ACCUM_WIDTH-1:0]  mac_term_ext;
    logic signed [ACCUM_WIDTH-1:0]  accum;
    logic signed [ACCUM_WIDTH-1:0]  accum_next;

    always_comb begin
        mac_term     = inputs_latched[input_idx] * weights_flat[neuron_idx * NUM_INPUTS + input_idx];
        mac_term_ext = {{(ACCUM_WIDTH - 2*DATA_WIDTH){mac_term[2*DATA_WIDTH-1]}}, mac_term};
        accum_next   = accum + mac_term_ext;
    end

    // Saturation pipeline (combinational; consumed in EMIT). Operates on the
    // committed `accum` value, which holds the full dot-product after the
    // last MAC of the current neuron has been registered.
    logic signed [ACCUM_WIDTH-1:0]  scaled;
    logic signed [OUTPUT_WIDTH-1:0] saturated;
    logic                           sat_pos_comb;
    logic                           sat_neg_comb;

    always_comb begin
        scaled = (accum >>> FRAC_BITS) +
                 $signed({{(ACCUM_WIDTH-DATA_WIDTH){biases[neuron_idx][DATA_WIDTH-1]}},
                          biases[neuron_idx]});
        sat_pos_comb = 1'b0;
        sat_neg_comb = 1'b0;

        if (scaled > $signed({{(ACCUM_WIDTH-OUTPUT_WIDTH){1'b0}}, MAX_VAL})) begin
            saturated = MAX_VAL;
            sat_pos_comb = 1'b1;
        end else if (scaled < $signed({{(ACCUM_WIDTH-OUTPUT_WIDTH){1'b1}}, MIN_VAL})) begin
            saturated = MIN_VAL;
            sat_neg_comb = 1'b1;
        end else begin
            saturated = scaled[OUTPUT_WIDTH-1:0];
        end
    end

    // FSM and datapath registers
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            state          <= IDLE;
            neuron_idx     <= '0;
            input_idx      <= '0;
            accum          <= '0;
            done           <= 1'b0;
            output_valid   <= 1'b0;
            sat_pos        <= 1'b0;
            sat_neg        <= 1'b0;
            output_current <= '0;
            output_idx     <= '0;
            for (int i = 0; i < NUM_INPUTS; i++) begin
                inputs_latched[i] <= '0;
            end
        end else begin
            unique case (state)
                IDLE: begin
                    done         <= 1'b0;
                    output_valid <= 1'b0;
                    sat_pos      <= 1'b0;
                    sat_neg      <= 1'b0;
                    if (start) begin
                        for (int i = 0; i < NUM_INPUTS; i++) begin
                            inputs_latched[i] <= inputs[i];
                        end
                        neuron_idx <= '0;
                        input_idx  <= '0;
                        accum      <= '0;
                        state      <= COMPUTING;
                    end
                end

                COMPUTING: begin
                    // One MAC per cycle, accumulated into `accum`. After the
                    // final MAC we transition to EMIT for one cycle to
                    // saturate, bias-add, and pulse output_valid.
                    output_valid <= 1'b0;
                    accum        <= accum_next;
                    if (input_idx == INPUT_IDX_WIDTH'(NUM_INPUTS - 1)) begin
                        state <= EMIT;
                    end else begin
                        input_idx <= input_idx + 1'b1;
                    end
                end

                EMIT: begin
                    // `accum` now holds the full dot-product for neuron_idx.
                    // The combinational saturation/bias pipeline is settled,
                    // so we just register the result.
                    output_current <= saturated;
                    output_idx     <= neuron_idx;
                    output_valid   <= 1'b1;
                    sat_pos        <= sat_pos_comb;
                    sat_neg        <= sat_neg_comb;

                    if (neuron_idx == IDX_WIDTH'(NUM_OUTPUTS - 1)) begin
                        done  <= 1'b1;
                        state <= IDLE;
                    end else begin
                        neuron_idx <= neuron_idx + 1'b1;
                        input_idx  <= '0;
                        accum      <= '0;
                        state      <= COMPUTING;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
