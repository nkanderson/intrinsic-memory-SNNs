// Nexys A7-100T board top for the lif-32-32 SNN configuration.
//
// Pins (see constraints/nexys_a7_general.xdc):
//   CLK100MHZ      : E3, 100 MHz system clock
//   CPU_RESETN     : C12, push-button (active-low). Inverted internally.
//   UART_TXD_IN    : C4, USB-UART data INTO the FPGA  (FT2232 -> RX core)
//   UART_RXD_OUT   : D4, USB-UART data OUT of the FPGA (TX core -> FT2232)
//   LED[15:0]      : status indicators, see assignments below
//
// LED map for first bring-up:
//   LED[0]   = busy        (high while an inference is in flight)
//   LED[1]   = action_q    (latched output of the last inference)
//   LED[14:2] = 0
//   LED[15]  = heartbeat   (~1.5 Hz blink — alive + clock running)
//
// Hardware-acceleration parameters mirror the cocotb Makefile target
// `cp_integration_lif-32-32` so the FPGA matches the simulated reference.

module board_top_lif_32_32 (
    input  wire        CLK100MHZ,
    input  wire        CPU_RESETN,
    input  wire        UART_TXD_IN,
    output wire        UART_RXD_OUT,
    output wire [15:0] LED
);

    // CPU_RESETN is active-low; invert to active-high for the internal
    // logic. No power-on reset stretcher is needed — Vivado initializes
    // every FF to its INIT value (0) at bitstream configuration, so the
    // design comes up in a known state without an explicit reset edge.
    wire reset_int;
    assign reset_int = ~CPU_RESETN;

    // --- Accelerator wrapper ------------------------------------------
    // Weight .mem files come from common/sv/cocotb/tests/weights/lif-32-16/;
    // add that directory to the Vivado project's source search path so
    // $readmemh resolves the basenames below.
    wire        accel_busy;
    wire        accel_done;
    wire        accel_action;

    top_uart_accel_wrapper #(
        .MODEL_TYPE       (0),                  // standard LIF
        .CLOCK_HZ         (100_000_000),
        .BAUD             (921_600),
        .NUM_INPUTS       (4),
        .HL1_SIZE         (32),
        .HL2_SIZE         (32),
        .NUM_ACTIONS      (2),
        .NUM_TIMESTEPS    (20),
        // BATCH_SIZE=4 (was 1): the q_accumulator now pipelines mem_sel /
        // multiply / accumulate stages and the neuron_membrane_buffer has a
        // sync read, so the membrane → DSP critical path is short enough that
        // BATCH_SIZE=4 closes timing at 100 MHz with HL2=32. Reduces inference
        // cycles per timestep from 32+ to 8+ (4 batches instead of 32).
        .Q_BATCH_SIZE     (4),
        .FC2_OUTPUT_WIDTH (24),
        .FC1_WEIGHTS_FILE ("fc1_weights.mem"),
        .FC1_BIAS_FILE    ("fc1_bias.mem"),
        .FC2_WEIGHTS_FILE ("fc2_weights.mem"),
        .FC2_BIAS_FILE    ("fc2_bias.mem"),
        .FC_OUT_WEIGHTS_FILE ("fc_out_weights.mem"),
        .FC_OUT_BIAS_FILE    ("fc_out_bias.mem")
    ) u_accel (
        .clk           (CLK100MHZ),
        .reset         (reset_int),
        .uart_rx_i     (UART_TXD_IN),
        .uart_tx_o     (UART_RXD_OUT),
        .status_busy   (accel_busy),
        .status_done   (accel_done),
        .status_action (accel_action)
    );

    // --- LED status -----------------------------------------------------
    // Latch the most recent action on each `done` so LED[1] holds the
    // last value rather than flashing for a single clock.
    logic action_q = 1'b0;
    always_ff @(posedge CLK100MHZ or posedge reset_int) begin
        if (reset_int) begin
            action_q <= 1'b0;
        end else if (accel_done) begin
            action_q <= accel_action;
        end
    end

    // ~1.5 Hz heartbeat: 100 MHz / 2^25 ~= 3 Hz toggle, 1.5 Hz on/off.
    logic [24:0] heartbeat_count = 25'd0;
    always_ff @(posedge CLK100MHZ) begin
        heartbeat_count <= heartbeat_count + 25'd1;
    end

    assign LED[0]    = accel_busy;
    assign LED[1]    = action_q;
    assign LED[14:2] = 13'd0;
    assign LED[15]   = heartbeat_count[24];

endmodule
