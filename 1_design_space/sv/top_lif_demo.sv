module top_lif_demo (
    input  wire        CLK100MHZ,
    // switch inputs
    input  wire [1:0]  sw,
    output logic [15:0] LED,
    output logic [6:0] seg,
    // decimal points (active-low)
    output logic       dp,
    // anode enables for each digit (active-low)
    output logic [7:0] an
);
    localparam int DATA_WIDTH = 16;
    localparam int MEMBRANE_WIDTH = 24;
    localparam int THRESHOLD = 8192;
    localparam int BETA = 115;
    localparam int RUN_STEPS = 256;

    // Expected results after RUN_STEPS updates (shown on 7-seg in hex):
    //   sw[1] = 0 -> current_in = 5000  (sub-threshold each step)
    //                With leak + reset-delay dynamics this settles to ~every-other-step spiking,
    //                so expected spike_count ~= 128 = 0x0080.
    //   sw[1] = 1 -> current_in = 9000  (supra-threshold each step)
    //                Neuron spikes every enabled update, so expected spike_count = 256 = 0x0100.
    // During normal operation LED[15] goes high when counting is complete (run_done=1).

    // 100 MHz / 1,000,000 = 100 Hz neuron update pulse
    localparam int STEP_DIV = 1_000_000;
    localparam int STEP_CNT_W = $clog2(STEP_DIV);
    localparam int RUN_CNT_W = $clog2(RUN_STEPS + 1);

    // 100 MHz / 25,000 = 4 kHz digit scan (1 kHz per digit for 4 digits)
    localparam int SEG_SCAN_DIV = 25_000;
    localparam int SEG_SCAN_CNT_W = $clog2(SEG_SCAN_DIV);

    logic [STEP_CNT_W-1:0] step_counter;
    logic step_pulse;
    logic step_sample_pending;

    logic signed [DATA_WIDTH-1:0] current_in;
    logic spike;
    logic signed [MEMBRANE_WIDTH-1:0] membrane;

    logic [15:0] spike_count;
    logic [RUN_CNT_W-1:0] step_count;
    logic run_done;

    logic [SEG_SCAN_CNT_W-1:0] seg_scan_counter;
    logic [1:0] seg_scan_sel;
    logic [3:0] seg_nibble;

    // Simple hardcoded stimulus selection
    // sw[0] = synchronous clear/reset/restart
    // sw[1] = current profile select (0: sub-threshold, 1: supra-threshold)
    always_comb begin
        if (sw[1]) begin
            current_in = 16'sd9000;
        end else begin
            current_in = 16'sd5000;
        end
    end

    // Update pulse generator
    always_ff @(posedge CLK100MHZ) begin
        if (sw[0]) begin
            step_counter <= '0;
            step_pulse <= 1'b0;
            step_count <= '0;
            run_done <= 1'b0;
            step_sample_pending <= 1'b0;
        end else if (step_counter == STEP_CNT_W'(STEP_DIV - 1)) begin
            step_counter <= '0;
            step_pulse <= ~run_done;

            if (!run_done) begin
                step_sample_pending <= 1'b1;
                if (step_count == RUN_CNT_W'(RUN_STEPS - 1)) begin
                    run_done <= 1'b1;
                end
                step_count <= step_count + 1'b1;
            end else begin
                step_sample_pending <= 1'b0;
            end
        end else begin
            step_counter <= step_counter + 1'b1;
            step_pulse <= 1'b0;
            step_sample_pending <= 1'b0;
        end
    end

    lif #(
        .THRESHOLD(THRESHOLD),
        .BETA(BETA),
        .DATA_WIDTH(DATA_WIDTH),
        .MEMBRANE_WIDTH(MEMBRANE_WIDTH)
    ) u_lif (
        .clk(CLK100MHZ),
        .reset(1'b0),
        .clear(sw[0]),
        .enable(step_pulse),
        .current(current_in),
        .spike_out(spike),
        .membrane_out(membrane)
    );

    // Spike counter over fixed runtime window (clear/restart with sw[0])
    // Note: spike updates in lif on the same clock edge as step_pulse, so use
    // step_sample_pending to sample spike one cycle later.
    always_ff @(posedge CLK100MHZ) begin
        if (sw[0]) begin
            spike_count <= '0;
        end else if (step_sample_pending && spike) begin
            spike_count <= spike_count + 1'b1;
        end
    end

    // LEDs for quick observability
    // LED[15] = run done
    // LED[14] = sampled spike pulse
    // LED[13:0] = low bits of spike count
    always_ff @(posedge CLK100MHZ) begin
        LED[15] <= run_done;
        LED[14] <= (step_sample_pending && spike);
        LED[13:0] <= spike_count[13:0];
    end

    // 4-digit seven-segment multiplexing to display full 16-bit spike count
    always_ff @(posedge CLK100MHZ) begin
        if (sw[0]) begin
            seg_scan_counter <= '0;
            seg_scan_sel <= '0;
        end else if (seg_scan_counter == SEG_SCAN_CNT_W'(SEG_SCAN_DIV - 1)) begin
            seg_scan_counter <= '0;
            seg_scan_sel <= seg_scan_sel + 1'b1;
        end else begin
            seg_scan_counter <= seg_scan_counter + 1'b1;
        end
    end

    always_comb begin
        an = 8'b1111_1111;  // all off by default (active-low)
        unique case (seg_scan_sel)
            2'd0: begin
                an = 8'b1111_1110;
                seg_nibble = spike_count[3:0];
            end
            2'd1: begin
                an = 8'b1111_1101;
                seg_nibble = spike_count[7:4];
            end
            2'd2: begin
                an = 8'b1111_1011;
                seg_nibble = spike_count[11:8];
            end
            default: begin
                an = 8'b1111_0111;
                seg_nibble = spike_count[15:12];
            end
        endcase
    end

    hex7seg u_hex (
        .hex(seg_nibble),
        .seg(seg)
    );

    assign dp = 1'b1;          // Off (active-low)

endmodule
