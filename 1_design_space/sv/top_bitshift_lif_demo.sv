module top_bitshift_lif_demo #(
    // Guard-bit sweep knobs for synthesis studies.
    // Defaults match validated precision behavior.
    parameter int BITSHIFT_ACCUM_GUARD_BITS = 5,
    parameter int BITSHIFT_NUMERATOR_GUARD_BITS = 1
) (
    input  wire        CLK100MHZ,
    input  wire [1:0]  sw,
    output logic [15:0] LED,
    output logic [6:0] seg,
    output logic       dp,
    output logic [7:0] an
);
    localparam int DATA_WIDTH = 16;
    localparam int MEMBRANE_WIDTH = 24;
    localparam int THRESHOLD = 8192;
    localparam int RUN_STEPS = 256;
    // Keep demo resource use modest to improve place/route success.
    // NOTE: This is a demo setting and does not reflect final model history depth.
    localparam int BITSHIFT_HISTORY_LENGTH = 32;

    localparam int STEP_DIV = 1_000_000;   // 100Hz update pulse from 100MHz clock
    localparam int STEP_CNT_W = $clog2(STEP_DIV);
    localparam int RUN_CNT_W = $clog2(RUN_STEPS + 1);

    localparam int SEG_SCAN_DIV = 25_000;  // 4kHz scan -> 1kHz per active digit
    localparam int SEG_SCAN_CNT_W = $clog2(SEG_SCAN_DIV);

    logic [STEP_CNT_W-1:0] step_counter;
    logic step_pulse;

    logic signed [DATA_WIDTH-1:0] current_in;
    logic spike;
    logic signed [MEMBRANE_WIDTH-1:0] membrane;
    logic neuron_busy;
    logic neuron_output_valid;

    logic [15:0] spike_count;
    logic [RUN_CNT_W-1:0] step_count;
    logic run_done;

    logic [SEG_SCAN_CNT_W-1:0] seg_scan_counter;
    logic [1:0] seg_scan_sel;
    logic [3:0] seg_nibble;

    // sw[0] = synchronous clear/reset/restart
    // sw[1] = current profile select
    //   0: lower drive (near threshold for bitshift model)
    //   1: higher drive (intended to produce visible spiking)
    always_comb begin
        if (sw[1]) begin
            current_in = 16'sd20000;
        end else begin
            current_in = 16'sd10000;
        end
    end

    always_ff @(posedge CLK100MHZ) begin
        if (sw[0]) begin
            step_counter <= '0;
            step_pulse <= 1'b0;
            step_count <= '0;
            run_done <= 1'b0;
        end else if (step_counter == STEP_CNT_W'(STEP_DIV - 1)) begin
            step_counter <= '0;
            step_pulse <= ~run_done;
        end else begin
            step_counter <= step_counter + 1'b1;
            step_pulse <= 1'b0;
        end

        if (!sw[0] && neuron_output_valid && !run_done) begin
            if (step_count == RUN_CNT_W'(RUN_STEPS - 1)) begin
                run_done <= 1'b1;
            end
            step_count <= step_count + 1'b1;
        end
    end

    bitshift_lif #(
        .THRESHOLD(THRESHOLD),
        .DATA_WIDTH(DATA_WIDTH),
        .MEMBRANE_WIDTH(MEMBRANE_WIDTH),
        .HISTORY_LENGTH(BITSHIFT_HISTORY_LENGTH),
        .ACCUM_GUARD_BITS(BITSHIFT_ACCUM_GUARD_BITS),
        .NUMERATOR_GUARD_BITS(BITSHIFT_NUMERATOR_GUARD_BITS)
    ) u_neuron (
        .clk(CLK100MHZ),
        .reset(1'b0),
        .clear(sw[0]),
        .enable(step_pulse),
        .current(current_in),
        .spike_out(spike),
        .membrane_out(membrane),
        .busy(neuron_busy),
        .output_valid(neuron_output_valid)
    );

    always_ff @(posedge CLK100MHZ) begin
        if (sw[0]) begin
            spike_count <= '0;
        end else if (neuron_output_valid && spike) begin
            spike_count <= spike_count + 1'b1;
        end
    end

    always_ff @(posedge CLK100MHZ) begin
        LED[15] <= run_done;
        LED[14] <= neuron_output_valid;
        LED[13:0] <= spike_count[13:0];
    end

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
        an = 8'b1111_1111;
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

    assign dp = 1'b1;

endmodule
