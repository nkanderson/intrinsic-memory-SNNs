module uart_rx #(
    parameter integer CLOCK_HZ = 100_000_000,
    parameter integer BAUD = 115_200
) (
    input  wire clk,
    input  wire reset,
    input  wire rx,
    output logic [7:0] data_out,
    output logic data_valid
);

    localparam integer CLKS_PER_BIT = CLOCK_HZ / BAUD;
    localparam integer CTR_W = (CLKS_PER_BIT > 1) ? $clog2(CLKS_PER_BIT) : 1;

    typedef enum logic [2:0] {
        RX_IDLE,
        RX_START,
        RX_DATA,
        RX_STOP
    } rx_state_t;

    rx_state_t state;
    logic [CTR_W-1:0] clk_count;
    logic [2:0] bit_idx;
    logic [7:0] data_shift;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= RX_IDLE;
            clk_count <= '0;
            bit_idx <= '0;
            data_shift <= '0;
            data_out <= '0;
            data_valid <= 1'b0;
        end else begin
            data_valid <= 1'b0;
            case (state)
                RX_IDLE: begin
                    clk_count <= '0;
                    bit_idx <= '0;
                    if (rx == 1'b0) begin
                        state <= RX_START;
                    end
                end

                RX_START: begin
                    if (clk_count == (CLKS_PER_BIT/2)-1) begin
                        clk_count <= '0;
                        if (rx == 1'b0) begin
                            state <= RX_DATA;
                        end else begin
                            state <= RX_IDLE;
                        end
                    end else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                RX_DATA: begin
                    if (clk_count == CLKS_PER_BIT-1) begin
                        clk_count <= '0;
                        data_shift[bit_idx] <= rx;
                        if (bit_idx == 3'd7) begin
                            bit_idx <= '0;
                            state <= RX_STOP;
                        end else begin
                            bit_idx <= bit_idx + 1'b1;
                        end
                    end else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                RX_STOP: begin
                    if (clk_count == CLKS_PER_BIT-1) begin
                        clk_count <= '0;
                        state <= RX_IDLE;
                        if (rx == 1'b1) begin
                            data_out <= data_shift;
                            data_valid <= 1'b1;
                        end
                    end else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                default: begin
                    state <= RX_IDLE;
                end
            endcase
        end
    end

endmodule
