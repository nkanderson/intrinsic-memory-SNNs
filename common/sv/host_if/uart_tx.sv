module uart_tx #(
    parameter integer CLOCK_HZ = 100_000_000,
    parameter integer BAUD = 115_200
) (
    input  wire clk,
    input  wire reset,
    input  wire start,
    input  wire [7:0] data_in,
    output logic tx,
    output logic busy,
    output logic done
);

    localparam integer CLKS_PER_BIT = CLOCK_HZ / BAUD;
    localparam integer CTR_W = (CLKS_PER_BIT > 1) ? $clog2(CLKS_PER_BIT) : 1;

    typedef enum logic [2:0] {
        TX_IDLE,
        TX_START,
        TX_DATA,
        TX_STOP
    } tx_state_t;

    tx_state_t state;
    logic [CTR_W-1:0] clk_count;
    logic [2:0] bit_idx;
    logic [7:0] data_latched;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= TX_IDLE;
            clk_count <= '0;
            bit_idx <= '0;
            data_latched <= '0;
            tx <= 1'b1;
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                TX_IDLE: begin
                    tx <= 1'b1;
                    busy <= 1'b0;
                    clk_count <= '0;
                    bit_idx <= '0;
                    if (start) begin
                        data_latched <= data_in;
                        busy <= 1'b1;
                        state <= TX_START;
                    end
                end

                TX_START: begin
                    tx <= 1'b0;
                    if (clk_count == CLKS_PER_BIT-1) begin
                        clk_count <= '0;
                        state <= TX_DATA;
                    end else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                TX_DATA: begin
                    tx <= data_latched[bit_idx];
                    if (clk_count == CLKS_PER_BIT-1) begin
                        clk_count <= '0;
                        if (bit_idx == 3'd7) begin
                            bit_idx <= '0;
                            state <= TX_STOP;
                        end else begin
                            bit_idx <= bit_idx + 1'b1;
                        end
                    end else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                TX_STOP: begin
                    tx <= 1'b1;
                    if (clk_count == CLKS_PER_BIT-1) begin
                        clk_count <= '0;
                        busy <= 1'b0;
                        done <= 1'b1;
                        state <= TX_IDLE;
                    end else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                default: begin
                    state <= TX_IDLE;
                end
            endcase
        end
    end

endmodule
