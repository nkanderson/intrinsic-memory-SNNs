// Simulation-only wrapper that ties uart_tx directly to uart_rx so cocotb
// can exercise byte fidelity through the full bit-serial path.

module tb_uart_loopback #(
    parameter integer CLOCK_HZ = 100_000_000,
    parameter integer BAUD = 115_200
) (
    input  wire clk,
    input  wire reset,
    input  wire [7:0] tx_data_in,
    input  wire tx_start,
    output wire tx_busy,
    output wire tx_done,
    output wire [7:0] rx_data_out,
    output wire rx_data_valid,
    output wire serial_line
);

    wire serial;

    uart_tx #(
        .CLOCK_HZ(CLOCK_HZ),
        .BAUD(BAUD)
    ) u_tx (
        .clk(clk),
        .reset(reset),
        .start(tx_start),
        .data_in(tx_data_in),
        .tx(serial),
        .busy(tx_busy),
        .done(tx_done)
    );

    uart_rx #(
        .CLOCK_HZ(CLOCK_HZ),
        .BAUD(BAUD)
    ) u_rx (
        .clk(clk),
        .reset(reset),
        .rx(serial),
        .data_out(rx_data_out),
        .data_valid(rx_data_valid)
    );

    assign serial_line = serial;

endmodule
