// Test file for setting up basic Makefile and Verilator simulation
module top (
    input logic clk,
    input logic rst,
    output logic [3:0] counter
);
  always_ff @(posedge clk or posedge rst) begin
    if (rst) counter <= 0;
    else counter <= counter + 1;
  end
endmodule
