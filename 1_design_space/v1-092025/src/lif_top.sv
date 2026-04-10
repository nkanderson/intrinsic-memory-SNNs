module lif_top (
    input logic clk,
    input logic rst_n,              // Active-low reset (common for FPGAs)
    input logic [7:0] sw,           // Switches for current input
    output logic [7:0] led,         // LEDs for membrane potential display
    output logic spike_led          // LED to indicate spike
);

    // Internal reset (convert active-low to active-high)
    logic rst;
    assign rst = ~rst_n;

    // LIF neuron instance
    lif #(
        .THRESHOLD(8'd200),
        .DECAY(8'd128),             // 0.5 decay factor
        .REFRACTORY_PERIOD(5'd10)   // Longer refractory period for visibility
    ) lif_inst (
        .clk(clk),
        .rst(rst),
        .current(sw),               // Use switches as current input
        .spike(spike_led)
    );

    // Display membrane potential on LEDs (for debugging)
    assign led = lif_inst.membrane_potential;

endmodule
