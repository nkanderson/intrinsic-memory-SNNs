module lif_tb;

    // Parameters
    parameter [7:0] THRESHOLD = 8'd200;
    parameter [7:0] DECAY = 8'd128; // Fixed-point representation of decay (e.g., 128/256 = 0.5)
    parameter [4:0] REFRACTORY_PERIOD = 5'd5;

    // Signals
    logic clk;
    logic rst;
    logic [7:0] current;
    logic spike;

    // Instantiate the lif module
    lif #(
        .THRESHOLD(THRESHOLD),
        .DECAY(DECAY),
        .REFRACTORY_PERIOD(REFRACTORY_PERIOD)
    ) uut (
        .clk(clk),
        .rst(rst),
        .current(current),
        .spike(spike)
    );

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk; // 10 time units clock period

    // Testbench logic
    initial begin
        // Initialize signals
        rst = 1;
        current = 8'd0;

        // Reset the system
        repeat (1) @(posedge clk);
        rst = 0;

        // Apply test stimulus
        repeat (1) @(posedge clk); current = 8'd50;  // Apply a small current
        repeat (2) @(posedge clk); current = 8'd100; // Increase current
        repeat (2) @(posedge clk); current = 8'd200; // Apply current equal to threshold
        repeat (2) @(posedge clk); current = 8'd250; // Apply a large current
        repeat (5) @(posedge clk); current = 8'd0;   // Remove current

        // Finish simulation
        repeat (10) @(posedge clk);
        $finish;
    end

    // Monitor signals
    initial begin
        $monitor("Time: %0t | clk: %b | rst: %b | current: %d | spike: %b | membrane_potential: %d",
                 $time, clk, rst, current, spike, uut.membrane_potential);
    end

endmodule
