module top_lapicque_icebreaker (
    input  clk,  // 12 MHz clock
    input  btn,  // push button (active low reset)
    // PMOD 1A pins for DIP switch input
    input P1A1, P1A2, P1A3, P1A4,
    input P1A7, P1A8, P1A9, P1A10,
    output led   // user LED (active high on iCEBreaker)
);

  logic [7:0] current;
  logic spike;
  logic rst;
  logic [7:0] dip_in;

  // DIP switch for adjusting input current
  // with very basic debouncing via a single FF
  always_ff @(posedge clk) dip_in <= {P1A10, P1A9, P1A8, P1A7,
                 P1A4, P1A3, P1A2, P1A1};

  //
  // Reset signal: active low button
  // Begin with a power-on reset, then allow for button reset
  //
  // Power-on reset logic: 4-bit counter to hold reset for ~16 clock cycles
  logic [3:0] por_cnt = 0;
  logic por_rst = 1;

  always_ff @(posedge clk) begin
    if (por_cnt != 4'hF) begin
      por_cnt <= por_cnt + 1;
      por_rst <= 1;
    end else begin
      por_rst <= 0;
    end
  end

  // Create a cleaner rst signal by running through 2 FFs
  bit rst_sync_0, rst_sync_1;
  always @(posedge clk) begin
    rst_sync_0 <= ~btn;  // invert btn (active low)
    rst_sync_1 <= rst_sync_0;
  end
  assign rst = por_rst | rst_sync_1;
  // End reset signal logic

  // Simple DIP switch input for current
  assign current = dip_in;  // Use DIP switch input

  lapicque neuron (
      .clk(clk),
      .rst(rst),
      .current(current),
      .spike(spike)
  );

  // LED pulse stretcher: ~200 ms at 12 MHz
  logic [21:0] led_counter = 0;  // enough bits for 200 ms (12 MHz * 0.2 = 2.4M)
  always_ff @(posedge clk) begin
    if (rst) begin
      led_counter <= 0;
    end else begin
      if (spike) begin
        led_counter <= 22'd2_400_000;  // 200 ms worth of 12 MHz ticks
      end else if (led_counter > 0) begin
        led_counter <= led_counter - 1;
      end
    end
  end

  // We may want to use the led_counter eventually if we can get
  // more granular control over the input current timing. But for now,
  // we just use the spike signal directly for the LED.
  assign led = spike; //(led_counter != 0);  // active-high LED

endmodule
