package accel_uart_pkg;

  // UART framing bytes
  localparam logic [7:0] SOF_HOST_TO_FPGA = 8'hA5;
  localparam logic [7:0] SOF_FPGA_TO_HOST = 8'h5A;

  // Opcodes
  localparam logic [7:0] OPCODE_WRITE = 8'h01;
  localparam logic [7:0] OPCODE_READ  = 8'h02;
  localparam logic [7:0] OPCODE_EXEC  = 8'h03;
  localparam logic [7:0] OPCODE_PING  = 8'h7F;

  // Status codes in responses
  localparam logic [7:0] ST_OK       = 8'h00;
  localparam logic [7:0] ST_BAD_CSUM = 8'h01;
  localparam logic [7:0] ST_BAD_CMD  = 8'h02;
  localparam logic [7:0] ST_BAD_ADDR = 8'h03;
  localparam logic [7:0] ST_BUSY     = 8'h04;
  localparam logic [7:0] ST_BAD_LEN  = 8'h05;

  // Register map (byte addresses)
  localparam logic [7:0] REG_CONTROL = 8'h00; // bit0=start (write 1 to pulse)
  localparam logic [7:0] REG_STATUS  = 8'h04; // bit0=done, bit1=busy
  localparam logic [7:0] REG_ACTION  = 8'h08; // bits[1:0]=selected_action

  localparam logic [7:0] REG_OBS0    = 8'h10; // int16 LE
  localparam logic [7:0] REG_OBS1    = 8'h14; // int16 LE
  localparam logic [7:0] REG_OBS2    = 8'h18; // int16 LE
  localparam logic [7:0] REG_OBS3    = 8'h1C; // int16 LE

  localparam logic [7:0] REG_VERSION = 8'hFC; // read-only, 0x0001

endpackage
