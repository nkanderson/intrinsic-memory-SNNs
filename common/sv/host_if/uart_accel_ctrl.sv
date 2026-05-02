module uart_accel_ctrl #(
    parameter integer NUM_INPUTS = 4,
    parameter integer ACTION_WIDTH = 2
) (
    input  wire clk,
    input  wire reset,

    // Byte stream in from UART RX
    input  wire [7:0] rx_data,
    input  wire rx_valid,

    // Byte stream out to UART TX
    output logic [7:0] tx_data,
    output logic tx_start,
    input  wire tx_busy,

    // Accelerator control/status
    output logic start_pulse,
    output logic signed [15:0] observations [0:NUM_INPUTS-1],
    input  wire accel_done,
    input  wire accel_busy,
    input  wire [ACTION_WIDTH-1:0] accel_action
);

    import accel_uart_pkg::*;

    localparam integer MAX_PAYLOAD = 64;
    localparam logic [15:0] IF_VERSION = 16'h0001;

    typedef enum logic [2:0] {
        RX_WAIT_SOF,
        RX_GET_OPCODE,
        RX_GET_ADDR,
        RX_GET_LEN,
        RX_GET_PAYLOAD,
        RX_GET_CSUM
    } rx_state_t;

    rx_state_t rx_state;

    logic [7:0] cmd_opcode;
    logic [7:0] cmd_addr;
    logic [7:0] cmd_len;
    logic [7:0] cmd_payload [0:MAX_PAYLOAD-1];
    logic [7:0] cmd_idx;
    logic [7:0] checksum_acc;

    logic [7:0] rsp_payload [0:MAX_PAYLOAD-1];
    logic [7:0] rsp_len;

    logic tx_active;
    logic [7:0] tx_frame [0:MAX_PAYLOAD+3];
    logic [7:0] tx_count;
    logic [7:0] tx_idx;
    logic frame_len_error;
    logic reg_ok;
    logic [7:0] reg_byte;

    // Deferred response queueing. The opcode handlers in RX_GET_CSUM update
    // rsp_payload via NBA, which would not be visible to queue_response if
    // it were called the same cycle (its RHS reads see start-of-cycle values).
    // Instead, the handler stages resp_pending/resp_pending_status/_len, and
    // a follow-up always_ff branch calls queue_response one cycle later, by
    // which time rsp_payload has committed.
    logic resp_pending;
    logic [7:0] resp_pending_status;
    logic [7:0] resp_pending_len;

    task automatic write_reg_byte(
        input  logic [7:0] reg_addr,
        input  logic [7:0] reg_data,
        output logic ok
    );
        ok = 1'b1;
        unique case (reg_addr)
            REG_CONTROL: begin
                if (reg_data[0] && !accel_busy) begin
                    start_pulse <= 1'b1;
                end
            end
            REG_OBS0: observations[0][7:0] <= reg_data;
            REG_OBS0 + 8'h01: observations[0][15:8] <= reg_data;
            REG_OBS1: observations[1][7:0] <= reg_data;
            REG_OBS1 + 8'h01: observations[1][15:8] <= reg_data;
            REG_OBS2: observations[2][7:0] <= reg_data;
            REG_OBS2 + 8'h01: observations[2][15:8] <= reg_data;
            REG_OBS3: observations[3][7:0] <= reg_data;
            REG_OBS3 + 8'h01: observations[3][15:8] <= reg_data;
            default: ok = 1'b0;
        endcase
    endtask

    task automatic read_reg_byte(
        input  logic [7:0] reg_addr,
        output logic [7:0] reg_data,
        output logic ok
    );
        reg_data = 8'h00;
        ok = 1'b1;
        unique case (reg_addr)
            REG_STATUS: begin
                reg_data[0] = accel_done;
                reg_data[1] = accel_busy;
            end
            REG_STATUS + 8'h01,
            REG_STATUS + 8'h02,
            REG_STATUS + 8'h03: begin
                reg_data = 8'h00;
            end
            REG_ACTION: begin
                for (int i = 0; i < ACTION_WIDTH; i++) begin
                    if (i < 8) begin
                        reg_data[i] = accel_action[i];
                    end
                end
            end
            REG_ACTION + 8'h01,
            REG_ACTION + 8'h02,
            REG_ACTION + 8'h03: begin
                reg_data = 8'h00;
            end
            REG_OBS0: reg_data = observations[0][7:0];
            REG_OBS0 + 8'h01: reg_data = observations[0][15:8];
            REG_OBS1: reg_data = observations[1][7:0];
            REG_OBS1 + 8'h01: reg_data = observations[1][15:8];
            REG_OBS2: reg_data = observations[2][7:0];
            REG_OBS2 + 8'h01: reg_data = observations[2][15:8];
            REG_OBS3: reg_data = observations[3][7:0];
            REG_OBS3 + 8'h01: reg_data = observations[3][15:8];
            REG_VERSION: reg_data = IF_VERSION[7:0];
            REG_VERSION + 8'h01: reg_data = IF_VERSION[15:8];
            REG_VERSION + 8'h02,
            REG_VERSION + 8'h03: reg_data = 8'h00;
            default: ok = 1'b0;
        endcase
    endtask

    task automatic queue_response(input logic [7:0] status_code, input logic [7:0] payload_len);
        logic [7:0] csum;
        csum = SOF_FPGA_TO_HOST ^ status_code ^ payload_len;

        tx_frame[0] <= SOF_FPGA_TO_HOST;
        tx_frame[1] <= status_code;
        tx_frame[2] <= payload_len;
        for (int i = 0; i < MAX_PAYLOAD; i++) begin
            if (i < payload_len) begin
                tx_frame[3 + i] <= rsp_payload[i];
                csum = csum ^ rsp_payload[i];
            end
        end
        tx_frame[3 + payload_len] <= csum;
        tx_count <= payload_len + 8'd4;
        tx_idx <= 8'd0;
        tx_active <= 1'b1;
    endtask

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            rx_state <= RX_WAIT_SOF;
            cmd_opcode <= 8'h00;
            cmd_addr <= 8'h00;
            cmd_len <= 8'h00;
            cmd_idx <= 8'h00;
            checksum_acc <= 8'h00;
            frame_len_error <= 1'b0;

            tx_data <= 8'h00;
            tx_start <= 1'b0;
            tx_active <= 1'b0;
            tx_count <= 8'h00;
            tx_idx <= 8'h00;
            rsp_len <= 8'h00;

            resp_pending <= 1'b0;
            resp_pending_status <= 8'h00;
            resp_pending_len <= 8'h00;

            start_pulse <= 1'b0;
            for (int i = 0; i < NUM_INPUTS; i++) begin
                observations[i] <= '0;
            end
            for (int i = 0; i < MAX_PAYLOAD; i++) begin
                cmd_payload[i] <= 8'h00;
                rsp_payload[i] <= 8'h00;
            end
        end else begin
            tx_start <= 1'b0;
            start_pulse <= 1'b0;

            // TX engine: send queued response bytes one at a time
            if (tx_active && !tx_busy) begin
                if (tx_idx < tx_count) begin
                    tx_data <= tx_frame[tx_idx];
                    tx_start <= 1'b1;
                    tx_idx <= tx_idx + 1'b1;
                end else begin
                    tx_active <= 1'b0;
                end
            end

            // Deferred response queueing (one cycle after the dispatch in
            // RX_GET_CSUM). By now any rsp_payload writes from that cycle
            // have committed and are visible to queue_response's RHS reads.
            if (resp_pending && !tx_active) begin
                resp_pending <= 1'b0;
                queue_response(resp_pending_status, resp_pending_len);
            end

            // RX command parser
            if (rx_valid) begin
                unique case (rx_state)
                    RX_WAIT_SOF: begin
                        if (rx_data == SOF_HOST_TO_FPGA) begin
                            checksum_acc <= rx_data;
                            cmd_idx <= 8'h00;
                            frame_len_error <= 1'b0;
                            rx_state <= RX_GET_OPCODE;
                        end
                    end

                    RX_GET_OPCODE: begin
                        cmd_opcode <= rx_data;
                        checksum_acc <= checksum_acc ^ rx_data;
                        rx_state <= RX_GET_ADDR;
                    end

                    RX_GET_ADDR: begin
                        cmd_addr <= rx_data;
                        checksum_acc <= checksum_acc ^ rx_data;
                        rx_state <= RX_GET_LEN;
                    end

                    RX_GET_LEN: begin
                        cmd_len <= rx_data;
                        checksum_acc <= checksum_acc ^ rx_data;
                        cmd_idx <= 8'h00;
                        // For READ, LEN encodes how many bytes to return; no
                        // payload follows on the wire. The bound check still
                        // applies so READ rejects LEN > MAX_PAYLOAD before
                        // overflowing tx_frame.
                        frame_len_error <= (rx_data > MAX_PAYLOAD);
                        if (rx_data == 8'h00 || cmd_opcode == OPCODE_READ) begin
                            rx_state <= RX_GET_CSUM;
                        end else begin
                            rx_state <= RX_GET_PAYLOAD;
                        end
                    end

                    RX_GET_PAYLOAD: begin
                        if (cmd_idx < MAX_PAYLOAD) begin
                            cmd_payload[cmd_idx] <= rx_data;
                        end
                        checksum_acc <= checksum_acc ^ rx_data;
                        if (cmd_idx == (cmd_len - 1'b1)) begin
                            rx_state <= RX_GET_CSUM;
                        end else begin
                            cmd_idx <= cmd_idx + 1'b1;
                        end
                    end

                    RX_GET_CSUM: begin
                        // Validate checksum and stage the response. The
                        // actual queue_response call happens one cycle later
                        // (see the deferred-response branch above) so any
                        // rsp_payload writes in this cycle are committed
                        // before queue_response reads them.
                        if (rx_data != checksum_acc) begin
                            rsp_len <= 8'h00;
                            resp_pending <= 1'b1;
                            resp_pending_status <= ST_BAD_CSUM;
                            resp_pending_len <= 8'h00;
                        end else if (frame_len_error) begin
                            rsp_len <= 8'h00;
                            resp_pending <= 1'b1;
                            resp_pending_status <= ST_BAD_LEN;
                            resp_pending_len <= 8'h00;
                        end else begin
                            reg_ok = 1'b1;
                            rsp_len <= 8'h00;

                            unique case (cmd_opcode)
                                OPCODE_PING: begin
                                    rsp_payload[0] <= 8'h50; // 'P'
                                    rsp_len <= 8'h01;
                                    resp_pending <= 1'b1;
                                    resp_pending_status <= ST_OK;
                                    resp_pending_len <= 8'h01;
                                end

                                OPCODE_EXEC: begin
                                    if (accel_busy) begin
                                        resp_pending <= 1'b1;
                                        resp_pending_status <= ST_BUSY;
                                        resp_pending_len <= 8'h00;
                                    end else begin
                                        start_pulse <= 1'b1;
                                        resp_pending <= 1'b1;
                                        resp_pending_status <= ST_OK;
                                        resp_pending_len <= 8'h00;
                                    end
                                end

                                OPCODE_WRITE: begin
                                    for (int i = 0; i < MAX_PAYLOAD; i++) begin
                                        if ((i < cmd_len) && reg_ok) begin
                                            write_reg_byte(cmd_addr + i[7:0], cmd_payload[i], reg_ok);
                                        end
                                    end
                                    resp_pending <= 1'b1;
                                    resp_pending_status <= reg_ok ? ST_OK : ST_BAD_ADDR;
                                    resp_pending_len <= 8'h00;
                                end

                                OPCODE_READ: begin
                                    reg_ok = 1'b1;
                                    for (int i = 0; i < MAX_PAYLOAD; i++) begin
                                        if ((i < cmd_len) && reg_ok) begin
                                            read_reg_byte(cmd_addr + i[7:0], reg_byte, reg_ok);
                                            if (reg_ok) begin
                                                rsp_payload[i] <= reg_byte;
                                            end
                                        end
                                    end
                                    resp_pending <= 1'b1;
                                    resp_pending_status <= reg_ok ? ST_OK : ST_BAD_ADDR;
                                    resp_pending_len <= reg_ok ? cmd_len : 8'h00;
                                end

                                default: begin
                                    resp_pending <= 1'b1;
                                    resp_pending_status <= ST_BAD_CMD;
                                    resp_pending_len <= 8'h00;
                                end
                            endcase
                        end

                        // Return parser to idle for next frame
                        rx_state <= RX_WAIT_SOF;
                        checksum_acc <= 8'h00;
                        cmd_idx <= 8'h00;
                    end

                    default: begin
                        rx_state <= RX_WAIT_SOF;
                    end
                endcase
            end
        end
    end

endmodule
