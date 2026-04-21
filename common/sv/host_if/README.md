# UART Host Interface (Accelerator Control)

This directory provides a minimal software-to-FPGA control plane for neural-network acceleration.

## Frame format

### Host -> FPGA

Bytes:

1. `SOF` = `0xA5`
2. `OPCODE`
3. `ADDR`
4. `LEN`
5. `PAYLOAD[LEN]`
6. `CSUM` = XOR of all prior bytes in the frame (`SOF ^ OPCODE ^ ADDR ^ LEN ^ PAYLOAD...`)

### FPGA -> Host

Bytes:

1. `SOF` = `0x5A`
2. `STATUS`
3. `LEN`
4. `PAYLOAD[LEN]`
5. `CSUM` = XOR of all prior bytes in the response

## Opcodes

- `0x01` `WRITE`: write `LEN` bytes beginning at `ADDR`
- `0x02` `READ`: read `LEN` bytes beginning at `ADDR`
- `0x03` `EXEC`: trigger one accelerator run
- `0x7F` `PING`: returns 1-byte payload `'P'`

## Status codes

- `0x00` `ST_OK`
- `0x01` `ST_BAD_CSUM`
- `0x02` `ST_BAD_CMD`
- `0x03` `ST_BAD_ADDR`
- `0x04` `ST_BUSY`
- `0x05` `ST_BAD_LEN`

## Register map (byte addresses)

- `0x00` `REG_CONTROL` (W): bit0 = start pulse
- `0x04` `REG_STATUS` (R): bit0 = done, bit1 = busy
- `0x08` `REG_ACTION` (R): selected action (`ACTION_WIDTH` bits)
- `0x10` `REG_OBS0` (RW, int16 little-endian)
- `0x14` `REG_OBS1` (RW, int16 little-endian)
- `0x18` `REG_OBS2` (RW, int16 little-endian)
- `0x1C` `REG_OBS3` (RW, int16 little-endian)
- `0xFC` `REG_VERSION` (R): `0x0001`

## Modules

- `accel_uart_pkg.sv`: protocol constants and register map
- `uart_rx.sv`: UART byte receiver
- `uart_tx.sv`: UART byte transmitter
- `uart_accel_ctrl.sv`: parser + register block + response framing
- `top_uart_accel_wrapper.sv`: thin wrapper to bind UART host IF with one accelerator top

## Integration notes

- `top_uart_accel_wrapper.sv` supports `MODEL_TYPE`:
  - `0` standard `neural_network`
  - `1` `neural_network_fractional`
  - `2` `neural_network_bitshift`
- Set `CLOCK_HZ` and `BAUD` to board clock/UART settings.
- `NUM_INPUTS` is expected to be 4 for the current fixed observation register map.
