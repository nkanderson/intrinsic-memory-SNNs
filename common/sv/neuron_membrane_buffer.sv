// Per-Neuron Membrane Buffer Module
// Stores membrane potentials for a single neuron across all timesteps
//
// This is a simple FIFO-like buffer that:
// - Writes: Stores one membrane value per cycle as the neuron progresses through timesteps
// - Reads: Allows random access by timestep for downstream processing
//
// Used between hidden layer 2 LIF neurons and the output Q-value computation.
// Each LIF neuron in hidden layer 2 gets its own instance of this buffer.
//
// Timing:
//   - Assert 'write_en' each cycle the neuron outputs a valid membrane value
//   - write_timestep indicates which timestep slot to write
//   - Read is combinational: provide read_timestep, get membrane_out immediately
//   - 'full' asserts when all timesteps have been written

module neuron_membrane_buffer #(
    parameter NUM_TIMESTEPS = 30,
    parameter MEMBRANE_WIDTH = 24
) (
    input wire clk,
    input wire reset,
    input wire clear,                                      // Clear buffer for new inference

    // Write interface (from LIF neuron)
    input wire write_en,                                   // Write enable
    input wire [$clog2(NUM_TIMESTEPS)-1:0] write_timestep, // Which timestep to write
    input wire signed [MEMBRANE_WIDTH-1:0] membrane_in,    // Membrane value to store

    // Read interface (from q_accumulator or linear_layer)
    input wire [$clog2(NUM_TIMESTEPS)-1:0] read_timestep,  // Which timestep to read
    output logic signed [MEMBRANE_WIDTH-1:0] membrane_out, // Membrane for requested timestep

    // Status
    output logic full                                      // All timesteps have been written
);

    // Storage for all timesteps
    logic signed [MEMBRANE_WIDTH-1:0] membrane_storage [0:NUM_TIMESTEPS-1];

    // Track which timesteps have been written
    logic [NUM_TIMESTEPS-1:0] written;

    // Combinational read - immediate output
    assign membrane_out = membrane_storage[read_timestep];

    // Full when all timesteps written
    assign full = (written == {NUM_TIMESTEPS{1'b1}});

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            written <= '0;
            for (int t = 0; t < NUM_TIMESTEPS; t++) begin
                membrane_storage[t] <= '0;
            end
        end else if (clear) begin
            written <= '0;
            // Don't need to clear storage - will be overwritten
        end else if (write_en) begin
            membrane_storage[write_timestep] <= membrane_in;
            written[write_timestep] <= 1'b1;
        end
    end

endmodule
