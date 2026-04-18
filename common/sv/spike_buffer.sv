// Spike Buffer Module
// Stores spike vectors from HL1 for each timestep
//
// This simplified version assumes synchronized HL1 processing where all neurons
// process the same timestep together, producing a complete spike vector each cycle.
// The neural_network module manages timestep synchronization.
//
// For the legacy version that handles staggered neuron starts from linear_layer,
// see spike_buffer_staggered.sv.
//
// Interface:
//   - write_en + write_timestep: Store 64-bit spike vector at specified timestep
//   - read_timestep: Combinational read of spike vector at specified timestep
//   - clear: Reset all storage for new inference

module spike_buffer #(
    parameter NUM_NEURONS = 64,
    parameter NUM_TIMESTEPS = 30
) (
    input wire clk,
    input wire reset,
    input wire clear,                                       // Synchronous clear for new inference
    input wire write_en,                                    // Store current spike vector
    input wire [$clog2(NUM_TIMESTEPS)-1:0] write_timestep,  // Which timestep to write
    input wire [NUM_NEURONS-1:0] spike_in,                  // Spike vector from all HL1 neurons
    input wire [$clog2(NUM_TIMESTEPS)-1:0] read_timestep,   // Which timestep to read
    output logic [NUM_NEURONS-1:0] spikes_out               // Spike vector for requested timestep
);

    // Spike storage: [timestep][neuron]
    logic [NUM_NEURONS-1:0] spike_storage [0:NUM_TIMESTEPS-1];

    // Combinational read
    assign spikes_out = spike_storage[read_timestep];

    always_ff @(posedge clk or posedge reset) begin
        spike_storage[write_timestep] <= spike_storage[write_timestep]; // Default to hold value
        if (reset) begin
            for (int t = 0; t < NUM_TIMESTEPS; t++) begin
                spike_storage[t] <= '0;
            end
        end else if (clear) begin
            // Clear all storage for new inference
            for (int t = 0; t < NUM_TIMESTEPS; t++) begin
                spike_storage[t] <= '0;
            end
        end else if (write_en) begin
            // Store spike vector at specified timestep
            spike_storage[write_timestep] <= spike_in;
        end
    end

endmodule
