// Spike Buffer Module (Staggered Input Version)
// Collects and synchronizes spikes from all neurons in a layer
//
// This is the legacy version that handles staggered neuron starts from linear_layer.
// For the simplified version used with synchronized HL1 processing in neural_network,
// see spike_buffer.sv.
//
// Neurons in a layer start at staggered times (one per cycle from linear_layer).
// This module uses the known timing pattern to collect spikes without per-neuron
// timestep signals, avoiding massive fan-in.
//
// Timing pattern (after start signal):
//   Cycle 0: neuron 0 outputs timestep 0
//   Cycle 1: neuron 0 outputs timestep 1, neuron 1 outputs timestep 0
//   Cycle K: neuron N outputs timestep (K-N) for all N <= K where (K-N) < NUM_TIMESTEPS
//
// The buffer stores spikes in a 2D array indexed by [timestep][neuron].
// All neurons complete timestep T at cycle (T + NUM_NEURONS - 1).
// When all neurons have completed timestep T, spikes_out provides that vector.
//
// Total collection time: NUM_NEURONS + NUM_TIMESTEPS - 1 cycles

module spike_buffer_staggered #(
    parameter NUM_NEURONS = 64,
    parameter NUM_TIMESTEPS = 30
) (
    input wire clk,
    input wire reset,
    input wire start,                           // Neuron 0's first spike is valid this cycle
    input wire [NUM_NEURONS-1:0] spike_in,      // Current spike from each neuron (directly connected)
    output logic [NUM_NEURONS-1:0] spikes_out,  // Spike vector for current output timestep
    output logic [4:0] timestep_out,            // Which timestep spikes_out is for
    output logic timestep_ready,                // spikes_out is valid
    output logic done                           // All timesteps have been output
);

    // Cycle counter since start (needs to count up to NUM_NEURONS + NUM_TIMESTEPS - 1)
    localparam CYCLE_WIDTH = $clog2(NUM_NEURONS + NUM_TIMESTEPS);
    logic [CYCLE_WIDTH-1:0] cycle_count;

    // Spike storage: [timestep][neuron]
    logic [NUM_NEURONS-1:0] spike_storage [0:NUM_TIMESTEPS-1];

    // Output timestep tracker
    logic [4:0] output_timestep;

    // State machine
    typedef enum logic [1:0] {
        IDLE,
        COLLECTING,
        OUTPUTTING,
        DONE_STATE
    } state_t;

    state_t state;

    // Compute when we can output a timestep
    // Timestep T is complete when cycle_count >= T + NUM_NEURONS - 1
    // i.e., the last neuron (NUM_NEURONS-1) has reached timestep T
    logic timestep_complete;
    assign timestep_complete = (cycle_count >= output_timestep + NUM_NEURONS - 1);

    // Output assignments
    assign spikes_out = spike_storage[output_timestep];
    assign timestep_out = output_timestep;
    assign timestep_ready = (state == COLLECTING || state == OUTPUTTING) && timestep_complete;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            cycle_count <= '0;
            output_timestep <= '0;
            done <= 1'b0;
            for (int t = 0; t < NUM_TIMESTEPS; t++) begin
                spike_storage[t] <= '0;
            end
        end else begin
            unique case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        // Clear storage and start collecting
                        // Clear timesteps 1 onwards completely, as timestep 0 gets first spike
                        for (int t = 1; t < NUM_TIMESTEPS; t++) begin
                            spike_storage[t] <= '0;
                        end
                        // For timestep 0: clear all neurons except neuron 0, which gets spike_in
                        spike_storage[0][NUM_NEURONS-1:1] <= '0;
                        spike_storage[0][0] <= spike_in[0];

                        cycle_count <= '0;
                        output_timestep <= '0;
                        state <= COLLECTING;
                    end
                end

                COLLECTING: begin
                    cycle_count <= cycle_count + 1'b1;

                    // Store spikes from each neuron based on cycle timing
                    // Neuron N is at timestep (cycle_count + 1 - N) if valid
                    // We're storing for the NEXT cycle, so use cycle_count + 1
                    for (int n = 0; n < NUM_NEURONS; n++) begin
                        // Check if neuron n is producing valid output this cycle
                        // Neuron n starts at cycle n, so at cycle C it's at timestep (C - n)
                        // Next cycle (C+1), we'll be storing: neuron n at timestep (C+1 - n)
                        // Next cycle and neuron timestep are used conceptually in the following logic,
                        // but since we can't create new variable and use blocking assignment here,
                        // replacing with direct calculation:
                        // next cycle = cycle_count + 1;
                        // neuron timestep = next_cycle - n;
                        if ((cycle_count + 1) >= n && (cycle_count + 1 - n) < NUM_TIMESTEPS) begin
                            spike_storage[(cycle_count + 1 - n)][n] <= spike_in[n];
                        end
                    end

                    // Advance output timestep when current one is complete
                    if (timestep_complete) begin
                        if (output_timestep == NUM_TIMESTEPS - 1) begin
                            // All timesteps output
                            done <= 1'b1;
                            state <= DONE_STATE;
                        end else begin
                            output_timestep <= output_timestep + 1'b1;
                        end
                    end

                    // Check if we've collected all spikes
                    if (cycle_count >= NUM_NEURONS + NUM_TIMESTEPS - 2) begin
                        state <= OUTPUTTING;
                    end
                end

                OUTPUTTING: begin
                    // All spikes collected, just outputting remaining timesteps
                    cycle_count <= cycle_count + 1'b1;

                    if (timestep_complete) begin
                        if (output_timestep == NUM_TIMESTEPS - 1) begin
                            done <= 1'b1;
                            state <= DONE_STATE;
                        end else begin
                            output_timestep <= output_timestep + 1'b1;
                        end
                    end
                end

                DONE_STATE: begin
                    // Hold until next start
                    if (start) begin
                        // Clear storage - same pattern as IDLE to avoid conflicts:
                        // Clear timesteps 1 onwards completely, as timestep 0 gets first spike
                        for (int t = 1; t < NUM_TIMESTEPS; t++) begin
                            spike_storage[t] <= '0;
                        end
                        spike_storage[0][NUM_NEURONS-1:1] <= '0;
                        spike_storage[0][0] <= spike_in[0];

                        cycle_count <= '0;
                        output_timestep <= '0;
                        done <= 1'b0;
                        state <= COLLECTING;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
