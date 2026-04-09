// Membrane Buffer Module
// Collects and synchronizes membrane potentials from all neurons in a layer
//
// Similar to spike_buffer but stores multi-bit membrane potentials instead of
// single-bit spikes. Used between the final hidden layer and output linear layer
// where we need membrane values (not spikes) for Q-value computation.
//
// Neurons in a layer start at staggered times (one per cycle from linear_layer).
// This module uses the known timing pattern to collect membrane potentials
// without per-neuron timestep signals, avoiding massive fan-in.
//
// Timing pattern (after start signal):
//   Cycle 0: neuron 0 outputs timestep 0
//   Cycle 1: neuron 0 outputs timestep 1, neuron 1 outputs timestep 0
//   Cycle K: neuron N outputs timestep (K-N) for all N <= K where (K-N) < NUM_TIMESTEPS
//
// The buffer stores membrane potentials in a 2D array indexed by [timestep][neuron].
// All neurons complete timestep T at cycle (T + NUM_NEURONS - 1).
// When all neurons have completed timestep T, membranes_out provides that vector.
//
// Total collection time: NUM_NEURONS + NUM_TIMESTEPS - 1 cycles

module membrane_buffer #(
    parameter NUM_NEURONS = 16,
    parameter NUM_TIMESTEPS = 30,
    parameter MEMBRANE_WIDTH = 24
) (
    input wire clk,
    input wire reset,
    input wire start,                                                    // Neuron 0's first output is valid this cycle
    input wire signed [MEMBRANE_WIDTH-1:0] membrane_in [0:NUM_NEURONS-1], // Current membrane from each neuron
    output logic signed [MEMBRANE_WIDTH-1:0] membranes_out [0:NUM_NEURONS-1], // Membrane vector for current output timestep
    output logic [4:0] timestep_out,                                     // Which timestep membranes_out is for
    output logic timestep_ready,                                         // membranes_out is valid
    output logic done                                                    // All timesteps have been output
);

    // Cycle counter since start (needs to count up to NUM_NEURONS + NUM_TIMESTEPS - 1)
    localparam CYCLE_WIDTH = $clog2(NUM_NEURONS + NUM_TIMESTEPS);
    logic [CYCLE_WIDTH-1:0] cycle_count;

    // Membrane storage: [timestep][neuron]
    logic signed [MEMBRANE_WIDTH-1:0] membrane_storage [0:NUM_TIMESTEPS-1][0:NUM_NEURONS-1];

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

    // Output assignments - directly wire from storage
    always_comb begin
        for (int n = 0; n < NUM_NEURONS; n++) begin
            membranes_out[n] = membrane_storage[output_timestep][n];
        end
    end
    assign timestep_out = output_timestep;
    assign timestep_ready = (state == COLLECTING || state == OUTPUTTING) && timestep_complete;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            cycle_count <= '0;
            output_timestep <= '0;
            done <= 1'b0;
            for (int t = 0; t < NUM_TIMESTEPS; t++) begin
                for (int n = 0; n < NUM_NEURONS; n++) begin
                    membrane_storage[t][n] <= '0;
                end
            end
        end else begin
            unique case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        // Clear storage and start collecting
                        // Clear timesteps 1 onwards completely
                        for (int t = 1; t < NUM_TIMESTEPS; t++) begin
                            for (int n = 0; n < NUM_NEURONS; n++) begin
                                membrane_storage[t][n] <= '0;
                            end
                        end
                        // For timestep 0: clear neurons 1+ and store neuron 0's value
                        for (int n = 1; n < NUM_NEURONS; n++) begin
                            membrane_storage[0][n] <= '0;
                        end
                        membrane_storage[0][0] <= membrane_in[0];

                        cycle_count <= '0;
                        output_timestep <= '0;
                        state <= COLLECTING;
                    end
                end

                COLLECTING: begin
                    cycle_count <= cycle_count + 1'b1;

                    // Store membrane potentials from each neuron based on cycle timing
                    // Neuron N is at timestep (cycle_count + 1 - N) next cycle
                    for (int n = 0; n < NUM_NEURONS; n++) begin
                        if ((cycle_count + 1) >= n && (cycle_count + 1 - n) < NUM_TIMESTEPS) begin
                            membrane_storage[(cycle_count + 1 - n)][n] <= membrane_in[n];
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

                    // Check if we've collected all membrane potentials
                    if (cycle_count >= NUM_NEURONS + NUM_TIMESTEPS - 2) begin
                        state <= OUTPUTTING;
                    end
                end

                OUTPUTTING: begin
                    // All potentials collected, just outputting remaining timesteps
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
                        // Clear storage - same pattern as IDLE to avoid conflicts
                        for (int t = 1; t < NUM_TIMESTEPS; t++) begin
                            for (int n = 0; n < NUM_NEURONS; n++) begin
                                membrane_storage[t][n] <= '0;
                            end
                        end
                        for (int n = 1; n < NUM_NEURONS; n++) begin
                            membrane_storage[0][n] <= '0;
                        end
                        membrane_storage[0][0] <= membrane_in[0];

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
