# CartPole SNN Hardware Specification

## Overview

This document specifies the hardware architecture for a Spiking Neural Network (SNN) trained with snnTorch to solve the CartPole reinforcement learning task. The network uses direct current injection (no spike encoding at input) to match the trained weights from snn_policy.py.

This initial plan reflects training with snnTorch's Leaky neuron model. Small adjustments (e.g. changes to reset mechanism) may be necessary for usage with an alternative model.

**Key Design Principle**: The hardware must exactly match the behavior of `snn_policy.py`'s forward pass:
```python
for _t in range(self.num_steps):
    h1 = self.fc1(observations)  # same input every timestep for a single forward pass
    spk1 = self.lif1(h1)         # LIF1 processes static h1, outputs spikes
    h2 = self.fc2(spk1)          # new spike vector to LIF2 each timestep
    spk2, mem2 = self.lif2(h2)   # LIF2 processes h2
    q_t = self.fc_out(mem2)      # Q-values from membrane
    out_accum += q_t             # Accumulate
q_values = out_accum / num_steps
```

This means:
- **Hidden Layer 1 LIFs** receive the **same current** every timestep
- **Hidden Layer 2 LIFs** receive **different current each timestep** (varies with HL1 spike vectors)

## Network Architecture

### Default Layer Configuration
- **Input Layer**: 4 continuous observations (cart position, cart velocity, pole angle, pole angular velocity)
- **Hidden Layer 1**: 64 LIF neurons
- **Hidden Layer 2**: 16 LIF neurons
- **Output Layer**: 2 Q-values (one per action: left/right)
- **Timesteps**: 30 timesteps per inference

**NOTE**: The hidden layer sizes may vary depending on the model. For the snnTorch Leaky neuron model, sizes of 64 and 16 for the hidden layers were necessary to generate reasonably good performance.

### Data Format
- **Fixed-Point Format**: QS2.13 (16-bit signed, 2 integer bits, 13 fractional bits)
  - Scale factor: 2^13 = 8192
  - Range: [-4.0, 3.9998779296875]
- **LIF Parameters**:
  - Threshold: 1.0 (8192 in fixed-point)
  - Beta: 0.9 (115 in Q1.7 format for 8-bit)
  - Membrane potential: 24-bit signed (extra headroom)
  - Reset mechanism: subtract (reset_delay=True)

## Module Specifications

### 1. `neural_network` Module (Top-Level)

**Purpose**: Top-level module that instantiates and coordinates all sub-modules to perform complete SNN inference. This is the interface for software interaction (cocotb tests, FPGA integration).

**Interface**:
```systemverilog
module neural_network #(
    // Network architecture
    parameter NUM_INPUTS = 4,
    parameter HL1_SIZE = 64,
    parameter HL2_SIZE = 16,
    parameter NUM_ACTIONS = 2,
    parameter NUM_TIMESTEPS = 30,
    // Fixed-point parameters
    parameter DATA_WIDTH = 16,
    parameter MEMBRANE_WIDTH = 24,
    parameter FRAC_BITS = 13,
    // LIF parameters, see lif.sv for details on format and interpretation of scaled values
    parameter THRESHOLD = 8192,
    parameter BETA = 115,
    // Weight files
    parameter FC1_WEIGHTS_FILE = "fc1_weights.mem",
    parameter FC1_BIAS_FILE = "fc1_bias.mem",
    parameter FC2_WEIGHTS_FILE = "fc2_weights.mem",
    parameter FC2_BIAS_FILE = "fc2_bias.mem",
    parameter FC_OUT_WEIGHTS_FILE = "fc_out_weights.mem",
    parameter FC_OUT_BIAS_FILE = "fc_out_bias.mem",
    // q_accumulator tuning
    parameter Q_BATCH_SIZE = 4
) (
    input wire clk,
    input wire reset,
    input wire start,
    input wire signed [DATA_WIDTH-1:0] observations [0:NUM_INPUTS-1],
    output logic selected_action,
    output logic done
);
```

**Behavior**:
1. When `start` asserted, latch `observations` and begin inference
2. Coordinate all sub-modules through the timestep loop
3. Assert `done` when `selected_action` is ready
4. Hold `selected_action` stable until next `start`

**Internal Structure**:
- Instantiates: `linear_layer` (×2), `lif` (×HL1_SIZE + HL2_SIZE), `spike_buffer`, `neuron_membrane_buffer` (×HL2_SIZE), `q_accumulator`
- HL1 current registers: 64 × 16-bit registers to latch fc1 outputs for reuse each timestep
- Central state machine manages timestep progression and inter-module coordination
- q_accumulator runs pipelined with HL2 processing

### 2. `linear_layer` Module

**Purpose**: Implements a fully connected layer (equivalent to nn.Linear in PyTorch)

**Interface**:
```systemverilog
module linear_layer #(
    parameter NUM_INPUTS = 4,
    parameter NUM_OUTPUTS = 64,
    parameter DATA_WIDTH = 16,
    parameter FRAC_BITS = 13,
    parameter WEIGHTS_FILE = "weights.mem",
    parameter BIAS_FILE = "bias.mem"
) (
    input wire clk,
    input wire reset,
    input wire start,                                    // Start computation
    input wire signed [DATA_WIDTH-1:0] inputs [0:NUM_INPUTS-1],
    output logic signed [DATA_WIDTH-1:0] output_current, // One current per cycle
    output logic [$clog2(NUM_OUTPUTS)-1:0] output_idx,  // Which neuron (0 to NUM_OUTPUTS-1)
    output logic output_valid,                           // Current output is valid
    output logic done                                    // All outputs computed
);
```

**Behavior**:
1. When `start` asserted, latch input values
2. Compute one output neuron per cycle:
   - `output_current = Σ(input[i] × weight[neuron_idx][i]) + bias[neuron_idx]`
3. Assert `output_valid` each cycle with corresponding `output_idx`
4. Assert `done` after NUM_OUTPUTS cycles
5. Outputs remain stable until next `start`

**Resource Usage**:
- NUM_INPUTS multipliers (e.g., 4 for input layer, 64 for hidden layer 2)
- 1 accumulator
- Weights storage: NUM_OUTPUTS × NUM_INPUTS × 16 bits
- Bias storage: NUM_OUTPUTS × 16 bits

**Timing**:
- Latency: 1 + NUM_OUTPUTS cycles (1 cycle to latch, then NUM_OUTPUTS cycles of output)
- Throughput: 1 output current per cycle (after first cycle)

### 3. `lif` Module (Single-Step)

**Purpose**: Leaky Integrate-and-Fire neuron with reset-by-subtraction. Processes one timestep per clock cycle with externally-managed timing.

**Interface**:
```systemverilog
module lif #(
    parameter THRESHOLD = 8192,       // Spike threshold (1.0 in QS2.13)
    parameter BETA = 115,             // Decay factor in Q1.7 format (115 ≈ 0.9)
    parameter DATA_WIDTH = 16,
    parameter MEMBRANE_WIDTH = 24
) (
    input wire clk,
    input wire reset,
    input wire clear,                                  // Clear membrane state for new inference
    input wire enable,                                 // Process one timestep
    input wire signed [DATA_WIDTH-1:0] current,       // Input current (QS2.13)
    output logic spike_out,                            // Spike output this timestep
    output logic signed [MEMBRANE_WIDTH-1:0] membrane_out  // Membrane potential after update
);
```

**Behavior**:
1. When `clear` asserted, reset membrane potential and spike history to zero
2. When `enable` asserted (and not `clear`), compute one timestep:
   - Apply decay: `decayed = (membrane × BETA) >> 7`
   - Apply reset: `reset_sub = spike_prev ? THRESHOLD : 0`
   - Update membrane: `membrane = decayed + current - reset_sub`
   - Generate spike: `spike = (membrane >= THRESHOLD)`
   - Store spike for next cycle's reset delay
3. Outputs are registered (valid on next clock edge after `enable`)

**State Update (per timestep)**:
```
decay_potential = (membrane_potential × BETA) >> 7
reset_subtract = (spike_prev) ? THRESHOLD : 0
membrane_potential = decay_potential + current - reset_subtract
spike = (membrane_potential >= THRESHOLD)
spike_prev = spike  // For next timestep's reset delay
```

**Resource Usage**:
- 1 multiplier (for beta decay)
- 24-bit membrane potential register
- 1-bit previous spike register

**Timing**:
- 1 cycle per timestep (externally driven by `enable`)
- Combinational path: current → membrane calculation
- Registered outputs: spike_out, membrane_out valid on next clock edge

**Note**: For the legacy version with internal timestep management, see `lif_timestep.sv`.

### 4. `spike_buffer` Module

**Purpose**: Store spike vectors from HL1 for each timestep. With synchronized HL1 processing, this becomes a simple register array.

**Interface**:
```systemverilog
module spike_buffer #(
    parameter NUM_NEURONS = 64,
    parameter NUM_TIMESTEPS = 30
) (
    input wire clk,
    input wire reset,
    input wire clear,                              // Clear for new inference
    input wire write_en,                           // Store current spike vector
    input wire [$clog2(NUM_TIMESTEPS)-1:0] write_timestep,  // Which timestep to write
    input wire [NUM_NEURONS-1:0] spike_in,         // Spike vector from all HL1 neurons
    input wire [$clog2(NUM_TIMESTEPS)-1:0] read_timestep,   // Which timestep to read
    output logic [NUM_NEURONS-1:0] spikes_out      // Spike vector for requested timestep
);
```

**Behavior**:
1. On `clear`, reset all storage
2. On `write_en`, store the 64-bit spike vector at `write_timestep`
3. Combinational read: `spikes_out = storage[read_timestep]`

**Note**: Since all HL1 LIFs process synchronously (same timestep together), we get a complete 64-bit spike vector each cycle—no complex staggered timing logic needed.

**Resource Usage**:
- Storage: NUM_NEURONS × NUM_TIMESTEPS bits (e.g., 64 × 30 = 1,920 bits)

### 5. `neuron_membrane_buffer` Module

**Purpose**: Per-neuron buffer storing membrane potentials across all timesteps

**Interface**:
```systemverilog
module neuron_membrane_buffer #(
    parameter NUM_TIMESTEPS = 30,
    parameter MEMBRANE_WIDTH = 24
) (
    input wire clk,
    input wire reset,
    input wire clear,                                      // Clear buffer for new inference
    input wire write_en,                                   // Write enable
    input wire [$clog2(NUM_TIMESTEPS)-1:0] write_timestep, // Which timestep to write
    input wire signed [MEMBRANE_WIDTH-1:0] membrane_in,    // Membrane value to store
    input wire [$clog2(NUM_TIMESTEPS)-1:0] read_timestep,  // Which timestep to read
    output logic signed [MEMBRANE_WIDTH-1:0] membrane_out, // Combinational read
    output logic full                                      // All timesteps written
);
```

**Behavior**:
1. Each LIF neuron in the final hidden layer gets its own instance
2. LIF writes membrane potential each timestep with `write_en` and `write_timestep`
3. `q_accumulator` reads via shared `read_timestep` (broadcast to all buffers)
4. Avoids fan-out issue of centralized buffer (no 384-wire bus to route)
5. Combinational read for low latency

**Resource Usage**:
- Per instance: NUM_TIMESTEPS × MEMBRANE_WIDTH bits = 30 × 24 = 720 bits
- 16 instances total: 11,520 bits (~1.4KB)

### 6. `q_accumulator` Module

**Purpose**: Compute Q-values by accumulating weighted membrane potentials across all timesteps

**Interface**:
```systemverilog
module q_accumulator #(
    parameter NUM_NEURONS = 16,
    parameter NUM_TIMESTEPS = 30,
    parameter NUM_ACTIONS = 2,
    parameter BATCH_SIZE = 4,            // Neurons processed per cycle
    parameter DATA_WIDTH = 16,
    parameter MEMBRANE_WIDTH = 24,
    parameter FRAC_BITS = 13,
    parameter WEIGHTS_FILE = "fc_out_weights.mem",
    parameter BIAS_FILE = "fc_out_bias.mem"
) (
    input wire clk,
    input wire reset,
    input wire start,
    output logic [$clog2(NUM_TIMESTEPS)-1:0] read_timestep, // Shared to all buffers
    input wire signed [MEMBRANE_WIDTH-1:0] membrane_in [0:NUM_NEURONS-1],
    output logic selected_action,
    output logic done
);
```

**Behavior**:
1. When `start` asserted, begin reading from all neuron membrane buffers
2. Process BATCH_SIZE neurons per cycle (parallel multipliers)
3. Accumulate: `Q[a] += Σ(w[a][n] × membrane[n][t]) + bias[a]` for each timestep
4. After all timesteps, divide by NUM_TIMESTEPS
5. Select action from full-precision divided Q-values: `selected_action = (Q[0] >= Q[1]) ? 0 : 1`
6. Assert `done` when `selected_action` is ready

**Note**: Q-values are not output because they routinely exceed the QS2.13 range (±4.0) and would saturate, losing the distinction between actions. The argmax is computed internally at full accumulator precision (~51 bits) where the values remain distinguishable.

**Resource Usage**:
- Multipliers: BATCH_SIZE × NUM_ACTIONS (e.g., 4 × 2 = 8 multipliers)
- Accumulators: NUM_ACTIONS wide accumulators (64-bit each)
- Weights: NUM_ACTIONS × NUM_NEURONS × 16 bits

**Timing**:
- Latency: NUM_TIMESTEPS × (NUM_NEURONS / BATCH_SIZE) + 2 cycles
- With defaults (BATCH_SIZE=4): 30 × 4 + 2 = 122 cycles
- Configurable: BATCH_SIZE=1 → 482 cycles, BATCH_SIZE=16 → 32 cycles

## Pipelined Execution Flow

The `neural_network` module coordinates all sub-modules to match snnTorch's forward pass exactly. The key insight is that **HL1 receives the same current each timestep** while **HL2 receives different current based on HL1's spikes**.

### Execution Strategy

The top-level state machine processes the network in phases:

```
IDLE → LOAD_HL1 → RUN_TIMESTEPS (with pipelined q_accumulator) → FINISH_Q → DONE
```

### Phase 1: Load Hidden Layer 1 Currents (Cycles 1-65)

**fc1 (linear_layer)**: Computes currents for all 64 HL1 neurons from observations.

```
Cycle 1:  Start fc1 with observations
Cycle 2:  fc1 outputs current[0] (output_valid=1), latch to hl1_current[0]
Cycle 3:  fc1 outputs current[1], latch to hl1_current[1]
...
Cycle 65: fc1 outputs current[63], latch to hl1_current[63], fc1.done=1
```

These currents are **latched** in registers (one per HL1 neuron) since they're reused every timestep.

**Latency**: 65 cycles

### Phase 2: Run Timesteps Loop with Pipelined Q-Accumulation

For each of the 30 timesteps, HL1, fc2, HL2, and q_accumulator operate in a coordinated pipeline:

**Timestep T processing (18 cycles per timestep)**:

```
Cycle T×18 + 66:     HL1 Step
                     - All 64 HL1 LIFs process in parallel (enable=1)
                     - Input: latched hl1_current[63:0] (same every timestep)
                     - Output: 64-bit spike vector
                     - Store spike vector in spike_buffer at timestep T

Cycle T×18 + 67:     Start fc2
                     - fc2.start=1 with spike_buffer output for timestep T

Cycles T×18 + 68-83: fc2 + HL2 Processing
                     - fc2 outputs 16 currents sequentially
                     - Each HL2 LIF processes as it receives its current
                     - Each HL2 LIF writes membrane to neuron_membrane_buffer[T]

Cycle T×18 + 83:     fc2.done=1, all HL2 neurons have processed timestep T
                     - Membrane buffers now have data for timestep T
                     - q_accumulator can process timestep T (pipelined)
```

**q_accumulator pipelining**:
- q_accumulator starts processing timestep 0 as soon as all HL2 membrane values for timestep 0 are ready
- Takes 4 cycles per timestep (4 batches of 4 neurons)
- Since HL2 takes 17 cycles per timestep, q_accumulator easily keeps up
- q_accumulator processes timestep T while HL2 works on timestep T+1

```
Timeline (simplified):
  Timestep 0: [HL1][----fc2+HL2----][q_acc T0]
  Timestep 1:                      [HL1][----fc2+HL2----][q_acc T1]
  Timestep 2:                                           [HL1][----fc2+HL2----][q_acc T2]
  ...
```

### Phase 3: Finish Q-Accumulation (Cycles ~606-609)

After timestep 29's HL2 completes, q_accumulator finishes processing timestep 29, then performs division and saturation.

```
Cycle ~606: q_accumulator processes last batch of timestep 29
Cycle ~607: Division by NUM_TIMESTEPS
Cycle ~608: Saturation and output
Cycle ~609: q_accumulator.done=1
```

### Phase 4: Done

```
Cycle ~610: neural_network.done=1, q_values stable
```

## Total Latency (Pipelined)

| Phase | Cycles | Cumulative |
|-------|--------|------------|
| Load HL1 currents (fc1) | 65 | 65 |
| Timestep loop (30 × 18) | 540 | 605 |
| Finish q_accumulator | ~4 | ~609 |
| **Total** | **~609** | |

@ 100MHz = **~6.1µs per inference**

Pipelining q_accumulator saves ~118 cycles compared to sequential execution (727 → 609 cycles, ~16% faster).

**Note**: This is fast enough for real-time CartPole control (typical requirement < 20ms response time)

### Timing Diagram (Pipelined)

```
Cycle:   1    65   66   83   84  101  ...  588  605  609
         |-----|----|----|----|----|------|----|----|---|
Phase:   [fc1  ][T0 ][T1 ][T2 ]...........[T29][q29][DIV]
                 │    │    │              │
                 └────┴────┴──q_acc───────┘ (pipelined)
```

### Latency Tradeoffs

**q_accumulator BATCH_SIZE** (pipelined execution):
| BATCH_SIZE | Multipliers | Cycles/Timestep | Can Pipeline? | Total Latency |
|------------|-------------|-----------------|---------------|---------------|
| 1          | 2           | 16              | Yes           | ~609 cycles   |
| 2          | 4           | 8               | Yes           | ~609 cycles   |
| 4 (default)| 8           | 4               | Yes           | ~609 cycles   |
| 8          | 16          | 2               | Yes           | ~609 cycles   |
| 16         | 32          | 1               | Yes           | ~609 cycles   |

With pipelining, q_accumulator BATCH_SIZE doesn't significantly affect total latency (as long as it can keep up with HL2's 17 cycles/timestep).

**Parallelizing fc2**: If fc2 outputs multiple currents per cycle, the timestep loop duration decreases proportionally, which would be the main lever for further latency reduction.

## Weight File Format

Weights exported from snn_policy.py must be formatted as:

### Linear Layer Weights
- **Format**: Row-major flattened array
- **File**: `fc1_weights.mem` (or similar)
- **Structure**:
  ```
  weight[0][0]  // neuron 0, input 0
  weight[0][1]  // neuron 0, input 1
  ...
  weight[0][N-1]  // neuron 0, last input
  weight[1][0]  // neuron 1, input 0
  ...
  ```
- **Encoding**: 16-bit hex, QS2.13 fixed-point

### Biases
- **Format**: 1D array
- **File**: `fc1_bias.mem` (or similar)
- **Structure**: One bias per neuron
- **Encoding**: 16-bit hex, QS2.13 fixed-point

### Weight File Configuration

Weight file paths are parameterized in `neural_network` and passed down to sub-modules. This allows different trained models to be loaded without modifying RTL.

For simulation, weight files can be specified via parameter overrides:
```bash
# cocotb/Makefile example
COMPILE_ARGS += -Pneural_network.FC1_WEIGHTS_FILE=\"path/to/fc1_weights.mem\"
```

For synthesis, a configuration file or top-level wrapper can set the paths.

## Resource Estimates (64-16 Network)

### Linear Layers
- **fc1**: 4 multipliers, 64×4×16 = 4,096 bits weights, 64×16 = 1,024 bits bias
- **fc2**: 64 multipliers, 16×64×16 = 16,384 bits weights, 16×16 = 256 bits bias

### q_accumulator (replaces fc_out linear layer)
- **Multipliers**: BATCH_SIZE × NUM_ACTIONS = 8 (with BATCH_SIZE=4)
- **Weights**: 2×16×16 = 512 bits
- **Biases**: 2×16 = 32 bits
- **Accumulators**: 2 × 64-bit = 128 bits

### LIF Neurons
- **Hidden Layer 1**: 64 instances × (1 multiplier + 24-bit membrane + 1-bit spike_prev) = 64 multipliers, 1,600 bits state
- **Hidden Layer 2**: 16 instances × (1 multiplier + 24-bit membrane + 1-bit spike_prev) = 16 multipliers, 400 bits state

### HL1 Current Registers
- **Purpose**: Store currents from fc1 for reuse across all timesteps
- **Size**: 64 neurons × 16 bits = 1,024 bits

### Membrane Buffers
- **neuron_membrane_buffer**: 16 instances × 30 timesteps × 24 bits = 11,520 bits (~1.4KB)

### Spike Buffer
- **spike_buffer**: 64 neurons × 30 timesteps = 1,920 bits
- Simplified design: just a register array with write/read timestep addressing

### Total Resource Usage
- **Multipliers**: 4 (fc1) + 64 (fc2) + 64 (HL1 LIF) + 16 (HL2 LIF) + 8 (q_accum) = **156 multipliers**
- **Memory**: 
  - Weights: ~22KB
  - Buffers/State: ~2KB
  - **Total**: ~24KB

## Internal Architecture

### Block Diagram

```
                    observations[3:0]
                          │
                          ▼
                    ┌──────────┐
                    │   fc1    │ (linear_layer)
                    │ 4→64     │
                    └────┬─────┘
                         │ currents[63:0] (latched)
                         ▼
              ┌──────────────────────┐
              │   HL1 LIF Array      │ (64 × lif)
              │   (same current      │
              │    each timestep)    │
              └──────────┬───────────┘
                         │ spikes[63:0]
                         ▼
                    ┌──────────┐
                    │   fc2    │ (linear_layer)
                    │ 64→16    │ (re-run each timestep)
                    └────┬─────┘
                         │ currents[15:0]
                         ▼
              ┌──────────────────────┐
              │   HL2 LIF Array      │ (16 × lif)
              │   (new current       │
              │    each timestep)    │
              └──────────┬───────────┘
                         │ membrane[15:0]
                         ▼
              ┌──────────────────────┐
              │  Membrane Buffers    │ (16 × neuron_membrane_buffer)
              └──────────┬───────────┘
                         │
                         ▼
                  ┌────────────┐
                  │q_accumulator│
                  │  (fc_out)   │
                  └──────┬─────┘
                         │
                         ▼
                  selected_action
```

### State Machine

```
                     ┌──────┐
         reset ──────│ IDLE │◄─────────────────────────┐
                     └──┬───┘                          │
                   start│                              │
                        ▼                              │
                  ┌──────────┐                         │
                  │ LOAD_HL1 │ fc1 running             │
                  └────┬─────┘                         │
               fc1.done│                               │
                       ▼                               │
               ┌─────────────┐                         │
          ┌────│RUN_TIMESTEPS│◄───┐                    │
          │    └──────┬──────┘    │                    │
          │           │           │                    │
          │    ┌──────▼──────┐    │                    │
          │    │  HL1_STEP   │    │  All 64 HL1 LIFs   │
          │    │             │    │  process in        │
          │    └──────┬──────┘    │  parallel          │
          │           │           │                    │
          │    ┌──────▼──────┐    │                    │
          │    │  FC2_START  │    │                    │
          │    └──────┬──────┘    │                    │
          │           │           │                    │
          │    ┌──────▼──────┐    │                    │
          │    │ HL2_PROCESS │    │ timestep<29        │
          │    │ (+q_acc     │    │                    │
          │    │  pipelined) │    │                    │
          │    └──────┬──────┘    │                    │
          │   fc2.done│           │                    │
          │           └───────────┘                    │
          │ timestep==29                               │
          │                                            │
          ▼                                            │
    ┌────────────┐                                     │
    │ FINISH_Q   │ q_accumulator finishes T29 + div   │
    └─────┬──────┘                                     │
    q.done│                                            │
          ▼                                            │
       ┌──────┐                                        │
       │ DONE │────────────────────────────────────────┘
       └──────┘
```

**Key changes from non-pipelined design**:
1. `HL1_STEP` processes all 64 neurons in parallel (synchronized)
2. `HL2_PROCESS` includes pipelined q_accumulator processing of the previous timestep
3. `FINISH_Q` replaces `Q_ACCUMULATE`—only needs to finish the last timestep's processing

## Software Integration

The `neural_network` module is designed to be the top-level interface for software interaction. It can be driven by:
1. **cocotb tests**: For functional verification using gymnasium environments
2. **FPGA integration**: For hardware-in-the-loop evaluation

**Note**: Action selection (`argmax(q_values)`) is performed **inside the hardware** (`q_accumulator`), not in software. This is necessary because the internal Q-values exceed the QS2.13 output range and would saturate to identical values, making a software argmax unreliable. The hardware computes the argmax at full internal precision (~51 bits).

### Hardware-Accelerated Policy Class

```python
class SNNPolicyHardware(nn.Module):
    """
    Hardware-accelerated SNN policy that offloads forward pass to FPGA.
    Maintains same interface as SNNPolicy for seamless integration with DQN training loop.
    """

    def __init__(self, n_observations, n_actions, fpga_interface):
        super().__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.fpga = fpga_interface  # FPGA communication interface

    def forward(self, observations):
        """
        Forward pass using hardware accelerator.

        Args:
            observations: [batch, 4] tensor of CartPole observations

        Returns:
            actions: [batch] tensor of selected actions (0 or 1)
        """
        batch_size = observations.size(0)
        actions = torch.zeros(batch_size, dtype=torch.long)

        # Process each sample in batch (hardware processes one at a time)
        for i in range(batch_size):
            obs = observations[i].cpu().numpy()

            # Convert to fixed-point and send to FPGA
            obs_fixed = self.float_to_qs2_13(obs)
            self.fpga.write_inputs(obs_fixed)
            self.fpga.start_inference()

            # Wait for completion and read selected action
            self.fpga.wait_done()
            actions[i] = self.fpga.read_selected_action()

        return actions

    def float_to_qs2_13(self, values):
        """Convert float array to QS2.13 fixed-point."""
        return np.round(values * 8192).astype(np.int16)

    def qs2_13_to_float(self, values):
        """Convert QS2.13 fixed-point to float array."""
        return values.astype(np.float32) / 8192.0
```

### cocotb Test Example

```python
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

@cocotb.test()
async def test_inference(dut):
    """Test a single inference through the neural network."""
    clock = Clock(dut.clk, 10, units="ns")  # 100 MHz
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Load observations (example CartPole state)
    observations = [0.01, -0.02, 0.03, 0.04]  # Example values
    for i, obs in enumerate(observations):
        dut.observations[i].value = float_to_fixed(obs)
    
    # Start inference
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for done
    while dut.done.value == 0:
        await RisingEdge(dut.clk)
    
    # Read selected action
    action = int(dut.selected_action.value)
    print(f"Selected action: {action}")
```

### Usage in Evaluation

```python
# Load trained weights (already quantized for hardware)
# Initialize FPGA with trained weights

# Create hardware policy
fpga_interface = FPGAInterface(device_path="/dev/fpga0")  # Platform-specific
policy_hw = SNNPolicyHardware(n_observations=4, n_actions=2, fpga_interface=fpga_interface)

# Use in existing evaluation loop (no changes needed)
env = gym.make("CartPole-v1")
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            actions = policy_hw(state_tensor)  # Hardware forward pass
        action = actions[0].item()  # Action selected in hardware
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
```

### Performance Validation

Compare hardware vs. software inference:
- Accuracy: Selected action should match software argmax for all test cases
- Latency: Hardware should be 6.17µs vs. software ~100µs (on CPU)
- Throughput: Measure frames per second in CartPole environment

### Testing Strategy

1. **Unit tests**: Verify fixed-point conversion accuracy
2. **Layer-by-layer tests**: Compare hardware vs. software outputs after each layer
3. **End-to-end tests**: Compare selected action vs. software argmax
4. **Episode tests**: Run full CartPole episodes, compare episode length distributions

## Notes and Future Enhancements

1. **LIF Module Variants**:
   - `lif.sv`: Single-step LIF with external timestep management (used in `neural_network`)
   - `lif_timestep.sv`: Legacy LIF with internal timestep loop (preserved for reference/testing)

2. **Fractional-Order Neurons**: If using fractional-order LIF neurons, each instance requires additional DSP resources for the Grünwald-Letnikov approximation

3. **Time-Multiplexing**: Current design uses parallel LIF arrays. For reduced resource usage, neurons could be time-multiplexed (e.g., 4 LIF instances processing 16 neurons over 4 cycles each).

4. **Reduced Precision**: Consider 8-bit or mixed-precision to reduce resource usage if accuracy permits

5. **Dynamic Timesteps**: Add early stopping mechanism if Q-values converge before 30 timesteps

6. **Membrane Potential Precision**: The output layer uses 24-bit membrane potentials but could potentially use reduced precision (16-bit) if testing shows acceptable accuracy

7. **fc2 Parallelization**: Processing multiple HL2 currents per cycle would reduce the timestep loop duration significantly
