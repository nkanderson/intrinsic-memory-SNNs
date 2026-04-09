# SNN-DQN CartPole Training

Deep Q-Network (DQN) implementation using Spiking Neural Networks (SNNs) for the CartPole-v1 environment. Original script based on an [example from the PyTorch documentation](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

## Quick Start

```bash
# Train with default baseline configuration
python main.py

# Train with live visualization
python main.py --human-render

# Load and evaluate a trained model (uses model's saved config)
python main.py --load models/dqn_baseline-best.pth --evaluate-only --human-render
```

## Configuration

Training hyperparameters and network settings are defined in YAML configuration files located in the `configs/` directory. When loading a saved model, its configuration is automatically loaded from the checkpoint - no config file is needed unless overriding specific parameters is necessary (as in the case of legacy checkpoint models).

### Configuration File Structure

```yaml
training:
  batch_size: 128          # Transitions sampled from replay buffer per optimization 
  gamma: 0.99              # Discount factor for future rewards
  eps_start: 0.9           # Initial exploration rate (epsilon-greedy)
  eps_end: 0.01            # Minimum exploration rate
  eps_decay: 2500          # Epsilon decay rate (higher = slower decay)
  tau: 0.005               # Target network soft update rate
  lr: 0.0003               # Learning rate for AdamW optimizer (3e-4)
  num_episodes: 600        # Total training episodes

snn:
  num_steps: 30            # SNN simulation timesteps per environment step
  beta: 0.9                # LIF neuron membrane decay rate
  surrogate_gradient_slope: 25  # Slope for fast_sigmoid surrogate gradient
  neuron_type: leaky       # Neuron type: 'leaky', 'leakysv', or 'fractional'
  hidden1_size: 64         # First hidden layer size
  hidden2_size: 16         # Second hidden layer size
  
  # Fractional-order LIF parameters (only used when neuron_type: fractional)
  alpha: 0.5               # Fractional order (0 < alpha < 1)
  lam: 0.111               # Decay rate parameter (lambda)
  history_length: 64       # Number of timesteps for GL memory
  dt: 1.0                  # Time step size
```

### Configuration Precedence

When loading a model with `--load`, the configuration precedence is:

1. **Checkpoint config** (highest priority) - Parameters saved in the model file
2. **Config file** - If `--config` is specified, fills in missing parameters from checkpoint (necessary for legacy models)
3. **Placeholder defaults** - Used when no config file and checkpoint is missing parameters

This allows you to:
- Load modern models without specifying a config file (checkpoint has everything)
- Use a config file to provide missing parameters for older checkpoints
- Train from scratch with a config file

### Example Available Configurations

- **`baseline.yaml`**: Standard leaky integrate-and-fire (LIF) neurons
- **`baseline-flif.yaml`**: Fractional-order LIF neurons with memory effects

## Command Line Arguments

### Configuration
- `--config`, `-c`: Path to YAML configuration file
  - Default: `configs/baseline.yaml` when training from scratch
  - Default: `None` when loading a model (uses checkpoint's saved config)
  - Can be specified when loading to override checkpoint parameters for legacy models

### Model Options
- `--load`: Load pre-trained model from file
- `--evaluate-only`: Only evaluate the loaded model without training

### Hardware & Visualization
- `--no-hw-acceleration`: Disable hardware acceleration (CUDA/MPS)
- `--human-render`: Show environment rendering and live training plot

## Usage Examples

### Training

```bash
# Basic training with baseline config
python main.py

# Train with custom configuration
python main.py --config configs/baseline.yaml

# Train with fractional LIF neurons
python main.py --config configs/baseline-flif.yaml

# Train with live visualization (slower but useful for debugging)
python main.py --human-render

# Train without hardware acceleration (CPU only)
python main.py --no-hw-acceleration
```

### Evaluation

```bash
# Evaluate best model with visualization (uses model's saved config)
python main.py --load models/dqn_baseline-best.pth --evaluate-only --human-render

# Evaluate final model
python main.py --load models/dqn_baseline-final.pth --evaluate-only

# Evaluate quantized model
python main.py --load models/dqn_baseline-best-quantized-QS2_5.pth --evaluate-only --human-render
```

### Resuming Training

```bash
# Continue training from a saved checkpoint (uses checkpoint's config)
python main.py --load models/dqn_baseline-best.pth

# Resume training but change training parameters (e.g., more episodes)
python main.py --load models/dqn_baseline-best.pth --config configs/extended-training.yaml
```

## Action Trace Scripts

Two helper scripts in `train/scripts/` support software vs hardware action-trace comparison.

- `export_action_trace.py`
  - Runs a trained PyTorch policy for fixed seeds and exports per-step trace rows.
  - Output columns include: seed, step, action, q0/q1, HL2 membrane summary, and observations.
  - Default output: `../metrics/cartpole_action_trace_sw.csv` (relative to `train/scripts/`).

- `compare_action_traces.py`
  - Compares software and hardware trace CSV files.
  - Reports, per seed, either:
    - actions identical through last overlapping step, or
    - first step where actions diverge (with software values, raw hardware values, and decoded hardware fixed-point values).

Hardware CSV note: `cartpole_action_trace_hw.csv` is generated by cocotb via `test_cartpole_action_trace` (run with `FULL_DEBUG=1`); see `../sv/cocotb/README.md` for the command flow.

Example workflow:

```bash
# From cartpole_snn/train/scripts

# 1) Export software trace from a checkpoint
python export_action_trace.py \
  --model ../models/dqn_fractional-32hl1-16hl2-8hist-best-generalization.pth \
  --seeds 42,49 \
  --output ../metrics/cartpole_action_trace_sw.csv

# 2) Compare against hardware trace exported by cocotb tests
python compare_action_traces.py \
  --sw ../metrics/cartpole_action_trace_sw.csv \
  --hw ../../sv/cocotb/results/cartpole_action_trace_hw.csv
```

## Output Files

Training produces two model checkpoints in the `models/` directory:

- **`models/dqn_<config-name>-best.pth`**: Model with highest average reward (over last 100 episodes)
- **`models/dqn_<config-name>-final.pth`**: Model from the final training episode

Each checkpoint contains:
- Policy network state (weights and biases)
- Target network state (weights and biases)
- Optimizer state (for resuming training)
- Episode number (training progress)
- Average reward (performance metric)
- **Full configuration** (all hyperparameters and network settings)
  - Training parameters (batch_size, gamma, learning rate, etc.)
  - SNN parameters (num_steps, beta, neuron_type, hidden sizes)
  - Fractional LIF parameters (if applicable: alpha, lam, history_length, dt)

This self-contained format means it is possible to evaluate or resume training from any checkpoint without needing the original config file.

## Weight Quantization

After training, model weights may be quantized for hardware deployment or to analyze the effects of reduced precision.

### Inspect Weights

Analyze weight distributions and get quantization format recommendations:

```bash
python scripts/manage_weights.py inspect models/dqn_config-baseline-best.pth
```

This displays:
- Overall weight statistics (min, max, mean, std)
- Suggested quantization format based on weight distribution
- Per-layer statistics

### Quantize Weights

Quantize weights to a specific fixed-point format and analyze quantization error (does not export or write out the quantized weights themselves, see export for weight file output):

```bash
# Quantize to QS1_6 format (8-bit signed: 1 sign + 1 int + 6 frac)
python scripts/manage_weights.py quantize models/dqn_config-baseline-best.pth --bits 8 --frac 6 --signed

# Quantize to Q3_5 format (8-bit unsigned: 3 int + 5 frac)
python scripts/manage_weights.py quantize models/dqn_config-baseline-best.pth --bits 8 --frac 5

# Show per-layer statistics
python scripts/manage_weights.py quantize models/dqn_config-baseline-best.pth --bits 8 --frac 6 --signed --verbose
```

The quantize command reports:
- Quantization configuration (scale factor, value range)
- Quantization error statistics:
  - **MSE** (Mean Squared Error): Average of squared differences `(original - quantized)²` across all weights
  - **mean_abs_error**: Average absolute difference `|original - quantized|` - typical error magnitude per weight
  - **max_abs_error**: Largest absolute difference for any single weight - worst-case quantization error
- Per-parameter statistics (with `--verbose`)

### Export Quantized Models

#### Export for PyTorch Evaluation

Export quantized weights as a PyTorch `.pth` file with dequantized float values:

```bash
python scripts/manage_weights.py export pytorch models/dqn_config-baseline-best.pth \
    --bits 8 --frac 6 --signed \
    --output models/dqn_config-baseline-best_quantized.pth
```

This creates a model checkpoint where weights represent the exact values that hardware will compute, but in PyTorch-compatible float format. It is then possible to evaluate performance degradation:

```bash
# Evaluate the quantized model
python main.py --load models/dqn_config-baseline-best_quantized.pth --evaluate-only --human-render
```

#### Export for Hardware Deployment

IN-PROGRESS: Export quantized weights as `.mem` files for Verilog/SystemVerilog hardware:

```bash
python scripts/manage_weights.py export hardware models/dqn_config-baseline-best.pth \
    --bits 8 --frac 6 --signed \
    --output weights_hw/
```

This generates:
- `fc1_weight.mem`, `fc1_bias.mem` - First linear layer
- `fc2_weight.mem`, `fc2_bias.mem` - Second linear layer  
- `fc_out_weight.mem`, `fc_out_bias.mem` - Output layer
- `metadata.json` - Quantization configuration

### Complete Quantization Workflow

```bash
# 1. Train model
python main.py --config configs/baseline.yaml

# 2. Inspect weights to determine optimal format
python scripts/manage_weights.py inspect models/dqn_baseline-best.pth

# 3. Quantize and check error
python scripts/manage_weights.py quantize models/dqn_baseline-best.pth --bits 8 --frac 6 --signed

# 4. Export for PyTorch validation
python scripts/manage_weights.py export pytorch models/dqn_baseline-best.pth \
    --bits 8 --frac 6 --signed \
    --output models/dqn_baseline-best_q8.pth

# 5. Evaluate quantized model
python main.py --load models/dqn_baseline-best_q8.pth --evaluate-only --human-render

# 6. If performance is acceptable, export for hardware
python scripts/manage_weights.py export hardware models/dqn_baseline-best.pth \
    --bits 8 --frac 6 --signed \
    --output weights_hw/
```

## Training Visualization

When using `--human-render`, the script displays:
- **Environment rendering**: Live CartPole simulation
- **Training plot**: Episode durations and 100-episode moving average

The plot updates during training and shows final results when complete.

## Parallel Optuna (CPU/GPU)

Use `optimize_multi_process.py` to run one shared Optuna study across multiple processes.

```bash
# CPU: run 8 worker processes
python optimize_multi_process.py \
  --device-mode cpu \
  --workers 8 \
  --neuron-type leaky \
  --n-trials 160

# GPU: run 4 workers pinned to GPUs 0-3
python optimize_multi_process.py \
  --device-mode gpu \
  --gpu-ids 0,1,2,3 \
  --workers 4 \
  --neuron-type leaky \
  --n-trials 160

# Auto mode: prefer GPU if CUDA is available, else CPU
python optimize_multi_process.py --device-mode auto --neuron-type fractional --n-trials 120
```

Notes:
- All workers share the same `--study-name` and `--storage`.
- `--n-trials` is split across workers as evenly as possible.
- With SQLite storage (`sqlite:///...`), keep CPU worker counts moderate (default cap: 8) to avoid DB write lock contention.
- Override SQLite cap only if needed: `--allow-oversubscribe-sqlite`.

After multi-process runs finish, export the final best config once from the complete study:

```bash
python optimize.py \
  --neuron-type <leaky|fractional|bitshift> \
  --study-name <your-study-name> \
  --storage <your-storage-uri> \
  --export-best
```

This avoids last-writer-wins behavior from multiple workers writing the same optimized config file.

## Tips for Hyperparameter Tuning

### Common Issues and Solutions

**Performance degrades after initial improvement:**
- Try reducing `tau` (e.g., 0.001) for more stable target network updates
- Increase `eps_decay` (e.g., 5000-10000) to maintain exploration longer
- Lower `lr` (e.g., 0.0001) for more stable learning

**Training is too slow:**
- Reduce `num_episodes` for faster experiments
- Ensure hardware acceleration is enabled (remove `--no-hw-acceleration`)
- Disable rendering (don't use `--human-render`)

**Agent not learning:**
- Check that `eps_decay` isn't too low (agent stops exploring too early)
- Verify `lr` isn't too small (learning too slow)
- Ensure `batch_size` provides sufficient samples

## Project Structure

```
cartpole_snn/train/
├── README.md                   # This file
├── main.py                     # Main training script
├── snn_policy.py               # SNN-based policy network
├── dqn_agent.py                # DQN agent implementation
├── leakysv.py                  # LeakySV neuron with refractory period
├── fractional_lif.py           # Fractional-order LIF neuron (FLIF)
├── utils.py                    # Utility functions (GL coefficients, etc.)
├── configs/                    # Configuration files
│   ├── baseline.yaml           # Standard LIF baseline
│   ├── baseline-flif.yaml      # Fractional-order LIF baseline
│   └── ...                     # Other experimental configs
├── scripts/                    # Utility scripts
│   ├── manage_weights.py       # Weight quantization and export tool
│   └── weights.py              # Weight manipulation library
└── models/                     # Saved model checkpoints
    ├── dqn_baseline-best.pth
    ├── dqn_baseline-final.pth
    └── ...
```

## Requirements

- Python 3.8+
- PyTorch
- snnTorch
- gymnasium
- PyYAML
- matplotlib

Install dependencies:
```bash
pip install -r ../../requirements-macos.txt  # macOS
# or
pip install -r ../../requirements.txt        # Linux/other
```
