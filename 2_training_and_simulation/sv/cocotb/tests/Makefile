# Makefile for running cocotb tests on SNN modules
#
# Usage:
#   make                                 # Run LIF tests (default)
#   make test_lif                        # Run LIF (single-step) tests
#   make test_fractional_lif             # Run fractional_lif (single-step) tests
#   make test_bitshift_lif               # Run bitshift_lif (single-step) tests
#   make test_lif_timestep               # Run LIF with internal timestep loop tests
#   make test_linear_layer               # Run linear_layer tests
#   make test_spike_buffer               # Run spike_buffer tests
#   make test_spike_buffer_staggered     # Run spike_buffer_staggered tests (legacy)
#   make test_membrane_buffer            # Run membrane_buffer tests
#   make test_neuron_membrane_buffer     # Run neuron_membrane_buffer tests
#   make cartpole_integration            # Run cartpole_integration tests
#   make cartpole_integration_fractional # Run cartpole_integration_fractional tests
#   make test_cartpole_integration_bitshift_custom_slow_decay # Run bitshift custom_slow_decay integration tests
#   make test_cartpole_integration_fractional_hist256 # Run cartpole_integration_fractional_hist256 tests
#   make clean                           # Clean build artifacts

# Default test module
TEST ?= lif

# Shared neuron source location (canonical, in-container volume path)
# docker-compose mounts ../../../common/sv/neurons -> /workspace/common/neurons
COMMON_NEURON_DIR ?= /workspace/common/neurons

# Configuration based on TEST variable
ifeq ($(TEST),lif)
    VERILOG_SOURCES = $(COMMON_NEURON_DIR)/lif.sv
    TOPLEVEL = lif
    MODULE = test_lif
else ifeq ($(TEST),lif_timestep)
    VERILOG_SOURCES = $(PWD)/../sv/lif_timestep.sv
    TOPLEVEL = lif_timestep
    MODULE = test_lif_timestep
else ifeq ($(TEST),fractional_lif)
    VERILOG_SOURCES = $(COMMON_NEURON_DIR)/fractional_lif.sv
    TOPLEVEL = fractional_lif
    MODULE = test_fractional_lif
else ifeq ($(TEST),fractional_lif_hist64)
    VERILOG_SOURCES = $(COMMON_NEURON_DIR)/fractional_lif.sv
    TOPLEVEL = fractional_lif
    MODULE = test_fractional_lif
else ifeq ($(TEST),bitshift_lif)
    VERILOG_SOURCES = $(COMMON_NEURON_DIR)/bitshift_lif.sv
    TOPLEVEL = bitshift_lif
    MODULE = test_bitshift_lif
else ifeq ($(TEST),bitshift_lif_mode0)
    VERILOG_SOURCES = $(COMMON_NEURON_DIR)/bitshift_lif.sv
    TOPLEVEL = bitshift_lif
    MODULE = test_bitshift_lif
else ifeq ($(TEST),bitshift_lif_mode1)
    VERILOG_SOURCES = $(COMMON_NEURON_DIR)/bitshift_lif.sv
    TOPLEVEL = bitshift_lif
    MODULE = test_bitshift_lif
else ifeq ($(TEST),bitshift_lif_mode2)
    VERILOG_SOURCES = $(COMMON_NEURON_DIR)/bitshift_lif.sv
    TOPLEVEL = bitshift_lif
    MODULE = test_bitshift_lif
else ifeq ($(TEST),bitshift_lif_mode3)
    VERILOG_SOURCES = $(COMMON_NEURON_DIR)/bitshift_lif.sv
    TOPLEVEL = bitshift_lif
    MODULE = test_bitshift_lif
else ifeq ($(TEST),linear_layer)
    VERILOG_SOURCES = $(PWD)/../sv/linear_layer.sv
    TOPLEVEL = linear_layer
    MODULE = test_linear_layer
else ifeq ($(TEST),spike_buffer)
    VERILOG_SOURCES = $(PWD)/../sv/spike_buffer.sv
    TOPLEVEL = spike_buffer
    MODULE = test_spike_buffer
else ifeq ($(TEST),spike_buffer_staggered)
    VERILOG_SOURCES = $(PWD)/../sv/spike_buffer_staggered.sv
    TOPLEVEL = spike_buffer_staggered
    MODULE = test_spike_buffer_staggered
else ifeq ($(TEST),membrane_buffer)
    VERILOG_SOURCES = $(PWD)/../sv/membrane_buffer.sv
    TOPLEVEL = membrane_buffer
    MODULE = test_membrane_buffer
else ifeq ($(TEST),neuron_membrane_buffer)
    VERILOG_SOURCES = $(PWD)/../sv/neuron_membrane_buffer.sv
    TOPLEVEL = neuron_membrane_buffer
    MODULE = test_neuron_membrane_buffer
else ifeq ($(TEST),q_accumulator)
    VERILOG_SOURCES = $(PWD)/../sv/q_accumulator.sv
    TOPLEVEL = q_accumulator
    MODULE = test_q_accumulator
else ifeq ($(TEST),neural_network)
    VERILOG_SOURCES = $(PWD)/../sv/neural_network.sv \
                      $(PWD)/../sv/linear_layer.sv \
                      $(COMMON_NEURON_DIR)/lif.sv \
                      $(PWD)/../sv/neuron_membrane_buffer.sv \
                      $(PWD)/../sv/q_accumulator.sv
    TOPLEVEL = neural_network
    MODULE = test_neural_network
else ifeq ($(TEST),cartpole_integration)
    # Full model integration test with gymnasium
    VERILOG_SOURCES = $(PWD)/../sv/neural_network.sv \
                      $(PWD)/../sv/linear_layer.sv \
                      $(COMMON_NEURON_DIR)/lif.sv \
                      $(PWD)/../sv/neuron_membrane_buffer.sv \
                      $(PWD)/../sv/q_accumulator.sv
    TOPLEVEL = neural_network
    MODULE = test_cartpole_integration
else ifeq ($(TEST),cartpole_integration_fractional)
    # Full model integration test with gymnasium using fractional LIF neurons
    VERILOG_SOURCES = $(PWD)/../sv/neural_network_fractional.sv \
                      $(PWD)/../sv/linear_layer.sv \
                      $(COMMON_NEURON_DIR)/fractional_lif.sv \
                      $(PWD)/../sv/neuron_membrane_buffer.sv \
                      $(PWD)/../sv/q_accumulator.sv
    TOPLEVEL = neural_network_fractional
    MODULE = test_cartpole_integration
else ifeq ($(TEST),cartpole_integration_fractional_hist8)
    # Full model integration test with gymnasium using fractional LIF neurons (history_length=8)
    VERILOG_SOURCES = $(PWD)/../sv/neural_network_fractional.sv \
                      $(PWD)/../sv/linear_layer.sv \
                      $(COMMON_NEURON_DIR)/fractional_lif.sv \
                      $(PWD)/../sv/neuron_membrane_buffer.sv \
                      $(PWD)/../sv/q_accumulator.sv
    TOPLEVEL = neural_network_fractional
    MODULE = test_cartpole_integration
else ifeq ($(TEST),cartpole_integration_fractional_hist256)
    # Full model integration test with gymnasium using fractional LIF neurons (history_length=256)
    VERILOG_SOURCES = $(PWD)/../sv/neural_network_fractional.sv \
                      $(PWD)/../sv/linear_layer.sv \
                      $(COMMON_NEURON_DIR)/fractional_lif.sv \
                      $(PWD)/../sv/neuron_membrane_buffer.sv \
                      $(PWD)/../sv/q_accumulator.sv
    TOPLEVEL = neural_network_fractional
    MODULE = test_cartpole_integration
else ifeq ($(TEST),cartpole_integration_bitshift_custom_slow_decay)
    # Full model integration test with gymnasium using bitshift custom_slow_decay neurons
    VERILOG_SOURCES = $(PWD)/../sv/neural_network_bitshift.sv \
                      $(PWD)/../sv/linear_layer.sv \
                      $(COMMON_NEURON_DIR)/bitshift_lif.sv \
                      $(PWD)/../sv/neuron_membrane_buffer.sv \
                      $(PWD)/../sv/q_accumulator.sv
    TOPLEVEL = neural_network_bitshift
    MODULE = test_cartpole_integration
else
    $(error Unknown TEST value: $(TEST). Use 'lif', 'fractional_lif', 'fractional_lif_hist64', 'bitshift_lif', 'bitshift_lif_mode0', 'bitshift_lif_mode1', 'bitshift_lif_mode2', 'bitshift_lif_mode3', 'lif_timestep', 'linear_layer', 'spike_buffer', 'spike_buffer_staggered', 'membrane_buffer', 'neuron_membrane_buffer', 'q_accumulator', 'neural_network', 'cartpole_integration', 'cartpole_integration_fractional', 'cartpole_integration_fractional_hist8', 'cartpole_integration_fractional_hist256', or 'cartpole_integration_bitshift_custom_slow_decay')
endif

# Simulator - use Icarus for all tests (event-driven, correct timing)
SIM := icarus

# Simulator-specific flags
# Icarus and Verilator differ in their naming of parameter overrides
ifeq ($(SIM),icarus)
    # Enable waveform tracing (FST format by default)
    WAVES = 1
    # Icarus uses -P for parameter overrides
    ifeq ($(TEST),linear_layer)
        COMPILE_ARGS += -Plinear_layer.NUM_INPUTS=4 -Plinear_layer.NUM_OUTPUTS=4
        COMPILE_ARGS += -Plinear_layer.WEIGHTS_FILE=\"$(PWD)/weights/test_weights.mem\"
        COMPILE_ARGS += -Plinear_layer.BIAS_FILE=\"$(PWD)/weights/test_bias.mem\"
    endif
    ifeq ($(TEST),spike_buffer)
        COMPILE_ARGS += -Pspike_buffer.NUM_NEURONS=8 -Pspike_buffer.NUM_TIMESTEPS=4
    endif
    ifeq ($(TEST),spike_buffer_staggered)
        COMPILE_ARGS += -Pspike_buffer_staggered.NUM_NEURONS=8 -Pspike_buffer_staggered.NUM_TIMESTEPS=4
    endif
    ifeq ($(TEST),fractional_lif)
        COMPILE_ARGS += -Pfractional_lif.HISTORY_LENGTH=8
        COMPILE_ARGS += -Pfractional_lif.COEFF_WIDTH=16 -Pfractional_lif.COEFF_FRAC_BITS=15
        COMPILE_ARGS += -Pfractional_lif.GL_COEFF_FILE=\"$(PWD)/weights/fractional_gl_coeffs_h8.mem\"
    endif
    ifeq ($(TEST),fractional_lif_hist64)
        # Match params from fractional_constants.txt for original 64-16 model
        COMPILE_ARGS += -Pfractional_lif.HISTORY_LENGTH=64
        COMPILE_ARGS += -Pfractional_lif.COEFF_WIDTH=16 -Pfractional_lif.COEFF_FRAC_BITS=15
        COMPILE_ARGS += -Pfractional_lif.C_SCALED=256 -Pfractional_lif.INV_DENOM=58988
        COMPILE_ARGS += -Pfractional_lif.GL_COEFF_FILE=\"$(PWD)/weights/fractional_order/gl_coefficients.mem\"
    endif
    ifeq ($(TEST),bitshift_lif)
        COMPILE_ARGS += -Pbitshift_lif.HISTORY_LENGTH=64
        COMPILE_ARGS += -Pbitshift_lif.SHIFT_WIDTH=8 -Pbitshift_lif.SHIFT_MODE=3
        COMPILE_ARGS += -Pbitshift_lif.CUSTOM_DECAY_RATE=3
        COMPILE_ARGS += -Pbitshift_lif.C_SCALED=256 -Pbitshift_lif.INV_DENOM=58982
    endif
    ifeq ($(TEST),bitshift_lif_mode0)
        COMPILE_ARGS += -Pbitshift_lif.HISTORY_LENGTH=64
        COMPILE_ARGS += -Pbitshift_lif.SHIFT_WIDTH=8 -Pbitshift_lif.SHIFT_MODE=0
        COMPILE_ARGS += -Pbitshift_lif.CUSTOM_DECAY_RATE=3
        COMPILE_ARGS += -Pbitshift_lif.C_SCALED=256 -Pbitshift_lif.INV_DENOM=58982
    endif
    ifeq ($(TEST),bitshift_lif_mode1)
        COMPILE_ARGS += -Pbitshift_lif.HISTORY_LENGTH=64
        COMPILE_ARGS += -Pbitshift_lif.SHIFT_WIDTH=8 -Pbitshift_lif.SHIFT_MODE=1
        COMPILE_ARGS += -Pbitshift_lif.CUSTOM_DECAY_RATE=3
        COMPILE_ARGS += -Pbitshift_lif.C_SCALED=256 -Pbitshift_lif.INV_DENOM=58982
    endif
    ifeq ($(TEST),bitshift_lif_mode2)
        COMPILE_ARGS += -Pbitshift_lif.HISTORY_LENGTH=64
        COMPILE_ARGS += -Pbitshift_lif.SHIFT_WIDTH=8 -Pbitshift_lif.SHIFT_MODE=2
        COMPILE_ARGS += -Pbitshift_lif.CUSTOM_DECAY_RATE=3
        COMPILE_ARGS += -Pbitshift_lif.C_SCALED=256 -Pbitshift_lif.INV_DENOM=58982
    endif
    ifeq ($(TEST),bitshift_lif_mode3)
        COMPILE_ARGS += -Pbitshift_lif.HISTORY_LENGTH=64
        COMPILE_ARGS += -Pbitshift_lif.SHIFT_WIDTH=8 -Pbitshift_lif.SHIFT_MODE=3
        COMPILE_ARGS += -Pbitshift_lif.CUSTOM_DECAY_RATE=3
        COMPILE_ARGS += -Pbitshift_lif.C_SCALED=256 -Pbitshift_lif.INV_DENOM=58982
    endif
    ifeq ($(TEST),membrane_buffer)
        COMPILE_ARGS += -Pmembrane_buffer.NUM_NEURONS=4 -Pmembrane_buffer.NUM_TIMESTEPS=3
    endif
    ifeq ($(TEST),neuron_membrane_buffer)
        COMPILE_ARGS += -Pneuron_membrane_buffer.NUM_TIMESTEPS=8
    endif
    ifeq ($(TEST),q_accumulator)
        COMPILE_ARGS += -Pq_accumulator.NUM_NEURONS=4 -Pq_accumulator.NUM_TIMESTEPS=4
        COMPILE_ARGS += -Pq_accumulator.NUM_ACTIONS=2 -Pq_accumulator.BATCH_SIZE=2
        COMPILE_ARGS += -Pq_accumulator.WEIGHTS_FILE=\"$(PWD)/weights/q_accum_weights.mem\"
        COMPILE_ARGS += -Pq_accumulator.BIAS_FILE=\"$(PWD)/weights/q_accum_bias.mem\"
    endif
    ifeq ($(TEST),neural_network)
        COMPILE_ARGS += -Pneural_network.NUM_INPUTS=4 -Pneural_network.HL1_SIZE=8
        COMPILE_ARGS += -Pneural_network.HL2_SIZE=4 -Pneural_network.NUM_ACTIONS=2
        COMPILE_ARGS += -Pneural_network.NUM_TIMESTEPS=4 -Pneural_network.Q_BATCH_SIZE=2
        COMPILE_ARGS += -Pneural_network.FC1_WEIGHTS_FILE=\"$(PWD)/weights/nn_fc1_weights.mem\"
        COMPILE_ARGS += -Pneural_network.FC1_BIAS_FILE=\"$(PWD)/weights/nn_fc1_bias.mem\"
        COMPILE_ARGS += -Pneural_network.FC2_WEIGHTS_FILE=\"$(PWD)/weights/nn_fc2_weights.mem\"
        COMPILE_ARGS += -Pneural_network.FC2_BIAS_FILE=\"$(PWD)/weights/nn_fc2_bias.mem\"
        COMPILE_ARGS += -Pneural_network.FC_OUT_WEIGHTS_FILE=\"$(PWD)/weights/nn_fc_out_weights.mem\"
        COMPILE_ARGS += -Pneural_network.FC_OUT_BIAS_FILE=\"$(PWD)/weights/nn_fc_out_bias.mem\"
    endif
    ifeq ($(TEST),cartpole_integration)
        # Full model: 64 HL1, 16 HL2, 30 timesteps
        COMPILE_ARGS += -Pneural_network.NUM_INPUTS=4 -Pneural_network.HL1_SIZE=64
        COMPILE_ARGS += -Pneural_network.HL2_SIZE=16 -Pneural_network.NUM_ACTIONS=2
        COMPILE_ARGS += -Pneural_network.NUM_TIMESTEPS=30 -Pneural_network.Q_BATCH_SIZE=4
        # FC2 output transport width must be wide enough to avoid clipping before FC_OUT.
        COMPILE_ARGS += -Pneural_network.FC2_OUTPUT_WIDTH=24
        COMPILE_ARGS += -Pneural_network.FC1_WEIGHTS_FILE=\"$(PWD)/weights/fc1_weights.mem\"
        COMPILE_ARGS += -Pneural_network.FC1_BIAS_FILE=\"$(PWD)/weights/fc1_bias.mem\"
        COMPILE_ARGS += -Pneural_network.FC2_WEIGHTS_FILE=\"$(PWD)/weights/fc2_weights.mem\"
        COMPILE_ARGS += -Pneural_network.FC2_BIAS_FILE=\"$(PWD)/weights/fc2_bias.mem\"
        COMPILE_ARGS += -Pneural_network.FC_OUT_WEIGHTS_FILE=\"$(PWD)/weights/fc_out_weights.mem\"
        COMPILE_ARGS += -Pneural_network.FC_OUT_BIAS_FILE=\"$(PWD)/weights/fc_out_bias.mem\"
    endif
    ifeq ($(TEST),cartpole_integration_fractional)
        # Fractional model from weights/fractional_order: 32 HL1, 16 HL2, 30 timesteps
        COMPILE_ARGS += -Pneural_network_fractional.NUM_INPUTS=4 -Pneural_network_fractional.HL1_SIZE=32
        COMPILE_ARGS += -Pneural_network_fractional.HL2_SIZE=16 -Pneural_network_fractional.NUM_ACTIONS=2
        COMPILE_ARGS += -Pneural_network_fractional.NUM_TIMESTEPS=30 -Pneural_network_fractional.Q_BATCH_SIZE=4
        # FC2 output transport width must be wide enough to avoid clipping before FC_OUT; 24 bits is the smallest stable value observed for this profile.
        COMPILE_ARGS += -Pneural_network_fractional.FC2_OUTPUT_WIDTH=24
        COMPILE_ARGS += -Pneural_network_fractional.HISTORY_LENGTH=64
        COMPILE_ARGS += -Pneural_network_fractional.COEFF_WIDTH=16 -Pneural_network_fractional.COEFF_FRAC_BITS=15
        COMPILE_ARGS += -Pneural_network_fractional.C_SCALED=256 -Pneural_network_fractional.INV_DENOM=58988
        COMPILE_ARGS += -Pneural_network_fractional.GL_COEFF_FILE=\"$(PWD)/weights/fractional_order/gl_coefficients.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC1_WEIGHTS_FILE=\"$(PWD)/weights/fractional_order/fc1_weights.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC1_BIAS_FILE=\"$(PWD)/weights/fractional_order/fc1_bias.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC2_WEIGHTS_FILE=\"$(PWD)/weights/fractional_order/fc2_weights.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC2_BIAS_FILE=\"$(PWD)/weights/fractional_order/fc2_bias.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC_OUT_WEIGHTS_FILE=\"$(PWD)/weights/fractional_order/fc_out_weights.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC_OUT_BIAS_FILE=\"$(PWD)/weights/fractional_order/fc_out_bias.mem\"
    endif
    ifeq ($(TEST),cartpole_integration_fractional_hist8)
        # Fractional model with history_length=8 from weights/fractional_hist8: 32 HL1, 16 HL2, 30 timesteps
        COMPILE_ARGS += -Pneural_network_fractional.NUM_INPUTS=4 -Pneural_network_fractional.HL1_SIZE=32
        COMPILE_ARGS += -Pneural_network_fractional.HL2_SIZE=16 -Pneural_network_fractional.NUM_ACTIONS=2
        COMPILE_ARGS += -Pneural_network_fractional.NUM_TIMESTEPS=30 -Pneural_network_fractional.Q_BATCH_SIZE=4
        # Same FC2 sizing rule as hist64: keep enough headroom to prevent saturation in observation sweeps.
        COMPILE_ARGS += -Pneural_network_fractional.FC2_OUTPUT_WIDTH=24
        COMPILE_ARGS += -Pneural_network_fractional.HISTORY_LENGTH=8
        COMPILE_ARGS += -Pneural_network_fractional.COEFF_WIDTH=16 -Pneural_network_fractional.COEFF_FRAC_BITS=15
        COMPILE_ARGS += -Pneural_network_fractional.C_SCALED=256 -Pneural_network_fractional.INV_DENOM=61388
        COMPILE_ARGS += -Pneural_network_fractional.GL_COEFF_FILE=\"$(PWD)/weights/fractional_hist8/gl_coefficients.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC1_WEIGHTS_FILE=\"$(PWD)/weights/fractional_hist8/fc1_weights.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC1_BIAS_FILE=\"$(PWD)/weights/fractional_hist8/fc1_bias.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC2_WEIGHTS_FILE=\"$(PWD)/weights/fractional_hist8/fc2_weights.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC2_BIAS_FILE=\"$(PWD)/weights/fractional_hist8/fc2_bias.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC_OUT_WEIGHTS_FILE=\"$(PWD)/weights/fractional_hist8/fc_out_weights.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC_OUT_BIAS_FILE=\"$(PWD)/weights/fractional_hist8/fc_out_bias.mem\"
    endif
    ifeq ($(TEST),cartpole_integration_fractional_hist256)
        # Fractional model with history_length=256 from weights/fractional_hist256: 16 HL1, 4 HL2, 30 timesteps
        COMPILE_ARGS += -Pneural_network_fractional.NUM_INPUTS=4 -Pneural_network_fractional.HL1_SIZE=16
        COMPILE_ARGS += -Pneural_network_fractional.HL2_SIZE=4 -Pneural_network_fractional.NUM_ACTIONS=2
        COMPILE_ARGS += -Pneural_network_fractional.NUM_TIMESTEPS=30 -Pneural_network_fractional.Q_BATCH_SIZE=4
        # Same FC2 sizing rule: keep enough headroom to prevent saturation in observation sweeps.
        COMPILE_ARGS += -Pneural_network_fractional.FC2_OUTPUT_WIDTH=24
        COMPILE_ARGS += -Pneural_network_fractional.HISTORY_LENGTH=256
        COMPILE_ARGS += -Pneural_network_fractional.COEFF_WIDTH=16 -Pneural_network_fractional.COEFF_FRAC_BITS=15
        COMPILE_ARGS += -Pneural_network_fractional.C_SCALED=256 -Pneural_network_fractional.INV_DENOM=57553
        COMPILE_ARGS += -Pneural_network_fractional.GL_COEFF_FILE=\"$(PWD)/weights/fractional_hist256/gl_coefficients.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC1_WEIGHTS_FILE=\"$(PWD)/weights/fractional_hist256/fc1_weights.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC1_BIAS_FILE=\"$(PWD)/weights/fractional_hist256/fc1_bias.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC2_WEIGHTS_FILE=\"$(PWD)/weights/fractional_hist256/fc2_weights.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC2_BIAS_FILE=\"$(PWD)/weights/fractional_hist256/fc2_bias.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC_OUT_WEIGHTS_FILE=\"$(PWD)/weights/fractional_hist256/fc_out_weights.mem\"
        COMPILE_ARGS += -Pneural_network_fractional.FC_OUT_BIAS_FILE=\"$(PWD)/weights/fractional_hist256/fc_out_bias.mem\"
    endif
    ifeq ($(TEST),cartpole_integration_bitshift_custom_slow_decay)
        # Bitshift custom_slow_decay model from weights/bitshift-custom_slow_decay: 32 HL1, 8 HL2, 10 timesteps
        # NOTE: This profile uses QS3.12 weights. Keep FRAC_BITS=12 and constants aligned
        # with train/configs/bitshift-custom_slow_decay-32hl1-8hl2-8hist.yaml + generated
        # fractional_constants.txt (C_SCALED=256, INV_DENOM=59823).
        COMPILE_ARGS += -Pneural_network_bitshift.NUM_INPUTS=4 -Pneural_network_bitshift.HL1_SIZE=32
        COMPILE_ARGS += -Pneural_network_bitshift.HL2_SIZE=8 -Pneural_network_bitshift.NUM_ACTIONS=2
        COMPILE_ARGS += -Pneural_network_bitshift.NUM_TIMESTEPS=10 -Pneural_network_bitshift.Q_BATCH_SIZE=4
        COMPILE_ARGS += -Pneural_network_bitshift.FRAC_BITS=12
        COMPILE_ARGS += -Pneural_network_bitshift.FC2_OUTPUT_WIDTH=24
        COMPILE_ARGS += -Pneural_network_bitshift.HISTORY_LENGTH=8
        COMPILE_ARGS += -Pneural_network_bitshift.SHIFT_WIDTH=8 -Pneural_network_bitshift.SHIFT_MODE=3
        COMPILE_ARGS += -Pneural_network_bitshift.CUSTOM_DECAY_RATE=3
        COMPILE_ARGS += -Pneural_network_bitshift.C_SCALED=256 -Pneural_network_bitshift.INV_DENOM=59823
        COMPILE_ARGS += -Pneural_network_bitshift.FC1_WEIGHTS_FILE=\"$(PWD)/weights/bitshift-custom_slow_decay/fc1_weights.mem\"
        COMPILE_ARGS += -Pneural_network_bitshift.FC1_BIAS_FILE=\"$(PWD)/weights/bitshift-custom_slow_decay/fc1_bias.mem\"
        COMPILE_ARGS += -Pneural_network_bitshift.FC2_WEIGHTS_FILE=\"$(PWD)/weights/bitshift-custom_slow_decay/fc2_weights.mem\"
        COMPILE_ARGS += -Pneural_network_bitshift.FC2_BIAS_FILE=\"$(PWD)/weights/bitshift-custom_slow_decay/fc2_bias.mem\"
        COMPILE_ARGS += -Pneural_network_bitshift.FC_OUT_WEIGHTS_FILE=\"$(PWD)/weights/bitshift-custom_slow_decay/fc_out_weights.mem\"
        COMPILE_ARGS += -Pneural_network_bitshift.FC_OUT_BIAS_FILE=\"$(PWD)/weights/bitshift-custom_slow_decay/fc_out_bias.mem\"
    endif
endif

# Verilator should be used when needing to trace unpacked arrays
ifeq ($(SIM),verilator)
    # Enable waveform tracing (VCD format)
    WAVES = 1
    EXTRA_ARGS += --trace --trace-fst
    # Verilator uses -G for parameter overrides
    ifeq ($(TEST),linear_layer)
        COMPILE_ARGS += -GNUM_INPUTS=4 -GNUM_OUTPUTS=4
        COMPILE_ARGS += -GWEIGHTS_FILE=\"$(PWD)/weights/test_weights.mem\"
        COMPILE_ARGS += -GBIAS_FILE=\"$(PWD)/weights/test_bias.mem\"
    endif
    ifeq ($(TEST),spike_buffer)
        COMPILE_ARGS += -GNUM_NEURONS=8 -GNUM_TIMESTEPS=4
    endif
    ifeq ($(TEST),spike_buffer_staggered)
        COMPILE_ARGS += -GNUM_NEURONS=8 -GNUM_TIMESTEPS=4
    endif
    ifeq ($(TEST),fractional_lif)
        COMPILE_ARGS += -GHISTORY_LENGTH=8
        COMPILE_ARGS += -GCOEFF_WIDTH=16 -GCOEFF_FRAC_BITS=15
        COMPILE_ARGS += -GGL_COEFF_FILE=\"$(PWD)/weights/fractional_gl_coeffs_h8.mem\"
    endif
    ifeq ($(TEST),fractional_lif_hist64)
				# Match params from fractional_constants.txt for original 64-16 model
        COMPILE_ARGS += -GHISTORY_LENGTH=64
        COMPILE_ARGS += -GCOEFF_WIDTH=16 -GCOEFF_FRAC_BITS=15
        COMPILE_ARGS += -GC_SCALED=256 -GINV_DENOM=58988
        COMPILE_ARGS += -GGL_COEFF_FILE=\"$(PWD)/weights/fractional_order/gl_coefficients.mem\"
    endif
    ifeq ($(TEST),bitshift_lif)
        COMPILE_ARGS += -GHISTORY_LENGTH=64
        COMPILE_ARGS += -GSHIFT_WIDTH=8 -GSHIFT_MODE=3
        COMPILE_ARGS += -GCUSTOM_DECAY_RATE=3
        COMPILE_ARGS += -GC_SCALED=256 -GINV_DENOM=58982
    endif
    ifeq ($(TEST),bitshift_lif_mode0)
        COMPILE_ARGS += -GHISTORY_LENGTH=64
        COMPILE_ARGS += -GSHIFT_WIDTH=8 -GSHIFT_MODE=0
        COMPILE_ARGS += -GCUSTOM_DECAY_RATE=3
        COMPILE_ARGS += -GC_SCALED=256 -GINV_DENOM=58982
    endif
    ifeq ($(TEST),bitshift_lif_mode1)
        COMPILE_ARGS += -GHISTORY_LENGTH=64
        COMPILE_ARGS += -GSHIFT_WIDTH=8 -GSHIFT_MODE=1
        COMPILE_ARGS += -GCUSTOM_DECAY_RATE=3
        COMPILE_ARGS += -GC_SCALED=256 -GINV_DENOM=58982
    endif
    ifeq ($(TEST),bitshift_lif_mode2)
        COMPILE_ARGS += -GHISTORY_LENGTH=64
        COMPILE_ARGS += -GSHIFT_WIDTH=8 -GSHIFT_MODE=2
        COMPILE_ARGS += -GCUSTOM_DECAY_RATE=3
        COMPILE_ARGS += -GC_SCALED=256 -GINV_DENOM=58982
    endif
    ifeq ($(TEST),bitshift_lif_mode3)
        COMPILE_ARGS += -GHISTORY_LENGTH=64
        COMPILE_ARGS += -GSHIFT_WIDTH=8 -GSHIFT_MODE=3
        COMPILE_ARGS += -GCUSTOM_DECAY_RATE=3
        COMPILE_ARGS += -GC_SCALED=256 -GINV_DENOM=58982
    endif
    ifeq ($(TEST),membrane_buffer)
        COMPILE_ARGS += -GNUM_NEURONS=4 -GNUM_TIMESTEPS=3
    endif
    ifeq ($(TEST),neuron_membrane_buffer)
        COMPILE_ARGS += -GNUM_TIMESTEPS=8
    endif
    ifeq ($(TEST),q_accumulator)
        COMPILE_ARGS += -GNUM_NEURONS=4 -GNUM_TIMESTEPS=4
        COMPILE_ARGS += -GNUM_ACTIONS=2 -GBATCH_SIZE=2
        COMPILE_ARGS += -GWEIGHTS_FILE=\"$(PWD)/weights/q_accum_weights.mem\"
        COMPILE_ARGS += -GBIAS_FILE=\"$(PWD)/weights/q_accum_bias.mem\"
    endif
    ifeq ($(TEST),neural_network)
        COMPILE_ARGS += -GNUM_INPUTS=4 -GHL1_SIZE=8
        COMPILE_ARGS += -GHL2_SIZE=4 -GNUM_ACTIONS=2
        COMPILE_ARGS += -GNUM_TIMESTEPS=4 -GQ_BATCH_SIZE=2
        COMPILE_ARGS += -GFC1_WEIGHTS_FILE=\"$(PWD)/weights/nn_fc1_weights.mem\"
        COMPILE_ARGS += -GFC1_BIAS_FILE=\"$(PWD)/weights/nn_fc1_bias.mem\"
        COMPILE_ARGS += -GFC2_WEIGHTS_FILE=\"$(PWD)/weights/nn_fc2_weights.mem\"
        COMPILE_ARGS += -GFC2_BIAS_FILE=\"$(PWD)/weights/nn_fc2_bias.mem\"
        COMPILE_ARGS += -GFC_OUT_WEIGHTS_FILE=\"$(PWD)/weights/nn_fc_out_weights.mem\"
        COMPILE_ARGS += -GFC_OUT_BIAS_FILE=\"$(PWD)/weights/nn_fc_out_bias.mem\"
    endif
    ifeq ($(TEST),cartpole_integration)
        # Full model: 64 HL1, 16 HL2, 30 timesteps
        COMPILE_ARGS += -GNUM_INPUTS=4 -GHL1_SIZE=64
        COMPILE_ARGS += -GHL2_SIZE=16 -GNUM_ACTIONS=2
        COMPILE_ARGS += -GNUM_TIMESTEPS=30 -GQ_BATCH_SIZE=4
        COMPILE_ARGS += -GFC1_WEIGHTS_FILE=\"$(PWD)/weights/fc1_weights.mem\"
        COMPILE_ARGS += -GFC1_BIAS_FILE=\"$(PWD)/weights/fc1_bias.mem\"
        COMPILE_ARGS += -GFC2_WEIGHTS_FILE=\"$(PWD)/weights/fc2_weights.mem\"
        COMPILE_ARGS += -GFC2_BIAS_FILE=\"$(PWD)/weights/fc2_bias.mem\"
        COMPILE_ARGS += -GFC_OUT_WEIGHTS_FILE=\"$(PWD)/weights/fc_out_weights.mem\"
        COMPILE_ARGS += -GFC_OUT_BIAS_FILE=\"$(PWD)/weights/fc_out_bias.mem\"
    endif
    ifeq ($(TEST),cartpole_integration_fractional)
        # Fractional model from weights/fractional_order: 32 HL1, 16 HL2, 30 timesteps
        COMPILE_ARGS += -GNUM_INPUTS=4 -GHL1_SIZE=32
        COMPILE_ARGS += -GHL2_SIZE=16 -GNUM_ACTIONS=2
        COMPILE_ARGS += -GNUM_TIMESTEPS=30 -GQ_BATCH_SIZE=4
        # FC2 output transport width must be wide enough to avoid clipping before FC_OUT; 24 bits is the smallest stable value observed for this profile.
        COMPILE_ARGS += -GFC2_OUTPUT_WIDTH=24
        COMPILE_ARGS += -GHISTORY_LENGTH=64
        COMPILE_ARGS += -GCOEFF_WIDTH=16 -GCOEFF_FRAC_BITS=15
        COMPILE_ARGS += -GC_SCALED=256 -GINV_DENOM=58988
        COMPILE_ARGS += -GGL_COEFF_FILE=\"$(PWD)/weights/fractional_order/gl_coefficients.mem\"
        COMPILE_ARGS += -GFC1_WEIGHTS_FILE=\"$(PWD)/weights/fractional_order/fc1_weights.mem\"
        COMPILE_ARGS += -GFC1_BIAS_FILE=\"$(PWD)/weights/fractional_order/fc1_bias.mem\"
        COMPILE_ARGS += -GFC2_WEIGHTS_FILE=\"$(PWD)/weights/fractional_order/fc2_weights.mem\"
        COMPILE_ARGS += -GFC2_BIAS_FILE=\"$(PWD)/weights/fractional_order/fc2_bias.mem\"
        COMPILE_ARGS += -GFC_OUT_WEIGHTS_FILE=\"$(PWD)/weights/fractional_order/fc_out_weights.mem\"
        COMPILE_ARGS += -GFC_OUT_BIAS_FILE=\"$(PWD)/weights/fractional_order/fc_out_bias.mem\"
    endif
    ifeq ($(TEST),cartpole_integration_fractional_hist8)
        # Fractional model with history_length=8 from weights/fractional_hist8: 32 HL1, 16 HL2, 30 timesteps
        COMPILE_ARGS += -GNUM_INPUTS=4 -GHL1_SIZE=32
        COMPILE_ARGS += -GHL2_SIZE=16 -GNUM_ACTIONS=2
        COMPILE_ARGS += -GNUM_TIMESTEPS=30 -GQ_BATCH_SIZE=4
        # Same FC2 sizing rule as hist64: keep enough headroom to prevent saturation in observation sweeps.
        COMPILE_ARGS += -GFC2_OUTPUT_WIDTH=24
        COMPILE_ARGS += -GHISTORY_LENGTH=8
        COMPILE_ARGS += -GCOEFF_WIDTH=16 -GCOEFF_FRAC_BITS=15
        COMPILE_ARGS += -GC_SCALED=256 -GINV_DENOM=61388
        COMPILE_ARGS += -GGL_COEFF_FILE=\"$(PWD)/weights/fractional_hist8/gl_coefficients.mem\"
        COMPILE_ARGS += -GFC1_WEIGHTS_FILE=\"$(PWD)/weights/fractional_hist8/fc1_weights.mem\"
        COMPILE_ARGS += -GFC1_BIAS_FILE=\"$(PWD)/weights/fractional_hist8/fc1_bias.mem\"
        COMPILE_ARGS += -GFC2_WEIGHTS_FILE=\"$(PWD)/weights/fractional_hist8/fc2_weights.mem\"
        COMPILE_ARGS += -GFC2_BIAS_FILE=\"$(PWD)/weights/fractional_hist8/fc2_bias.mem\"
        COMPILE_ARGS += -GFC_OUT_WEIGHTS_FILE=\"$(PWD)/weights/fractional_hist8/fc_out_weights.mem\"
        COMPILE_ARGS += -GFC_OUT_BIAS_FILE=\"$(PWD)/weights/fractional_hist8/fc_out_bias.mem\"
    endif
    ifeq ($(TEST),cartpole_integration_fractional_hist256)
        # Fractional model with history_length=256 from weights/fractional_hist256: 16 HL1, 4 HL2, 30 timesteps
        COMPILE_ARGS += -GNUM_INPUTS=4 -GHL1_SIZE=16
        COMPILE_ARGS += -GHL2_SIZE=4 -GNUM_ACTIONS=2
        COMPILE_ARGS += -GNUM_TIMESTEPS=30 -GQ_BATCH_SIZE=4
        # Same FC2 sizing rule: keep enough headroom to prevent saturation in observation sweeps.
        COMPILE_ARGS += -GFC2_OUTPUT_WIDTH=24
        COMPILE_ARGS += -GHISTORY_LENGTH=256
        COMPILE_ARGS += -GCOEFF_WIDTH=16 -GCOEFF_FRAC_BITS=15
        COMPILE_ARGS += -GC_SCALED=256 -GINV_DENOM=57553
        COMPILE_ARGS += -GGL_COEFF_FILE=\"$(PWD)/weights/fractional_hist256/gl_coefficients.mem\"
        COMPILE_ARGS += -GFC1_WEIGHTS_FILE=\"$(PWD)/weights/fractional_hist256/fc1_weights.mem\"
        COMPILE_ARGS += -GFC1_BIAS_FILE=\"$(PWD)/weights/fractional_hist256/fc1_bias.mem\"
        COMPILE_ARGS += -GFC2_WEIGHTS_FILE=\"$(PWD)/weights/fractional_hist256/fc2_weights.mem\"
        COMPILE_ARGS += -GFC2_BIAS_FILE=\"$(PWD)/weights/fractional_hist256/fc2_bias.mem\"
        COMPILE_ARGS += -GFC_OUT_WEIGHTS_FILE=\"$(PWD)/weights/fractional_hist256/fc_out_weights.mem\"
        COMPILE_ARGS += -GFC_OUT_BIAS_FILE=\"$(PWD)/weights/fractional_hist256/fc_out_bias.mem\"
    endif
    ifeq ($(TEST),cartpole_integration_bitshift_custom_slow_decay)
        # Bitshift custom_slow_decay model from weights/bitshift-custom_slow_decay: 32 HL1, 8 HL2, 10 timesteps
        # NOTE: Keep these parameters in sync with the Icarus block above.
        COMPILE_ARGS += -GNUM_INPUTS=4 -GHL1_SIZE=32
        COMPILE_ARGS += -GHL2_SIZE=8 -GNUM_ACTIONS=2
        COMPILE_ARGS += -GNUM_TIMESTEPS=10 -GQ_BATCH_SIZE=4
        COMPILE_ARGS += -GFRAC_BITS=12
        COMPILE_ARGS += -GFC2_OUTPUT_WIDTH=24
        COMPILE_ARGS += -GHISTORY_LENGTH=8
        COMPILE_ARGS += -GSHIFT_WIDTH=8 -GSHIFT_MODE=3
        COMPILE_ARGS += -GCUSTOM_DECAY_RATE=3
        COMPILE_ARGS += -GC_SCALED=256 -GINV_DENOM=59823
        COMPILE_ARGS += -GFC1_WEIGHTS_FILE=\"$(PWD)/weights/bitshift-custom_slow_decay/fc1_weights.mem\"
        COMPILE_ARGS += -GFC1_BIAS_FILE=\"$(PWD)/weights/bitshift-custom_slow_decay/fc1_bias.mem\"
        COMPILE_ARGS += -GFC2_WEIGHTS_FILE=\"$(PWD)/weights/bitshift-custom_slow_decay/fc2_weights.mem\"
        COMPILE_ARGS += -GFC2_BIAS_FILE=\"$(PWD)/weights/bitshift-custom_slow_decay/fc2_bias.mem\"
        COMPILE_ARGS += -GFC_OUT_WEIGHTS_FILE=\"$(PWD)/weights/bitshift-custom_slow_decay/fc_out_weights.mem\"
        COMPILE_ARGS += -GFC_OUT_BIAS_FILE=\"$(PWD)/weights/bitshift-custom_slow_decay/fc_out_bias.mem\"
    endif
endif

# Output directory for simulation results
RESULTS_DIR = ../results
SIM_BUILD = $(RESULTS_DIR)/sim_build

# Export variables for cocotb
export COCOTB_RESULTS_FILE = $(RESULTS_DIR)/results.xml
export COCOTB_TEST_MODULES = $(MODULE)
ifneq ($(origin TESTCASE), undefined)
    export COCOTB_TESTCASE = $(TESTCASE)
    undefine TESTCASE
endif

# Include cocotb's make rules
include $(shell cocotb-config --makefiles)/Makefile.sim

# Convenience targets
test_lif:
	$(MAKE) TEST=lif

test_fractional_lif:
	$(MAKE) TEST=fractional_lif

test_fractional_lif_hist64:
	$(MAKE) TEST=fractional_lif_hist64

test_bitshift_lif:
	$(MAKE) TEST=bitshift_lif

# SHIFT_MODE=0 (simple): [0,1,2,3,...]
test_bitshift_lif_mode0:
	$(MAKE) TEST=bitshift_lif_mode0

# SHIFT_MODE=1 (slow_decay): [0,1,1,2,2,3,3,...]
test_bitshift_lif_mode1:
	$(MAKE) TEST=bitshift_lif_mode1

# SHIFT_MODE=2 (custom): [0,1,3,4,5xR,6xR,...], R=CUSTOM_DECAY_RATE
test_bitshift_lif_mode2:
	$(MAKE) TEST=bitshift_lif_mode2

# SHIFT_MODE=3 (custom_slow_decay): [0,1,3,4,5x3,6x4,7x5,...]
test_bitshift_lif_mode3:
	$(MAKE) TEST=bitshift_lif_mode3

test_lif_timestep:
	$(MAKE) TEST=lif_timestep

test_linear_layer:
	$(MAKE) TEST=linear_layer

test_spike_buffer:
	$(MAKE) TEST=spike_buffer

test_spike_buffer_staggered:
	$(MAKE) TEST=spike_buffer_staggered

test_membrane_buffer:
	$(MAKE) TEST=membrane_buffer

test_neuron_membrane_buffer:
	$(MAKE) TEST=neuron_membrane_buffer

test_q_accumulator:
	$(MAKE) TEST=q_accumulator

test_neural_network:
	$(MAKE) TEST=neural_network

test_cartpole_integration:
	$(MAKE) TEST=cartpole_integration

test_cartpole_integration_fractional:
	$(MAKE) TEST=cartpole_integration_fractional

test_cartpole_integration_fractional_hist8:
	$(MAKE) TEST=cartpole_integration_fractional_hist8

test_cartpole_integration_fractional_hist256:
	$(MAKE) TEST=cartpole_integration_fractional_hist256

test_cartpole_integration_bitshift_custom_slow_decay:
	$(MAKE) TEST=cartpole_integration_bitshift_custom_slow_decay

# Clean up generated files
clean::
	rm -rf $(RESULTS_DIR)/*
	rm -rf __pycache__
	rm -rf .pytest_cache

.PHONY: clean test_lif test_fractional_lif test_fractional_lif_hist64 test_bitshift_lif test_bitshift_lif_mode0 test_bitshift_lif_mode1 test_bitshift_lif_mode2 test_bitshift_lif_mode3 test_lif_timestep test_linear_layer test_spike_buffer test_spike_buffer_staggered test_membrane_buffer test_neuron_membrane_buffer test_q_accumulator test_neural_network test_cartpole_integration test_cartpole_integration_fractional test_cartpole_integration_fractional_hist8 test_cartpole_integration_fractional_hist256 test_cartpole_integration_bitshift_custom_slow_decay
