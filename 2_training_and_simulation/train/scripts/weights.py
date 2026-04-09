"""
Weight management for SNN-DQN models.

Provides inspection, quantization, and export functionality for trained model weights.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import math


class Weights:
    """
    Manages weight inspection, quantization, and export for SNN-DQN models.

    Usage:
        weights = Weights('models/dqn_config-baseline-best.pth')

        # Inspect weights
        stats = weights.inspect()

        # Quantize to QS1_6 format
        quant_data = weights.quantize(bits=8, fractional_bits=6, signed=True)

        # Export for PyTorch validation
        weights.export_pytorch(bits=8, fractional_bits=6, signed=True)

        # Export for hardware deployment
        weights.export_hardware(bits=8, fractional_bits=6, signed=True)
    """

    def __init__(self, model_path: str):
        """
        Load model checkpoint and extract weights/config.

        Args:
            model_path: Path to .pth checkpoint file
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load checkpoint
        self.checkpoint = torch.load(
            self.model_path, map_location="cpu", weights_only=False
        )

        # Extract policy network state dict
        if "policy_net_state_dict" in self.checkpoint:
            self.state_dict = self.checkpoint["policy_net_state_dict"]
        else:
            raise ValueError("Checkpoint does not contain 'policy_net_state_dict'")

        # Extract configuration if available
        self.config = self.checkpoint.get("config", {})
        self.episode = self.checkpoint.get("episode", 0)
        self.avg_reward = self.checkpoint.get("avg_reward", 0.0)

    def inspect(self) -> Dict:
        """
        Analyze weights and biases in each layer.

        Returns:
            Dict with structure:
            {
                'layers': {
                    'layer_name': {
                        'weight': {'shape': tuple, 'min': float, 'max': float,
                                   'mean': float, 'std': float, 'values': tensor},
                        'bias': {...} or None
                    }
                },
                'summary': {
                    'total_params': int,
                    'weight_min': float,
                    'weight_max': float,
                    'weight_mean': float,
                    'weight_std': float,
                    'bias_min': float,
                    'bias_max': float
                },
                'suggested_format': {
                    'bits': int,
                    'fractional_bits': int,
                    'integer_bits': int,
                    'signed': bool,
                    'format_name': str  # e.g., 'QS1_6'
                }
            }
        """
        layers = {}
        all_weights = []
        all_biases = []
        total_params = 0

        # Analyze each layer
        for name, param in self.state_dict.items():
            if "weight" in name:
                layer_base = name.replace(".weight", "")
                if layer_base not in layers:
                    layers[layer_base] = {}

                weight_data = param.detach().cpu()
                layers[layer_base]["weight"] = {
                    "shape": tuple(weight_data.shape),
                    "min": float(weight_data.min()),
                    "max": float(weight_data.max()),
                    "mean": float(weight_data.mean()),
                    "std": float(weight_data.std()),
                    "values": weight_data,
                }
                all_weights.append(weight_data.flatten())
                total_params += weight_data.numel()

            elif "bias" in name:
                layer_base = name.replace(".bias", "")
                if layer_base not in layers:
                    layers[layer_base] = {}

                bias_data = param.detach().cpu()
                layers[layer_base]["bias"] = {
                    "shape": tuple(bias_data.shape),
                    "min": float(bias_data.min()),
                    "max": float(bias_data.max()),
                    "mean": float(bias_data.mean()),
                    "std": float(bias_data.std()),
                    "values": bias_data,
                }
                all_biases.append(bias_data.flatten())
                total_params += bias_data.numel()

        # Compute summary statistics
        all_weights_tensor = torch.cat(all_weights) if all_weights else torch.tensor([])
        all_biases_tensor = torch.cat(all_biases) if all_biases else torch.tensor([])

        summary = {
            "total_params": total_params,
            "weight_min": (
                float(all_weights_tensor.min()) if len(all_weights_tensor) > 0 else 0.0
            ),
            "weight_max": (
                float(all_weights_tensor.max()) if len(all_weights_tensor) > 0 else 0.0
            ),
            "weight_mean": (
                float(all_weights_tensor.mean()) if len(all_weights_tensor) > 0 else 0.0
            ),
            "weight_std": (
                float(all_weights_tensor.std()) if len(all_weights_tensor) > 0 else 0.0
            ),
            "bias_min": (
                float(all_biases_tensor.min()) if len(all_biases_tensor) > 0 else 0.0
            ),
            "bias_max": (
                float(all_biases_tensor.max()) if len(all_biases_tensor) > 0 else 0.0
            ),
        }

        # Suggest quantization format
        suggested = self.suggest_format(target_bits=8)

        return {"layers": layers, "summary": summary, "suggested_format": suggested}

    def suggest_format(self, target_bits: int = 8) -> Dict:
        """
        Analyze weight ranges and recommend quantization format.

        Args:
            target_bits: Target total bit width (default: 8)

        Returns:
            Dict with recommended format:
            {
                'bits': int,
                'fractional_bits': int,
                'integer_bits': int,
                'signed': bool,
                'format_name': str,
                'range': tuple,
                'resolution': float
            }
        """
        # Get weight range
        all_values = []
        for name, param in self.state_dict.items():
            all_values.append(param.detach().cpu().flatten())

        all_values = torch.cat(all_values)
        min_val = float(all_values.min())
        max_val = float(all_values.max())
        abs_max = max(abs(min_val), abs(max_val))

        # Determine if signed is needed
        signed = min_val < 0

        # Calculate required integer bits (including sign bit if signed)
        if signed:
            # Need to represent values from -abs_max to +abs_max
            # For signed, range is [-2^(int_bits), 2^(int_bits) - resolution]
            integer_bits = math.ceil(math.log2(abs_max + 1)) if abs_max > 0 else 1
            sign_bit = 1
        else:
            # Unsigned: range is [0, 2^(int_bits) - resolution]
            integer_bits = math.ceil(math.log2(max_val + 1)) if max_val > 0 else 1
            sign_bit = 0

        # Fractional bits = total - integer - sign
        fractional_bits = target_bits - integer_bits - sign_bit

        # Ensure we have at least 1 fractional bit for precision
        if fractional_bits < 1:
            fractional_bits = 1
            integer_bits = target_bits - fractional_bits - sign_bit

        # Calculate actual range and resolution
        if signed:
            range_min = -(2**integer_bits)
            range_max = (2**integer_bits) - (2**-fractional_bits)
            format_name = f"QS{integer_bits}_{fractional_bits}"
        else:
            range_min = 0
            range_max = (2**integer_bits) - (2**-fractional_bits)
            format_name = f"Q{integer_bits}_{fractional_bits}"

        resolution = 2**-fractional_bits

        return {
            "bits": target_bits,
            "fractional_bits": fractional_bits,
            "integer_bits": integer_bits,
            "signed": signed,
            "format_name": format_name,
            "range": (range_min, range_max),
            "resolution": resolution,
        }

    def quantize(
        self, bits: int = 8, fractional_bits: int = 6, signed: bool = True
    ) -> Dict:
        """
        Quantize weights to fixed-point format.

        Args:
            bits: Total number of bits (default: 8)
            fractional_bits: Number of fractional bits (default: 6)
            signed: Whether format is signed (default: True)

        Returns:
            Dict with:
            {
                'quantized_weights': Dict[str, np.ndarray],  # Integer arrays
                'quantization_error': {
                    'mse': float,
                    'max_abs_error': float,
                    'mean_abs_error': float
                },
                'config': {
                    'bits': int,
                    'fractional_bits': int,
                    'integer_bits': int,
                    'signed': bool,
                    'format_name': str,
                    'scale_factor': float,
                    'range': tuple
                }
            }
        """
        # Calculate format parameters
        integer_bits = bits - fractional_bits - (1 if signed else 0)
        scale_factor = 2**fractional_bits

        if signed:
            min_int = -(2 ** (bits - 1))
            max_int = (2 ** (bits - 1)) - 1
            format_name = f"QS{integer_bits}_{fractional_bits}"
            range_min = min_int / scale_factor
            range_max = max_int / scale_factor
        else:
            min_int = 0
            max_int = (2**bits) - 1
            format_name = f"Q{integer_bits}_{fractional_bits}"
            range_min = 0
            range_max = max_int / scale_factor

        quantized_weights = {}
        original_values = []
        quantized_values = []

        # Quantize each parameter
        for name, param in self.state_dict.items():
            weight_float = param.detach().cpu().numpy()

            # Scale to integer
            weight_scaled = weight_float * scale_factor

            # Round and saturate to valid range
            weight_int = np.round(weight_scaled).astype(np.int32)
            weight_int = np.clip(weight_int, min_int, max_int)

            # Convert to appropriate integer type
            if signed:
                if bits == 8:
                    weight_int = weight_int.astype(np.int8)
                elif bits == 16:
                    weight_int = weight_int.astype(np.int16)
                elif bits == 32:
                    weight_int = weight_int.astype(np.int32)
            else:
                if bits == 8:
                    weight_int = weight_int.astype(np.uint8)
                elif bits == 16:
                    weight_int = weight_int.astype(np.uint16)
                elif bits == 32:
                    weight_int = weight_int.astype(np.uint32)

            quantized_weights[name] = weight_int

            # Track for error calculation
            original_values.append(weight_float.flatten())
            dequantized = weight_int.astype(np.float32) / scale_factor
            quantized_values.append(dequantized.flatten())

        # Calculate quantization error
        original_all = np.concatenate(original_values)
        quantized_all = np.concatenate(quantized_values)

        errors = original_all - quantized_all
        mse = float(np.mean(errors**2))
        max_abs_error = float(np.max(np.abs(errors)))
        mean_abs_error = float(np.mean(np.abs(errors)))

        return {
            "quantized_weights": quantized_weights,
            "quantization_error": {
                "mse": mse,
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
            },
            "config": {
                "bits": bits,
                "fractional_bits": fractional_bits,
                "integer_bits": integer_bits,
                "signed": signed,
                "format_name": format_name,
                "scale_factor": scale_factor,
                "range": (range_min, range_max),
            },
        }

    def export_pytorch(
        self,
        output_path: Optional[str] = None,
        bits: int = 8,
        fractional_bits: int = 6,
        signed: bool = True,
    ) -> str:
        """
        Export quantized weights as PyTorch-compatible .pth file.

        Process:
        1. Quantize weights to integers
        2. Convert integers back to floats: float_val = int_val / (2^fractional_bits)
        3. Save as .pth with same structure as original model

        These floats represent the exact values that hardware will compute,
        but stored in a format PyTorch/snnTorch can load and use.

        Args:
            output_path: Output file path (auto-generated if None)
            bits: Total number of bits
            fractional_bits: Number of fractional bits
            signed: Whether format is signed

        Returns:
            Path to saved .pth file
        """
        # Quantize weights
        quant_data = self.quantize(bits, fractional_bits, signed)
        quantized_ints = quant_data["quantized_weights"]
        scale_factor = quant_data["config"]["scale_factor"]
        format_name = quant_data["config"]["format_name"]

        # Convert quantized integers back to floats (dequantize)
        dequantized_state_dict = {}
        for name, int_weights in quantized_ints.items():
            # Convert to float, maintaining quantized precision
            float_weights = int_weights.astype(np.float32) / scale_factor
            # Ensure result is an ndarray (division can return scalar for 0-d arrays)
            if not isinstance(float_weights, np.ndarray):
                float_weights = np.array(float_weights, dtype=np.float32)
            dequantized_state_dict[name] = torch.from_numpy(float_weights)

        # Create new checkpoint with quantized weights
        quantized_checkpoint = {
            "policy_net_state_dict": dequantized_state_dict,
            "target_net_state_dict": dequantized_state_dict,  # Same as policy
            "config": self.config,
            "episode": self.episode,
            "avg_reward": self.avg_reward,
            "quantization_info": {
                "format": format_name,
                "bits": bits,
                "fractional_bits": fractional_bits,
                "integer_bits": quant_data["config"]["integer_bits"],
                "signed": signed,
                "quantization_error": quant_data["quantization_error"],
            },
        }

        # Generate output path if not provided
        if output_path is None:
            stem = self.model_path.stem
            parent = self.model_path.parent
            output_path = parent / f"{stem}-quantized-{format_name}.pth"
        else:
            output_path = Path(output_path)

        # Save quantized checkpoint
        torch.save(quantized_checkpoint, output_path)

        return str(output_path)

    def export_hardware(
        self,
        output_path: Optional[str] = None,
        bits: int = 8,
        fractional_bits: int = 6,
        signed: bool = True,
    ) -> str:
        """
        Export quantized weights as .mem files for hardware deployment.

        Process:
        1. Quantize weights to integers
        2. Convert integers to hex strings
        3. Generate .mem file(s) in format for $readmemh in SystemVerilog

        Format example (.mem file):
        // Layer: fc1.weight
        // Shape: [128, 4]
        // Format: QS1_6 (8-bit signed, 1 integer bit, 6 fractional bits)
        // Range: [-2.0, 1.984375], Resolution: 0.015625
        21  // weight[0][0] = 0x21 = 33 decimal = 0.515625
        FF  // weight[0][1] = 0xFF = -1 decimal = -0.015625
        ...

        Generates separate .mem files for each layer parameter.

        Args:
            output_path: Output directory path (auto-generated if None)
            bits: Total number of bits
            fractional_bits: Number of fractional bits
            signed: Whether format is signed

        Returns:
            Path to output directory containing .mem files
        """
        # Quantize weights
        quant_data = self.quantize(bits, fractional_bits, signed)
        quantized_ints = quant_data["quantized_weights"]
        config = quant_data["config"]
        format_name = config["format_name"]

        # Generate output directory if not provided
        if output_path is None:
            stem = self.model_path.stem
            parent = self.model_path.parent
            output_dir = parent / f"{stem}-hardware-{format_name}"
        else:
            output_dir = Path(output_path)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export each parameter to separate .mem file
        # Map layer names to expected hardware filenames
        layer_name_map = {
            "fc1.weight": "fc1_weights",
            "fc1.bias": "fc1_bias",
            "fc2.weight": "fc2_weights",
            "fc2.bias": "fc2_bias",
            "fc_out.weight": "fc_out_weights",
            "fc_out.bias": "fc_out_bias",
        }

        for name, int_weights in quantized_ints.items():
            # Skip LIF parameters (threshold, beta, etc.) - they're not used in hardware export
            if name not in layer_name_map:
                continue

            # Use mapped filename for hardware compatibility
            filename = layer_name_map[name] + ".mem"
            filepath = output_dir / filename

            # Flatten weights for sequential memory layout
            flat_weights = int_weights.flatten()

            # Convert to hex based on bit width and signedness
            if bits <= 8:
                hex_width = 2
                mask = 0xFF
            elif bits <= 16:
                hex_width = 4
                mask = 0xFFFF
            else:
                hex_width = 8
                mask = 0xFFFFFFFF

            # Write .mem file
            with open(filepath, "w") as f:
                # Write header with metadata
                f.write(f"// Parameter: {name}\n")
                f.write(f"// Shape: {list(int_weights.shape)}\n")
                f.write(
                    f"// Format: {format_name} ({bits}-bit {'signed' if signed else 'unsigned'}, "
                    f"{config['integer_bits']} integer bits, {fractional_bits} fractional bits)\n"
                )
                f.write(
                    f"// Range: [{config['range'][0]:.6f}, {config['range'][1]:.6f}], "
                    f"Resolution: {config['range'][1] - config['range'][0]:.6f}\n"
                )
                f.write(f"// Total values: {len(flat_weights)}\n")
                f.write("//\n")

                # Write hex values (one per line for $readmemh)
                for i, val in enumerate(flat_weights):
                    # Convert numpy type to Python int to avoid overflow issues with bitwise ops
                    val_int = int(val)

                    # Convert to unsigned representation for hex output
                    if signed and val_int < 0:
                        # Two's complement
                        hex_val = val_int & mask
                    else:
                        hex_val = val_int & mask

                    # Convert to float for comment
                    float_val = float(val) / config["scale_factor"]

                    # Write hex value with optional comment
                    f.write(f"{hex_val:0{hex_width}X}  // [{i}] = {float_val:.6f}\n")

        return str(output_dir)
