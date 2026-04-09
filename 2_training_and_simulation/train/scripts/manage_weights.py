"""
CLI tool for managing SNN-DQN model weights.

Subcommands:
    inspect     - Analyze weight distributions and suggest quantization format
    quantize    - Quantize weights and report errors
    export      - Export weights for PyTorch or hardware deployment

Examples:
    # Inspect weights
    python manage_weights.py inspect models/dqn_config-baseline-best.pth

    # Quantize to QS1_6 format
    python manage_weights.py quantize models/dqn_config-baseline-best.pth --bits 8 --frac 6 --signed

    # Export for PyTorch validation
    python manage_weights.py export pytorch models/dqn_config-baseline-best.pth

    # Export for hardware deployment
    python manage_weights.py export hardware models/dqn_config-baseline-best.pth --bits 8 --frac 6
"""

import argparse
from pathlib import Path

# Import Weights from same directory
from weights import Weights


def format_stats_table(stats: dict, indent: int = 0) -> str:
    """Format statistics dict as aligned table."""
    lines = []
    prefix = " " * indent

    # Find max key length for alignment
    max_key_len = max(len(str(k)) for k in stats.keys())

    for key, value in stats.items():
        if isinstance(value, float):
            lines.append(f"{prefix}{key:<{max_key_len}} : {value:>12.6f}")
        elif isinstance(value, int):
            lines.append(f"{prefix}{key:<{max_key_len}} : {value:>12,}")
        elif isinstance(value, tuple):
            lines.append(f"{prefix}{key:<{max_key_len}} : {str(value):>12}")
        else:
            lines.append(f"{prefix}{key:<{max_key_len}} : {str(value):>12}")

    return "\n".join(lines)


def cmd_inspect(args):
    """Inspect weights and display statistics."""
    weights = Weights(args.model)
    data = weights.inspect()

    print(f"\n{'='*70}")
    print(f"Weight Inspection: {args.model}")
    print(f"{'='*70}\n")

    # Model metadata
    if weights.episode > 0:
        print("Model Info:")
        print(f"  Episode:     {weights.episode}")
        print(f"  Avg Reward:  {weights.avg_reward:.2f}")
        print()

    # Summary statistics
    print("Overall Statistics:")
    print(format_stats_table(data["summary"], indent=2))
    print()

    # Suggested format
    suggested = data["suggested_format"]
    print("Suggested Quantization Format:")
    print(f"  Format:      {suggested['format_name']}")
    sign_bits = "1 sign" if suggested["signed"] else "0 sign"
    print(
        f"  Bits:        {suggested['bits']} "
        f"({suggested['integer_bits']} int + {suggested['fractional_bits']} frac + {sign_bits})"
    )
    print(f"  Range:       [{suggested['range'][0]:.6f}, {suggested['range'][1]:.6f}]")
    print(f"  Resolution:  {suggested['resolution']:.8f}")
    print()

    # Layer-by-layer statistics
    if args.verbose:
        print("Layer-by-Layer Statistics:")
        print(f"{'─'*70}\n")

        for layer_name, layer_data in data["layers"].items():
            print(f"Layer: {layer_name}")

            if "weight" in layer_data:
                weight = layer_data["weight"]
                print("  Weight:")
                print(f"    Shape:  {weight['shape']}")
                print(f"    Range:  [{weight['min']:.6f}, {weight['max']:.6f}]")
                print(f"    Mean:   {weight['mean']:.6f}")
                print(f"    Std:    {weight['std']:.6f}")

            if "bias" in layer_data and layer_data["bias"] is not None:
                bias = layer_data["bias"]
                print("  Bias:")
                print(f"    Shape:  {bias['shape']}")
                print(f"    Range:  [{bias['min']:.6f}, {bias['max']:.6f}]")
                print(f"    Mean:   {bias['mean']:.6f}")
                print(f"    Std:    {bias['std']:.6f}")

            print()


def cmd_quantize(args):
    """Quantize weights and display errors."""
    weights = Weights(args.model)

    # Determine format name
    if args.signed:
        integer_bits = args.bits - args.frac - 1
        format_name = f"QS{integer_bits}_{args.frac}"
    else:
        integer_bits = args.bits - args.frac
        format_name = f"Q{integer_bits}_{args.frac}"

    print(f"\n{'='*70}")
    print(f"Weight Quantization: {format_name}")
    print(f"{'='*70}\n")

    print(f"Model:  {args.model}")
    print(
        f"Format: {format_name} ({args.bits}-bit {'signed' if args.signed else 'unsigned'})"
    )
    print(
        f"Bits:   {integer_bits} integer + {args.frac} fractional + {'1 sign' if args.signed else '0 sign'}"
    )
    print()

    # Perform quantization
    result = weights.quantize(
        bits=args.bits, fractional_bits=args.frac, signed=args.signed
    )

    # Display configuration
    config = result["config"]
    print("Quantization Configuration:")
    print(f"  Scale Factor: {config['scale_factor']:.2f}")
    print(f"  Range:        [{config['range'][0]:.6f}, {config['range'][1]:.6f}]")
    print()

    # Display errors
    error = result["quantization_error"]
    print("Quantization Error:")
    print(format_stats_table(error, indent=2))
    print()

    # Display per-layer statistics if verbose
    if args.verbose:
        print("Per-Parameter Statistics:")
        print(f"{'─'*70}\n")

        for name, quant_vals in result["quantized_weights"].items():
            print(f"Parameter: {name}")
            print(f"  Shape:     {quant_vals.shape}")
            print(f"  Int Range: [{quant_vals.min()}, {quant_vals.max()}]")
            print(f"  Unique:    {len(set(quant_vals.flatten()))}")
            print()


def cmd_export_pytorch(args):
    """Export quantized weights as PyTorch .pth file."""
    weights = Weights(args.model)

    # Determine format name
    if args.signed:
        integer_bits = args.bits - args.frac - 1
        format_name = f"QS{integer_bits}_{args.frac}"
    else:
        integer_bits = args.bits - args.frac
        format_name = f"Q{integer_bits}_{args.frac}"

    print(f"\n{'='*70}")
    print(f"Export PyTorch: {format_name}")
    print(f"{'='*70}\n")

    print(f"Model:  {args.model}")
    print(
        f"Format: {format_name} ({args.bits}-bit {'signed' if args.signed else 'unsigned'})"
    )
    print()

    # Export
    output_path = weights.export_pytorch(
        output_path=args.output,
        bits=args.bits,
        fractional_bits=args.frac,
        signed=args.signed,
    )

    print(f"✓ Exported to: {output_path}")
    print()
    print("This .pth file contains dequantized float weights that represent")
    print("the exact values hardware will compute, but in PyTorch-compatible format.")
    print()


def cmd_export_hardware(args):
    """Export quantized weights as .mem files for hardware."""
    weights = Weights(args.model)

    # Determine format name
    if args.signed:
        integer_bits = args.bits - args.frac - 1
        format_name = f"QS{integer_bits}_{args.frac}"
    else:
        integer_bits = args.bits - args.frac
        format_name = f"Q{integer_bits}_{args.frac}"

    print(f"\n{'='*70}")
    print(f"Export Hardware: {format_name}")
    print(f"{'='*70}\n")

    print(f"Model:  {args.model}")
    print(
        f"Format: {format_name} ({args.bits}-bit {'signed' if args.signed else 'unsigned'})"
    )
    print()

    # Export
    output_dir = weights.export_hardware(
        output_path=args.output,
        bits=args.bits,
        fractional_bits=args.frac,
        signed=args.signed,
    )

    print(f"✓ Exported to: {output_dir}")
    print()
    print("Generated .mem files for $readmemh in SystemVerilog:")

    # List generated files
    output_path = Path(output_dir)
    mem_files = sorted(output_path.glob("*.mem"))
    for mem_file in mem_files:
        size_kb = mem_file.stat().st_size / 1024
        print(f"  - {mem_file.name:<30} ({size_kb:.2f} KB)")
    print()


def cmd_export(args):
    """Route to appropriate export function."""
    if args.format == "pytorch":
        cmd_export_pytorch(args)
    elif args.format == "hardware":
        cmd_export_hardware(args)


def main():
    parser = argparse.ArgumentParser(
        description="Manage weights for SNN-DQN models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")
    subparsers.required = True

    # Inspect subcommand
    inspect_parser = subparsers.add_parser(
        "inspect", help="Analyze weight distributions and suggest quantization format"
    )
    inspect_parser.add_argument("model", help="Path to .pth model file")
    inspect_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show per-layer statistics"
    )
    inspect_parser.set_defaults(func=cmd_inspect)

    # Quantize subcommand
    quantize_parser = subparsers.add_parser(
        "quantize", help="Quantize weights and report errors"
    )
    quantize_parser.add_argument("model", help="Path to .pth model file")
    quantize_parser.add_argument(
        "--bits", type=int, default=8, help="Total number of bits (default: 8)"
    )
    quantize_parser.add_argument(
        "--frac", type=int, default=6, help="Number of fractional bits (default: 6)"
    )
    quantize_parser.add_argument(
        "--signed",
        action="store_true",
        default=True,
        help="Use signed format (default: True)",
    )
    quantize_parser.add_argument(
        "--unsigned", dest="signed", action="store_false", help="Use unsigned format"
    )
    quantize_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show per-parameter statistics"
    )
    quantize_parser.set_defaults(func=cmd_quantize)

    # Export subcommand
    export_parser = subparsers.add_parser(
        "export", help="Export weights for PyTorch or hardware"
    )
    export_parser.add_argument(
        "format",
        choices=["pytorch", "hardware"],
        help="Export format: pytorch (.pth) or hardware (.mem)",
    )
    export_parser.add_argument("model", help="Path to .pth model file")
    export_parser.add_argument(
        "-o", "--output", help="Output path (auto-generated if not specified)"
    )
    export_parser.add_argument(
        "--bits", type=int, default=8, help="Total number of bits (default: 8)"
    )
    export_parser.add_argument(
        "--frac", type=int, default=6, help="Number of fractional bits (default: 6)"
    )
    export_parser.add_argument(
        "--signed",
        action="store_true",
        default=True,
        help="Use signed format (default: True)",
    )
    export_parser.add_argument(
        "--unsigned", dest="signed", action="store_false", help="Use unsigned format"
    )
    export_parser.set_defaults(func=cmd_export)

    # Parse and execute
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
