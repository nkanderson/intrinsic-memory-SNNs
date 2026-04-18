"""
Generate GL coefficients and constants for fractional_lif.sv

This script computes:
1. Grünwald-Letnikov coefficients (g_1 to g_{H-1}) for the fractional derivative
2. Precomputed constants C_SCALED and INV_DENOM for the membrane update

The relationship between standard LIF beta and fractional lambda is:
    λ = (1 - β) / β

For β = 0.9 (default): λ = 0.1 / 0.9 ≈ 0.111

Usage:
    python generate_coefficients.py --alpha 0.5 --lam 0.111 --history-length 64 --output-dir ../sv/weights/
    python generate_coefficients.py --alpha 0.5 --beta 0.9 --history-length 64 --output-dir ../sv/weights/
    python generate_coefficients.py --alpha 0.5 --lam 0.111 --constants-only
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path when running this script directly.
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.scripts.utils import compute_gl_coefficients


def beta_to_lam(beta: float) -> float:
    """
    Convert standard LIF beta (membrane decay) to fractional lambda (leakage).

    The relationship is: λ = (1 - β) / β

    This provides approximate equivalence between:
    - Standard LIF: mem[n] = β * mem[n-1] + input
    - Fractional LIF: mem[n] = (input - C * Σ g_k * mem[n-k]) / (C + λ)

    Args:
        beta: Membrane decay rate from standard LIF (0 < beta < 1)

    Returns:
        lambda: Leakage parameter for fractional LIF
    """
    assert 0 < beta < 1, f"beta must be in (0, 1), got {beta}"
    return (1.0 - beta) / beta


def compute_fractional_constants(
    alpha: float,
    lam: float,
    dt: float = 1.0,
    c_frac_bits: int = 8,
    inv_denom_frac_bits: int = 16,
) -> dict:
    """
    Compute precomputed constants for fractional LIF hardware.

    Args:
        alpha: Fractional order (0 < alpha <= 1)
        lam: Leakage parameter (λ >= 0)
        dt: Discrete timestep (default 1.0)
        c_frac_bits: Fractional bits for C_SCALED (Q8.8 by default)
        inv_denom_frac_bits: Fractional bits for INV_DENOM (Q0.16 by default)

    Returns:
        Dict with C_SCALED, INV_DENOM, and their float equivalents
    """
    # C = 1 / dt^α
    C = 1.0 / (dt**alpha)

    # Denominator = C + λ
    denom = C + lam

    # Inverse denominator = 1 / (C + λ)
    inv_denom = 1.0 / denom

    # Scale to fixed-point
    c_scale = 1 << c_frac_bits
    inv_denom_scale = 1 << inv_denom_frac_bits

    C_SCALED = int(round(C * c_scale))
    INV_DENOM = int(round(inv_denom * inv_denom_scale))

    return {
        "C": C,
        "C_SCALED": C_SCALED,
        "c_frac_bits": c_frac_bits,
        "lam": lam,
        "denom": denom,
        "inv_denom": inv_denom,
        "INV_DENOM": INV_DENOM,
        "inv_denom_frac_bits": inv_denom_frac_bits,
    }


def quantize_coefficients_magnitude(
    coeffs,
    bits: int = 16,
    frac_bits: int = 15,
) -> list:
    """
    Quantize GL coefficient magnitudes |g_k| to unsigned fixed-point.

    Args:
        coeffs: Tensor of GL coefficients (g_1 to g_{H-1})
        bits: Total bit width
        frac_bits: Number of fractional bits

    Returns:
        List of quantized unsigned integer values
    """
    scale = 1 << frac_bits
    max_val = (1 << bits) - 1

    # Convert to numpy and quantize magnitudes
    coeffs_np = coeffs.numpy() if hasattr(coeffs, "numpy") else np.array(coeffs)
    magnitudes = np.abs(coeffs_np)
    quantized = np.round(magnitudes * scale).astype(np.int64)

    # Clamp to representable unsigned range
    quantized = np.clip(quantized, 0, max_val)

    return quantized.tolist()


def write_mem_file(
    filepath: Path,
    values: list,
    bits: int,
    frac_bits: int,
    alpha: float,
    lam: float,
    history_length: int,
):
    """
    Write quantized coefficient magnitudes to .mem file for $readmemh.

    Args:
        filepath: Output file path
        values: List of quantized integer values
        bits: Bit width per value
        frac_bits: Fractional bits
        alpha: Fractional order (for documentation)
        lam: Leakage parameter (for documentation)
        history_length: History length (for documentation)
    """
    if bits <= 8:
        hex_width = 2
        mask = 0xFF
    elif bits <= 16:
        hex_width = 4
        mask = 0xFFFF
    else:
        hex_width = 8
        mask = 0xFFFFFFFF

    scale = 1 << frac_bits

    with open(filepath, "w") as f:
        # Header
        f.write("// GL Coefficient Magnitudes for Fractional LIF\n")
        f.write(f"// alpha = {alpha}, lambda = {lam:.6f}\n")
        f.write(f"// History length = {history_length}, coefficients = {len(values)}\n")
        f.write(f"// Format: QU{bits-frac_bits}.{frac_bits} ({bits}-bit unsigned)\n")
        f.write(
            f"// Contains |g_1| to |g_{{{history_length}-1}}| (g_0 = 1 is implicit)\n"
        )
        f.write("// Assumes 0 < alpha <= 1, where g_k (k>=1) are non-positive\n")
        f.write("//\n")

        # Write values
        for i, val in enumerate(values):
            hex_val = val & mask

            # Magnitude float value for comment
            float_val = val / scale

            f.write(f"{hex_val:0{hex_width}X}  // |g_{i+1}| = {float_val:.6f}\n")


def write_constants_header(
    filepath: Path,
    constants: dict,
    alpha: float,
    beta: float | None,
    history_length: int,
    coeff_bits: int,
    coeff_frac_bits: int,
):
    """
    Write a header file with precomputed constants for reference.

    Args:
        filepath: Output file path
        constants: Dict from compute_fractional_constants
        alpha: Fractional order
        beta: Original beta if provided, or None
        history_length: History length
        coeff_bits: Coefficient bit width
        coeff_frac_bits: Coefficient fractional bits
    """
    with open(filepath, "w") as f:
        f.write("// Precomputed constants for fractional_lif.sv\n")
        f.write("// Generated by generate_coefficients.py\n")
        f.write("//\n")
        f.write("// Fractional parameters:\n")
        f.write(f"//   alpha = {alpha}\n")
        if beta is not None:
            f.write(f"//   beta = {beta} (standard LIF equivalent)\n")
            f.write(f"//   lambda = (1 - beta) / beta = {constants['lam']:.6f}\n")
        else:
            f.write(f"//   lambda = {constants['lam']:.6f}\n")
        f.write(f"//   C = 1 / dt^alpha = {constants['C']:.6f}\n")
        f.write(f"//   denominator = C + lambda = {constants['denom']:.6f}\n")
        f.write(f"//   1 / denominator = {constants['inv_denom']:.6f}\n")
        f.write("//\n")
        f.write("// SystemVerilog parameters:\n")
        f.write(f"//   parameter HISTORY_LENGTH = {history_length};\n")
        f.write(f"//   parameter COEFF_WIDTH = {coeff_bits};\n")
        f.write(f"//   parameter COEFF_FRAC_BITS = {coeff_frac_bits};\n")
        f.write(
            f"//   parameter [15:0] C_SCALED = 16'd{constants['C_SCALED']};  "
            f"// Q8.{constants['c_frac_bits']}\n"
        )
        f.write(
            f"//   parameter [15:0] INV_DENOM = 16'd{constants['INV_DENOM']};  "
            f"// Q0.{constants['inv_denom_frac_bits']}\n"
        )
        f.write("//\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GL coefficients for fractional_lif.sv"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Fractional order (0 < alpha <= 1), default: 0.5",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=None,
        help="Leakage parameter lambda (>= 0). If not provided, computed from --beta",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9,
        help="Standard LIF beta for computing lambda = (1-beta)/beta, default: 0.9",
    )
    parser.add_argument(
        "--dt", type=float, default=1.0, help="Discrete timestep, default: 1.0"
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=64,
        help="Number of history values (H), default: 64",
    )
    parser.add_argument(
        "--coeff-bits", type=int, default=16, help="Coefficient bit width, default: 16"
    )
    parser.add_argument(
        "--coeff-frac-bits",
        type=int,
        default=15,
        help="Coefficient fractional bits, default: 15 (QU1.15 with 16-bit coeffs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated files (required unless --constants-only)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for output filenames (e.g., 'hl1_' for layer-specific)",
    )
    parser.add_argument(
        "--constants-only",
        action="store_true",
        help="Compute and print only fractional constants (skip GL coefficient generation)",
    )

    args = parser.parse_args()

    # Determine lambda
    if args.lam is not None:
        lam = args.lam
        beta_used = None
    else:
        lam = beta_to_lam(args.beta)
        beta_used = args.beta
        print(f"Computing lambda from beta={args.beta}: lambda = {lam:.6f}")

    if not args.constants_only and args.output_dir is None:
        parser.error("--output-dir is required unless --constants-only is specified")

    # Compute constants (used in both full and constants-only modes)
    constants = compute_fractional_constants(
        alpha=args.alpha,
        lam=lam,
        dt=args.dt,
    )

    print("\nConstants:")
    print(f"  alpha = {args.alpha}")
    print(f"  lambda = {lam:.6f}")
    print(f"  dt = {args.dt}")
    print(f"  C = {constants['C']:.6f} -> C_SCALED = {constants['C_SCALED']} (Q8.8)")
    print(
        f"  1/(C+λ) = {constants['inv_denom']:.6f} -> INV_DENOM = {constants['INV_DENOM']} (Q0.16)"
    )

    print("\n" + "=" * 60)
    print("SystemVerilog parameters (copy to fractional_lif instantiation):")
    print("=" * 60)
    print(f"    .HISTORY_LENGTH({args.history_length}),")
    print(f"    .COEFF_WIDTH({args.coeff_bits}),")
    print(f"    .COEFF_FRAC_BITS({args.coeff_frac_bits}),")
    print(f"    .C_SCALED(16'd{constants['C_SCALED']}),")
    print(f"    .INV_DENOM(16'd{constants['INV_DENOM']}),")

    if args.constants_only:
        if args.output_dir is not None:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            constants_file = output_dir / f"{args.prefix}fractional_constants.txt"
            write_constants_header(
                constants_file,
                constants,
                alpha=args.alpha,
                beta=beta_used,
                history_length=args.history_length,
                coeff_bits=args.coeff_bits,
                coeff_frac_bits=args.coeff_frac_bits,
            )
            print(f"\nWrote constants to {constants_file}")
        return

    # This magnitude-only flow relies on coefficient sign behavior for 0 < alpha <= 1
    assert (
        0 < args.alpha <= 1.0
    ), "Magnitude-only coefficient export requires 0 < alpha <= 1.0"

    # Compute GL coefficients (g_0 to g_{H-1})
    print(
        f"Computing GL coefficients for alpha={args.alpha}, H={args.history_length}..."
    )
    coeffs = compute_gl_coefficients(args.alpha, args.history_length)

    # We need g_1 to g_{H-1} (skip g_0 = 1)
    coeffs_to_store = coeffs[1:]

    print(f"  g_0 = {coeffs[0].item():.6f} (implicit, not stored)")
    print(f"  g_1 = {coeffs[1].item():.6f}")
    print(f"  g_2 = {coeffs[2].item():.6f}")
    print(f"  g_{{H-1}} = {coeffs[-1].item():.6f}")

    # Validate sign pattern and quantize magnitudes
    coeffs_np = (
        coeffs_to_store.numpy()
        if hasattr(coeffs_to_store, "numpy")
        else np.array(coeffs_to_store)
    )
    if np.any(coeffs_np > 1e-12):
        raise ValueError(
            "Expected g_k <= 0 for k>=1 when 0<alpha<=1; found positive coefficient"
        )

    quantized = quantize_coefficients_magnitude(
        coeffs_to_store,
        bits=args.coeff_bits,
        frac_bits=args.coeff_frac_bits,
    )

    # Write output files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coeff_file = output_dir / f"{args.prefix}gl_coefficients.mem"
    constants_file = output_dir / f"{args.prefix}fractional_constants.txt"

    write_mem_file(
        coeff_file,
        quantized,
        bits=args.coeff_bits,
        frac_bits=args.coeff_frac_bits,
        alpha=args.alpha,
        lam=lam,
        history_length=args.history_length,
    )
    print(f"\nWrote {len(quantized)} coefficients to {coeff_file}")

    write_constants_header(
        constants_file,
        constants,
        alpha=args.alpha,
        beta=beta_used,
        history_length=args.history_length,
        coeff_bits=args.coeff_bits,
        coeff_frac_bits=args.coeff_frac_bits,
    )
    print(f"Wrote constants to {constants_file}")

    # Print SystemVerilog parameters for copy-paste
    print(f'    .GL_COEFF_FILE("{args.prefix}gl_coefficients.mem")')


if __name__ == "__main__":
    main()
