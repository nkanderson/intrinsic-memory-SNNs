"""
Compare Grunwald-Letnikov binomial coefficients with bit-shift approximations.

This script analyzes whether bit-shift operations (which are computationally
efficient in hardware) can approximate the GL coefficients used in fractional-
order derivatives.
"""

import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path is set
from utils import compute_gl_coefficients  # noqa: E402


def get_gl_coefficients(alpha: float, history_length: int) -> list[float]:
    """
    Get GL binomial coefficients for a given fractional order alpha.

    Args:
        alpha: Fractional order (0 < alpha <= 1 typically)
        history_length: Number of coefficients to compute

    Returns:
        List of GL coefficient values [g_0, g_1, ..., g_{H-1}]
    """
    coeffs = compute_gl_coefficients(alpha, history_length)
    return coeffs.tolist()


def simple_bitshift(history_length: int) -> list[int]:
    """
    Generate sequence of bit-shift amounts (exponents).

    This produces the sequence: [0, 1, 2, 3, 4, ...]
    which represents right shifts by k bits for k=0..history_length-1.

    In hardware, value >> k is equivalent to dividing by 2^k.

    Args:
        history_length: Number of shift amounts to generate

    Returns:
        List of shift amounts [0, 1, 2, 3, ...]
    """
    return list(range(history_length))


def slow_decay_bitshift(history_length: int) -> list[int]:
    """
    Generate slow-decay bit-shift amounts with special case for first coefficient.

    This produces: [0, 1, 1, 2, 2, 3, 3, ...]

    The first coefficient (shift 0) does not repeat. Starting from the second position,
    each shift amount appears twice before incrementing.

    Args:
        history_length: Number of shift amounts to generate

    Returns:
        List of shift amounts where amounts repeat (except first)
    """
    shift_amounts = []
    for k in range(history_length):
        if k == 0:
            # First coefficient: shift by 0
            shift_amount = 0
        else:
            # For k >= 1: shift by (k+1)//2, so k=1,2 -> shift 1; k=3,4 -> shift 2, etc.
            shift_amount = (k + 1) // 2
        shift_amounts.append(shift_amount)
    return shift_amounts


def custom_bitshift(history_length: int, decay_rate: int = 3) -> list[int]:
    """
    Generate custom bit-shift amounts with specific repetition pattern.

    Pattern:
    - shift 0 once (k=0)
    - shift 1 once (k=1)
    - skip shift 2
    - shift 3 once (k=2)
    - shift 4 once (k=3)
    - shift 5 decay_rate times (e.g. k=4,5,6 for decay_rate = 3)
    - shift 6 decay_rate times (e.g. k=7,8,9 for decay_rate = 3)
    - shift 7 decay_rate times (e.g. k=10,11,12 for decay_rate = 3)
    - and so on...

    This produces shift amounts: [0, 1, 3, 4, 5, 5, 5, 6, 6, 6, ...]

    Args:
        history_length: Number of shift amounts to generate
        decay_rate: Number of times to repeat each shift >= 5

    Returns:
        List of custom bit-shift amounts following the specified pattern
    """
    shift_sequence = [0, 1, 3, 4]  # Initial shifts: 2^0, 2^-1, 2^-3, 2^-4

    # Build the full sequence by adding shifts >= 5 decay_rate times each
    shift = 5
    while len(shift_sequence) < history_length:
        shift_sequence.extend([shift] * decay_rate)
        shift += 1

    return shift_sequence[:history_length]


def custom_slow_decay_bitshift(history_length: int) -> list[int]:
    """
    Generate custom slow-decay bit-shift amounts with incrementing repetition.

    Pattern:
    - shift 0 once (k=0)
    - shift 1 once (k=1)
    - skip shift 2
    - shift 3 once (k=2)
    - shift 4 once (k=3)
    - shift 5 repeats 3 times (k=4,5,6)
    - shift 6 repeats 4 times (k=7,8,9,10)
    - shift 7 repeats 5 times (k=11,12,13,14,15)
    - shift N repeats (N-2) times for N >= 5

    This produces shift amounts: [0, 1, 3, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, ...]

    Args:
        history_length: Number of shift amounts to generate

    Returns:
        List of custom slow-decay bit-shift amounts following the specified pattern
    """
    shift_sequence = [0, 1, 3, 4]  # Initial shifts: 2^0, 2^-1, 2^-3, 2^-4

    # Build the full sequence: shift N repeats (N-2) times for N >= 5
    shift = 5
    while len(shift_sequence) < history_length:
        repeat_count = shift - 2  # shift 5 -> 3 times, shift 6 -> 4 times, etc.
        shift_sequence.extend([shift] * repeat_count)
        shift += 1

    return shift_sequence[:history_length]


def bitshift_amounts_to_sequence(shift_amounts: list[int]) -> list[float]:
    """
    Convert bit-shift amounts to coefficient values.

    Takes a list of shift amounts (integers) and converts them to the actual
    coefficient values by computing 1.0 / (2**shift) for each shift amount.

    Args:
        shift_amounts: List of integer shift amounts

    Returns:
        List of coefficient values computed from shift amounts

    Example:
        >>> bitshift_amounts_to_sequence([0, 1, 2, 3])
        [1.0, 0.5, 0.25, 0.125]
    """
    return [1.0 / (2**shift) for shift in shift_amounts]


def simple_sequence(history_length: int) -> list[float]:
    """Generate coefficient sequence using simple_bitshift."""
    return bitshift_amounts_to_sequence(simple_bitshift(history_length))


def slow_decay_sequence(history_length: int) -> list[float]:
    """Generate coefficient sequence using slow_decay_bitshift."""
    return bitshift_amounts_to_sequence(slow_decay_bitshift(history_length))


def custom_sequence(history_length: int, decay_rate: int = 3) -> list[float]:
    """
    Generate coefficient sequence using custom_bitshift.

    Args:
        history_length: Number of values to generate
        decay_rate: Number of times to repeat each value >= 2^-5
    """
    return bitshift_amounts_to_sequence(custom_bitshift(history_length, decay_rate))


def custom_slow_decay_sequence(history_length: int) -> list[float]:
    """Generate coefficient sequence using custom_slow_decay_bitshift."""
    return bitshift_amounts_to_sequence(custom_slow_decay_bitshift(history_length))


# TODO: Add plot showing divergence between GL coefficients and bit-shift values
# over time, as history_length increases to values as high as 256-512.

# TODO: Add method for finding closest bit-shift approximation to a given GL coefficient.
# This may be used to identify a particular bit shift sequence for approximating GL coefficients
# with a given alpha value. It may also be worthwhile testing an implementation with a simple
# incrementing bit shift, as compared to a more complex custom sequence.


def main():
    """Compare GL coefficients with bit-shift approximations."""
    # Example usage
    alpha = 0.5
    history_length = 32

    print(f"Comparing GL coefficients (alpha={alpha}) with bit-shift approximations")
    print(f"History length: {history_length}\n")

    gl_coeffs = get_gl_coefficients(alpha, history_length)
    bitshift_vals = simple_sequence(history_length)
    slow_decay_vals = slow_decay_sequence(history_length)
    # TODO: Compare different decay rates for the custom sequence
    custom_vals = custom_sequence(history_length, 4)
    custom_slow_decay_vals = custom_slow_decay_sequence(history_length)

    # Regular bit-shift comparison
    print("=" * 85)
    print("REGULAR BIT-SHIFT COMPARISON (2^0, 2^-1, 2^-2, ...)")
    print("=" * 85)
    print(
        f"{'Index':<6} {'|GL Coeff|':<20} {'Bit-Shift':<20} {'Difference':<20} "
        f"{'Rel Error %':<15}"
    )
    print("-" * 85)

    for k in range(history_length):
        gl_mag = abs(gl_coeffs[k])
        bitshift_val = bitshift_vals[k]
        diff = gl_mag - bitshift_val
        # Calculate relative error as percentage
        rel_error = (diff / gl_mag * 100) if gl_mag != 0 else 0
        print(
            f"{k:<6} {gl_mag:<20.10f} {bitshift_val:<20.10f} {diff:<20.10f} "
            f"{rel_error:<15.2f}"
        )

    # Slow-decay bit-shift comparison
    print("\n" + "=" * 85)
    print(
        "SLOW-DECAY BIT-SHIFT COMPARISON (2^0, 2^-1, 2^-1, 2^-2, 2^-2, 2^-3, 2^-3, ...)"
    )
    print("=" * 85)
    print(
        f"{'Index':<6} {'|GL Coeff|':<20} {'Slow-Decay':<20} {'Difference':<20} "
        f"{'Rel Error %':<15}"
    )
    print("-" * 85)

    for k in range(history_length):
        gl_mag = abs(gl_coeffs[k])
        slow_decay_val = slow_decay_vals[k]
        diff = gl_mag - slow_decay_val
        # Calculate relative error as percentage
        rel_error = (diff / gl_mag * 100) if gl_mag != 0 else 0
        print(
            f"{k:<6} {gl_mag:<20.10f} {slow_decay_val:<20.10f} {diff:<20.10f} "
            f"{rel_error:<15.2f}"
        )

    # Custom bit-shift comparison
    print("\n" + "=" * 85)
    print("CUSTOM BIT-SHIFT COMPARISON (2^0, 2^-1, 2^-3, 2^-4, 2^-5×3, 2^-6×3, ...)")
    print("=" * 85)
    print(
        f"{'Index':<6} {'|GL Coeff|':<20} {'Custom':<20} {'Difference':<20} "
        f"{'Rel Error %':<15}"
    )
    print("-" * 85)

    for k in range(history_length):
        gl_mag = abs(gl_coeffs[k])
        custom_val = custom_vals[k]
        diff = gl_mag - custom_val
        # Calculate relative error as percentage
        rel_error = (diff / gl_mag * 100) if gl_mag != 0 else 0
        print(
            f"{k:<6} {gl_mag:<20.10f} {custom_val:<20.10f} {diff:<20.10f} "
            f"{rel_error:<15.2f}"
        )

    # Custom slow-decay bit-shift comparison
    print("\n" + "=" * 85)
    print(
        "CUSTOM SLOW-DECAY COMPARISON "
        "(2^0, 2^-1, 2^-3, 2^-4, 2^-5×3, 2^-6×4, 2^-7×5, ...)"
    )
    print("=" * 85)
    print(
        f"{'Index':<6} {'|GL Coeff|':<20} {'Custom Slow':<20} {'Difference':<20} "
        f"{'Rel Error %':<15}"
    )
    print("-" * 85)

    for k in range(history_length):
        gl_mag = abs(gl_coeffs[k])
        custom_slow_val = custom_slow_decay_vals[k]
        diff = gl_mag - custom_slow_val
        # Calculate relative error as percentage
        rel_error = (diff / gl_mag * 100) if gl_mag != 0 else 0
        print(
            f"{k:<6} {gl_mag:<20.10f} {custom_slow_val:<20.10f} {diff:<20.10f} "
            f"{rel_error:<15.2f}"
        )


if __name__ == "__main__":
    main()
