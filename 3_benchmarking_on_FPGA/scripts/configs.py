"""Registry of stage-3 FPGA benchmark configurations.

Pointers only — numeric parameters (HL1, HL2, NUM_TIMESTEPS, HISTORY_LENGTH,
INV_DENOM, GL_COEFF_FILE, SHIFT_*, weight paths) live in their canonical
locations:
  - cocotb Makefile cp_integration_<config> block (source of truth for sim)
  - 3_benchmarking_on_FPGA/sv/board_top_<config>.sv (source of truth for synth)

Adding a new config means adding a CONFIGS entry pointing at:
  - the cp_integration Makefile target (already present or to be added)
  - the board_top SV file (or None if not yet authored)
  - the XDC (or None)
  - the weights dir (relative to common/sv/cocotb/tests/weights/)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    name: str
    neuron_type: str
    cocotb_test_target: str
    board_top: str | None
    xdc: str | None
    weights_dir: str | None
    display_label: str

    @property
    def has_synth_artifacts(self) -> bool:
        return self.board_top is not None and self.xdc is not None


CONFIGS: list[Config] = [
    Config(
        name="lif-64-16",
        neuron_type="lif",
        cocotb_test_target="cp_integration_lif-64-16",
        board_top="board_top_lif_64_16",
        xdc="nexys_a7_lif_64_16.xdc",
        weights_dir="lif-64-16",
        display_label="LIF 64-16",
    ),
    Config(
        name="lif-32-16",
        neuron_type="lif",
        cocotb_test_target="cp_integration_lif-32-16",
        board_top="board_top_lif_32_16",
        xdc="nexys_a7_lif_32_16.xdc",
        weights_dir="lif-32-16",
        display_label="LIF 32-16",
    ),
    Config(
        name="frac-32-4-16",
        neuron_type="fractional",
        cocotb_test_target="cp_integration_frac-32-4-16",
        board_top="board_top_fractional_32_4_16",
        xdc="nexys_a7_fractional_32_4_16.xdc",
        weights_dir="fractional-32-4-16",
        display_label="Frac 32-4 (H=16)",
    ),
    Config(
        name="frac-16-4-32",
        neuron_type="fractional",
        cocotb_test_target="cp_integration_frac-16-4-32",
        board_top="board_top_fractional_16_4_32",
        xdc="nexys_a7_fractional_16_4_32.xdc",
        weights_dir="fractional-16-4-32",
        display_label="Frac 16-4 (H=32)",
    ),
    Config(
        name="bitshift-custom_slow_decay",
        neuron_type="bitshift",
        cocotb_test_target="cp_integration_bitshift_custom_slow_decay",
        board_top="board_top_bitshift_custom_slow_decay",
        xdc="nexys_a7_bitshift_custom_slow_decay.xdc",
        weights_dir="bitshift-custom_slow_decay",
        display_label="Bitshift slow",
    ),
    Config(
        name="frac-32-8-8-q2_13",
        neuron_type="fractional",
        cocotb_test_target="cp_integration_frac-32-8-8-QS2_13",
        board_top="board_top_fractional_32_8_8",
        xdc="nexys_a7_general.xdc",
        weights_dir="fractional-32-8-8/q2_13",
        display_label="Frac 32-8 Q2.13",
    ),
    Config(
        name="bitshift-64-32-4",
        neuron_type="bitshift",
        cocotb_test_target="cp_integration_bitshift-64-32-4",
        board_top="board_top_bitshift_64_32_4",
        xdc="nexys_a7_general.xdc",
        weights_dir="bitshift-64-32-4",
        display_label="Bitshift 64-32-4",
    ),
    Config(
        name="lif-32-32",
        neuron_type="lif",
        cocotb_test_target="cp_integration_leaky-32-32",
        board_top="board_top_lif_32_32",
        xdc="nexys_a7_general.xdc",
        weights_dir="leaky-32-32",
        display_label="LIF 32-32",
    ),
]


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE3_ROOT = REPO_ROOT / "3_benchmarking_on_FPGA"
RESULTS_ROOT = STAGE3_ROOT / "results"
CONSTRAINTS_ROOT = STAGE3_ROOT / "constraints"
SV_ROOT = STAGE3_ROOT / "sv"
WEIGHTS_ROOT = REPO_ROOT / "common" / "sv" / "cocotb" / "tests" / "weights"
COMMON_SV_ROOT = REPO_ROOT / "common" / "sv"


def get(name: str) -> Config:
    for c in CONFIGS:
        if c.name == name:
            return c
    raise KeyError(f"Unknown config: {name!r}. Known: {[c.name for c in CONFIGS]}")


def all_with_synth_artifacts() -> list[Config]:
    return [c for c in CONFIGS if c.has_synth_artifacts]
