"""
Plot training metrics from CartPole SNN runs.

Default behavior:
- Input:  train/metrics directory
- Output: train/images/training directory
- Plot: episode reward (light), running_avg_100 (line), generalization_avg (markers)

If a directory is provided as input, all CSV files in that directory are combined
into one multi-config comparison plot.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


@dataclass
class MetricSeries:
    label: str
    episode: np.ndarray
    episode_reward: np.ndarray
    running_avg_100: np.ndarray
    generalization_avg: np.ndarray
    generalization_seeds: np.ndarray


def _parse_float(value: str | None) -> float:
    if value is None:
        return np.nan
    value = value.strip()
    if value == "":
        return np.nan
    return float(value)


def _parse_generalization_seeds(value: str | None) -> float:
    """Parse generalization_seeds into a numeric indicator/count.

    Supports:
    - numeric values (e.g., "30")
    - pipe-delimited lists (e.g., "42|49|56|...")
    """
    if value is None:
        return np.nan
    value = value.strip()
    if value == "":
        return np.nan

    # Newer metric files may store a pipe-delimited list; treat as count.
    if "|" in value:
        items = [token for token in value.split("|") if token.strip()]
        return float(len(items)) if items else np.nan

    return float(value)


def _label_from_filename(path: Path) -> str:
    name = path.stem
    if name.startswith("dqn_"):
        name = name[4:]
    if name.endswith("-training-metrics"):
        name = name[: -len("-training-metrics")]

    parts = [p for p in name.split("-") if p]
    if not parts:
        return name

    # Find architecture/history tokens by suffix, keep all prior tokens as neuron-type descriptor.
    hl1_idx = next((i for i, p in enumerate(parts) if p.endswith("hl1")), None)
    hl2_idx = next((i for i, p in enumerate(parts) if p.endswith("hl2")), None)
    hist_idx = next((i for i, p in enumerate(parts) if p.endswith("hist")), None)

    if hl1_idx is None or hl2_idx is None:
        return name

    neuron_desc = "-".join(parts[:hl1_idx])
    if not neuron_desc:
        neuron_desc = parts[0]

    # Human-readable neuron descriptor, e.g.:
    # bitshift-custom_slow_decay -> Bitshift Custom Slow Decay
    neuron_words: list[str] = []
    for segment in neuron_desc.split("-"):
        for sub in segment.split("_"):
            if sub:
                neuron_words.append(sub.capitalize())
    neuron_label = " ".join(neuron_words) if neuron_words else neuron_desc.capitalize()

    hl1_token = parts[hl1_idx]
    hl2_token = parts[hl2_idx]
    hl1_size = hl1_token[: -len("hl1")]
    hl2_size = hl2_token[: -len("hl2")]

    label_parts = [
        neuron_label,
        f"Hidden layer 1 size {hl1_size}",
        f"Hidden layer 2 size {hl2_size}",
    ]
    if hist_idx is not None and hist_idx > hl2_idx:
        hist_token = parts[hist_idx]
        hist_size = hist_token[: -len("hist")]
        label_parts.append(f"History length {hist_size}")

    return " | ".join(label_parts)


def load_metrics_csv(path: Path) -> MetricSeries:
    episodes: list[float] = []
    rewards: list[float] = []
    running: list[float] = []
    generalization: list[float] = []
    generalization_seeds: list[float] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"episode", "episode_reward"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: missing required columns {sorted(missing)}")

        for row in reader:
            episodes.append(_parse_float(row.get("episode")))
            rewards.append(_parse_float(row.get("episode_reward")))
            running.append(_parse_float(row.get("running_avg_100")))
            generalization.append(_parse_float(row.get("generalization_avg")))
            generalization_seeds.append(
                _parse_generalization_seeds(row.get("generalization_seeds"))
            )

    return MetricSeries(
        label=_label_from_filename(path),
        episode=np.array(episodes, dtype=float),
        episode_reward=np.array(rewards, dtype=float),
        running_avg_100=np.array(running, dtype=float),
        generalization_avg=np.array(generalization, dtype=float),
        generalization_seeds=np.array(generalization_seeds, dtype=float),
    )


def _finite_mask(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr)


def _generalization_mask(series: MetricSeries) -> np.ndarray:
    # Plot only when generalization was actually run.
    # Preferred signal: generalization_seeds > 0.
    # Fallback: finite generalization_avg if seeds column is unavailable.
    avg_ok = np.isfinite(series.generalization_avg)
    seeds_ok = np.isfinite(series.generalization_seeds)
    if seeds_ok.any():
        return avg_ok & seeds_ok & (series.generalization_seeds > 0)
    return avg_ok


def _legend_multiline(label: str) -> str:
    """Compact legend label with line breaks."""
    return label.replace(" | ", "\n")


def plot_single(
    series: MetricSeries,
    output_path: Path,
    show_reward: bool,
    show_running_avg: bool,
    show_generalization: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    if show_reward:
        ax.plot(
            series.episode,
            series.episode_reward,
            color="tab:blue",
            alpha=0.25,
            linewidth=1.0,
            label="Episode reward",
        )

    if show_running_avg:
        mask = _finite_mask(series.running_avg_100)
        if mask.any():
            ax.plot(
                series.episode[mask],
                series.running_avg_100[mask],
                color="tab:blue",
                linewidth=2.5,
                label="Running avg (100)",
            )

    if show_generalization:
        mask = _generalization_mask(series)
        if mask.any():
            ax.scatter(
                series.episode[mask],
                series.generalization_avg[mask],
                color="tab:orange",
                marker="o",
                s=45,
                label="Generalization avg",
                zorder=3,
            )
            ax.plot(
                series.episode[mask],
                series.generalization_avg[mask],
                color="tab:orange",
                linewidth=1.25,
                alpha=0.8,
                linestyle="--",
            )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_multi(
    series_list: Iterable[MetricSeries],
    output_path: Path,
    show_reward: bool,
    show_running_avg: bool,
    show_generalization: bool,
    jitter_generalization: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    color_handles: list[Line2D] = []
    marker_styles = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    for idx, series in enumerate(series_list):
        color = f"C{idx % 10}"
        marker = marker_styles[idx % len(marker_styles)]

        if show_reward:
            ax.plot(
                series.episode,
                series.episode_reward,
                color=color,
                alpha=0.12,
                linewidth=0.8,
            )

        if show_running_avg:
            mask = _finite_mask(series.running_avg_100)
            if mask.any():
                ax.plot(
                    series.episode[mask],
                    series.running_avg_100[mask],
                    color=color,
                    linewidth=2.0,
                )
                color_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linewidth=3,
                        label=_legend_multiline(series.label),
                    )
                )

        if show_generalization:
            mask = _generalization_mask(series)
            if mask.any():
                x_vals = series.episode[mask]
                y_vals = series.generalization_avg[mask]

                if jitter_generalization:
                    rng = np.random.default_rng(12345 + idx)
                    x_vals = x_vals + rng.normal(loc=0.0, scale=0.18, size=len(x_vals))

                ax.scatter(
                    x_vals,
                    y_vals,
                    facecolors="none",
                    edgecolors=color,
                    marker=marker,
                    s=42,
                    linewidths=1.4,
                    alpha=0.75,
                )
                ax.plot(
                    x_vals,
                    y_vals,
                    color=color,
                    linewidth=1.1,
                    alpha=0.75,
                    linestyle="--",
                )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.25)

    series_anchor_y = 0.63
    if color_handles:
        config_legend = ax.legend(
            handles=color_handles,
            loc="upper left",
            fontsize=8,
            framealpha=0.9,
            title="Configuration (color)",
            title_fontsize=9,
        )
        ax.add_artist(config_legend)
        # Place the series legend directly beneath config legend.
        # Approximate per-entry height in axes coords for compact stacked layout.
        series_anchor_y = max(0.12, 1.0 - (0.12 + 0.075 * len(color_handles)))

    style_handles: list[Line2D] = []
    if show_running_avg:
        style_handles.append(
            Line2D([0], [0], color="black", linewidth=2.2, label="Running average (100)")
        )
    if show_generalization:
        style_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.8,
                label="Generalization average",
            )
        )
    if style_handles:
        ax.legend(
            handles=style_handles,
            loc="upper left",
            bbox_to_anchor=(0.0, series_anchor_y),
            fontsize=8,
            framealpha=0.9,
            title="Series type",
            title_fontsize=9,
        )

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def discover_csvs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.glob("*.csv") if p.is_file())
    raise FileNotFoundError(f"Input path not found: {input_path}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    train_dir = script_dir.parent

    default_input = train_dir / "metrics"
    default_output = train_dir / "images" / "training"

    parser = argparse.ArgumentParser(description="Visualize training metrics CSV files")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(default_input),
        help="CSV file or directory of CSV files (default: train/metrics)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output),
        help="Output image directory (default: train/images/training)",
    )
    parser.add_argument(
        "--no-reward",
        action="store_true",
        help="Hide raw episode reward trace",
    )
    parser.add_argument(
        "--no-running-avg",
        action="store_true",
        help="Hide running_avg_100",
    )
    parser.add_argument(
        "--no-generalization",
        action="store_true",
        help="Hide generalization_avg markers",
    )
    parser.add_argument(
        "--jitter-generalization",
        action="store_true",
        help="Apply small x-jitter to generalization markers/line in multi-config plots",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    csv_files = discover_csvs(input_path)
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_path}")

    show_reward = not args.no_reward
    show_running_avg = not args.no_running_avg
    show_generalization = not args.no_generalization

    if not (show_reward or show_running_avg or show_generalization):
        raise ValueError("At least one plot series must be enabled")

    all_series = [load_metrics_csv(path) for path in csv_files]

    if len(all_series) == 1:
        series = all_series[0]
        output_path = output_dir / f"{series.label}.png"
        plot_single(
            series,
            output_path,
            show_reward=show_reward,
            show_running_avg=show_running_avg,
            show_generalization=show_generalization,
        )
        print(f"Saved: {output_path}")
    else:
        output_path = output_dir / "training_comparison.png"
        plot_multi(
            all_series,
            output_path,
            show_reward=show_reward,
            show_running_avg=show_running_avg,
            show_generalization=show_generalization,
            jitter_generalization=args.jitter_generalization,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
