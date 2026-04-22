"""
visualize_training_metrics.py
Visualizes SNN RL training data from a CSV log file.

Usage:
    python visualize_training_metrics.py metrics/bitshift.csv
    python visualize_training_metrics.py metrics/bitshift.csv --ema-alpha 0.1 --output images/training/bitshift-training.png
    python visualize_training_metrics.py metrics/bitshift.csv --roll-window 100 --output images/training/bitshift-training.png
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_EMA_ALPHA = 0.05  # lower = smoother / more lag
DEFAULT_ROLL_WINDOW = 50  # window for rolling-std band on training panel
SCATTER_ALPHA = 0.18
BAND_ALPHA = 0.20
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG = "#0f1117"
PANEL_BG = "#161b22"
SPINE_COLOR = "#30363d"
TICK_COLOR = "#c9d1d9"
RAW_COLOR = "#8b949e"
ACCENT_TRAIN = "#58a6ff"
ACCENT_GEN = "#3fb950"
ACCENT_BEST = "#f78166"
ACCENT_EPS = "#d2a8ff"


def ema(series, alpha):
    return series.ewm(alpha=alpha, adjust=False).mean()


def parse_pipe_floats(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return None
    try:
        return [float(v) for v in str(cell).split("|")]
    except ValueError:
        return None


def load(path):
    df = pd.read_csv(path, header=0)
    df.columns = df.columns.str.strip()
    rename = {
        "episode": "episode",
        "episode_steps": "episode_steps",
        "episode_reward": "episode_reward",
        "epsilon": "epsilon",
        "running_avg_100": "running_avg",
        "generalization_avg": "gen_avg",
        "generalization_seeds": "gen_seeds",
        "generalization_rewards": "gen_rewards",
        "best_running_avg_100": "best_running_avg",
        "best_generalization_avg": "best_gen_avg",
        "saved_best_running_model": "saved_running",
        "saved_best_generalization_model": "saved_gen",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "gen_rewards" in df.columns:
        reward_lists = df["gen_rewards"].apply(parse_pipe_floats)
    else:
        reward_lists = pd.Series([None] * len(df), index=df.index)

    df["gen_min"] = reward_lists.apply(lambda r: float(np.min(r)) if r else np.nan)
    df["gen_max"] = reward_lists.apply(lambda r: float(np.max(r)) if r else np.nan)
    df["gen_p25"] = reward_lists.apply(
        lambda r: float(np.percentile(r, 25)) if r else np.nan
    )
    df["gen_p75"] = reward_lists.apply(
        lambda r: float(np.percentile(r, 75)) if r else np.nan
    )
    df["gen_std"] = reward_lists.apply(lambda r: float(np.std(r)) if r else np.nan)
    df["gen_n"] = reward_lists.apply(lambda r: len(r) if r else np.nan)
    return df


def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TICK_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TICK_COLOR)
    ax.yaxis.label.set_color(TICK_COLOR)
    ax.title.set_color(TICK_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
    ax.grid(color=SPINE_COLOR, linewidth=0.5, linestyle="--", alpha=0.6)


def plot(df, ema_alpha, roll_window, output):
    eps = df["episode"]
    rewards = df["episode_reward"]

    smooth = ema(rewards, ema_alpha)
    roll_std = rewards.rolling(roll_window, min_periods=5).std()
    band_hi = smooth + roll_std
    band_lo = smooth - roll_std

    gen_mask = df["gen_avg"].notna()
    gen_eps = eps[gen_mask]
    gen_avg = df["gen_avg"][gen_mask]
    gen_min = df["gen_min"][gen_mask]
    gen_max = df["gen_max"][gen_mask]
    gen_p25 = df["gen_p25"][gen_mask]
    gen_p75 = df["gen_p75"][gen_mask]
    best_gen = df["best_gen_avg"][gen_mask]

    has_minmax = gen_min.notna().any()
    has_iqr = gen_p25.notna().any()

    saved_run_eps = eps[df["saved_running"] == 1]
    saved_gen_df = df[df["saved_gen"] == 1]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1], "hspace": 0.06},
    )
    fig.patch.set_facecolor(DARK_BG)
    ax_train, ax_gen, ax_eps_panel = axes
    for ax in axes:
        style_ax(ax)

    # ── Panel 1: Training ────────────────────────────────────────────────────
    ax_train.scatter(
        eps,
        rewards,
        s=5,
        color=RAW_COLOR,
        alpha=SCATTER_ALPHA,
        linewidths=0,
        zorder=1,
        label="Episode reward",
    )
    ax_train.fill_between(
        eps,
        band_lo,
        band_hi,
        color=ACCENT_TRAIN,
        alpha=BAND_ALPHA,
        zorder=2,
        label=f"EMA ± rolling std ({roll_window} ep)",
    )
    ax_train.plot(
        eps,
        smooth,
        color=ACCENT_TRAIN,
        linewidth=2.0,
        zorder=3,
        label=f"EMA  α={ema_alpha}",
    )
    ax_train.plot(
        eps,
        df["best_running_avg"],
        color=ACCENT_BEST,
        linewidth=1.1,
        linestyle="--",
        alpha=0.75,
        zorder=3,
        label="Best running avg",
    )
    for x in saved_run_eps:
        ax_train.axvline(x, color=ACCENT_BEST, linewidth=0.8, alpha=0.35, zorder=2)

    ax_train.set_ylabel("Reward", fontsize=10)
    ax_train.set_title("Training Episode Rewards", fontsize=11, pad=6)
    ax_train.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.25,
        labelcolor=TICK_COLOR,
        facecolor=PANEL_BG,
        edgecolor=SPINE_COLOR,
    )

    # ── Panel 2: Generalization ───────────────────────────────────────────────
    if has_minmax:
        ax_gen.fill_between(
            gen_eps,
            gen_min,
            gen_max,
            color=ACCENT_GEN,
            alpha=0.12,
            zorder=1,
            label="Seed min–max range",
        )
    if has_iqr:
        ax_gen.fill_between(
            gen_eps,
            gen_p25,
            gen_p75,
            color=ACCENT_GEN,
            alpha=0.28,
            zorder=2,
            label="Seed IQR (25–75%)",
        )
    ax_gen.plot(
        gen_eps,
        gen_avg,
        color=ACCENT_GEN,
        linewidth=2.0,
        zorder=3,
        marker="o",
        markersize=4,
        label="Gen mean",
    )
    ax_gen.plot(
        gen_eps,
        best_gen,
        color=ACCENT_BEST,
        linewidth=1.1,
        linestyle="--",
        alpha=0.8,
        zorder=3,
        label="Best gen avg",
    )
    if not saved_gen_df.empty:
        ax_gen.scatter(
            saved_gen_df["episode"],
            saved_gen_df["best_gen_avg"],
            color=ACCENT_BEST,
            s=70,
            zorder=5,
            marker="*",
            label="Best gen model saved",
        )

    ax_gen.set_ylabel("Generalization Reward", fontsize=10)
    ax_gen.set_title("Generalization Evaluations (across seeds)", fontsize=11, pad=6)
    ax_gen.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.25,
        labelcolor=TICK_COLOR,
        facecolor=PANEL_BG,
        edgecolor=SPINE_COLOR,
    )

    # ── Panel 3: Epsilon ──────────────────────────────────────────────────────
    ax_eps_panel.plot(eps, df["epsilon"], color=ACCENT_EPS, linewidth=1.5, zorder=2)
    ax_eps_panel.fill_between(
        eps, 0, df["epsilon"], color=ACCENT_EPS, alpha=0.15, zorder=1
    )
    ax_eps_panel.set_ylabel("ε", fontsize=10)
    ax_eps_panel.set_xlabel("Episode", fontsize=10)
    ax_eps_panel.set_title("Exploration Rate (ε)", fontsize=11, pad=6)
    ax_eps_panel.set_ylim(bottom=0)

    fig.suptitle("SNN RL Training Run", fontsize=13, color=TICK_COLOR, y=0.998)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot SNN RL training logs.")
    parser.add_argument("csv", help="Path to the training CSV file")
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=DEFAULT_EMA_ALPHA,
        help=f"EMA smoothing factor (default {DEFAULT_EMA_ALPHA})",
    )
    parser.add_argument(
        "--roll-window",
        type=int,
        default=DEFAULT_ROLL_WINDOW,
        help=f"Rolling std window (default {DEFAULT_ROLL_WINDOW})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Save to file instead of displaying (e.g. plot.png)",
    )
    args = parser.parse_args()

    df = load(args.csv)
    n_gen = df["gen_avg"].notna().sum()
    print(
        f"Loaded {len(df)} episodes | {n_gen} gen checkpoints | "
        f"ep {df['episode'].min()}–{df['episode'].max()}"
    )
    plot(df, args.ema_alpha, args.roll_window, args.output)


if __name__ == "__main__":
    main()
