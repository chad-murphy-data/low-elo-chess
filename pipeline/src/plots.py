"""
Visualization: generate all analysis plots.

Produces:
1. Blunder rate by ELO band
2. Blunder rate by clock remaining × ELO band
3. Centipawn loss by move distance
4. Blunder rate by move number
5. Defended vs undefended blunder rate by ELO
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Consistent style
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("viridis", 5)
ELO_BAND_ORDER = ["500-700", "700-900", "900-1100", "1100-1300", "1300-1500"]


def _save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_blunder_rate_by_elo(df, fig_dir):
    """Bar chart: blunder rate by ELO band."""
    data = (
        df.groupby("elo_band")["is_blunder"]
        .agg(["mean", "count", "sem"])
        .reindex(ELO_BAND_ORDER)
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        data.index, data["mean"], yerr=data["sem"] * 1.96,
        capsize=4, color=PALETTE, edgecolor="black", linewidth=0.5,
    )

    # Add count labels
    for bar, (_, row) in zip(bars, data.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"n={int(row['count']):,}", ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("ELO Band")
    ax.set_ylabel("Blunder Rate (CPL > 100)")
    ax.set_title("Blunder Rate by ELO Band")
    ax.set_ylim(0, None)

    _save(fig, fig_dir / "blunder_rate_by_elo.png")


def plot_blunder_by_clock(df, fig_dir):
    """Grouped bar chart: blunder rate by clock bin × ELO band."""
    clock_order = ["<30s", "30-60s", "1-2min", "2-5min", "5min+"]

    subset = df.dropna(subset=["clock_bin", "elo_band"])
    if len(subset) == 0:
        print("  Skipping clock plot: no clock data")
        return

    pivot = (
        subset.groupby(["elo_band", "clock_bin"])["is_blunder"]
        .mean()
        .unstack(fill_value=np.nan)
    )

    # Reorder
    pivot = pivot.reindex(index=ELO_BAND_ORDER, columns=clock_order).dropna(
        how="all"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, width=0.8, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("ELO Band")
    ax.set_ylabel("Blunder Rate")
    ax.set_title("Blunder Rate by Clock Remaining and ELO Band")
    ax.legend(title="Clock Remaining", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    _save(fig, fig_dir / "blunder_by_clock_and_elo.png")


def plot_cpl_by_move_distance(df, fig_dir):
    """Box plot: centipawn loss by move distance."""
    subset = df.dropna(subset=["move_distance", "centipawn_loss"])
    # Cap CPL at 500 for visualization
    subset = subset.copy()
    subset["centipawn_loss_capped"] = subset["centipawn_loss"].clip(upper=500)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Group by distance
    distances = sorted(subset["move_distance"].unique())
    distances = [d for d in distances if d <= 7]  # max chessboard distance is 7

    box_data = [
        subset[subset["move_distance"] == d]["centipawn_loss_capped"]
        for d in distances
    ]

    bp = ax.boxplot(
        box_data, labels=[str(int(d)) for d in distances],
        patch_artist=True, showfliers=False, medianprops={"color": "black"},
    )
    for patch, color in zip(bp["boxes"], sns.color_palette("viridis", len(distances))):
        patch.set_facecolor(color)

    ax.set_xlabel("Move Distance (Chebyshev)")
    ax.set_ylabel("Centipawn Loss (capped at 500)")
    ax.set_title("Centipawn Loss by Move Distance")

    # Add mean line
    means = [
        subset[subset["move_distance"] == d]["centipawn_loss"].mean()
        for d in distances
    ]
    ax.plot(range(1, len(distances) + 1), means, "r--o", label="Mean CPL", markersize=4)
    ax.legend()

    _save(fig, fig_dir / "cpl_by_move_distance.png")


def plot_blunder_rate_by_move_number(df, fig_dir):
    """Line chart: blunder rate by move number, colored by ELO band."""
    subset = df.dropna(subset=["move_number", "is_blunder", "elo_band"])
    if len(subset) == 0:
        return

    subset = subset.copy()
    # Bin move numbers in groups of 5
    subset["move_bin"] = (subset["move_number"] // 10) * 5 + 5  # midpoint of 10-ply bins
    subset = subset[subset["move_bin"] <= 50]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, band in enumerate(ELO_BAND_ORDER):
        band_df = subset[subset["elo_band"] == band]
        if len(band_df) == 0:
            continue
        rates = band_df.groupby("move_bin")["is_blunder"].mean()
        ax.plot(rates.index, rates.values, "o-", label=band,
                color=PALETTE[i], linewidth=2, markersize=4)

    ax.set_xlabel("Move Number (ply, binned)")
    ax.set_ylabel("Blunder Rate")
    ax.set_title("Blunder Rate by Move Number and ELO Band")
    ax.legend(title="ELO Band")

    _save(fig, fig_dir / "blunder_by_move_number.png")


def plot_defended_vs_undefended(df, fig_dir):
    """Grouped bar chart: blunder rate for defended vs undefended pieces by ELO."""
    if "piece_is_defended" not in df.columns:
        print("  Skipping defended/undefended plot: column missing")
        return

    subset = df.dropna(subset=["piece_is_defended", "elo_band"]).copy()
    subset["defended"] = subset["piece_is_defended"].apply(
        lambda x: "Defended" if x else "Undefended"
    )

    pivot = (
        subset.groupby(["elo_band", "defended"])["is_blunder"]
        .mean()
        .unstack(fill_value=np.nan)
    )
    pivot = pivot.reindex(index=ELO_BAND_ORDER).dropna(how="all")

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, width=0.7, edgecolor="black", linewidth=0.5,
               color=["#2ecc71", "#e74c3c"])

    ax.set_xlabel("ELO Band")
    ax.set_ylabel("Blunder Rate")
    ax.set_title("Blunder Rate: Defended vs Undefended Pieces by ELO Band")
    ax.legend(title="Piece Status")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    _save(fig, fig_dir / "defended_vs_undefended.png")


def generate_all_plots(csv_path="data/moves_features.csv", fig_dir="results/figures"):
    """Generate all plots from the feature dataset."""
    fig_path = Path(fig_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    print("Loading data for plotting...")
    df = pd.read_csv(csv_path)

    # Prepare data — derive is_blunder from centipawn_loss
    df = df.dropna(subset=["eval_before", "eval_after", "centipawn_loss"])

    # Exclude near-mate
    if "is_near_mate" in df.columns:
        df = df[df["is_near_mate"] != True]

    df["is_blunder"] = (df["centipawn_loss"] > 100).astype(float)

    # Convert booleans
    for col in ["in_check", "is_capture", "is_check_given", "piece_is_defended"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add ELO band
    def elo_band(elo):
        if pd.isna(elo):
            return None
        elo = int(elo)
        for lo, hi in [(500, 700), (700, 900), (900, 1100),
                       (1100, 1300), (1300, 1500)]:
            if lo <= elo < hi:
                return f"{lo}-{hi}"
        return None

    df["elo_band"] = df["player_elo"].apply(elo_band)
    df = df.dropna(subset=["elo_band"])

    # Add clock bins
    def clock_bin(secs):
        if pd.isna(secs):
            return None
        if secs < 30:
            return "<30s"
        elif secs < 60:
            return "30-60s"
        elif secs < 120:
            return "1-2min"
        elif secs < 300:
            return "2-5min"
        else:
            return "5min+"

    df["clock_bin"] = df["clock_remaining"].apply(clock_bin)

    print(f"Plotting with {len(df)} observations...")

    print("Plot 1: Blunder rate by ELO band")
    plot_blunder_rate_by_elo(df, fig_path)

    print("Plot 2: Blunder rate by clock × ELO")
    plot_blunder_by_clock(df, fig_path)

    print("Plot 3: CPL by move distance")
    plot_cpl_by_move_distance(df, fig_path)

    print("Plot 4: Blunder rate by move number")
    plot_blunder_rate_by_move_number(df, fig_path)

    print("Plot 5: Defended vs undefended blunder rate")
    plot_defended_vs_undefended(df, fig_path)

    print("All plots generated.")


if __name__ == "__main__":
    generate_all_plots()
