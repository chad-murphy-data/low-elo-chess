"""
Visualization: generate all analysis plots.

Produces:
1. Blunder rate by ELO band
2. Blunder rate by clock remaining × ELO band
3. Centipawn loss by move distance
4. Hung piece rate by piece recency
5. Missed capture rate by hanging piece recency
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


def plot_hung_piece_by_recency(df, fig_dir):
    """Bar chart: rate of creating hanging pieces by piece recency bins."""
    subset = df.dropna(subset=["moves_since_piece_last_moved"])
    if len(subset) == 0:
        print("  Skipping recency plot: no recency data")
        return

    subset = subset.copy()
    subset["recency_bin"] = pd.cut(
        subset["moves_since_piece_last_moved"],
        bins=[0, 2, 5, 10, 20, 200],
        labels=["0-2", "3-5", "6-10", "11-20", "21+"],
        right=True,
    )

    rates = (
        subset.groupby("recency_bin", observed=True)["created_hanging_piece"]
        .agg(["mean", "count", "sem"])
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        rates.index, rates["mean"], yerr=rates["sem"] * 1.96,
        capsize=4, color=sns.color_palette("rocket", len(rates)),
        edgecolor="black", linewidth=0.5,
    )

    for bar, (_, row) in zip(bars, rates.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"n={int(row['count']):,}", ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("Plies Since Piece Last Moved")
    ax.set_ylabel("Rate of Creating Hanging Piece")
    ax.set_title("Hanging Piece Creation Rate by Piece Recency")

    _save(fig, fig_dir / "hung_piece_by_recency.png")


def plot_missed_capture_by_recency(df, fig_dir):
    """Bar chart: missed capture rate by how long ago the hanging piece moved."""
    opp_hanging = df[df["opponent_had_hanging_piece"] == 1].copy()
    opp_hanging = opp_hanging.dropna(subset=["missed_capture_recency"])

    if len(opp_hanging) < 30:
        print("  Skipping missed capture plot: insufficient data")
        return

    opp_hanging["captured_it"] = opp_hanging["is_capture"].astype(int)
    opp_hanging["missed_it"] = 1 - opp_hanging["captured_it"]

    opp_hanging["recency_bin"] = pd.cut(
        opp_hanging["missed_capture_recency"],
        bins=[0, 2, 5, 10, 20, 200],
        labels=["0-2", "3-5", "6-10", "11-20", "21+"],
        right=True,
    )

    rates = (
        opp_hanging.groupby("recency_bin", observed=True)["missed_it"]
        .agg(["mean", "count", "sem"])
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        rates.index, rates["mean"], yerr=rates["sem"] * 1.96,
        capsize=4, color=sns.color_palette("mako", len(rates)),
        edgecolor="black", linewidth=0.5,
    )

    for bar, (_, row) in zip(bars, rates.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"n={int(row['count']):,}", ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("Plies Since Hanging Piece Last Moved")
    ax.set_ylabel("Miss Rate (did NOT capture)")
    ax.set_title("Missed Capture Rate by Hanging Piece Recency\n(Availability Heuristic Test)")

    _save(fig, fig_dir / "missed_capture_by_recency.png")


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


def plot_punishment_rate_by_elo(df, fig_dir):
    """Bar chart: hanging piece punishment rate by ELO band."""
    opp_hanging = df[df["opponent_had_hanging_piece"] == 1].copy()
    if len(opp_hanging) < 30:
        print("  Skipping punishment plot: insufficient data")
        return

    opp_hanging["punished"] = opp_hanging["is_capture"].astype(int)

    rates = (
        opp_hanging.groupby("elo_band")["punished"]
        .agg(["mean", "count", "sem"])
        .reindex(ELO_BAND_ORDER)
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        rates.index, rates["mean"], yerr=rates["sem"] * 1.96,
        capsize=4, color=PALETTE[:len(rates)],
        edgecolor="black", linewidth=0.5,
    )

    for bar, (_, row) in zip(bars, rates.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"n={int(row['count']):,}", ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("ELO Band")
    ax.set_ylabel("Punishment Rate (captured hanging piece)")
    ax.set_title("Hanging Piece Punishment Rate by ELO Band")
    ax.set_ylim(0, 1)

    _save(fig, fig_dir / "punishment_rate_by_elo.png")


def generate_all_plots(csv_path="data/moves_features.csv", fig_dir="results/figures"):
    """Generate all plots from the feature dataset."""
    fig_path = Path(fig_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    print("Loading data for plotting...")
    df = pd.read_csv(csv_path)

    # Prepare data
    bool_cols = [
        "is_blunder", "is_mistake", "is_inaccuracy", "is_near_mate",
        "in_check", "has_hanging_piece_before", "created_hanging_piece",
        "opponent_had_hanging_piece", "is_capture", "is_check_given",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)  # handle NA bools

    # Exclude near-mate
    df = df[df["is_near_mate"] != 1.0]
    df = df.dropna(subset=["is_blunder"])

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

    print("Plot 4: Hung piece rate by recency")
    plot_hung_piece_by_recency(df, fig_path)

    print("Plot 5: Missed capture rate by recency")
    plot_missed_capture_by_recency(df, fig_path)

    print("Plot 6: Blunder rate by move number")
    plot_blunder_rate_by_move_number(df, fig_path)

    print("Plot 7: Punishment rate by ELO")
    plot_punishment_rate_by_elo(df, fig_path)

    print("All plots generated.")


if __name__ == "__main__":
    generate_all_plots()
