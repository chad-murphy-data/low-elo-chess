"""
Research dashboard -- three-panel layout for chess blunder hazard model.

Panel 1: Exploratory Landscape  (blunder by ELO/piece, hang vs punishment)
Panel 2: Hazard Model           (survival curves, blunder arrival vs clock)
Panel 3: Live Coefficient Plot  (logistic regression with sample-size slider)
"""

import base64
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from flask import Flask, jsonify, render_template_string, request

from src.dashboard_template import TEMPLATE

app = Flask(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))

ELO_BANDS = ["500-700", "700-900", "900-1100", "1100-1300", "1300-1500"]
PIECE_TYPES = ["pawn", "knight", "bishop", "rook", "queen"]

# ---------------------------------------------------------------------------
# Cached dataframe (reloaded when CSV file modification time changes)
# ---------------------------------------------------------------------------
_df_cache = {"mtime": 0, "df": None}


def _load_df():
    """Load and cache the feature CSV. Re-reads if file was modified."""
    csv_path = DATA_DIR / "moves_features.csv"
    if not csv_path.exists():
        return None
    mtime = csv_path.stat().st_mtime
    if _df_cache["mtime"] == mtime and _df_cache["df"] is not None:
        return _df_cache["df"]

    df = pd.read_csv(csv_path)
    # Convert booleans
    for col in ["is_near_mate", "is_capture", "piece_is_defended",
                "is_executing_pattern", "is_two_move_attack_target"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without eval data, exclude near-mate
    df = df.dropna(subset=["eval_before", "eval_after", "centipawn_loss"])
    df = df[df["is_near_mate"] != 1.0].copy()

    # Derive blunder labels from centipawn_loss
    df["is_blunder"] = (df["centipawn_loss"] > 100).astype(float)

    # Add elo_band
    def _elo_band(elo):
        if pd.isna(elo):
            return None
        elo = int(elo)
        for lo, hi in [(500, 700), (700, 900), (900, 1100),
                       (1100, 1300), (1300, 1500)]:
            if lo <= elo < hi:
                return f"{lo}-{hi}"
        return None

    df["elo_band"] = df["player_elo"].apply(_elo_band)

    _df_cache["mtime"] = mtime
    _df_cache["df"] = df
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    hw = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return float(center), float(max(0, center - hw)), float(min(1, center + hw))


def _loess_smooth(x, y, frac=0.3, n_out=80):
    """Simple LOESS-like smoothing via weighted moving average.

    Returns (xs, ys) arrays suitable for plotting a smooth curve.
    Not a true LOESS but avoids extra dependencies. Uses Gaussian kernel.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return x.tolist(), y.tolist()

    xs = np.linspace(x.min(), x.max(), n_out)
    ys = np.zeros(n_out)
    h = frac * (x.max() - x.min())
    if h == 0:
        return xs.tolist(), y.mean() * np.ones(n_out).tolist()
    for i, xq in enumerate(xs):
        w = np.exp(-0.5 * ((x - xq) / h) ** 2)
        w_sum = w.sum()
        if w_sum > 0:
            ys[i] = (w * y).sum() / w_sum
        else:
            ys[i] = np.nan
    return xs.tolist(), ys.tolist()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(TEMPLATE)


@app.route("/api/status")
def api_status():
    """Basic status endpoint."""
    df = _load_df()
    csv_path = DATA_DIR / "moves_features.csv"
    pgn_dir = DATA_DIR / "pgn"
    pgn_count = len(list(pgn_dir.glob("*.pgn"))) if pgn_dir.exists() else 0
    has_model = (RESULTS_DIR / "model_summary.txt").exists()

    if df is not None and has_model:
        phase = "complete"
    elif df is not None:
        phase = "analyzing"
    elif pgn_count > 0:
        phase = "processing"
    else:
        phase = "waiting"

    return jsonify({
        "phase": phase,
        "n_moves": len(df) if df is not None else 0,
        "n_games": int(df["game_id"].nunique()) if df is not None else 0,
        "pgn_files": pgn_count,
    })


@app.route("/api/figure/<filename>")
def api_figure(filename):
    """Serve a figure as base64 JSON."""
    if not filename.endswith(".png") or "/" in filename or ".." in filename:
        return jsonify({"error": "invalid filename"}), 400
    fig_dir = (RESULTS_DIR / "figures").resolve()
    fig_path = (fig_dir / filename).resolve()
    if not str(fig_path).startswith(str(fig_dir)):
        return jsonify({"error": "invalid filename"}), 400
    if not fig_path.exists():
        return jsonify({"error": "not found"}), 404
    with open(fig_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return jsonify({"data": b64, "filename": filename})


# ===================================================================
# PANEL 1: Exploratory Landscape
# ===================================================================

@app.route("/api/panel1")
def api_panel1():
    """Panel 1 data: blunder rate by ELO/piece, hang vs punishment rates."""
    df = _load_df()
    if df is None or len(df) < 50:
        return jsonify({"ready": False})

    valid = df.dropna(subset=["elo_band"]).copy()

    # --- Chart A: Blunder rate by ELO, smooth curves per piece type ---
    # Group into ELO bins of width 50 for smooth curves
    valid["elo_bin"] = (valid["player_elo"] // 50) * 50

    piece_curves = {}
    for pt in PIECE_TYPES:
        pt_df = valid[valid["piece_type_moved"] == pt]
        if len(pt_df) < 50:
            continue
        # Aggregate blunder rate per elo_bin
        agg = pt_df.groupby("elo_bin")["is_blunder"].agg(["mean", "count"])
        agg = agg[agg["count"] >= 5]
        if len(agg) < 5:
            continue
        xs, ys = _loess_smooth(agg.index.values, agg["mean"].values, frac=0.25)
        piece_curves[pt] = {"elo": xs, "rate": [round(v, 5) for v in ys]}

    # --- Chart B: Defended vs Undefended blunder rate by ELO ---
    defended_curve = {}
    undefended_curve = {}
    if "piece_is_defended" in valid.columns:
        for label, mask, curve_dict in [
            ("defended", valid["piece_is_defended"] == 1, defended_curve),
            ("undefended", valid["piece_is_defended"] == 0, undefended_curve),
        ]:
            sub = valid[mask]
            if len(sub) < 50:
                continue
            agg = sub.groupby("elo_bin")["is_blunder"].agg(["mean", "count"])
            agg = agg[agg["count"] >= 5]
            if len(agg) >= 5:
                xs, ys = _loess_smooth(agg.index.values, agg["mean"].values, frac=0.25)
                curve_dict["elo"] = xs
                curve_dict["rate"] = [round(v, 5) for v in ys]

    return jsonify({
        "ready": True,
        "n_moves": len(valid),
        "piece_curves": piece_curves,
        "defended_curve": defended_curve,
        "undefended_curve": undefended_curve,
    })


# ===================================================================
# PANEL 2: Hazard Model
# ===================================================================

@app.route("/api/panel2")
def api_panel2():
    """Panel 2 data: survival curves and blunder arrival scatter."""
    df = _load_df()
    if df is None or len(df) < 50:
        return jsonify({"ready": False})

    valid = df.dropna(subset=["elo_band", "game_id"]).copy()

    # --- Chart A: Kaplan-Meier survival curves ---
    # "Time to first blunder" per game per color, measured in move number
    # For each game+color, find the first move where is_blunder==1
    blunders = valid[valid["is_blunder"] == 1]
    first_blunder = blunders.groupby(["game_id", "player_color"])["move_number"].min().reset_index()
    first_blunder.columns = ["game_id", "player_color", "first_blunder_move"]

    # Get max move per game+color (for censoring)
    max_move = valid.groupby(["game_id", "player_color"])["move_number"].max().reset_index()
    max_move.columns = ["game_id", "player_color", "max_move"]

    # Get ELO for each game+color
    elo_info = valid.groupby(["game_id", "player_color"]).agg(
        player_elo=("player_elo", "first"),
        elo_band=("elo_band", "first"),
    ).reset_index()

    game_df = max_move.merge(elo_info, on=["game_id", "player_color"])
    game_df = game_df.merge(first_blunder, on=["game_id", "player_color"], how="left")

    # Event = 1 if blundered, 0 if censored (no blunder)
    game_df["event"] = (~game_df["first_blunder_move"].isna()).astype(int)
    game_df["time"] = game_df["first_blunder_move"].fillna(game_df["max_move"])

    # Compute KM curves per ELO band
    survival_curves = {}
    try:
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()

        for band in ELO_BANDS:
            band_df = game_df[game_df["elo_band"] == band]
            if len(band_df) < 10:
                continue
            kmf.fit(band_df["time"], event_observed=band_df["event"])
            timeline = kmf.survival_function_.index.tolist()
            sf = kmf.survival_function_.iloc[:, 0].tolist()
            # Downsample for JSON (every other point, cap at 60 points)
            step = max(1, len(timeline) // 60)
            survival_curves[band] = {
                "time": [round(t, 1) for t in timeline[::step]],
                "survival": [round(s, 4) for s in sf[::step]],
                "n": len(band_df),
            }
    except ImportError:
        # Fallback: simple empirical survival
        for band in ELO_BANDS:
            band_df = game_df[game_df["elo_band"] == band]
            if len(band_df) < 10:
                continue
            blundered = band_df[band_df["event"] == 1]["time"].sort_values()
            n = len(band_df)
            times = sorted(blundered.unique())
            surv = []
            remaining = n
            for t in times:
                events_at_t = (blundered == t).sum()
                remaining -= events_at_t
                surv.append(remaining / n)
            step = max(1, len(times) // 60)
            survival_curves[band] = {
                "time": [round(float(t), 1) for t in times[::step]],
                "survival": [round(s, 4) for s in surv[::step]],
                "n": len(band_df),
            }

    # --- Chart B: Blunder arrival vs clock remaining (scatter + LOESS) ---
    # For each game+color, first blunder: (clock_remaining, move_number)
    blunder_with_clock = blunders.dropna(subset=["clock_remaining"])
    first_bc = blunder_with_clock.groupby(
        ["game_id", "player_color"]
    ).agg(
        first_move=("move_number", "min"),
    ).reset_index()

    # Get clock at first blunder
    scatter_data = first_bc.merge(
        blunder_with_clock[["game_id", "player_color", "move_number",
                            "clock_remaining", "player_elo"]],
        left_on=["game_id", "player_color", "first_move"],
        right_on=["game_id", "player_color", "move_number"],
        how="inner",
    )

    # Add elo_band
    def _elo_band(elo):
        if pd.isna(elo):
            return None
        elo = int(elo)
        for lo, hi in [(500, 700), (700, 900), (900, 1100),
                       (1100, 1300), (1300, 1500)]:
            if lo <= elo < hi:
                return f"{lo}-{hi}"
        return None

    scatter_data["elo_band"] = scatter_data["player_elo"].apply(_elo_band)
    scatter_data = scatter_data.dropna(subset=["elo_band"])

    # Build scatter points and LOESS per band
    scatter_by_band = {}
    loess_by_band = {}
    for band in ELO_BANDS:
        bdf = scatter_data[scatter_data["elo_band"] == band]
        if len(bdf) < 5:
            continue
        # Sample down to max 200 points for JSON
        if len(bdf) > 200:
            bdf = bdf.sample(200, random_state=42)
        scatter_by_band[band] = {
            "clock": bdf["clock_remaining"].round(0).tolist(),
            "move": bdf["first_move"].tolist(),
        }
        # LOESS smooth
        if len(bdf) >= 10:
            xs, ys = _loess_smooth(
                bdf["clock_remaining"].values, bdf["first_move"].values, frac=0.4
            )
            loess_by_band[band] = {"clock": xs, "move": [round(v, 1) for v in ys]}

    # --- Clock effect crossover detection ---
    # For each ELO band, correlation between clock_remaining and is_blunder
    clock_effect = {}
    for band in ELO_BANDS:
        bdf = valid[(valid["elo_band"] == band)].dropna(subset=["clock_remaining"])
        if len(bdf) < 50:
            continue
        corr = bdf["clock_remaining"].corr(bdf["is_blunder"].astype(float))
        clock_effect[band] = round(float(corr), 4) if not np.isnan(corr) else 0.0

    return jsonify({
        "ready": True,
        "survival_curves": survival_curves,
        "scatter_by_band": scatter_by_band,
        "loess_by_band": loess_by_band,
        "clock_effect": clock_effect,
    })


# ===================================================================
# PANEL 3: Live Coefficient Plot
# ===================================================================

@app.route("/api/panel3")
def api_panel3():
    """Panel 3 data: logistic regression coefficients at a given sample size.

    Query params:
        n: sample size (int) -- subsample the data to this many rows
           if 0 or missing, uses full dataset
    """
    df = _load_df()
    if df is None or len(df) < 100:
        return jsonify({"ready": False})

    valid = df.dropna(subset=["elo_band", "clock_remaining"]).copy()

    requested_n = request.args.get("n", 0, type=int)
    total_n = len(valid)
    if requested_n > 0 and requested_n < total_n:
        valid = valid.sample(requested_n, random_state=42)

    actual_n = len(valid)
    if actual_n < 100:
        return jsonify({"ready": False})

    # Add interaction term
    valid["elo_x_clock"] = valid["player_elo"] * valid["clock_remaining"]

    features = [
        "player_elo", "clock_remaining", "move_number",
        "moves_since_piece_last_moved", "elo_x_clock",
    ]

    subset = valid.dropna(subset=features + ["is_blunder"])
    if len(subset) < 100:
        return jsonify({"ready": False})

    X = subset[features].astype(float)
    y = subset["is_blunder"].astype(int)
    X_sm = sm.add_constant(X)

    try:
        result = sm.Logit(y, X_sm).fit(disp=0, maxiter=100)
    except Exception as e:
        return jsonify({"ready": False, "error": str(e)})

    coefficients = []
    for feat in features:
        coef = float(result.params[feat])
        ci_lo = float(result.conf_int().loc[feat, 0])
        ci_hi = float(result.conf_int().loc[feat, 1])
        pval = float(result.pvalues[feat])

        # Color logic: green if CI excludes zero in predicted direction,
        # grey if CI includes zero, red if wrong sign
        expected_sign = {
            "player_elo": -1,        # higher ELO = fewer blunders
            "clock_remaining": -1,   # more time = fewer blunders
            "move_number": 1,        # later moves = more blunders
            "moves_since_piece_last_moved": 1,  # rusty pieces = more blunders
            "elo_x_clock": -1,       # interaction: clock matters MORE at higher ELO
        }.get(feat, 0)

        if ci_lo > 0 and expected_sign > 0:
            color = "green"
        elif ci_hi < 0 and expected_sign < 0:
            color = "green"
        elif (ci_lo > 0 and expected_sign < 0) or (ci_hi < 0 and expected_sign > 0):
            color = "red"
        else:
            color = "grey"

        coefficients.append({
            "name": feat,
            "coef": round(coef, 6),
            "ci_lo": round(ci_lo, 6),
            "ci_hi": round(ci_hi, 6),
            "p": round(pval, 5),
            "color": color,
        })

    # Also compute the elo:clock interaction with ELO bands separately
    # to detect WHERE clock starts mattering
    clock_by_band = {}
    for band in ELO_BANDS:
        bdf = valid[valid["elo_band"] == band].dropna(subset=["clock_remaining", "is_blunder"])
        if len(bdf) < 50:
            continue
        Xb = sm.add_constant(bdf[["clock_remaining"]].astype(float))
        yb = bdf["is_blunder"].astype(int)
        try:
            res = sm.Logit(yb, Xb).fit(disp=0, maxiter=50)
            clock_by_band[band] = {
                "coef": round(float(res.params["clock_remaining"]), 6),
                "p": round(float(res.pvalues["clock_remaining"]), 5),
                "ci_lo": round(float(res.conf_int().loc["clock_remaining", 0]), 6),
                "ci_hi": round(float(res.conf_int().loc["clock_remaining", 1]), 6),
                "n": len(bdf),
            }
        except Exception:
            pass

    return jsonify({
        "ready": True,
        "n": actual_n,
        "total_n": total_n,
        "pseudo_r2": round(float(result.prsquared), 5),
        "coefficients": coefficients,
        "clock_by_band": clock_by_band,
    })


# ---------------------------------------------------------------------------
# Dashboard launcher
# ---------------------------------------------------------------------------

def run_dashboard(host="127.0.0.1", port=5050, data_dir="data", results_dir="results"):
    """Start the dashboard server."""
    global DATA_DIR, RESULTS_DIR
    DATA_DIR = Path(data_dir)
    RESULTS_DIR = Path(results_dir)
    os.environ["DATA_DIR"] = data_dir
    os.environ["RESULTS_DIR"] = results_dir

    print(f"Dashboard: http://{host}:{port}")
    print(f"  Watching: {DATA_DIR}/ and {RESULTS_DIR}/")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chess Blunder Analysis Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    run_dashboard(args.host, args.port, args.data_dir, args.results_dir)
