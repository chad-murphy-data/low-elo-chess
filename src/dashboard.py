"""
Live dashboard — serves a localhost page that polls for pipeline progress.

Run in one terminal:  python -m src.dashboard
Run pipeline in another:  python run_pipeline.py

The dashboard reads from disk (checkpoint, CSV, PNGs, model output) and
updates automatically every few seconds.
"""

import base64
import json
import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string

from src.dashboard_template import TEMPLATE

app = Flask(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))

ELO_BANDS = ["500-700", "700-900", "900-1100", "1100-1300", "1300-1500"]


def _read_checkpoint():
    """Read collection checkpoint and return progress dict."""
    cp_path = DATA_DIR / "collection_checkpoint.json"
    if not cp_path.exists():
        return None
    try:
        with open(cp_path) as f:
            data = json.load(f)
        bands = {}
        for key, users in data.get("users_by_band", {}).items():
            bands[key] = len(users)
        return {
            "bands": bands,
            "processed_users": len(data.get("processed_users", [])),
            "total_users": sum(bands.values()),
        }
    except (json.JSONDecodeError, KeyError):
        return None


def _count_pgn_files():
    """Count downloaded PGN files."""
    pgn_dir = DATA_DIR / "pgn"
    if not pgn_dir.exists():
        return 0
    return len(list(pgn_dir.glob("*.pgn")))


def _read_dataset_stats():
    """Read basic stats from the feature CSV."""
    csv_path = DATA_DIR / "moves_features.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        # Convert booleans for aggregation
        for col in ["is_blunder", "is_mistake", "is_inaccuracy", "is_near_mate"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        stats = {
            "total_moves": len(df),
            "total_games": df["game_id"].nunique(),
            "blunder_rate": None,
            "mistake_rate": None,
            "elo_bands": {},
            "elo_stats": {},
            "piece_distribution": {},
        }

        valid = df[df["is_near_mate"] != 1.0].dropna(subset=["is_blunder"])
        if len(valid) > 0:
            stats["blunder_rate"] = round(valid["is_blunder"].mean() * 100, 1)
            stats["mistake_rate"] = round(valid["is_mistake"].mean() * 100, 1)

            # Per-band blunder rates
            def elo_band(elo):
                if pd.isna(elo):
                    return None
                elo = int(elo)
                for lo, hi in [(500, 700), (700, 900), (900, 1100),
                               (1100, 1300), (1300, 1500)]:
                    if lo <= elo < hi:
                        return f"{lo}-{hi}"
                return None

            valid = valid.copy()
            valid["elo_band"] = valid["player_elo"].apply(elo_band)
            band_stats = (
                valid.dropna(subset=["elo_band"])
                .groupby("elo_band")["is_blunder"]
                .agg(["mean", "count"])
            )
            for band in ELO_BANDS:
                if band in band_stats.index:
                    stats["elo_bands"][band] = {
                        "blunder_rate": round(band_stats.loc[band, "mean"] * 100, 1),
                        "count": int(band_stats.loc[band, "count"]),
                    }

            # ELO distribution
            elo_desc = valid["player_elo"].describe()
            stats["elo_stats"] = {
                k: round(float(v), 1) for k, v in elo_desc.items()
                if k in ["mean", "std", "min", "max", "25%", "50%", "75%"]
            }

            # Piece type distribution
            if "piece_type_moved" in valid.columns:
                piece_counts = valid["piece_type_moved"].value_counts().to_dict()
                stats["piece_distribution"] = {
                    k: int(v) for k, v in piece_counts.items()
                }

        return stats
    except Exception:
        return None


def _read_model_summary():
    """Read model summary text file."""
    path = RESULTS_DIR / "model_summary.txt"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return None


def _read_findings():
    """Read findings markdown."""
    path = RESULTS_DIR / "findings.md"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return None


def _list_figures():
    """List available figure files with base64 data for embedding."""
    fig_dir = RESULTS_DIR / "figures"
    if not fig_dir.exists():
        return []
    figures = []
    for png in sorted(fig_dir.glob("*.png")):
        figures.append({
            "name": png.stem.replace("_", " ").title(),
            "filename": png.name,
        })
    return figures


def _read_figure_b64(filename):
    """Read a figure as base64 for embedding in the page."""
    fig_path = RESULTS_DIR / "figures" / filename
    if not fig_path.exists():
        return None
    with open(fig_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


@app.route("/")
def index():
    return render_template_string(TEMPLATE)


@app.route("/api/status")
def api_status():
    """JSON endpoint polled by the dashboard."""
    checkpoint = _read_checkpoint()
    pgn_count = _count_pgn_files()
    dataset = _read_dataset_stats()
    model_summary = _read_model_summary()
    findings = _read_findings()
    figures = _list_figures()

    # Determine pipeline phase
    if model_summary and figures:
        phase = "complete"
    elif dataset:
        phase = "analyzing"
    elif pgn_count > 0:
        phase = "processing"
    elif checkpoint:
        phase = "collecting"
    else:
        phase = "waiting"

    return jsonify({
        "phase": phase,
        "collection": {
            "checkpoint": checkpoint,
            "pgn_files": pgn_count,
        },
        "dataset": dataset,
        "model_summary": model_summary,
        "findings": findings,
        "figures": figures,
    })


@app.route("/api/figure/<filename>")
def api_figure(filename):
    """Serve a figure as base64 JSON."""
    # Sanitize: only allow simple filenames ending in .png
    if not filename.endswith(".png") or "/" in filename or ".." in filename:
        return jsonify({"error": "invalid filename"}), 400
    # Resolve and verify the path stays within the figures directory
    fig_dir = (RESULTS_DIR / "figures").resolve()
    fig_path = (fig_dir / filename).resolve()
    if not str(fig_path).startswith(str(fig_dir)):
        return jsonify({"error": "invalid filename"}), 400
    b64 = _read_figure_b64(filename)
    if b64 is None:
        return jsonify({"error": "not found"}), 404
    return jsonify({"data": b64, "filename": filename})


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
